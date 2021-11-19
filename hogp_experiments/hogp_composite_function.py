#!/usr/bin/env python3

r"""
Implements a generalized version (added batch evaluation, pending points, KG)
of what is described in the following paper:

R. Astudillo and P. Frazier. Bayesian optimization of composite functions.
ICML 2019
"""

import logging
import math
import time
import warnings
from typing import Dict, List

import numpy as np
import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import qExpectedImprovement, qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, HigherOrderGP
from botorch.models.higher_order_gp import FlattenedStandardize
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_torch
from botorch.sampling import IIDNormalSampler
from botorch.utils.transforms import normalize

# from botorch_fb.models.fidelity.fidelity_utils import warmstart_initialization
from gpytorch import settings as gpt_settings
from gpytorch.mlls import ExactMarginalLogLikelihood

from data import prepare_data
from parser import parse
from utils import (
    fit_model,
    fit_hogp_model,
    gen_rand_points,
    get_suggested_point,
    optimize_ei,
    optimize_kg,
)

ZEROISH = 1e-7


def main(
    seed: int = 0,
    n_trials: int = 50,
    n_init: int = 5,
    n_batches: int = 10,
    batch_size: int = 3,
    num_fantasies: int = 64,
    num_restarts: int = 16,
    mc_samples: int = 256,
    raw_samples: int = 512,
    partial_restarts: int = 12,
    batch_limit: int = 4,
    init_batch_limit: int = 8,
    maxiter: int = 200,
    s_size: int = 3,
    t_size: int = 4,
    device: str = "cpu",
    problem: str = "environmental",
    max_cf_batch_size: int = 100,
    output: str = None,
) -> List[Dict[str, List[float]]]:
    runner = run_trial_cpu if device == "cpu" else run_trial_gpu

    logging.basicConfig(filename=output[:-3] + ".log", level=logging.DEBUG)

    return [
        runner(
            seed=seed,
            n_init=n_init,
            n_batches=n_batches,
            batch_size=batch_size,
            num_fantasies=num_fantasies,
            num_restarts=num_restarts,
            mc_samples=mc_samples,
            raw_samples=raw_samples,
            partial_restarts=partial_restarts,
            batch_limit=batch_limit,
            maxiter=maxiter,
            problem=problem,
            init_batch_limit=init_batch_limit,
            s_size=s_size,
            t_size=t_size,
            max_cf_batch_size=max_cf_batch_size,
        )
        for _ in range(n_trials)
    ]


def run_trial_cpu(
    n_init: int,
    n_batches: int,
    batch_size: int,
    seed: int = 0,
    num_fantasies: int = 32,
    num_restarts: int = 8,
    mc_samples: int = 256,
    raw_samples: int = 512,
    partial_restarts: int = 6,
    batch_limit: int = 4,
    init_batch_limit: int = 8,
    maxiter: int = 200,
    s_size: int = 3,
    t_size: int = 4,
    problem: str = "environmental",
    max_cf_batch_size: int = 100,
):
    return run_trial(
        n_init=n_init,
        n_batches=n_batches,
        batch_size=batch_size,
        num_fantasies=num_fantasies,
        num_restarts=num_restarts,
        mc_samples=mc_samples,
        raw_samples=raw_samples,
        partial_restarts=partial_restarts,
        batch_limit=batch_limit,
        maxiter=maxiter,
        s_size=s_size,
        t_size=t_size,
        problem=problem,
        init_batch_limit=init_batch_limit,
        max_cf_batch_size=max_cf_batch_size,
    )


def run_trial_gpu(
    n_init: int,
    n_batches: int,
    batch_size: int,
    seed: int = 0,
    num_fantasies: int = 32,
    num_restarts: int = 8,
    mc_samples: int = 256,
    raw_samples: int = 512,
    partial_restarts: int = 6,
    batch_limit: int = 4,
    maxiter: int = 200,
    s_size: int = 3,
    t_size: int = 4,
    problem: str = "environmental",
    init_batch_limit: int = 8,
    max_cf_batch_size: int = 100,
):
    return run_trial(
        seed=seed,
        n_init=n_init,
        n_batches=n_batches,
        batch_size=batch_size,
        num_fantasies=num_fantasies,
        num_restarts=num_restarts,
        mc_samples=mc_samples,
        raw_samples=raw_samples,
        partial_restarts=partial_restarts,
        batch_limit=batch_limit,
        maxiter=maxiter,
        s_size=s_size,
        t_size=t_size,
        device="cuda",
        problem=problem,
        init_batch_limit=init_batch_limit,
        max_cf_batch_size=max_cf_batch_size,
    )


def run_trial(
    n_init: int,
    n_batches: int,
    batch_size: int,
    seed: int = 0,
    num_fantasies: int = 32,
    num_restarts: int = 8,
    mc_samples: int = 256,
    raw_samples: int = 512,
    partial_restarts: int = 6,
    batch_limit: int = 4,
    init_batch_limit: int = 8,
    maxiter: int = 200,
    s_size: int = 3,
    t_size: int = 4,
    device: str = "cpu",
    problem: str = "environmental",
    max_cf_batch_size: int = 100,
) -> Dict[str, List[float]]:

    torch.random.manual_seed(seed)
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)

    device = torch.device("cpu") if device == "cpu" else torch.device("cuda:0")
    dtype = torch.float

    print("Using ", device)

    c_batched, objective, bounds, num_samples = prepare_data(
        problem, s_size, t_size, device, dtype
    )

    train_X_init = gen_rand_points(bounds, n_init)
    train_Y_init = c_batched(train_X_init)
    num_Y_outputs = np.prod(train_Y_init.shape[1:])

    # these will keep track of the points explored

    if num_Y_outputs <= max_cf_batch_size:
        models_used = (
            "rnd",
            "ei",
            "ei_hogp_cf",
            "ei_hogp_cf_smooth",
            "rnd_cf",
            "ei_cf",
        )
    else:
        logging.info(
            "--- Not using RND-CF or EI-CF because number of outputs > threshold ----"
        )
        models_used = ("rnd", "ei", "ei_hogp_cf", "ei_hogp_cf_smooth")
    train_X = {k: train_X_init.clone() for k in models_used}
    train_Y = {k: train_Y_init.clone() for k in train_X}
    suggested = {k: [] for k in train_X}

    # run the BO loop
    for i in range(n_batches):
        tic = time.time()

        # get best observations, log status
        best_f = {k: objective(v).max().detach() for k, v in train_Y.items()}

        logging.info(
            f"It {i+1:>2}/{n_batches}, best obs.: "
            ", ".join([f"{k}: {v:.3f}" for k, v in best_f.items()])
        )

        # generate random candidates
        cands = {}
        cands["rnd"] = gen_rand_points(bounds, batch_size)
        cands["rnd_cf"] = gen_rand_points(bounds, batch_size)

        optimize_acqf_kwargs = {
            "q": batch_size,
            "num_restarts": num_restarts,
            "raw_samples": raw_samples,
            "options": {
                "batch_limit": batch_limit,
                "maxiter": maxiter,
                "nonnegative": True,
                "method": "L-BFGS-B",
                "init_batch_limit": init_batch_limit,
            },
        }
        sampler = IIDNormalSampler(num_samples=mc_samples)

        model_rnd = fit_model(
            normalize(train_X["rnd"], bounds), objective(train_Y["rnd"]).unsqueeze(-1),
        )
        suggested["rnd"].append(
            get_suggested_point(model_rnd, bounds, None, num_samples=num_samples)
        )
        del model_rnd
        torch.cuda.empty_cache()

        if num_Y_outputs <= max_cf_batch_size:
            model_rnd_cf = fit_model(
                normalize(train_X["rnd_cf"], bounds), train_Y["rnd_cf"]
            )
            suggested["rnd_cf"].append(
                get_suggested_point(
                    model_rnd_cf, bounds, objective, num_samples=num_samples
                )
            )
            del model_rnd_cf
            torch.cuda.empty_cache()

        model_ei = fit_model(
            normalize(train_X["ei"], bounds), objective(train_Y["ei"]).unsqueeze(-1)
        )
        suggested["ei"].append(
            get_suggested_point(model_ei, bounds, None, num_samples=num_samples)
        )
        # generate qEI candidate (single output modeling)
        qEI = qExpectedImprovement(model_ei, best_f=best_f["ei"], sampler=sampler)
        cands["ei"] = optimize_ei(qEI, bounds, **optimize_acqf_kwargs)
        del model_ei
        torch.cuda.empty_cache()

        if num_Y_outputs <= max_cf_batch_size:
            model_ei_cf = fit_model(normalize(train_X["ei_cf"], bounds), train_Y["ei_cf"])
            suggested["ei_cf"].append(
                get_suggested_point(
                    model_ei_cf, bounds, objective, num_samples=num_samples
                )
            )
            # generate qEI candidate (multi-output modeling)
            qEI_cf = qExpectedImprovement(
                model_ei_cf, best_f=best_f["ei_cf"], sampler=sampler, objective=objective,
            )
            cands["ei_cf"] = optimize_ei(qEI_cf, bounds, **optimize_acqf_kwargs)
            del model_ei_cf
            torch.cuda.empty_cache()

        model_ei_hogp_cf = fit_hogp_model(
            normalize(train_X["ei_hogp_cf"], bounds), train_Y["ei_hogp_cf"]
        )
        suggested["ei_hogp_cf"].append(
            get_suggested_point(
                model_ei_hogp_cf, bounds, objective, num_samples=num_samples
            )
        )
        # generate qEI candidate (multi-output modeling)
        qEI_hogp_cf = qExpectedImprovement(
            model_ei_hogp_cf,
            best_f=best_f["ei_hogp_cf"],
            sampler=sampler,
            objective=objective,
        )
        cands["ei_hogp_cf"] = optimize_ei(qEI_hogp_cf, bounds, **optimize_acqf_kwargs)
        del model_ei_hogp_cf
        torch.cuda.empty_cache()

        model_ei_hogp_cf_smooth = fit_hogp_model(
            normalize(train_X["ei_hogp_cf_smooth"], bounds),
            train_Y["ei_hogp_cf_smooth"],
            "gp",
        )
        suggested["ei_hogp_cf_smooth"].append(
            get_suggested_point(
                model_ei_hogp_cf_smooth, bounds, objective, num_samples=num_samples
            )
        )
        # generate qEI candidate (multi-output modeling)
        qEI_hogp_cf_smooth = qExpectedImprovement(
            model_ei_hogp_cf_smooth,
            best_f=best_f["ei_hogp_cf_smooth"],
            sampler=sampler,
            objective=objective,
        )
        cands["ei_hogp_cf_smooth"] = optimize_ei(
            qEI_hogp_cf_smooth, bounds, **optimize_acqf_kwargs
        )
        del model_ei_hogp_cf_smooth
        torch.cuda.empty_cache()

        # fit models
        # models = {
        #     # "kg": fit_model(
        #     #     normalize(train_X["kg"], bounds), objective(train_Y["kg"]).unsqueeze(-1)
        #     # ),
        #     # "kg_cf": fit_model(normalize(train_X["kg_cf"], bounds), train_Y["kg_cf"]),
        # }

        # get suggested points (max posterior mean)

        # # get full solution for KG for warm-starting
        # soln_kg = get_suggested_point(
        #     models["kg"], bounds, None, return_best_only=False
        # )
        # soln_kg_cf = get_suggested_point(
        #     models["kg_cf"], bounds, objective, return_best_only=False
        # )
        # suggested["kg"].append(soln_kg[0, 0])
        # suggested["kg_cf"].append(soln_kg_cf[0, 0])

        # generate qKG candidate (single output modeling)
        # qKG = qKnowledgeGradient(
        #     models["kg"], num_fantasies=num_fantasies, inner_sampler=sampler
        # )
        # cands["kg"] = optimize_kg(
        #     qKG,
        #     bounds,
        #     current_soln=soln_kg,
        #     partial_restarts=partial_restarts,
        #     **optimize_acqf_kwargs,
        # )

        # # generate qEI candidate (multi-output modeling)
        # qKG_cf = qKnowledgeGradient(
        #     models["kg_cf"],
        #     num_fantasies=num_fantasies,
        #     inner_sampler=sampler,
        #     objective=objective,
        # )
        # cands["kg_cf"] = optimize_kg(
        #     qKG_cf,
        #     bounds,
        #     current_soln=soln_kg_cf,
        #     partial_restarts=partial_restarts,
        #     **optimize_acqf_kwargs,
        # )

        # make observations and update data
        for k, Xold in train_X.items():
            Xnew = cands[k]
            if Xnew.shape[0] > 0:
                train_X[k] = torch.cat([Xold, Xnew])
                train_Y[k] = torch.cat([train_Y[k], c_batched(Xnew)])

        logging.info(f"Wall time: {time.time() - tic:1f}")
        torch.cuda.empty_cache()

    sugg_vals = {}
    for k, v in suggested.items():
        if len(v) > 0:
            sugg_vals[k] = objective(c_batched(torch.stack(v))).tolist()
        else:
            sugg_vals[k] = v
    return sugg_vals, train_X


if __name__ == "__main__":
    args = parse()
    result = main(**vars(args))
    torch.save(result, args.output)
