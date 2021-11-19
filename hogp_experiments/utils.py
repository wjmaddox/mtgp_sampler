import math
import torch
import numpy as np

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
from botorch.utils.sampling import draw_sobol_samples

# from botorch_fb.models.fidelity.fidelity_utils import warmstart_initialization
from gpytorch import settings as gpt_settings
from gpytorch.mlls import ExactMarginalLogLikelihood

ZEROISH = 1e-7


def fit_model(train_X, train_Y):
    if train_X.ndim != train_Y.ndim:
        train_Y = train_Y.reshape(train_Y.shape[0], np.prod(train_Y.shape[1:]))

    model = FixedNoiseGP(
        train_X,
        train_Y,
        torch.full_like(train_Y, ZEROISH),
        # input_transform=Normalize(train_X.shape[-1]),
        outcome_transform=Standardize(train_Y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def fit_hogp_model(train_X, train_Y, latent_init="default", fixed_noise=True):
    model = HigherOrderGP(
        train_X,
        train_Y,
        outcome_transform=FlattenedStandardize(train_Y.shape[1:]),
        latent_init=latent_init,
    )
    if fixed_noise:
        model.likelihood.noise = ZEROISH
        model.likelihood.raw_noise.detach_()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # init the likelihood to be similar to ZEROISH
    # model.likelihood.noise = 1e-3
    if np.prod(train_Y.shape[1:]) > 500:
        print("Large number of output dims, using torch")
        fit_gpytorch_torch(mll, options={"lr": 0.01, "maxiter": 3000})
    else:
        fit_gpytorch_model(mll)
    return model


def gen_rand_points(bounds, num_samples):
    points_nlzd = torch.rand(num_samples, bounds.shape[-1]).to(bounds)
    return bounds[0] + (bounds[1] - bounds[0]) * points_nlzd


def get_suggested_point(
    model, bounds, objective, return_best_only=True, num_samples=32, fixed_features=None,
):
    bounds_nlzd = (bounds - bounds[0]) / (bounds[1] - bounds[0])
    sampler = IIDNormalSampler(num_samples=num_samples)

    acqf = qSimpleRegret(model, objective=objective, sampler=sampler)

    batch_initial_conditions = draw_sobol_samples(bounds=bounds, q=1, n=1)

    try:
        with gpt_settings.fast_computations(covar_root_decomposition=False):
            sugg_cand_nlzd, _ = optimize_acqf(
                acqf,
                bounds_nlzd,
                q=1,
                num_restarts=1,
                raw_samples=num_samples,
                return_best_only=return_best_only,
                batch_initial_conditions=batch_initial_conditions,
                fixed_features=fixed_features,
            )
        if return_best_only:
            sugg_cand_nlzd = sugg_cand_nlzd[0]
        return bounds[0] + (bounds[1] - bounds[0]) * sugg_cand_nlzd
    except RuntimeError:
        # if e.message == "probability tensor contains either `inf`, `nan` or element < 0":
        print("Warning: optimization failed")
        try:
            X = model.train_inputs[0]
            post_samples = sampler(model.posterior(X))
            if objective is not None:
                obj_values = objective(post_samples)
            else:
                obj_values = post_samples
            best_obj = obj_values.max(dim=(0))[0].argmax()

            if return_best_only:
                sugg_cand_nlzd = X[best_obj]
            return bounds[0] + (bounds[1] - bounds[0]) * sugg_cand_nlzd
        except RuntimeError:
            return bounds[0] + (bounds[1] - bounds[0]) * X[0]


def optimize_ei(qEI, bounds, **options):
    bounds_nlzd = (bounds - bounds[0]) / (bounds[1] - bounds[0])

    batch_initial_conditions = draw_sobol_samples(
        bounds=bounds,
        q=options["q"],
        n=options["num_restarts"] * options["options"]["batch_limit"],
    )

    try:
        with gpt_settings.fast_computations(covar_root_decomposition=False):
            cands_nlzd, _ = optimize_acqf(
                qEI,
                bounds_nlzd,
                batch_initial_conditions=batch_initial_conditions,
                **options
            )
        return bounds[0] + (bounds[1] - bounds[0]) * cands_nlzd
    except:
        print("--- Warning BO Loop failed; trying a random point instead ---")
        rand_cand = torch.rand(
            options["q"],
            qEI.model.train_inputs[0].shape[-1],
            device=bounds.device,
            dtype=bounds.dtype,
        )
        return bounds[0] + (bounds[1] - bounds[0]) * rand_cand


def optimize_kg(qKG, bounds, current_soln=None, **options):
    bounds_nlzd = (bounds - bounds[0]) / (bounds[1] - bounds[0])
    batch_initial_conditions = warmstart_initialization(
        acq_function=qKG,
        bounds=bounds_nlzd,
        q=options["q"],
        num_restarts=options.get("num_restarts"),
        raw_samples=options.get("raw_samples"),
        current_soln=current_soln,
        partial_restarts=options.get("partial_restarts"),
    )
    with gpt_settings.fast_pred_var(), gpt_settings.fast_computations(
        covar_root_decomposition=False, log_prob=False, solves=False
    ), gpt_settings.max_cholesky_size(1000):
        cands_nlzd, _ = optimize_acqf(
            acq_function=qKG,
            bounds=bounds_nlzd,
            q=options["q"],
            num_restarts=options.get("num_restarts"),
            raw_samples=options.get("raw_samples"),
            options=options.get("options", {}),
            batch_initial_conditions=batch_initial_conditions,
        )
    cands_nlzd = qKG.extract_candidates(cands_nlzd.detach().unsqueeze(0))
    return bounds[0] + (bounds[1] - bounds[0]) * cands_nlzd[0]
