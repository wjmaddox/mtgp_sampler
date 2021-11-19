import torch
import argparse
import sys
import logging
import time

from botorch.models.contextual_multioutput import LCEMGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.acquisition.objective import LinearMCObjective
from botorch.sampling import IIDNormalSampler
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.priors import GammaPrior

from sampling_mtgps import KroneckerLCEMGP, MatheronLCEMGP

sys.path.append("../../ContextualBO/benchmarks")
from get_synthetic_problem import get_benchmark_problem


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument(
        "--problem",
        type=str,
        default="Branin2D",
        choices=["Branin2D", "Hartmann6D", "Branin1DEmbedding", "Hartmann5DEmbedding"],
    )
    parser.add_argument("--n_batch", type=int, default=50)
    parser.add_argument("--contexts", type=int, default=10)
    parser.add_argument("--mc_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_init", type=int, default=10)
    return parser.parse_args()


def optimize_acqf_and_get_candidate(acq_func, bounds, batch_size):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "init_batch_limit": 5},
    )
    # observe new values
    new_x = candidates.detach()
    return new_x


def fit_model(model):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_model(mll)
    fit_gpytorch_torch(mll, options={"maxiter": 250})
    # [print(name, par) for name, par in model.named_parameters()]


def construct_acqf(model, objective, num_samples, best_f):
    sampler = IIDNormalSampler(num_samples=num_samples)
    qEI = qExpectedImprovement(
        model=model, best_f=best_f, sampler=sampler, objective=objective,
    )
    return qEI


def main(
    seed: int = 0,
    device: str = "cuda",
    output: str = "results.pt",
    problem: str = "Branin2D",
    n_batch: int = 50,
    contexts: int = 10,
    mc_samples: int = 64,
    batch_size: int = 2,
    n_init: int = 10,
) -> None:

    logging.basicConfig(filename=output[:-3] + ".log", level=logging.DEBUG)

    torch.random.manual_seed(seed)

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device(
            "cuda:0" if torch.cuda.is_available() and device == "cuda" else "cpu"
        ),
    }

    benchmark_problem = get_benchmark_problem(problem, num_contexts=contexts)
    objective = LinearMCObjective(torch.tensor(benchmark_problem.context_weights)).to(
        tkwargs["device"]
    )

    gen_new = lambda X: -benchmark_problem.evaluation_function_vectorized(
        X.view(1, -1).repeat(contexts, 1).numpy()
    )
    gen_new_batch = lambda batch: torch.stack(
        [torch.from_numpy(gen_new(X.cpu())) for X in batch]
    ).to(batch)

    def gen_new_for_lcem(batch):
        reshaped_output = gen_new_batch(batch).view(-1, 1)
        context_dims = (
            torch.arange(contexts, **tkwargs)
            .unsqueeze(0)
            .repeat(batch.shape[0], 1)
            .view(-1, 1)
        )
        context_stacked_batch = torch.cat(
            (batch.repeat(1, contexts).view(-1, batch.shape[-1]), context_dims), dim=-1
        )
        return context_stacked_batch, reshaped_output

    bounds = torch.tensor(
        [x["bounds"] for x in benchmark_problem.base_parameters], **tkwargs
    ).t()
    bounds_w_taskdim = torch.cat(
        (bounds, torch.tensor([[0.0], [contexts]], **tkwargs)), dim=-1
    )

    init_x = (bounds[1] - bounds[0]) * torch.rand(
        n_init, bounds.shape[1], **tkwargs
    ) + bounds[0]
    init_y = gen_new_batch(init_x)

    init_y_lcem = init_y.view(-1, 1)
    context_dims = (
        torch.arange(contexts, **tkwargs)
        .unsqueeze(0)
        .repeat(init_x.shape[0], 1)
        .view(-1, 1)
    )
    init_x_lcem = torch.cat(
        (init_x.repeat(1, contexts).view(-1, init_x.shape[-1]), context_dims), dim=-1
    )

    data_dict = {}
    data_dict["lcem"] = [init_x_lcem, init_y_lcem]
    data_dict["lcem_matheron"] = [init_x, init_y]
    data_dict["lcem_kronecker"] = [init_x, init_y]

    elapsed_time = {}
    best_achieved = {}
    has_failed = {}
    for key, _ in data_dict.items():
        elapsed_time[key] = []
        best_achieved[key] = []
        has_failed[key] = False

    for step in range(n_batch):
        torch.cuda.empty_cache()

        models_dict = {}
        models_dict["lcem"] = LCEMGP(
            *data_dict["lcem"],
            task_feature=-1,
            outcome_transform=Standardize(m=init_y_lcem.shape[-1]),
            # input_transform=Normalize(d=init_x_lcem.shape[-1],bounds=bounds_w_taskdim),
        )
        mt_likelihood = MultitaskGaussianLikelihood(
            rank=0,
            num_tasks=contexts,
            # noise_constraint=Interval(1e-7, 4.0),
            noise_prior=GammaPrior(1.1, 0.05),
            has_global_noise=True,
            has_task_noise=False,
        )

        models_dict["lcem_matheron"] = MatheronLCEMGP(
            *data_dict["lcem_matheron"],
            outcome_transform=Standardize(m=init_y.shape[-1]),
            # input_transform=Normalize(d=init_x.shape[-1],bounds=bounds),
            likelihood=mt_likelihood,
        )
        mt_likelihood2 = MultitaskGaussianLikelihood(
            rank=0,
            num_tasks=contexts,
            # noise_constraint=Interval(1e-7, 4.0),
            noise_prior=GammaPrior(1.1, 0.05),
            has_global_noise=True,
            has_task_noise=False,
        )
        models_dict["lcem_kronecker"] = KroneckerLCEMGP(
            *data_dict["lcem_kronecker"],
            outcome_transform=Standardize(m=init_y.shape[-1]),
            # input_transform=Normalize(d=init_x.shape[-1],bounds=bounds),
            likelihood=mt_likelihood2,
        )

        # construct acqusition functions
        for key, model in models_dict.items():
            start = time.time()
            if not has_failed[key]:
                try:
                    fit_model(model)
                except Exception as e:
                    logging.info(e)
                    logging.info(f"{key} failed during model fitting.")
                    has_failed[key] = True

            fit_time = time.time()

            best_f = best_achieved[key][-1] if step > 0 else init_y.max()

            start_acqf = time.time()
            if not has_failed[key]:
                try:
                    acqf = construct_acqf(model, objective, mc_samples, best_f)
                    new_x = optimize_acqf_and_get_candidate(acqf, bounds, batch_size)
                except RuntimeError as e:
                    if "memory" in str(e):
                        try:
                            torch.cuda.empty_cache()
                            acqf = construct_acqf(
                                model, objective, int(mc_samples / 2), best_f
                            )
                            new_x = optimize_acqf_and_get_candidate(
                                acqf, bounds, batch_size
                            )
                        except RuntimeError as e:
                            print(
                                "Warning, model ",
                                key,
                                "has failed. Removing from operations.",
                            )
                            has_failed[key] = True

            end = time.time()

            if not has_failed[key]:
                if key != "lcem":
                    new_y = gen_new_batch(new_x)
                else:
                    new_x, new_y = gen_new_for_lcem(new_x)

                curr_x, curr_y = data_dict[key]
                data_dict[key] = [
                    torch.cat((curr_x, new_x), dim=0),
                    torch.cat((curr_y, new_y), dim=0),
                ]
                elapsed_time[key].append([end - start_acqf, fit_time - start])

        if all(value for value in has_failed.values()):
            logging.info("Ending early")
            break

        best_lcemk = objective(data_dict["lcem_kronecker"][1]).max()
        best_lcemm = objective(data_dict["lcem_matheron"][1]).max()
        # ninds = (step + 1) * batch_size + n_init
        # best_lcem = objective(data_dict["lcem"][1].reshape(ninds, contexts)).max()
        if not has_failed["lcem"]:
            ninds = (step + 1) * batch_size + n_init
            best_lcem = objective(data_dict["lcem"][1].reshape(ninds, contexts)).max()
            best_achieved["lcem"].append(best_lcem.cpu())

        if not has_failed["lcem_matheron"]:
            best_achieved["lcem_matheron"].append(best_lcemm.cpu())
        if not has_failed["lcem_kronecker"]:
            best_achieved["lcem_kronecker"].append(best_lcemk.cpu())

        last_times = [kk[-1] for _, kk in elapsed_time.items()]
        logging.info(
            f"Step: {step} Best Achieved: {best_lcem} {best_lcemm} {best_lcemk} Last Times: {last_times[0]} {last_times[1]} {last_times[2]}"
        )

    cpu_data_dict = {}
    for key, m in data_dict.items():
        cpu_data_dict[key] = [x.cpu().detach() for x in m]

    torch.save(
        {
            "time": elapsed_time,
            "data": cpu_data_dict,
            "best": best_achieved,
            "failed": has_failed,
        },
        output,
    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))
