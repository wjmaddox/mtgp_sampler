import torch
import argparse
import sys
import logging
import time
import signal

from botorch.models.contextual_multioutput import LCEMGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.models import FixedNoiseMultiTaskGP
from botorch.acquisition.objective import LinearMCObjective
from botorch.sampling import IIDNormalSampler
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.priors import GammaPrior
import matplotlib.pyplot as plt
from gpytorch import settings

from sampling_mtgps import KroneckerMultiTaskGP, MatheronMultiTaskGP

sys.path.append("../../ContextualBO/benchmarks")
from get_synthetic_problem import get_benchmark_problem


# using resource
import resource
from numpy import int64


def limit_memory():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(soft, hard)
    # hard = int64(127 * (1000**9))
    # soft = int64(120 * (1000**9))
    hard = 28000000000
    soft = 20000000000
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def make_and_fit_model(x, y):
    model = KroneckerMultiTaskGP(x, y, lik_rank=0, has_second_noise=False,)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_torch(mll, options={"maxiter": 25})

    return model


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument(
        "--problem",
        type=str,
        default="Hartmann5DEmbedding",
        choices=["Branin2D", "Hartmann6D", "Branin1DEmbedding", "Hartmann5DEmbedding"],
    )
    parser.add_argument("--contexts", type=int, default=10)
    parser.add_argument("--love", action="store_true")
    parser.add_argument("--n_init", type=int, default=100)
    parser.add_argument("--nt_max", type=int, default=310)
    return parser.parse_args()


def main(
    n_init: int = 100,
    seed: int = 0,
    contexts: int = 10,
    problem="Hartmann5DEmbedding",
    love: bool = False,
    device: str = "cuda",
    output: str = "result.pt",
    nt_max: int = 310,
    # model: str = "hadamard",
):
    if love:
        max_cholesky_size = 4
        use_love = True
    else:
        max_cholesky_size = 5000
        use_love = False

    torch.random.manual_seed(seed)

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device(
            "cuda:0" if torch.cuda.is_available() and device == "cuda" else "cpu"
        ),
    }

    benchmark_problem = get_benchmark_problem(problem, num_contexts=contexts)

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

    model1 = make_and_fit_model(init_x, init_y)

    def compute_posterior_samples(test_x, orig_model, samples=256, model="kronecker"):
        torch.cuda.empty_cache()

        if model != "hadamard":
            if model == "kronecker":
                minit = KroneckerMultiTaskGP
            elif model == "matheron":
                minit = MatheronMultiTaskGP

            m = minit(init_x, init_y, lik_rank=0, has_second_noise=False,)
            m.load_state_dict(orig_model.state_dict())
        else:
            m = FixedNoiseMultiTaskGP(
                init_x_lcem,
                init_y_lcem,
                model1.likelihood._shaped_noise_covar(init_x.shape).diag().view(-1, 1),
                task_feature=-1,
            )

            # now load states
            m.task_covar_module.covar_factor.data = (
                orig_model.covar_module.task_covar_module.covar_factor.data
            )
            m.task_covar_module.var = orig_model.covar_module.task_covar_module.var.data
            m.covar_module.base_kernel.lengthscale = (
                orig_model.covar_module.data_covar_module.lengthscale.data
            )
            m.covar_module.outputscale = 1.0
            m.mean_module.constant.data = orig_model.mean_module(test_x).mean().data

        # global run
        # while run:
        try:
            start = time.time()
            res = m.posterior(test_x).rsample(torch.Size((samples,)))
            end = time.time()
            return end - start
        except Exception as e:
            print(e)
            del m
            return 1e10

        # if not run:
        #     run = True
        #     return 1e-10

    # ntest_setup = [5, 10, 20, 30, 40, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500]
    ntest_setup = torch.arange(5, nt_max, 25)
    res_list = []
    for ntest in ntest_setup:
        with settings.max_cholesky_size(max_cholesky_size), settings.fast_pred_samples(
            use_love
        ), settings.fast_computations(covar_root_decomposition=use_love):
            print(ntest)
            curr = []
            test_x = (bounds[1] - bounds[0]) * torch.rand(
                ntest, bounds.shape[1], **tkwargs
            ) + bounds[0]

            # if model == "kronecker":
            curr.append(compute_posterior_samples(test_x, model1, model="kronecker"))
            # elif model == "hadamard":
            curr.append(compute_posterior_samples(test_x, model1, model="hadamard"))
            # elif model == "matheron":
            curr.append(compute_posterior_samples(test_x, model1, model="matheron"))
            res_list.append(curr)

    torch.save(torch.tensor(res_list), output)


if __name__ == "__main__":
    args = parse()
    limit_memory()
    main(**vars(args))
