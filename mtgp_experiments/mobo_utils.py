import torch
import argparse

from sampling_mtgps.matheron_mtgp import KroneckerMultiTaskGP, MatheronMultiTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex

from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument("--problem", type=str, default="c2dtlz2")
    parser.add_argument("--n_batch", type=int, default=50)
    parser.add_argument("--mc_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def optimize_qehvi_and_get_observation(
    problem, model, train_obj, train_con, sampler, twkwargs, standard_bounds, batch_size
):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # compute feasible observations
    n_obj = train_obj.shape[-1]
    n_con = train_con.shape[-1]

    is_feas = (train_con <= 0).all(dim=-1)
    # compute points that are better than the known reference point
    better_than_ref = (train_obj > problem.ref_point).all(dim=-1)
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(
        num_outcomes=problem.num_objectives,
        # use observations that are better than the specified reference point and feasible
        Y=train_obj[better_than_ref & is_feas],
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        # specify that the constraint is on the last outcome
        constraints=[
            lambda Z, dim=dim: Z[..., dim] for dim in range(n_obj, n_obj + n_con)
        ],
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={
            "batch_limit": 5,
            "maxiter": 200,
            "init_batch_limit": 20,
            "nonnegative": True,
        },
        sequential=True,
    )
    # observe new values
    # new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
    new_x = candidates.detach()
    new_obj = problem(new_x)
    # negative values imply feasibility in botorch
    new_con = -problem.evaluate_slack(new_x)
    return new_x, new_obj, new_con


def generate_initial_data(problem, n, method="sobol"):
    # generate training data
    if method == "sobol":
        train_x = draw_sobol_samples(
            bounds=problem.bounds, n=1, q=n, seed=torch.randint(1000000, (1,)).item()
        ).squeeze(0)
        train_obj = problem(train_x)
        train_con = -problem.evaluate_slack(train_x)
    elif method == "rejection":
        train_x = torch.zeros(0, 0)
        while train_x.shape[0] < n:
            new_x = (problem.bounds[1] - problem.bounds[0]) * torch.rand(
                n,
                problem.bounds.shape[-1],
                device=problem.bounds.device,
                dtype=problem.bounds.dtype,
            ) + problem.bounds[0]
            new_con = -problem.evaluate_slack(new_x)
            new_obj = problem(new_x)

            # keep only feasible outcomes
            if train_x.shape[0] <= n / 2:
                # negative implies feasible
                tokeep = (new_con <= 0).sum(-1) == new_con.shape[-1]
                new_x = new_x[tokeep]
                new_con = new_con[tokeep]
                new_obj = new_obj[tokeep]

                if tokeep.sum() > 0:
                    if train_x.shape[0] == 0:
                        train_x = new_x
                        train_con = new_con
                        train_obj = new_obj
                    else:
                        train_x = torch.cat((train_x, new_x), dim=0)
                        train_con = torch.cat((train_con, new_con), dim=0)
                        train_obj = torch.cat((train_obj, new_obj), dim=0)
            else:
                train_x = torch.cat((train_x, new_x), dim=0)
                train_con = torch.cat((train_con, new_con), dim=0)
                train_obj = torch.cat((train_obj, new_obj), dim=0)
        train_x, train_con, train_obj = train_x[:n], train_con[:n], train_obj[:n]

    # train_x = (problem.bounds[1] - problem.bounds[0]) * torch.rand(n, 6, **tkwargs) + problem.bounds[0]
    # negative values imply feasibility in botorch

    return train_x, train_obj, train_con


def initialize_model(train_x, train_obj, train_con, bounds=None, model_type="ind"):
    if model_type == "ind":
        gp_class = SingleTaskGP
    elif model_type == "mmtgp":
        gp_class = MatheronMultiTaskGP
    elif model_type == "mtgp":
        gp_class = KroneckerMultiTaskGP

    # define models for objective and constraint
    train_y = torch.cat([train_obj, train_con], dim=-1)
    model = gp_class(
        train_x,
        train_y,
        outcome_transform=Standardize(m=train_y.shape[-1]),
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def get_constrained_mc_objective(train_obj, train_con, scalarization):
    """Initialize a ConstrainedMCObjective for qParEGO"""
    n_obj = train_obj.shape[-1]
    n_con = train_con.shape[-1]
    # assume first outcomes of the model are the objectives, the rest constraints
    def objective(Z):
        return scalarization(Z[..., :n_obj])

    constrained_obj = ConstrainedMCObjective(
        objective=objective,
        constraints=[
            lambda Z, dim=dim: Z[..., dim] for dim in range(n_obj, n_obj + n_con)
        ],  # index the constraint
    )
    return constrained_obj


def optimize_qparego_and_get_observation(
    problem, model, train_obj, train_con, sampler, tkwargs, standard_bounds, batch_size
):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization 
    of the qParEGO acquisition function, and returns a new candidate and observation."""
    acq_func_list = []
    for _ in range(batch_size):
        # sample random weights
        weights = sample_simplex(problem.num_objectives, **tkwargs).squeeze()
        # construct augmented Chebyshev scalarization
        scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)
        # initialize ConstrainedMCObjective
        constrained_objective = get_constrained_mc_objective(
            train_obj=train_obj, train_con=train_con, scalarization=scalarization
        )
        train_y = torch.cat([train_obj, train_con], dim=-1)
        acq_func = qExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=constrained_objective,
            best_f=constrained_objective(train_y).max(),
            sampler=sampler,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "init_batch_limit": 20},
    )
    # observe new values
    # new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
    new_x = candidates.detach()
    new_obj = problem(new_x)
    # negative values imply feasibility in botorch
    new_con = -problem.evaluate_slack(new_x)
    return new_x, new_obj, new_con


def generate_problem(problem, tkwargs):
    if problem == "c2dtlz2":
        from botorch.test_functions.multi_objective import C2DTLZ2

        d = 12
        M = 2
        problem = C2DTLZ2(dim=d, num_objectives=M, negate=True).to(**tkwargs)

        standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
        standard_bounds[1] = 1
    elif problem == "osy":
        from botorch.test_functions.multi_objective import OSY

        d = 6
        M = 2
        problem = OSY(negate=True).to(**tkwargs)
        standard_bounds = problem.bounds
    elif problem == "ripple":
        from botorch.test_functions.multi_objective import SwitchRipple

        d = 6
        M = 3
        problem = SwitchRipple(negate=True).to(**tkwargs)
        standard_bounds = problem.bounds

    return problem, standard_bounds
