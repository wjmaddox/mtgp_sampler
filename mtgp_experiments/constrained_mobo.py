import time
import warnings
import torch
import argparse
import logging

from botorch import fit_gpytorch_model
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.optim.fit import fit_gpytorch_torch
from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from mobo_utils import (
    parse,
    optimize_qehvi_and_get_observation,
    generate_initial_data,
    initialize_model,
    get_constrained_mc_objective,
    optimize_qparego_and_get_observation,
    generate_problem,
)


def main(
    output: str = "results.pt",
    device: str = "cuda",
    problem: str = "c2dtlz2",
    seed: int = 0,
    n_batch: int = 50,
    mc_samples: int = 128,
    batch_size: int = 2,
):
    logging.basicConfig(filename=output[:-3] + ".log", level=logging.DEBUG)

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device(
            "cuda:0" if torch.cuda.is_available() and device == "cuda" else "cpu"
        ),
    }

    problem, standard_bounds = generate_problem(problem, tkwargs)
    d = problem.dim

    verbose = True
    hv = Hypervolume(ref_point=problem.ref_point)

    torch.manual_seed(seed)

    model_types = [
        "rnd",
        "qparego",
        "qparego_ind",
        "qehvi",
        "qehvi_ind",
    ]

    train_x_qparego, train_obj_qparego, train_con_qparego = generate_initial_data(
        problem, n=2 * (d + 1), method="rejection"
    )

    # compute pareto front
    is_feas = (train_con_qparego <= 0).all(dim=-1)
    feas_train_obj = train_obj_qparego[is_feas]
    pareto_mask = is_non_dominated(feas_train_obj)
    pareto_y = feas_train_obj[pareto_mask]
    # compute hypervolume

    volume = hv.compute(pareto_y)

    hvs_dict = {}
    data_dict = {}
    for m in model_types:
        # assign volume
        hvs_dict[m] = [volume]
        # assign initial data
        data_dict[m] = [train_x_qparego, train_obj_qparego, train_con_qparego]

    logging.info("Beginning trial with {n_batch} iterations")
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, n_batch + 1):
        model_dict = {}
        for m in model_types:
            # and models
            if m == "qparego" or m == "qehvi":
                model_dict[m] = initialize_model(
                    *data_dict[m], bounds=standard_bounds, model_type="mmtgp"
                )
            elif m == "qparego_ind" or m == "qehvi_ind":
                model_dict[m] = initialize_model(
                    *data_dict[m], bounds=standard_bounds, model_type="ind"
                )
        #             elif m == "qparego_mtgp" or m == "qehvi_mtgp":
        #                 model_dict[m] = initialize_model(
        #                     *data_dict[m], bounds=standard_bounds, model_type="mtgp"
        #                 )

        t0 = time.time()

        # fit the models
        for m in model_types[1:]:
            # fit_gpytorch_model(model_dict[m][0])
            fit_gpytorch_torch(model_dict[m][0], options={"maxiter": 250})

        # define sampler and get new data points
        new_points_dict = {}
        new_points_dict["rnd"] = generate_initial_data(problem, n=batch_size)
        for m in model_types[1:]:
            sampler = IIDNormalSampler(num_samples=mc_samples)
            if m[:7] == "qparego":
                optim_fn = optimize_qparego_and_get_observation
            else:
                optim_fn = optimize_qehvi_and_get_observation

            new_x, new_obj, new_con = optim_fn(
                problem,
                model_dict[m][1],
                *data_dict[m][1:],
                sampler,
                tkwargs,
                standard_bounds,
                batch_size,
            )
            new_points_dict[m] = [new_x, new_obj, new_con]
            torch.cuda.empty_cache()

        for m in model_types:
            data_dict[m][0] = torch.cat([data_dict[m][0], new_points_dict[m][0]])
            data_dict[m][1] = torch.cat([data_dict[m][1], new_points_dict[m][1]])
            data_dict[m][2] = torch.cat([data_dict[m][2], new_points_dict[m][2]])

        for m in model_types:
            _, train_obj, train_con = data_dict[m]
            # compute pareto front
            is_feas = (train_con <= 0).all(dim=-1)
            feas_train_obj = train_obj[is_feas]
            pareto_mask = is_non_dominated(feas_train_obj)
            pareto_y = feas_train_obj[pareto_mask]
            # compute feasible hypervolume
            volume = hv.compute(pareto_y)
            hvs_dict[m].append(volume)

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        del model_dict

        t1 = time.time()

        if verbose:
            logging.info(
                f"\nBatch {iteration:>2}: Hypervolume = "
                + str([[k, max(x)] for k, x in hvs_dict.items()])
                + f"time = {t1-t0:>4.2f}."  # f"({hvs_dict["rnd"]:>4.2f}, {hvs_dict["qparego"]:>4.2f})"#, {hvs_dict["qehvi"]:>4.2f}, {hvs_dict["qparego_ind"]:>4.2f}, {hvs_dict["qehvi_ind"]:>4.2f})"
            )
        else:
            logging.info(".")

    cpu_data_dict = {}
    for m in model_types:
        cpu_data_dict[m] = [x.cpu().detach() for x in data_dict[m]]

    torch.save({"hvs": hvs_dict, "data": cpu_data_dict}, output)


if __name__ == "__main__":
    args = parse()
    main(**vars(args))
