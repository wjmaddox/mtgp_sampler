import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, help="(int) seed", default=10,
    )
    parser.add_argument(
        "--n_trials", type=int, help="(int) number of initial points", default=1,
    )
    parser.add_argument(
        "--n_init", type=int, help="(int) number of initial points", default=5,
    )
    parser.add_argument(
        "--n_batches", type=int, help="(int) number of initial points", default=10,
    )
    parser.add_argument(
        "--batch_size", type=int, help="(int) number of initial points", default=3,
    )
    parser.add_argument(
        "--num_fantasies", type=int, help="(int) number of initial points", default=64,
    )
    parser.add_argument(
        "--num_restarts", type=int, help="(int) number of initial points", default=16,
    )
    parser.add_argument(
        "--mc_samples", type=int, help="(int) number of initial points", default=256,
    )
    parser.add_argument(
        "--raw_samples", type=int, help="(int) number of initial points", default=512,
    )
    parser.add_argument(
        "--partial_restarts", type=int, help="(int) number of initial points", default=12,
    )
    parser.add_argument(
        "--batch_limit", type=int, help="(int) number of initial points", default=4,
    )
    parser.add_argument(
        "--init_batch_limit", type=int, help="(int) number of initial points", default=8,
    )
    parser.add_argument(
        "--maxiter", type=int, help="(int) number of initial points", default=200,
    )
    parser.add_argument(
        "--s_size", type=int, help="(int) number of initial points", default=3,
    )
    parser.add_argument(
        "--t_size", type=int, help="(int) number of initial points", default=4,
    )
    parser.add_argument(
        "--device", type=str, help="(int) number of initial points", default="gpu",
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="(int) number of initial points",
        default="environmental",
    )
    parser.add_argument(
        "--max_cf_batch_size",
        type=int,
        help="(int) number of initial points",
        default=100,
    )
    parser.add_argument(
        "--output", type=str, help="(str) output file", default="result.pt"
    )
    return parser.parse_args()
