""" Generate realizations of a scattering spectra model estimated on snp daily prices from 2000 to 2014. 
See tutorial.ipynb from https://github.com/RudyMorel/scattering_spectra on how to set the few arguments of function 'generate' """
from pathlib import Path
import argparse

from scatspectra import SPDaily, generate


def get_args():

    parser = argparse.ArgumentParser(description='')

    # multiprocessing arguments
    parser.add_argument('-ntot', type=int, default=1, help="Total number of tasks")
    parser.add_argument('-tid', type=int, default=0, help="Task ID")

    # model arguments
    parser.add_argument('-J', type=int, default=9, help="Number of scales")
    parser.add_argument('-R', type=int, default=32768, help="Number of realizations")
    parser.add_argument('--epsilon', type=float, default=1e-2, help="Tolerance for optimization")
    parser.add_argument('--verbose', action='store_true', help="Print optimization details")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    # data
    snp = SPDaily(start='03-01-2000', end='31-12-2014')
    path = Path(__file__).parent

    # generation path
    GEN_PATH = Path(__file__).parents[1] / '_cache' / 'snp_generation'

    # generation
    x_gen = generate(
        x=snp,
        gen_log_returns=True,
        R=args.R//args.ntot,
        J=args.J,
        tol_optim=args.epsilon,
        max_iterations=1000,
        cache_path=GEN_PATH,
        verbose=args.verbose,
        load_cache=False,
        cuda=True,
    )

    print("FINISHED")
