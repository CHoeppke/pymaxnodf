from multiprocessing import Pool
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import greedySolver2
import hill_climb
import multiprocessing
import numpy as np
import pandas as pd
import simulatedAnnealing
import time

# This script solves optimises the NODF-metric for the
# network considered in the paper. This purpose of this
# script is to make the algorithm accessible and reproduction
# of our results easy.

if __name__ == "__main__":
    # Read in the list of the network configurations considered.
    fname = "webs.csv"
    DF = pd.read_csv(fname)
    print(DF)

