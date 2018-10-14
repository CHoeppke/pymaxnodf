from multiprocessing import Pool
from os import listdir
from os.path import isfile, join
from texttable import Texttable
from tqdm import tqdm
import csv
import greedySolver2
import hill_climb
import multiprocessing
import numpy as np
import simulatedAnnealing
import time
import toolbox

# This script contains an algorithm to optimise the NODF
# metric by computing a first estimate for the marginal totals
# and then applying a greedy hill climb approach to improve
# the result.


def optimise(NodesA, NodesB, Edges, verbose = True):
    mtx = greedySolver2.greedySolve(NodesA, NodesB, Edges)
    # mtx = hill_climb.full_hill_climb(mtx, multithread = True )
    bestMTX = simulatedAnnealing.sim_anneal_opt(mtx)
    return bestMTX

if __name__ == "__main__":
    NodesA = 131
    NodesB = 666
    Edges = 2933
    t1 = time.time()
    mtx = optimise(NodesA, NodesB, Edges)
    t2 = time.time()
    print(toolbox.nodf(mtx), t2 - t1)

