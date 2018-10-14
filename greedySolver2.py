from multiprocessing import Pool
from os import listdir
from os.path import isfile, join
from scipy import ndimage
from texttable import Texttable
from tqdm import tqdm
import hill_climb
import multiprocessing
import numpy as np
import random
import time
import toolbox as tb

# Runs over the matrix and finds all zeros that are next to
# at least one 'one'-entry (diagonals count).
# Highly vectorised and efficient!
def mtxSearch3(mtx):
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    mtx2 = ndimage.convolve(mtx, k, mode='constant')
    mtx3 = (mtx2 <= 4.0).astype(float) * (mtx >= 0.5).astype(float)
    k2 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    mtx4 = ndimage.convolve(mtx3, k2, mode='constant')
    mtx5 = (mtx4 >= 0.5).astype(float)
    mtx6 = (mtx4 == 2.0).astype(float) * (mtx == 0.0).astype(float) * mtx5
    positionList = np.array(np.where(mtx6 == 1.0)).T
    return positionList

def greedySolveAdaptive(NodesA, NodesB, Edges, R = 2, verbose = False):
    Steplimit = max(NodesA, NodesB)
    mtx = np.zeros((NodesA, NodesB))
    for j in range(NodesA):
        mtx[j, 0] = 1.0
    for j in range(1, NodesB):
        mtx[0, j] = 1.0
    mtx[1,1] = 1.0

    EdgesLeft = (int)(round(Edges - NodesA - NodesB))
    [MT, Fill, deg_min, neg_delta, sums] = tb.init_nodf(mtx)
    old_nodf = -100.0
    for i in tqdm(range(EdgesLeft)):
        potentialPositions = mtxSearch3(mtx)
        newNODFList = []
        opt_nodf = -100
        opt_pos = [-1, -1]
        for pos in potentialPositions:
            nodf = tb.test_nodf_one_link_added(mtx, MT, Fill, deg_min, neg_delta,sums, pos)
            # print(pos[0] + 1, pos[1] + 1, nodf)
            if(nodf> opt_nodf):
                opt_pos = pos
                opt_nodf = nodf
            elif(nodf == opt_nodf):
                score_old = ((opt_pos[0] / NodesA)**2) + ((opt_pos[1] / NodesB)**2)
                score_new = ((pos[0] / NodesA)**2) + ((pos[1] / NodesB)**2)
                if(score_new < score_old):
                    opt_pos = pos
                    opt_nodf = nodf
        nodf, sums = tb.nodf_one_link_added(mtx, MT, Fill, deg_min, neg_delta, sums,opt_pos)
        if(verbose):
            print(nodf)

    return mtx

def test_multiplePositions(params):
    mtx = params[0]
    MT = params[1]
    Fill = params[2]
    deg_min = params[3]
    neg_delta = params[4]
    sums = params[5]
    posList = params[6]
    res_list = []
    for pos in posList:
        nodf = tb.test_nodf_one_link_added(mtx, MT, Fill, deg_min, neg_delta, sums, pos)
        res_list.append([pos, nodf])

    return res_list

def greedySolve(NodesA, NodesB, Edges, verbose = False):
    mtx = np.zeros((NodesA, NodesB))
    for j in range(NodesA):
        mtx[j, 0] = 1.0
    for j in range(1, NodesB):
        mtx[0, j] = 1.0
    mtx[1,1] = 1.0

    EdgesLeft = (int)(round(Edges - NodesA - (NodesB - 1) -1))
    [MT, Fill, deg_min, neg_delta, sums] = tb.init_nodf(mtx)
    nproc = multiprocessing.cpu_count()
    for i in tqdm(range(EdgesLeft)):
        potentialPositions = mtxSearch3(mtx)
        myarg = [mtx, MT, Fill, deg_min, neg_delta, sums, potentialPositions]
        res_list = test_multiplePositions(myarg)
        opt_nodf = -100
        opt_pos = [-1, -1]
        for pos, nodf in res_list:
            if(nodf > opt_nodf):
                opt_pos = pos
                opt_nodf = nodf
            elif(nodf == opt_nodf):
                score_old = (opt_pos[0] / NodesA) + (opt_pos[1] / NodesB)
                score_new = (pos[0] / NodesA) + (pos[1] / NodesB)
                if(score_new < score_old):
                    opt_pos = pos
                    opt_nodf = nodf
        #print("--->" + str(opt_pos + 1))

        a, sums = tb.nodf_one_link_added(mtx, MT, Fill, deg_min, neg_delta, sums, opt_pos)
    return mtx

def greedyAddLink(mtx, MT, F, DM, ND, S, edges_left):
    NodesA, NodesB = mtx.shape
    exp = min(2*edges_left, 64)
    pot_positions = mtxSearch3(mtx)
    nodf_list = np.zeros(len(pot_positions))
    for i in range(len(pot_positions)):
        pos = pot_positions[i]
        nodf = tb.test_nodf_one_link_added(mtx, MT, F, DM, ND, S, pos)
        nodf_list[i] = nodf**exp
    # Normalise the list:
    nodf_list = nodf_list / nodf_list.sum()
    # Sample one position:
    idx = np.random.choice(len(pot_positions), p = nodf_list)
    pos = pot_positions[idx]
    # Add the link:
    a, S = tb.nodf_one_link_added(mtx, MT, F, DM, ND, S, pos)

def greedy_gen(params):
    """
    Generate an initial graph with the intention of using it later in a
    more sophisticated algorithm like simmulated annealing or a genetic
    algorithm. This function is not deterministic. It rather tests all
    potential positions for a new entry and then adds one using the nodf
    values to create a probability distribution.
    """
    NodesA, NodesB, Edges = params
    #Initial fill
    mtx = np.zeros((NodesA, NodesB))
    mtx[0,:] = 1.0
    mtx[:,0] = 1.0
    mtx[1,1] = 1.0
    edges_left = Edges - NodesA - NodesB

    MT, F, DM, ND, S = tb.init_nodf(mtx)
    for i in range(edges_left):
        #greedy add a link
        greedyAddLink(mtx, MT, F, DM, ND, S, edges_left - i)
    return mtx

if(__name__ == "__main__"):
    NodesA = 131
    NodesB = 666
    Edges = 2933
    mtx = greedySolve(NodesA, NodesB, Edges, verbose = False)
    print(mtx.sum())
    print(tb.nodf(mtx))
