from multiprocessing import Pool
from tqdm import tqdm
import greedySolver2
import numpy as np
import toolbox as tb
import multiprocessing


def cost(nodf):
    """
    A very simple cost function
    """
    return 1.0 - nodf

def greedy_remove_link(mtx):
    """
    Removes a link to maximize the nodf metric.
    Return: The new nodf metric and if the matrix has changed
            return opt_mtx
    """
    print("Greedy remove link")
    NodesA, NodesB = mtx.shape
    opt_mtx = np.zeros((NodesA, NodesB))
    opt_nodf = -100.0
    oList = tb.get_valid_ones(mtx)
    MT, F, DM, ND, S = tb.init_nodf(mtx) # Highly optimised but awkward
    for pos in oList:
        # Compute new NODF
        new_nodf, S= tb.nodf_one_link_removed(mtx, MT, F, DM, ND, S, pos)
        if(new_nodf > opt_nodf):
            opt_mtx = np.copy(mtx)
            opt_nodf = new_nodf
        x, S = tb.nodf_one_link_added(mtx, MT, F, DM, ND, S, pos)
    return opt_mtx

def greedy_add_link(mtx):
    """
    Removes a link to maximize the nodf metric.
    Return: The new nodf metric and if the matrix has changed
            return opt_mtx
    """
    print("Greedy add link")
    NodesA, NodesB = mtx.shape
    opt_mtx = np.zeros((NodesA, NodesB))
    opt_nodf = -100.0
    posList = tb.get_promising_zeros(mtx)
    MT, F, DM, ND, S = tb.init_nodf(mtx) # Highly optimised but awkward
    for pos in posList:
        # Compute new NODF
        new_nodf, S= tb.nodf_one_link_added(mtx, MT, F, DM, ND, S, pos)
        if(new_nodf > opt_nodf):
            opt_mtx = np.copy(mtx)
            opt_nodf= new_nodf
        # Restore the matrix
        x, S = tb.nodf_one_link_removed(mtx, MT, F, DM, ND, S, pos)
    return opt_mtx

def hill_climb(mtx, depth = 2):
    NodesA, NodesB = mtx.shape
    old_nodf = -100.0
    new_nodf = tb.nodf(mtx)
    old_mtx = np.copy(mtx)
    counter = 0
    while(new_nodf > old_nodf and counter < 5):
        counter = counter + 1
        old_mtx = np.copy(mtx)
        old_nodf = new_nodf
        for i in range(depth):
            mtx = greedy_remove_link(mtx)
        for i in range(depth):
            mtx = greedy_add_link(mtx)
        new_nodf = tb.nodf(mtx)
    return old_mtx

def hill_climb_test_positions(mtx, posSwapList):
    MT, F, DM, ND, S = tb.init_nodf(mtx)
    opt_mtx = np.copy(mtx)
    opt_nodf = tb.nodf(mtx)
    for oPos, zPos in posSwapList:
        n1, S = tb.nodf_one_link_removed(mtx, MT, F, DM, ND, S, oPos)
        n1, S = tb.nodf_one_link_added(mtx, MT, F, DM, ND, S, zPos)
        if(n1 > opt_nodf):
            # Update the optimal matrix and params
            opt_mtx = np.copy(mtx)
            opt_nodf = n1
        n1, S = tb.nodf_one_link_removed(mtx, MT, F, DM, ND, S, zPos)
        a, S = tb.nodf_one_link_added(mtx, MT, F, DM, ND, S, oPos)
    return opt_mtx, opt_nodf

def hill_climb_step(mtx, R):
    NodesA, NodesB = mtx.shape
    oPosList = tb.get_valid_ones(mtx)
    # initialise nodf computations
    opt_mtx = np.copy(mtx)
    opt_nodf = tb.nodf(mtx)

    # List positions to test:
    posSwapList = []

    for oidx in range(len(oPosList)):
        oPos = oPosList[oidx]
        xpos, ypos = oPos
        for xshift, yshift in zip(range(-R, R+1), range(-R, R+1)):
            # Test if new position is valid:
            xnew = xpos + xshift
            ynew = ypos + yshift
            if(xnew >= 0 and xnew< NodesA and ynew>= 0 and ynew < NodesB):
                # Test if there is a 0
                if(mtx[xnew, ynew] == 0.0):
                    # perform the swap:
                    zPos = [xnew, ynew]
                    posSwapList.append([oPos, zPos])
    processes = multiprocessing.cpu_count()
    posLists = [[] for i in range(processes)]
    for i in range(len(posSwapList)):
        j = i % processes
        posLists[j].append(posSwapList[i])
    argList = []
    for i in range(processes):
        argList.append([np.copy(mtx), posLists[i]])

    with Pool(processes) as pool:
        resList = pool.starmap(hill_climb_test_positions, argList, chunksize = 32)

    for new_mtx, new_nodf in resList:
        if(new_nodf > opt_nodf):
            opt_nodf = new_nodf
            opt_mtx = np.copy(new_mtx)

    mtx = np.copy(opt_mtx)
    return mtx

def hill_climb_step_st(mtx, R):
    NodesA, NodesB = mtx.shape
    oPosList = tb.get_valid_ones(mtx)
    # initialise nodf computations
    opt_mtx = np.copy(mtx)
    opt_nodf = tb.nodf(mtx)

    # List positions to test:
    posSwapList = []

    for oidx in range(len(oPosList)):
        oPos = oPosList[oidx]
        xpos, ypos = oPos
        for xshift, yshift in zip(range(-R, R+1), range(-R, R+1)):
            # Test if new position is valid:
            xnew = xpos + xshift
            ynew = ypos + yshift
            if(xnew >= 0 and xnew< NodesA and ynew>= 0 and ynew < NodesB):
                # Test if there is a 0
                if(mtx[xnew, ynew] == 0.0):
                    # perform the swap:
                    zPos = [xnew, ynew]
                    posSwapList.append([oPos, zPos])
    opt_mtx, opt_nodf = hill_climb_test_positions(mtx, posSwapList)
    mtx = np.copy(opt_mtx)
    return mtx

def full_hill_climb(mtx, R = 2, multithread = False):
    NodesA, NodesB = mtx.shape
    old_nodf = -100.0
    count = 0
    while(old_nodf < tb.nodf(mtx)):
        count = count + 1
        old_nodf = tb.nodf(mtx)
        if(multithread):
            mtx = hill_climb_step(mtx, R = R)
        else:
            mtx = hill_climb_step_st(mtx, R = R)
    return mtx

if(__name__ == "__main__"):
    NodesA = 14
    NodesB = 13
    Edges = 52
    mtx1 = greedySolver2.greedySolve(NodesA, NodesB, Edges)
    mtx2 = np.copy(mtx1)
    mtx1 = full_hill_climb(mtx1, R = 2)
    mtx2 = full_hill_climb(mtx2, R = 3)
    nodf1 = tb.nodf(mtx1)
    nodf2 = tb.nodf(mtx2)
    print("NODF: {} -> {}".format(nodf1, nodf2))
