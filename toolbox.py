from NDSparseMatrix import NDSparseMatrix
from multiprocessing import Pool
from nestedness_calculator import NestednessCalculator
from scipy import ndimage
from scipy.sparse import dok_matrix
from scipy.special import comb
import itertools
import numpy as np
# This file will collect all auxiliary functions that
# are often used in other scripts

# Compute marginal totals of a matrix mtx
def computeMarginalTotals(mtx):
    NodesA, NodesB = mtx.shape
    mt_0 = np.dot(mtx, np.ones((NodesB, 1))).reshape((NodesA))
    mt_t = np.dot(np.transpose(mtx), np.ones((NodesA, 1))).reshape((NodesB))
    return [mt_0, mt_t]

# Compute the number of jumps in a seq of marginal totals
def countMTJumpcount(mt):
    counter = 0
    NodesA = len(mt)
    for i in range(1,NodesA):
        if(mt[i-1] > mt[i]):
            counter = counter + 1
    return counter

# Compute Jumps
def jumps(mt):
    Nodes = len(mt)
    myjumps = np.zeros((Nodes, Nodes))
    for i in range(Nodes):
        for j in range(i+1, Nodes):
            if(mt[i] > mt[j]):
                myjumps[i, j] = 1
    return myjumps

def isValid(mtx):
    mt_0, mt_t = computeMarginalTotals(mtx)

    valid1 = np.all(mt_0 >= 1.0)
    valid2 = np.all(mt_t >= 1.0)
    return (valid1 and valid2)

def init_nodf(mtx):
    NodesA, NodesB = mtx.shape
    norm = (NodesA*(NodesA -1) + NodesB * (NodesB - 1))/ (2.0)
    mt_0, mt_t = computeMarginalTotals(mtx)
    F0, Ft = compute_fill_factors(mtx)

    deg_mtx0 = mt_0 * np.ones_like(F0)
    deg_mtxt = mt_t * np.ones_like(Ft)

    deg_min0 = np.minimum(deg_mtx0, deg_mtx0.T)
    deg_mint = np.minimum(deg_mtxt, deg_mtxt.T)

    neg_delta0 = (mt_0 < mt_0[:, np.newaxis])
    n_pairs0 = F0[neg_delta0] / (deg_min0[neg_delta0])

    neg_deltat = (mt_t < mt_t[:, np.newaxis])
    n_pairst = Ft[neg_deltat] / (deg_mint[neg_deltat])

    sum0 = n_pairs0.sum()
    sumt = n_pairst.sum()
    # prepare result:
    MT = [mt_0, mt_t]
    Fill = [F0, Ft]
    deg_min = [deg_min0, deg_mint]
    neg_delta = [neg_delta0, neg_deltat]
    sums = [sum0, sumt]
    return [MT, Fill, deg_min, neg_delta, sums]

def get_paired_nestedness(mat, rows=True):
    if rows:
        # consider rows
        po_mat = np.dot(mat, mat.T)
        degrees = mat.sum(axis=1)
    else:
        # consider cols
        po_mat = np.dot(mat.T, mat)
        degrees = mat.sum(axis=0)

    assert len(degrees) == len(po_mat)
    neg_delta = (degrees != degrees[:, np.newaxis])
    deg_matrix = degrees * np.ones_like(po_mat)
    deg_minima = np.minimum(deg_matrix, deg_matrix.T)
    n_pairs = po_mat[neg_delta] / (2.0 * deg_minima[neg_delta])
    return n_pairs.sum()

def nodf(mat):
    NodesA, NodesB = mat.shape
    fac = ((NodesA-1)*NodesA + (NodesB-1)*NodesB) / 2.0
    n_pairs_rows = get_paired_nestedness(mat, rows=True)
    n_pairs_cols = get_paired_nestedness(mat, rows=False)
    nodf = (n_pairs_rows + n_pairs_cols) / fac
    return nodf

def nodf_multithreading(mtx_list):
    with Pool(4) as pool:
        res_list = pool.map(nodf, mtx_list)
    return res_list

# Computes if a list contains non increasing values
def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

# Computes the partial overlaps of a binary matrix
# Uses nested for loops (possible slow)
def partialOverlaps(mtx):
    A1 = np.dot(mtx, mtx.T)
    A2 = np.dot(mtx.T, mtx)
    mt_0, mt_t = computeMarginalTotals(mtx)

    sizeA1x, sizeA1y = A1.shape
    sizeA2x, sizeA2y = A2.shape
    for i in range(sizeA1x):
        for j in range(sizeA1y):
            t = max(i,j)
            A1[i,j] = A1[i,j] / mt_0[t]

    for i in range(sizeA2x):
        for j in range(sizeA2y):
            t = max(i,j)
            A2[i,j] = A2[i,j] / mt_t[t]
    return [A1, A2]

def compute_fill_factors(mtx):
    F0 = np.dot(mtx, mtx.T)
    Ft = np.dot(mtx.T, mtx)
    return [F0, Ft]

def minimumfromMTX(mtx):
    item = np.min(mtx)
    itemindex = np.where(mtx==item)
    return [item, [itemindex[0][0], itemindex[1][0]]]

def maximumfromMTX(mtx):
    item = np.max(mtx)
    itemindex = np.where(mtx==item)
    return [item, [itemindex[0][0], itemindex[1][0]]]

# swap a zero and a one entry in the matrix and compute the
# new nodf measure
def swapEntriesNODF(params):
    mtx = params[0]
    zpos = params[1]
    opos = params[2]
    #po_0 = params[3]
    #po_t = params[4]
    #mt_0 = params[5]
    #mt_t = params[6]
    mymtx = np.copy(mtx)
    mymtx[zpos[0], zpos[1]] = 1.0
    mymtx[opos[0], opos[1]] = 0.0
    if(isValidMatrix(mymtx)):
        newnodf = toolbox.nodf(mymtx)
        return newnodf
    else:
        return -100.0

def del_link_nodf(mtx, po_0, po_t, mt_0, mt_t, pos):
    pos_x, pos_y = pos
    mt_0[pos_x] = mt_0[pos_x] - 1
    mt_t[pos_y] = mt_t[pos_y] - 1
    if(mt_0[pos_x] < 0 or mt_t[pos_y] < 0):
        return -100.0
    else:
        return -1

# Get all the zero and one entry positons in the matrix
# and put them into a usable format
def getZerosAndOnes(mtx):
    zList = np.where(mtx == 0.0)
    oList = np.where(mtx == 1.0)
    myZList = np.array(zList).T
    myOList = np.array(oList).T
    return [myZList, myOList]

def get_valid_ZO_list(mtx):
    """
    Find lists containing all the zero and one positions
    of the matrix where x_idx > 0 and y_idx > 0
    """
    NodesA, NodesB = mtx.shape
    zList = np.where(mtx[1:NodesA, 1:NodesB] == 0.0)
    oList = np.where(mtx[1:NodesA, 1:NodesB] == 1.0)
    myZList = np.array(zList).T + np.array([1,1])
    myOList = np.array(oList).T + np.array([1,1])
    return [myZList, myOList]

def get_valid_ones(mtx):
    NodesA, NodesB = mtx.shape
    #mt_0, mt_t = computeMarginalTotals(mtx)
    #valid = np.outer(mt_0 > 1, mt_t > 1).astype(float)
    sub_mtx = mtx[1:, 1:]
    oList = np.where(sub_mtx == 1.0)
    myOList = (np.array(oList).T + np.array([1, 1])).tolist()
    return myOList

def get_valid_zeros(mtx):
    zList = np.where(mtx == 0.0)
    myZList = (np.array(zList).T).tolist()
    return myZList

def get_promising_zeros(mtx, R=3):
    """
    Runs over the matrix and finds all indices of zeros that
    are neighboring a one. The same rule as in the greedy
    algorithm is used to determine if a position is promising.
    Result: positionList
    """
    N = 2*R + 1
    k = np.ones((N, N))
    k[R, R] = 0
    mtx2 = ndimage.convolve(mtx, k, mode='constant')
    mtx6 = ((mtx2 >= 0.5).astype(float) * (mtx == 0.0)).astype(float)
    positionList = np.array(np.where(mtx6 == 1.0)).T
    return positionList.tolist()

# Swaps a zero and a one entry in a given matrix
def findNeighborMtx(mtx):
    zList = get_valid_zeros(mtx)
    oList = get_valid_ones(mtx)
    idx1 = np.random.randint(0, len(zList))
    idx2 = np.random.randint(0, len(oList))
    zEntry = zList[idx1]
    oEntry = oList[idx2]
    mtx[zEntry[0], zEntry[1]] = 1.0
    mtx[oEntry[0], oEntry[1]] = 0.0
    return mtx

# Computes all the neighbor matricies:
def findAllNeighborMtx(mtx):
    nb_list = []
    sub_mtx = mtx[1:, 1:]
    zList = np.array(np.where(sub_mtx == 0.0)).T
    oList = np.array(np.where(sub_mtx == 1.0)).T
    for i in range(len(zList)):
        for j in range(len(oList)):
            mymtx = np.copy(mtx)
            zEntry = zList[i,:] + [1,1]
            oEntry = oList[j,:] + [1,1]
            mymtx[zEntry[0], zEntry[1]] = 1.0
            mymtx[oEntry[0], oEntry[1]] = 0.0
            nb_list.append(mymtx)

def modifyMTconfig(params):
    mt_0 = params[0]
    mt_t = params[1]
    o_0 = params[2]
    t_0 = params[3]
    o_t = params[4]
    t_t = params[5]
    # Compute the new config:
    new_mt_0 = np.copy(mt_0)
    new_mt_t = np.copy(mt_t)
    new_mt_0[t_0] = new_mt_0[t_0] + 1
    new_mt_0[o_0] = new_mt_0[o_0] - 1
    new_mt_t[t_t] = new_mt_t[t_t] + 1
    new_mt_t[o_t] = new_mt_t[o_t] - 1
    # Will always return true if the class of bipartite graphs
    # corresponding to this mt configuration is not empty
    valid1 = non_increasing(new_mt_0)
    valid2 = non_increasing(new_mt_t)
    valid3 = False
    if(valid1 and valid2):
        valid3 = mt_config_feasable(new_mt_0, new_mt_t)
    valid = (valid1 and valid2 and valid3)
    result = [valid, (new_mt_0, new_mt_t)]
    return result

# Input a mt configuration.
# List of marginal total configurations that differ by a maximum of one
# in every entry from the original mt config.
def findNeighborMT(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    Edges = np.sum(mt_0)
    new_mt_0 = []
    new_mt_t = []
    neighborList = []

    # Want to get new marginal total configurations by moving a value of 1
    # from one entry of mt_0 to another. For this I created the
    # from to list
    fromToList_0 = []
    for i in range(1, len(mt_0)):
        for j in range(1, len(mt_0)):
            fromToList_0.append([i,j])
    fromToList_t = []
    for i in range(1, len(mt_t)):
        for j in range(1, len(mt_t)):
            fromToList_t.append([i,j])

    # Use multithreading to make this faster:
    parameters = []
    for (o_0, t_0) in fromToList_0:
        for (o_t, t_t) in fromToList_t:
            # Compute the new config:
            entry = [mt_0, mt_t, o_0, t_0, o_t, t_t]
            parameters.append(entry)
    with Pool(4) as pool:
        neighborList = pool.map(modifyMTconfig, parameters)
    validNeighborList = []
    for [valid, neighbor] in neighborList:
        if(valid):
            validNeighborList.append(neighbor)
    return validNeighborList

def findOneNeighborMT(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    Edges = np.sum(mt_0)
    new_mt_0 = []
    new_mt_t = []

    # Want to get new marginal total configurations by moving a value of 1
    # from one entry of mt_0 to another. For this I created the
    # from to list
    fromToList_0 = []
    for i in range(1, len(mt_0)):
        for j in range(1, len(mt_0)):
            fromToList_0.append([i,j])
    fromToList_t = []
    for i in range(1, len(mt_t)):
        for j in range(1, len(mt_t)):
            fromToList_t.append([i,j])

    valid = False
    while(not valid):
        idx1 = np.random.randint(len(fromToList_0))
        [o_0, t_0] = fromToList_0[idx1]
        idx2 = np.random.randint(len(fromToList_t))
        [o_t, t_t] = fromToList_t[idx2]
        params = [mt_0, mt_t, o_0, t_0, o_t, t_t]
        [valid1, [new_mt_0, new_mt_t]] = modifyMTconfig(params)
        valid2 = non_increasing(new_mt_0)
        valid3 = non_increasing(new_mt_t)
        valid = (valid1 and valid2 and valid3)
    return [new_mt_0, new_mt_t]

# Input a mt configuration. Will output True if and only if
# the class of bipartite graphs corresponding to the given
# marginal total configuration is not empty
def mt_config_feasable(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    Edges = int(round(np.sum(mt_0)))
    mtx = np.zeros((NodesA, NodesB))
    for i in range(NodesA):
        mtx[i,0] = 1
    for i in range(NodesB):
        mtx[0,i] = 1

    EdgesLeft = Edges - NodesA - NodesB + 1
    for i in range(EdgesLeft):
        new_mt_0, new_mt_t = computeMarginalTotals(mtx)
        mt_left_0 = mt_0 - new_mt_0
        mt_left_t = mt_t - new_mt_t
        scores = np.outer(mt_left_0, mt_left_t)
        scores = scores * (mtx  < 0.5).astype(float)
        idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        mtx[idx[0], idx[1]] = 1

    s = np.sum(mtx)
    valid1 = np.all(mt_0 > 0)
    valid2 = np.all( mt_t > 0)
    valid3 = (s == Edges)
    valid = valid1 and valid2 and valid3
    return valid

# Input an mt configuration. Procedure of checking the class
# of bipartite matricies corresponding to the given marginal
# total configuration is non empty also gives a matrix that
# has close to optimal nestedness according to the NODF metric.
# This function is computationally equivalent to calling mt_config_feasable.
def estimate_mt_quality(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    Edges = int(round(np.sum(mt_0)))
    mtx = np.zeros((NodesA, NodesB))
    for i in range(NodesA):
        mtx[i,0] = 1
    for i in range(NodesB):
        mtx[0,i] = 1

    EdgesLeft = Edges - NodesA - NodesB + 1
    for i in range(EdgesLeft):
        new_mt_0, new_mt_t = computeMarginalTotals(mtx)
        mt_left_0 = mt_0 - new_mt_0
        mt_left_t = mt_t - new_mt_t
        scores = np.outer(mt_left_0, mt_left_t)
        scores = scores * (mtx  < 0.5).astype(float)
        idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        mtx[idx[0], idx[1]] = 1

    s = np.sum(mtx)
    valid1 = np.all(mt_0 > 0)
    valid2 = np.all( mt_t > 0)
    valid3 = (s == Edges)
    valid = valid1 and valid2 and valid3
    if valid:
        return nodf(mtx)
    else:
        return -100.0

# Returns the acceptance probability of a new solution
def acceptProb(cost_old, cost_new, temp):
    if(cost_new < cost_old):
        result = 1.0
    else:
        a = -1.0*(cost_new - cost_old) / temp
        result = np.exp(a)
    return result

def upperNODFEstimate(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    Edges = np.sum(mt_0)

    estimateA = 0
    estimateB = 0
    for i in range(NodesA):
        for j in range(i+1, NodesA):
            if(mt_0[i] > mt_0[j]):
                estimateA = estimateA + 1

    for i in range(NodesB):
        for j in range(i+1, NodesB):
            if(mt_t[i] > mt_t[j]):
                estimateB = estimateB + 1

    fac = 2.0 / (NodesA *(NodesA - 1) + NodesB * (NodesB - 1))
    val = (estimateA + estimateB) * fac
    return val

def estimate_mt_quality2(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    Edges = np.sum(mt_0)
    fac_0, fac_t = computeFillFactors(mt_0, mt_t)

    estimateA = 0
    estimateB = 0
    for i in range(NodesA):
        for j in range(i+1, NodesA):
            if(mt_0[i] > mt_0[j]):
                estimateA = estimateA + fac_0[i,j]

    for i in range(NodesB):
        for j in range(i+1, NodesB):
            if(mt_t[i] > mt_t[j]):
                estimateB = estimateB + fac_t[i,j]

    fac = 2.0 / (NodesA *(NodesA - 1) + NodesB * (NodesB - 1))
    val = (estimateA + estimateB) * fac
    return val

def mixed_mt_quality_est(mt_0, mt_t):
    estimate1 = estimate_mt_quality(mt_0, mt_t)
    estimate2 = estimate_mt_quality2(mt_0, mt_t)
    estimate3 = upperNODFEstimate(mt_0, mt_t)
    est = estimate1 + estimate2 + estimate3
    return est / 3.0

def computeFillFactors(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    fill_mt_0 = np.zeros((NodesB, 1))
    fill_mt_t = np.zeros((NodesA, 1))
    for i in range(NodesB):
        fill_mt_0[i] = np.sum(mt_0 >= (i + 1))
    for i in range(NodesA):
        fill_mt_t[i] = np.sum(mt_t >= (i + 1))

    fac_0 = np.ones((NodesA, NodesA))
    fac_t = np.ones((NodesB, NodesB))
    for i in range(NodesA):
        for j in range(NodesA):
            if(fill_mt_t[i] > mt_0[i]):
                fac_0[i,j] = mt_0[i] / fill_mt_t[i]

    for i in range(NodesB):
        for j in range(NodesB):
            if(fill_mt_0[i] > mt_t[i]):
                fac_t[i,j] = mt_t[i] / fill_mt_0[i]

    return fac_0, fac_t

def list_valid_mt_configs(Nodes, Edges, maxEntry):
    if(Nodes == 1):
        if(1 <= Edges and Edges <= maxEntry):
            return [[Edges]]
        else:
            return None
    elif(Nodes > 1):
        myList = []
        for i in range(1, maxEntry + 1):
            listofsublists = list_valid_mt_configs(Nodes - 1, Edges - i, i)
            if listofsublists is not None:
                for sublist in listofsublists:
                    sublist.insert(0, i)
                    myList.append(sublist)
        return myList
    else:
        return None
    return None

# Computes the number of valid marginal total configurations
# for a given number of Nodes and Edges using dynamic programming
# and sparse matrices to save computing resources and memory
def num_valid_mt_configs(NodesA, NodesB, Edges):
    # Create the field for Dynamic programming
    Field = dok_matrix((NodesA, NodesB, Edges), dtype=np.int)

    # Initial step:
    for i in range(1, NodesB + 1):
        lim = min(i, Edges)
        for j in range(1, lim + 1):
            Field[0, i - 1, j - 1] = 1
    # Dynamic programming to avoid recursion:
    for k in range(2, NodesA + 1):
        for e in range(1, Edges + 1):
            for i in range(1, NodesB + 1):
                myVal = 0.0
                if(i >= 2):
                    myVal = myVal + Field[k-1, i-2, e-1]
                if(k - 2 >= 0 and e - i -1 >=0):
                    myVal = myVal + Field[k - 2, i - 1, e -i -1]
                Field[k-1, i-1, e-1] = np.round(myVal)
    return Field[NodesA - 1, NodesB - 1, Edges -1]

# Computes the mt configuration based on its index in the valid_mt_configs
# list. This computes the entire number of marginal totals for every
# mt configuration that is required. Note that this can be inefficient
def get_mt_config_by_idx(NodesA, NodesB, Edges, idx):
    # Create the field for Dynamic programming
    Field = NDSparseMatrix()
    # Initial step:
    for i in range(1, NodesB + 1):
        lim = min(i, Edges)
        for j in range(1, lim + 1):
            Field.addValue((0, i - 1, j - 1) , 1)
    # Dynamic programming to avoid recursion:
    for k in range(2, NodesA + 1):
        for e in range(1, Edges + 1):
            for i in range(1, NodesB + 1):
                myVal = 0.0
                if(i >= 2):
                    myVal = myVal + Field.readValue((k-1, i-2, e-1))
                if(k - 2 >= 0 and e - i -1 >=0):
                    myVal = myVal + Field.readValue((k-2, i-1, e-1-i))
                Field.addValue((k-1, i-1, e-1), np.round(myVal))
    # Now reassemble the mt configuration:
    num = Field.readValue((NodesA - 1, NodesB - 1, Edges -1))
    myIdx = idx
    cVal = NodesB
    nNodes = NodesA
    nEdges = Edges
    mt = []
    while(nNodes > 0):
        if(cVal >= 2):
            boundaryIdx = Field.readValue((nNodes - 1, cVal - 2, nEdges -1))
            if(myIdx < boundaryIdx):
                # Do not chose the current value and
                # move down one row
                cVal = cVal - 1
            else:
                # Choose the current value and move right
                mt.append(cVal)
                nNodes = nNodes - 1
                nEdges = nEdges - cVal
                myIdx = myIdx - boundaryIdx
        else:
            # This means cVal == 1
            mt.append(cVal)
            nNodes = nNodes - 1

    return mt

# The score is defined as the contribution that would be achieved by
# a matrix in satisfying the given mt config in either rows or cols
# if perfect partial overlaps could be achieved.
# This does not take into consideration the mt_t configuration.
def score(mt):
    Nodes = len(mt)
    Edges = np.sum(mt)

    Score = 0
    for i in range(Nodes):
        for j in range(i+1, Nodes):
            if(mt[i] > mt[j]):
                Score = Score + 1
    return Score

def partial_overlap_factors(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    po_fac_0 = np.zeros((NodesA, NodesA))
    po_fac_t = np.zeros((NodesB, NodesB))
    for i in range(NodesA):
        for j in range(i+1, NodesA):
            if(mt_0[i] > mt_0[j]):
                po_fac_0[i,j] = 1.0
            else:
                po_fac_0[i,j] = 0.0

    for i in range(NodesB):
        for j in range(i+1, NodesB):
            if(mt_t[i] > mt_t[j]):
                po_fac_t[i,j] = 1.0
            else:
                po_fac_t[i,j] = 0.0
    return po_fac_0, po_fac_t

def partial_overlap_effciecies(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    po_fac_0 = np.zeros((NodesA, NodesA))
    po_fac_t = np.zeros((NodesB, NodesB))
    for i in range(NodesA):
        for j in range(i+1, NodesA):
            if(mt_0[i] > mt_0[j]):
                val1 = comb(NodesB, mt_0[j])
                val2 = comb(mt_0[i], mt_0[j])
                efficiency = val2 / val1
                po_fac_0[i,j] = efficiency
            else:
                po_fac_0[i,j] = 0.0

    for i in range(NodesB):
        for j in range(i+1, NodesB):
            if(mt_t[i] > mt_t[j]):
                val1 = comb(NodesA, mt_t[j])
                val2 = comb(mt_t[i], mt_t[j])
                efficiency = val2 / val1
                po_fac_t[i,j] = efficiency
            else:
                po_fac_t[i,j] = 0.0
    return po_fac_0, po_fac_t

def compute_optimal_locks(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    numGraphs_0 = 1
    numGraphs_t = 1
    for i in range(NodesA):
        numGraphs_0 = numGraphs_0 * comb(NodesB, mt_0[i])

    for i in range(NodesB):
        numGraphs_t = numGraphs_t * comb(NodesA, mt_t[i])
    numGraphs = min(numGraphs_0, numGraphs_t)
    po_eff_0, po_eff_t = partial_overlap_effciecies(mt_0, mt_t)
    print(po_eff_0)
    print(po_eff_t)

    # Now extract row and column locks until the number of graphs
    # left is minimal
    locks_0 = np.zeros((NodesA, NodesA))
    locks_t = np.zeros((NodesB, NodesB))

    for i in range(NodesA):
        locks_0[i, i] = 1.0

    for i in range(NodesB):
        locks_t[i, i] = 1.0

    while (numGraphs > 1):
        print(numGraphs)
        [eff_0, [xidx_0, yidx_0]] =  maximumfromMTX(po_eff_0)
        [eff_t, [xidx_t, yidx_t]] =  maximumfromMTX(po_eff_t)
        if(eff_0 >= eff_t):
            po_eff_0[xidx_0, yidx_0] = 0.0
            if(numGraphs >= 1/ eff_0):
                locks_0[xidx_0, yidx_0] = 1.0
                numGraphs = numGraphs * eff_0
                old_lock = (np.matmul(locks_0, locks_0) >= 1.0).astype(float)
                new_lock = (np.matmul(old_lock, locks_0) >= 1.0).astype(float)
                while(not np.allclose(new_lock, old_lock)):
                    old_lock = new_lock
                    new_lock = (np.matmul(old_lock, locks_0) >= 1.0).astype(float)
                locks_0 = new_lock
                po_eff_0 = po_eff_0 * (np.ones((NodesA, NodesA)) - locks_0)
            else:
                numGraphs = 1
        else:
            po_eff_t[xidx_t, yidx_t] = 0.0
            if(numGraphs >= 1/ eff_t):
                locks_t[xidx_t, yidx_t] = 1.0
                numGraphs = numGraphs * eff_t
                old_lock = (np.matmul(locks_t, locks_t) >= 1.0).astype(float)
                new_lock = (np.matmul(old_lock, locks_t) >= 1.0).astype(float)
                while(not np.allclose(new_lock, old_lock)):
                    old_lock = new_lock
                    new_lock = (np.matmul(old_lock, locks_t) >= 1.0).astype(float)
                locks_t = new_lock
                po_eff_t = po_eff_t * (np.ones((NodesB, NodesB)) - locks_t)
            else:
                numGraphs = 1
    return locks_0, locks_t

def estimate_optimal_po(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    locks_0, locks_t = compute_optimal_locks(mt_0, mt_t)
    po_0 = np.zeros((NodesA, NodesA))
    po_t = np.zeros((NodesB, NodesB))
    for i in range(NodesA):
        for j in range(i+1, NodesA):
            val = (mt_0[j] - 1.0) / mt_0[j]
            po_0[i,j] = val
    for i in range(NodesB):
        for j in range(i+1, NodesB):
            val = (mt_t[j] - 1.0) / mt_t[j]
            po_t[i,j] = val

    for i in range(NodesA):
        for j in range(NodesA):
            po_0[i,j] = max(po_0[i,j], locks_0[i,j])

    for i in range(NodesB):
        for j in range(NodesB):
            po_t[i,j] = max(po_t[i,j], locks_t[i,j])

    return po_0, po_t

# Good try bu did not work as expected.
# I believe it is possible to compute exact estimates this way
# but it is again a NP Hard problem. (Exponential runtime)
def upperNODFEstimate2(mt_0, mt_t):
    NodesA = len(mt_0)
    NodesB = len(mt_t)
    Edges = np.sum(mt_0)
    po_0, po_t = estimate_optimal_po(mt_0, mt_t)

    estimateA = 0
    estimateB = 0
    for i in range(NodesA):
        for j in range(i+1, NodesA):
            if(mt_0[i] > mt_0[j]):
                estimateA = estimateA + po_0[i,j]

    for i in range(NodesB):
        for j in range(i+1, NodesB):
            if(mt_t[i] > mt_t[j]):
                estimateB = estimateB + po_t[i,j]

    fac = 2.0 / (NodesA *(NodesA - 1) + NodesB * (NodesB - 1))
    val = (estimateA + estimateB) * fac
    return val

def optimize_mtx_order(mtx):
    NodesA, NodesB = mtx.shape
    inputMtx = np.copy(mtx)
    #compute marginal totals for the rows
    MT_0 = np.matmul(mtx, np.ones((NodesB, 1))).flatten()
    arrIdxs = MT_0.argsort()
    mtx = mtx[arrIdxs[::-1],:]

    # now compute the same for the cols
    MT_T = np.matmul(np.transpose(mtx), np.ones((NodesA,1))).flatten()
    arrIdxs = MT_T.argsort()
    mtx = mtx[:,arrIdxs[::-1]]
    return mtx

# Required to make nodf_one_link_add / removed work
def get_contributions(F, neg_delta, deg_min, idx):
    A1 = F[idx, : ][neg_delta[idx,:]] / (deg_min[idx,:][neg_delta[idx, :]])
    A2 = F[: , idx][neg_delta[:,idx]] / (deg_min[:,idx][neg_delta[:, idx]])
    A3 = F[idx,idx][neg_delta[idx, idx]] / (deg_min[idx,idx][neg_delta[idx, idx]])
    return A1.sum() + A2.sum()

# Note: These functions are not self repairing any more!
def nodf_one_link_removed(mtx, MT, F, deg_min, neg_delta, sums, idx):
    """
    Efficient way to remove a link in a matrix and compute the resulting nodf value.
    Important: The user needs to ensure that mtx[idx] == 1.0.
    This method will not check for this for efficiency reasons and compute
    wrong results in case this assumption is violated.
    Note that the input parameters will be modified during the method.
    """
    mt_0, mt_t = MT
    F0, Ft = F
    deg_min0, deg_mint = deg_min
    neg_delta0, neg_deltat = neg_delta
    sum0, sumt = sums
    xidx, yidx = idx
    # compute norm
    NodesA, NodesB = mtx.shape
    norm = (NodesA*(NodesA -1) + NodesB * (NodesB - 1))/ (2.0)
    # finally modify the actual matrix:
    mtx[xidx, yidx] = 0.0
    #subtract old contribution from sum:
    old_contrib_0 = get_contributions(F0, neg_delta0, deg_min0, xidx)
    old_contrib_t = get_contributions(Ft, neg_deltat, deg_mint, yidx)
    # modify marginal totals
    mt_0[xidx] = mt_0[xidx] - 1
    mt_t[yidx] = mt_t[yidx] - 1
    # modify degree mtx:
    m0 = mt_0[xidx] * np.ones_like(mt_0)
    mt = mt_t[yidx] * np.ones_like(mt_t)
    deg_min0[xidx, :] = np.minimum(m0, mt_0)
    deg_min0[:, xidx] = np.minimum(m0, mt_0)
    deg_mint[yidx, :] = np.minimum(mt, mt_t)
    deg_mint[:, yidx] = np.minimum(mt, mt_t)
    # modify neg_deltas:
    neg_delta0[xidx, :] = (m0 > mt_0)
    neg_delta0[:, xidx] = (m0 < mt_0)
    neg_deltat[yidx, :] = (mt > mt_t)
    neg_deltat[:, yidx] = (mt < mt_t)
    # modify fill factors
    F0[:,xidx] = F0[:,xidx] - mtx[:,yidx]
    F0[xidx,:] = F0[xidx,:] - mtx[:,yidx].T
    F0[xidx,xidx] = F0[xidx,xidx] -1
    # modify fill factors
    Ft[:,yidx] = Ft[:,yidx] - mtx[xidx,:].T
    Ft[yidx,:] = Ft[yidx,:] - mtx[xidx,:]
    Ft[yidx,yidx] = Ft[yidx,yidx] - 1
    #compute new contributions
    new_contrib_0 = get_contributions(F0, neg_delta0, deg_min0, xidx)
    new_contrib_t = get_contributions(Ft, neg_deltat, deg_mint, yidx)
    # compute nodf
    sum0 = sum0 - old_contrib_0 + new_contrib_0
    sumt = sumt - old_contrib_t + new_contrib_t
    nodf = (sum0 + sumt) / norm
    # package up the results:
    MT = [mt_0, mt_t]
    F = [F0, Ft]
    deg_min = [deg_min0, deg_mint]
    neg_delta = [neg_delta0, neg_deltat]
    sums = [sum0, sumt]
    ###################
    return nodf, sums

# Note: These functions are not self repairing any more!
def nodf_one_link_added(mtx, MT, F, deg_min, neg_delta, sums, idx):
    """
    Efficient way to add a link in a matrix and compute the resulting nodf value.
    Important: The user needs to ensure that mtx[idx] == 0.0.
    This method will not check for this for efficiency reasons and compute
    wrong results in case this assumption is violated.
    Note that the input parameters will be modified during the method.
    """
    mt_0, mt_t = MT
    F0, Ft = F
    deg_min0, deg_mint = deg_min
    neg_delta0, neg_deltat = neg_delta
    sum0, sumt = sums
    xidx, yidx = idx
    # compute norm
    NodesA, NodesB = mtx.shape
    norm = (NodesA*(NodesA -1) + NodesB * (NodesB - 1))/ (2.0)
    # modify the actual matrix:
    mtx[xidx, yidx] = 1.0
    #subtract old contribution from sum:
    old_contrib_0 = get_contributions(F0, neg_delta0, deg_min0, xidx)
    old_contrib_t = get_contributions(Ft, neg_deltat, deg_mint, yidx)
    # modify marginal totals
    mt_0[xidx] = mt_0[xidx] + 1
    mt_t[yidx] = mt_t[yidx] + 1
    # modify degree mtx:
    m0 = mt_0[xidx] * np.ones_like(mt_0)
    mt = mt_t[yidx] * np.ones_like(mt_t)
    deg_min0[xidx, :] = np.minimum(m0, mt_0)
    deg_min0[:, xidx] = np.minimum(m0, mt_0)
    deg_mint[yidx, :] = np.minimum(mt, mt_t)
    deg_mint[:, yidx] = np.minimum(mt, mt_t)
    # modify neg_deltas:
    neg_delta0[xidx, :] = (m0 > mt_0)
    neg_delta0[:, xidx] = (m0 < mt_0)
    neg_deltat[yidx, :] = (mt > mt_t)
    neg_deltat[:, yidx] = (mt < mt_t)
    # modify fill factors
    F0[:,xidx] = F0[:,xidx] + mtx[:,yidx]
    F0[xidx,:] = F0[xidx,:] + mtx[:,yidx].T
    F0[xidx,xidx] = F0[xidx,xidx] - 1
    # modify fill factors
    Ft[:,yidx] = Ft[:,yidx] + mtx[xidx,:].T
    Ft[yidx,:] = Ft[yidx,:] + mtx[xidx,:]
    Ft[yidx,yidx] = Ft[yidx,yidx] - 1
    #compute new contributions
    new_contrib_0 = get_contributions(F0, neg_delta0, deg_min0, xidx)
    new_contrib_t = get_contributions(Ft, neg_deltat, deg_mint, yidx)
    # compute nodf
    sum0 = sum0 - old_contrib_0 + new_contrib_0
    sumt = sumt - old_contrib_t + new_contrib_t
    nodf = (sum0 + sumt) / norm
    # package up the results:
    MT = [mt_0, mt_t]
    F = [F0, Ft]
    deg_min = [deg_min0, deg_mint]
    neg_delta = [neg_delta0, neg_deltat]
    sums = [sum0, sumt]
    ###################
    return nodf, sums

# Note: This function is self repairing again!
def test_nodf_one_link_added(mtx, MT, F, deg_min, neg_delta, sums, idx):
    """
    Efficient way to add a link in a matrix and compute the resulting nodf value.
    Important: The user needs to ensure that mtx[idx] == 0.0.
    This method will not check for this for efficiency reasons and compute
    wrong results in case this assumption is violated.
    Note that the input parameters will be modified during the method.
    """
    mt_0, mt_t = MT
    F0, Ft = F
    deg_min0, deg_mint = deg_min
    neg_delta0, neg_deltat = neg_delta
    sum0, sumt = sums
    xidx, yidx = idx
    # compute norm
    NodesA, NodesB = mtx.shape
    norm = (NodesA*(NodesA -1) + NodesB * (NodesB - 1))/ (2.0)
    #subtract old contribution from sum:
    old_contrib_0 = get_contributions(F0, neg_delta0, deg_min0, xidx)
    old_contrib_t = get_contributions(Ft, neg_deltat, deg_mint, yidx)
    sum0 = sum0 - old_contrib_0
    sumt = sumt - old_contrib_t
    # modify marginal totals
    mt_0[xidx] = mt_0[xidx] + 1
    mt_t[yidx] = mt_t[yidx] + 1
    # modify degree mtx:
    m0 = mt_0[xidx] * np.ones_like(mt_0)
    mt = mt_t[yidx] * np.ones_like(mt_t)
    deg_min0[xidx, :] = np.minimum(m0, mt_0)
    deg_min0[:, xidx] = np.minimum(m0, mt_0)
    deg_mint[yidx, :] = np.minimum(mt, mt_t)
    deg_mint[:, yidx] = np.minimum(mt, mt_t)
    # modify neg_deltas:
    neg_delta0[xidx, :] = (m0 > mt_0)
    neg_delta0[:, xidx] = (m0 < mt_0)
    neg_deltat[yidx, :] = (mt > mt_t)
    neg_deltat[:, yidx] = (mt < mt_t)
    # modify fill factors
    F0[:,xidx] = F0[:,xidx] + mtx[:,yidx]
    F0[xidx,:] = F0[xidx,:] + mtx[:,yidx].T
    F0[xidx,xidx] = F0[xidx,xidx] -1
    # modify fill factors
    Ft[:,yidx] = Ft[:,yidx] + mtx[xidx,:].T
    Ft[yidx,:] = Ft[yidx,:] + mtx[xidx,:]
    Ft[yidx,yidx] = Ft[yidx,yidx] -1
    #compute new contributions
    new_contrib_0 = get_contributions(F0, neg_delta0, deg_min0, xidx)
    new_contrib_t = get_contributions(Ft, neg_deltat, deg_mint, yidx)
    # compute nodf
    sum0 = sum0 + new_contrib_0
    sumt = sumt + new_contrib_t
    nodf = (sum0 + sumt) / norm
    ###################
    # repair eveything:
    # modify marginal totals
    mt_0[xidx] = mt_0[xidx] - 1
    mt_t[yidx] = mt_t[yidx] - 1
    # modify degree mtx:
    m0 = mt_0[xidx] * np.ones_like(mt_0)
    mt = mt_t[yidx] * np.ones_like(mt_t)
    deg_min0[xidx, :] = np.minimum(m0, mt_0)
    deg_min0[:, xidx] = np.minimum(m0, mt_0)
    deg_mint[yidx, :] = np.minimum(mt, mt_t)
    deg_mint[:, yidx] = np.minimum(mt, mt_t)
    # modify neg_deltas:
    neg_delta0[xidx, :] = (m0 > mt_0)
    neg_delta0[:, xidx] = (m0 < mt_0)
    neg_deltat[yidx, :] = (mt > mt_t)
    neg_deltat[:, yidx] = (mt < mt_t)
    # modify fill factors
    F0[:,xidx] = F0[:,xidx] - mtx[:,yidx]
    F0[xidx,:] = F0[xidx,:] - mtx[:,yidx].T
    F0[xidx,xidx] = F0[xidx,xidx] +1
    # modify fill factors
    Ft[:,yidx] = Ft[:,yidx] - mtx[xidx,:].T
    Ft[yidx,:] = Ft[yidx,:] - mtx[xidx,:]
    Ft[yidx,yidx] = Ft[yidx,yidx] +1
    # fix the sums:
    sum0 = sum0 - new_contrib_0 + old_contrib_0
    sumt = sumt - new_contrib_t + old_contrib_t
    return nodf

def genRandGraph(NodesA, NodesB, Edges):
    graph = np.zeros((NodesA, NodesB))
    #Initial fill:
    graph[:, 0] = 1.0
    graph[0, :] = 1.0
    filled = NodesA + NodesB - 1
    # Fill the rest:
    rows = list(range(1, NodesA))
    cols = list(range(1, NodesB))
    idxes = list(itertools.product(rows, cols))
    probs = np.zeros(len(idxes))
    for i in range(len(idxes)):
        xidx, yidx = idxes[i]
        probs[i] = 1.0 / (xidx*xidx + yidx*yidx)
    probs = probs / (probs.sum())

    choices = np.random.choice(len(idxes), Edges - filled, replace = False, p = probs)
    #choices = np.random.choice(len(idxes), Edges - filled, replace = False)
    for choice in choices:
        idxX, idxY = idxes[choice]
        graph[idxX, idxY] = 1.0
    return graph

def getRandIdx(mtx, val):
    val = -1.0
    xpos = -1
    ypos = -1
    NodesA, NodesB = mtx.shape
    while(val != val):
        xpos = np.random.randint(0, NodesA)
        ypos = np.random.randint(0, NodesB)
        val = mtx[xpos, ypos]
    return [xpos, ypos]

if __name__ == "__main__":
    print("Just test the toolbox!")
