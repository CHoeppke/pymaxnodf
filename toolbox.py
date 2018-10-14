from multiprocessing import Pool
from scipy import ndimage
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

# Initialises meta-data for NODF-computations
def init_nodf(mtx):
    NodesA, NodesB = mtx.shape
    norm = (NodesA*(NodesA -1) + NodesB * (NodesB - 1))/ (2.0)
    mt_0, mt_t = computeMarginalTotals(mtx)
    F0 = np.dot(mtx, mtx.T)
    Ft = np.dot(mtx.T, mtx)

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

# Helper function for the function nodf.
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

# Highly optimised method for computing the NODF-metic directly.
# Note that by definition of NODF the runtime of this method will still
# be cubic.
def nodf(mat):
    NodesA, NodesB = mat.shape
    fac = ((NodesA-1)*NodesA + (NodesB-1)*NodesB) / 2.0
    n_pairs_rows = get_paired_nestedness(mat, rows=True)
    n_pairs_cols = get_paired_nestedness(mat, rows=False)
    nodf = (n_pairs_rows + n_pairs_cols) / fac
    return nodf

# This methods facilitates calling the nodf method using multithreading.
# The default number of threads is chosen based on the number of
# threads available on the current machine.
def nodf_multithreading(mtx_list, numThreads = 0):
    with Pool(numThreads) as pool:
        res_list = pool.map(nodf, mtx_list)
    return res_list

# Computes positions of "1"-entries that when moved still yield a
# matrix on which the NODF metric is well defined.
def get_valid_ones(mtx):
    NodesA, NodesB = mtx.shape
    #mt_0, mt_t = computeMarginalTotals(mtx)
    #valid = np.outer(mt_0 > 1, mt_t > 1).astype(float)
    sub_mtx = mtx[1:, 1:]
    oList = np.where(sub_mtx == 1.0)
    myOList = (np.array(oList).T + np.array([1, 1])).tolist()
    return myOList

# Computes the "0"-entires that when flipped to a "1" state are
# most likely in improving the NODF-metric.
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

# Returns the acceptance probability of a new solution
def acceptProb(cost_old, cost_new, temp):
    if(cost_new < cost_old):
        result = 1.0
    else:
        a = -1.0*(cost_new - cost_old) / temp
        result = np.exp(a)
    return result

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

if __name__ == "__main__":
    # This script only contains helper functions and is not meant to be
    # executable
    pass
