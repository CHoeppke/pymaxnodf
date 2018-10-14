import greedySolver2
import simulatedAnnealing
import time
import toolbox

# This script contains an algorithm to optimise the NODF
# metric by computing a first estimate for the marginal totals
# and then applying a greedy hill climb approach to improve
# the result.


def optimise(NodesA, NodesB, Edges, verbose = True):
    mtx = greedySolver2.greedySolve(NodesA, NodesB, Edges)
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

