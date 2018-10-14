import pandas as pd
import algSimulatedAnneal
import toolbox as tb

# This script solves optimises the NODF-metric for the
# network considered in the paper. This purpose of this
# script is to make the algorithm accessible and reproduction
# of our results easy.

if __name__ == "__main__":
    # Read in the list of the network configurations considered.
    fname = "webs.csv"
    DF = pd.read_csv(fname)
    for index, row in DF.iterrows():
        rows = row['rows']
        cols = row['columns']
        links = row['links']
        NODFsong = row['song']
        print("Simulating web with config ({}, {}, {}).".format(rows, cols, links))
        mtx = algSimulatedAnneal.optimise(rows, cols, links)
        NODFSa = tb.nodf(mtx)
        print("NODF_song = {}\t, NODF_SA = {}.".format(NODFsong, NODFSa))
