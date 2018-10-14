# pymaxnodf
Python package for finding graphs which maximise the NODF-metric.

## Installation:
Please install using

    git clone https://github.com/CHoeppke/pymaxnodf.git

The package pymaxnodf depends on

    pandas
    tqdm
    numpy
    scipy


## How to use
The goal of this project is to make a modified-simulated annealing algorithm
designed for NDOF-maximisation accessible. The algorithm is implemented in
python3. The main function of pymaxnodf is

    algSimulatedAnneal.optimise(rows, cols, links)

which may be used to compute the biadjacency matrix of a Bipartite graph
that maximises the NODF-metric. Note that the matrix which is returned by

    algSimulatedAnneal.optimise(rows, cols, links)

may not be truly optimal as simulated annealing algorithms are inherently
stochastic.


### Testing the algorithm
To test the algorithm it is recommended to call

    python3 main.py

from the command line. This will compute maximal NODF-values for 59
plant-pollinator networks from the web-of-life.es database for ecological
networks.
