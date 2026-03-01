# This is an example on how to generate samples from the Ising model

import mcmc
import Ising
import Clusters_Ising

## Generate a problem to solve.
# This generate a Ising instance with a size of the lattice N 
N = 10

Ising = Ising.Ising(N, seed=None)

## Display initial configuration
# print("Initial configuration:")
# print(Ising.s)
# Ising.display()

## Optimize it.
s, diag = mcmc.mcmc(Ising,
                samples = 1, wait=20, burn_in=10000, beta=0.4,
                seed = None, debug_delta_cost = False) # set to True to enable the check

## Display the final configuration
# print("Final configuration:")   
# print(best)
# Ising.display()

## Analysis of the clusters
# Find the clusters of same sign spins and compute their sizes
res = Clusters_Ising.find_clusters(s[0])
