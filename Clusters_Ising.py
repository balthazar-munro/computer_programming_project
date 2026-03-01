import numpy as np

def find_clusters(sample):
    N = sample.shape[0]
    mask = np.ones(N**2, dtype=bool)

    done = set()
    todo = set()

    cluster_sizes = []
    while mask.sum() > 0:

        # choose cluster seed and initialize the exploration queue
        s0 = np.random.choice(np.arange(N**2)[mask])
        i0 = s0//N; j0 = s0%N

        sign = sample[i0,j0] # find the sign of the spins in the cluster
        todo.add(s0)
        count = 1

        ## TODO: complete the cluster growth part algorithm


        cluster_sizes.append(count)

    return cluster_sizes
