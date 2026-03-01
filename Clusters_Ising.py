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
        done = set()
        todo = set()
        todo.add(s0)
        count = 1

        while len(todo) > 0:
            s = todo.pop()
            i = s // N
            j = s % N
            for (di, dj) in [(0,1), (1,0), (-1,0), (0,-1)]:
                ii = (i + di) % N
                jj = (j + dj) % N
                ss = N * ii + jj
                if sample[ii, jj] != sign:
                    continue
                if ss not in done and ss not in todo and mask[ss]:
                    todo.add(ss)
                    count += 1
            done.add(s)
            mask[s] = False

        cluster_sizes.append(count)

    return cluster_sizes
