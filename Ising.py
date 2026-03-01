import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

class Ising:
    def __init__(self, N, seed = None):
        self.N = N

        ## Optionally set up the random number generator state
        if seed is not None:
            np.random.seed(seed)
    
        # s is the initial configuration vector
        # The spins are arranged over a NxN lattice
        s = np.ones((N,N), dtype=int)
        self.s = s
        self.init_config()

        pass

    ## Initialize (or reset) the current configuration
    def init_config(self):
        N = self.N
        self.s = np.random.choice([-1,1], size=(N,N))

    ## Definition of the cost function
    # Here you need to complete the function computing the cost as described in the pdf file
    # The cost function depends on the value of the spins and the values of their nearest neighbors
    def cost(self):
        s = self.s
        # Each nearest-neighbor pair counted once: right shift + down shift
        return -(np.sum(s * np.roll(s, 1, axis=0)) + np.sum(s * np.roll(s, 1, axis=1)))

    ## Propose a valid random move. 
    def propose_move(self):
        N = self.N
        move = (np.random.choice(N), np.random.choice(N))
        return move
    
    ## Modify the current configuration, accepting the proposed move
    def accept_move(self, move):
        self.s[move] *= -1

    ## Compute the extra cost of the move (new-old, negative means convenient)
    # Here you need complete the compute_delta_cost function as explained in the pdf file
    # Check the constant in front of the delta cost
    def compute_delta_cost(self, move):
        i, j = move
        N = self.N
        s = self.s
        neighbors = (s[(i-1) % N, j] + s[(i+1) % N, j] +
                     s[i, (j-1) % N] + s[i, (j+1) % N])
        return 2 * s[i, j] * neighbors
    
    ## Make an entirely independent duplicate of the current object.
    def copy(self):
        return deepcopy(self)
    
    ## The display function is used for having a graphical representation of the configurations in white and black
    def display(self):
        plt.clf()
        plt.imshow(self.s, cmap='gray', vmin=-1, vmax=1)
        plt.show()
        

