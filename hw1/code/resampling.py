'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        
        ok so inputs is just N,4 particles wher each particle is (x,y,heading, weight)
        output is just same N,4 heading with the new resampled particle set
        """
        
        """
        TODO : Add your code here
        """
        # X_bar_resampled =  np.zeros_like(X_bar)
        # return X_bar_resampled
        
        # normalized weights to get probs
        w = X_bar[:, 3]
        w = w / np.sum(w)
        
        # now we sample indices
        N = X_bar.shape[0]
        indices = np.random.choice(N, size=N, p=w)
        
        # now we just index original particles
        X_bar_resampled = X_bar[indices].copy()
        
        # reset weights to uniform
        X_bar_resampled[:, 3] = 1 / N
        
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        
        same inputs and outputs as 
        """
        """
        TODO : Add your code here
        """
        # return X_bar_resampled
        
        # normalize weights 
        w = X_bar[:, 3]
        w = w / np.sum(w)
        
        # calc cum weights
        cum_weights = np.cumsum(w) # should be length N, [-1] = 1
        
        # calc random starting point
        N = X_bar.shape[0]
        r = np.random.uniform(0, 1/N)
        
        # init resampled particles
        X_bar_resampled =  np.zeros_like(X_bar)
        
        # now we step by 1/N 
        # keep track of curr particle in old particles
        curr_idx= 0
        for k in range(N):
            u = r + k/N # calc sampling point
            
            # now we have to find which particle owns this sampling point
            # so basically smallest index where weights[curr_idx] >=u
            while curr_idx < N-1 and u > cum_weights[curr_idx]: 
                curr_idx += 1
                
            # update new particles
            X_bar_resampled[k] = X_bar[curr_idx]
            
        # reset weights
        X_bar_resampled[:, 3] = 1/N
        
        return X_bar_resampled
                
            

