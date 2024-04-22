from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from itertools import combinations, product
from scipy.optimize import linprog, minimize
import pandas as pd
import scipy.spatial.distance as distance

class ReduceExact():
    """
    Reduction class to reduce scenarios by their energy score.
    Applies the exact method, which is exhaustive and evaluates the distance for all possible scenario combinations.
    """

    def __init__(self, x, cdn, w=None, p=1, dist='energy', parallel=True, COMB=None, distx=None, verbose=1):
        """
        :param x: array of scenarios
        :param cdn: number of reduced scenarios
        :param w: weights for the energy score (probabilities)
        :param p: p-norm for the energy score
        :param parallel: whether to use parallel processing (n_jobs=-1)
        """
        self.x = x
        self.cdn = cdn
        self.w = w
        self.p = p
        self.dist = dist
        self.parallel = parallel
        self.distx = distx
        self.verbose=verbose

        # Determine number of scenarios to select
        if self.cdn is None:
            self.cdn = int(np.floor(np.log(self.x.shape[0])))

        # Normalize weights
        if self.w is None:
            self.w = np.ones(self.x.shape[0])/self.x.shape[0]
        else:
            self.w = self.w/np.sum(self.w)

        if COMB is None:
            # Compute combinations of scenarios to select
            self.COMB = np.array(list(combinations(range(x.shape[0]), self.cdn))).T
        else:
            # Option to provide your own combination list to try so we can easily apply it to the forward algorithm too
            self.COMB = COMB

        if distx is None:
            if self.dist == 'energy':
                # Compute distance matrix
                self.distx = np.power(np.array(np.asmatrix(distance.cdist(self.x, self.x, metric='euclidean'))), self.p)
            elif self.dist == 'wasserstein':
                # For wasserstein keep the original scenario set
                # self.distx = self.x
                self.distx = np.power(np.array(np.asmatrix(distance.cdist(self.x, self.x, metric='euclidean'))), self.p)
        else:
            # Option to provide your own distance matrix to try so we can efficiently apply it to the forward algorithm too
            self.distx = distx


    def reduce_exact(self):
        # Compute distance matrix
        self.optz = np.zeros(self.COMB.shape[1]) # optimal value of the objective function
        self.optc = np.zeros((self.COMB.shape[1], self.cdn)) # optimal weights of the scenarios
        
        if not self.parallel:
            if self.verbose:
                iter = tqdm(range(self.optz.shape[0]))
            else:
                iter = range(self.optz.shape[0])
            for i_z in iter:
                index_array = self.COMB[:,i_z]
                out = self.get_val(index_array)
                self.optz[i_z] = out[0]
                self.optc[i_z,:] = out[1]
        else:
            index_arrays = [self.COMB[:,i_z] for i_z in range(self.optz.shape[0])]
            out = Parallel(n_jobs=-1, verbose=self.verbose)(delayed(self.get_val)(i_arr) for i_arr in index_arrays)
            for i_z in range(self.optz.shape[0]):
                self.optz[i_z] = out[i_z][0]
                self.optc[i_z,:] = out[i_z][1]

        # Return the optimal set of scenarios and their weights
        self.idx = np.argmin(self.optz)
        self.res = np.column_stack((self.COMB[:,self.idx], self.optc[self.idx,:]))
        self.res = self.res[np.argsort(self.res[:, 0]), :]
        self.res[:, 1] = self.res[:, 1]/np.sum(np.abs(self.res[:, 1]))
        self.make_df()

    def get_val(self, i_arr):
        if self.dist == 'energy':
            return self.get_energy_score_weights_(i_arr)
        elif self.dist == 'wasserstein':
            return self.get_wasserstein_distance_weights_(i_arr)

    def get_energy_score_weights_(self, i_arr, x0=None):
        # Function to minimize for each combination of scenarios to select (i_z)
        b = 2*np.dot(self.distx, self.w)
        A_z = self.distx[i_arr,:][:,i_arr]
        b_z = b[i_arr]
        
        if x0 is None:
            x0 = np.ones(self.cdn)/self.cdn
        
        bounds = [(0, 1) for i in range(self.cdn)]
        res = minimize(fun=lambda x: np.dot(x, b_z) - np.dot(x, np.dot(A_z, x)),
                        x0=x0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints={'type': 'eq', 'fun': lambda x: np.sum(x)-1})
        x_z = res.x
        obj_val = np.power(res.fun, 1/self.p)
        return [obj_val, x_z]
    
    def get_wasserstein_distance_weights_(self, index_array):
        # Get the number of elements in the distance matrix
        xn = self.distx.shape[0]
        
        # Create an index set for the complement of index_array
        J = np.delete(np.arange(xn), index_array)
        
        # Create a distance matrix for the elements in index_array and J
        d = self.distx[index_array[:, None], J]
        
        # Find the index of the minimum distance for each element in index_array
        ji = index_array[np.apply_along_axis(np.argmin, 0, d)]
        
        # Create a copy of the initial weight array
        wr = self.w.copy()
        
        # Add the weights of the elements in J to the weights of the elements in index_array
        for i, j in enumerate(ji):
            wr[j] += self.w[J[i]]
        
        # Select the initial weights of the elements in J
        www = self.w[J]
        
        # Calculate the Wasserstein distance weights by minimizing the sum of the weighted distances
        weights = [
            np.power(np.dot(www, np.apply_along_axis(np.min, 0, d)), 1/self.p),
            wr[index_array]
        ]
        return weights
        # xn = self.distx.shape[0]
        # J = np.delete(np.arange(xn), index_array)
        # d = self.distx[index_array[:,None],J]
        # ji = index_array[np.apply_along_axis(np.argmin, 0, d)]
        # wr = self.w.copy()
        # for i, j in enumerate(ji):
        #     wr[j] += self.w[J[i]]
        # www = self.w[J]
        # return [np.power(np.dot(www, np.apply_along_axis(np.min, 0, d)), 1/self.p), wr[index_array]]

    def make_df(self):
        # Make a dataframe with the optimal set of scenarios and their weights
        self.clusters = pd.DataFrame(index=range(self.res.shape[0]), columns=range(self.x.shape[1]))
        self.cluster_idx = [int(self.res[i, 0]) for i in range(self.res.shape[0])]
        self.clusters.iloc[:,:] = self.x[self.cluster_idx,:]
        self.clusters['weight'] = self.res[:, 1]


class ReduceForward():
    """
    Reduction class to reduce scenarios by their energy score.
    Applies the forward method, which is greedy and selects the scenario with the lowest energy score.
    """

    def __init__(self, x, cdn, w=None, dist='energy', p=1, parallel=True, verbose=True):
        """
        :param x: array of scenarios
        :param cdn: number of reduced scenarios
        :param parallel: whether to use parallel processing (n_jobs=-1)
        """
        self.x = x
        self.cdn = cdn
        self.w = w
        self.dist = dist
        self.p = p
        self.parallel = parallel
        self.tracker = {}
        self.weights = None
        self.reduced_df = None
        self.verbose = verbose

        # Determine number of scenarios to select
        if self.cdn is None:
            self.cdn = int(np.floor(np.log(self.x.shape[0])))

        # Normalize weights
        if self.w is None:
            self.w = np.ones(self.x.shape[0])/self.x.shape[0]
        else:
            self.w = self.w/np.sum(self.w)

        # Compute distance matrix
        self.distx = np.array(np.asmatrix(distance.cdist(self.x, self.x, metric='euclidean')))
        self.distx = np.power(self.distx, p)

    def reduce_forward(self):
        self.reduced_set = set()
        self.candidate_set =set({i for i in range(self.x.shape[0])})

        # Select the first scenario
        while len(self.reduced_set) < self.cdn:
            scenarios = self.select_scenario(return_scenarios=True)
            new = [s for s in scenarios if s not in self.reduced_set][0]
            self.reduced_set.add(new)
            self.candidate_set.remove(new)

        # Return the optimal set of scenarios and their weights
        self.res = np.column_stack((np.array(list(self.reduced_set)), self.weights))
        self.res = self.res[np.argsort(self.res[:, 0]), :]
        self.res[:, 1] = self.res[:, 1]/np.sum(np.abs(self.res[:, 1]))

        self.make_df()

    def select_scenario(self, return_scenarios=True):
        # Make a new array with the combinations of scenarios to select
        # Array rows consists of scenarios, columns consist of combinations
        # The reduced set fills the first rows, the new candidates fill the last row

        comb_array = np.zeros((len(self.reduced_set) + 1, len(self.candidate_set)))
        
        for i in range(len(self.reduced_set)):
            comb_array[i, :] = np.ones(comb_array.shape[1]) * list(self.reduced_set)[i]
        comb_array[-1, :] = np.array(list(self.candidate_set))
        comb_array = comb_array.astype(int)

        # Get the exact solution for each combination
        clusterer = ReduceExact(x=self.x,
                                cdn=comb_array.shape[0], 
                                w=self.w, 
                                p=self.p, 
                                parallel=self.parallel, 
                                COMB=comb_array, 
                                distx=self.distx,
                                dist=self.dist,
                                verbose=self.verbose)

        clusterer.reduce_exact()
        results = clusterer.res
        score = min(clusterer.optz)
        scenario_set = results[:, 0].astype(int)
        self.weights = results[:, 1]
        step = len(self.reduced_set)
        self.tracker[step] = {'scenario_set': scenario_set, 'weights': self.weights, 'score': score}

        if return_scenarios:
            return scenario_set
        
    def make_df(self):
        # Make a dataframe with the optimal set of scenarios and their weights
        self.clusters = pd.DataFrame(index=range(self.res.shape[0]), columns=range(self.x.shape[1]))
        self.cluster_idx = [int(self.res[i, 0]) for i in range(self.res.shape[0])]
        self.clusters.iloc[:,:] = self.x[self.cluster_idx,:]
        self.clusters['weight'] = self.res[:, 1]

