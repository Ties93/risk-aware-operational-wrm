import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from py_banshee.rankcorr import bn_rankcorr
from py_banshee.prediction import inference
from py_banshee.sample_bn import generate_samples
import seaborn as sns

class BN():
    
    def __init__(self, historic_data, distributionloader, R=None, parent_cell=None, names=None, varname=None, n=48, threshold=0.5):
        self.data = historic_data
        self.distributionloader = distributionloader
        self.parent_cell = parent_cell
        self.names = names
        self.n = n
        self.varname = varname
        self.threshold = threshold
        self.R = R

    def make_structure(self, method='rankcorr'):
        if not self.parent_cell:
            self.make_structure_names()

        if not self.R:
            if method=='rankcorr':
                self.R = bn_rankcorr(parent_cell=self.parent_cell, data=self.data, var_names=self.names, is_data=True, plot=False)  
            else:
                raise NotImplementedError('Only rankcorr is implemented')
        
        if self.threshold:
            self.R = np.where(self.R > self.threshold, self.R, 0)

    def make_structure_names(self):
        # Name the nodes
        names  = [self.varname]*self.n
        names = [names[i]+str(i+1) for i in range(self.n)]
        
        days=int(self.n/24)
        
        if days==1:
            # Define structure by giving each node a parent cell, except node 0.
            parent_cell = [[] if i==0 else [0, i-1] for i in range(self.n)]
            parent_cell[1] = [0]
            
        if days>1:
            # Define structure by giving each node a parent cell, except node 0.
            parent_cell = [[] if i==0 else [int(np.floor(i/24))*24, i-1] for i in range(self.n)]
            
            parent_cell[1] = [0]
            for i in range(24, self.n, 24):
                parent_cell[i] = [i-1]
            
            for i in range(24, self.n):
                parent_cell[i] =  [i-24] + parent_cell[i]
                
            parent_cell[25] = [1, 24]
        
        self.parent_cell = parent_cell
        self.names = names
        
    def get_marginals(self):
        if not self.copula:
            self.make_copula()
            
        self.marginals = {n: self.copula.getMarginal(i) for i, n in enumerate(self.names)}
    
    def get_distribution(self, index, hour):
        """Returns the distribution for the given index and hour."""
        return self.distributionloader.get_distribution(index, hour)
    
    def get_forecast_distributions(self, index):
        """Returns the distributions for the given index."""
        return [self.get_distribution(index, hour) for hour in range(1, self.n+1)]
    
    def sample_BN(self, index, N_samples=1000):
        """Samples the BN for the given index."""
        distributions, parameters = zip(*self.get_forecast_distributions(index))
        distributions = [self.distributionloader.distributions[d].name for d in distributions]
        return generate_samples(
                    R=self.R,
                    n=N_samples,
                    names=self.names,
                    data=[],
                    empirical_data=False,
                    distributions=list(distributions),
                    parameters=list(parameters)
                )
    
    def __call__(self, i, N_samples=1000):
        return self.sample_BN(self.distributionloader.index[i], N_samples)
    
    def plot_scenarios(self, index, N_samples=100, ax=None, title=None):
        """Plots the scenarios for the given index."""
        if not ax:
            fig, ax = plt.subplots(figsize=(4, 2))

        if type(index) == int:
            index = self.distributionloader.index[index]
            
        scenarios = self.sample_BN(index, N_samples)
        ax.plot(scenarios.T, lw=0.5, color='C0', alpha=0.5)
        if not title:
            ax.set_title(f'Scenarios for index {index}')
        else:
            ax.set_title(title)
        
        ax.set_xticks([i for i in np.arange(0, self.n+1, 6)] + [self.n-1])
        ax.set_xticklabels([i for i in np.arange(1, self.n+2, 6)] + [self.n])
        ax.set_xlabel('Forecast horizon [hours]')
        ax.set_xlim(0, self.n-0.9)
        ax.set_ylim(self.distributionloader.valmin, self.distributionloader.valmax)

    def plot_R(self, ax=None):
        """Plots the correlations of the BN."""
        if not ax:
            fig, ax = plt.subplots(figsize=(4, 4))

        sns.heatmap(self.R, ax=ax, cmap='RdBu_r', vmin=-1, vmax=1, square=True, cbar=False)
        ax.set_xticks(np.arange(1, len(self.names)+1, 6))
        ax.set_yticks(np.arange(1, len(self.names)+1, 6))
        ax.set_xticklabels(self.names[::6], rotation=90)
        ax.set_yticklabels(self.names[::6], rotation=0)
        
        return ax
    
    def plot_scenario_correlation(self, index, N_samples=1000, ax=None):
        """Plots the correlation of the scenarios for the given index."""
        if not ax:
            fig, ax = plt.subplots(figsize=(4, 4))

        if type(index) == int:
            index = self.distributionloader.index[index]
            
        scenarios = self.sample_BN(index, N_samples).T
        R = np.corrcoef(scenarios)
        sns.heatmap(R, ax=ax, cmap='RdBu_r', vmin=-1, vmax=1, square=True, cbar=False)
        ax.set_xticks(np.arange(1, len(self.names)+1, 6))
        ax.set_yticks(np.arange(1, len(self.names)+1, 6))
        ax.set_xticklabels(self.names[::6], rotation=90)
        ax.set_yticklabels(self.names[::6], rotation=0)
        
        return ax
    
    def plot_scenarios_R(self, index, N_samples=1000, ax=None):
        """Plots the scenarios and the correlation of the scenarios for the given index."""
        if not ax:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

        self.plot_R(ax=ax[0])
        self.plot_scenario_correlation(index, N_samples, ax=ax[1])

        fig.tight_layout()
        
        return ax
    
class IDM_BN():
    """
    Class for the IDM BN. Conditionalizes IDM prices on DAM prices.
    """
    def __init__(self, bn_data, bn_structure=None, normalize_idm=True, bn_data_is_normalized=False, pred_hor=24, R=None, mask=True, mask_method='std', mask_factor=3, mask_by_hour=False):
        self.bn_data = bn_data.copy()
        self.bn_structure = bn_structure
        self.normalize = normalize_idm
        self.pred_hor = pred_hor
        self.bn_data_is_normalized = bn_data_is_normalized

        # Normalize the data
        if self.normalize:
            self.normalize_data()
        
        # Include the current stap to the amount of prices
        self.n_dam_prices = pred_hor+1
        self.n_intraday_prices = pred_hor+1

        # Initiate the structure
        if self.bn_structure is None:
            self.make_structure()

        self.R = R
        # Initiate the BN
        if self.R is None:
            self.R = bn_rankcorr(parent_cell=self.bn_structure, data=self.bn_data, var_names=self.bn_data.columns, is_data=True)

        self.make_mask(mask, mask_factor, mask_method, mask_by_hour)
    
    def make_mask(self, mask, factor=4, how='std', mask_by_hour=False):
        # Masks the output of the ID price of the BN to 'factor' times the standard deviation of the DAM price
        if mask and not mask_by_hour:
            if how == 'std':
                mask_std = self.bn_data.loc[:, [f'IDM{h}' for h in range(self.pred_hor+1)]].values.flatten().std()
                mask_mean = self.bn_data.loc[:, [f'IDM{h}' for h in range(self.pred_hor+1)]].values.flatten().mean()

                self.mask = (mask_mean - factor*mask_std, mask_mean + factor*mask_std)
            elif how == 'minmax':
                mask_min = self.bn_data.loc[:, [f'IDM{h}' for h in range(self.pred_hor+1)]].values.flatten().min()
                mask_max = self.bn_data.loc[:, [f'IDM{h}' for h in range(self.pred_hor+1)]].values.flatten().max()

                mask_min = min(mask_min*factor, mask_min/factor)
                mask_max = max(mask_max*factor, mask_max/factor)
                self.mask = (mask_min, mask_max)
            else:
                raise ValueError(f'Unknown how: {how}')
        elif mask and mask_by_hour:
            if how == 'std':
                mask_stds = self.bn_data.loc[:, [f'IDM{h}' for h in range(self.pred_hor+1)]].std().values
                mask_means = self.bn_data.loc[:, [f'IDM{h}' for h in range(self.pred_hor+1)]].mean().values
                self.mask = (mask_means - factor*mask_stds, mask_means + factor*mask_stds)
            elif how == 'minmax':
                mask_mins = self.bn_data.loc[:, [f'IDM{h}' for h in range(self.pred_hor+1)]].min().values
                mask_maxs = self.bn_data.loc[:, [f'IDM{h}' for h in range(self.pred_hor+1)]].max().values

                mask_mins = np.minimum(mask_mins*factor, mask_mins/factor)
                mask_maxs = np.maximum(mask_maxs*factor, mask_maxs/factor)
                self.mask = (mask_mins, mask_maxs)
        else:
            self.mask = None

    def apply_mask(self, data):
        if self.mask is None:
            return data
        
        if self.mask[0].shape[0] == 1:
            data[data<self.mask[0]] = self.mask[0]
            data[data>self.mask[1]] = self.mask[1]
        else:
            h0 = self.n_intraday_prices - data.shape[1]
            for i in range(data.shape[1]):
                # Loop through all the hours
                min_mask = self.mask[0][i+h0]
                max_mask = self.mask[1][i+h0]

                data[:, i][data[:, i]<min_mask] = min_mask
                data[:, i][data[:, i]>max_mask] = max_mask
                
        return data
    
    def make_structure(self):
        # Initiate the structure
        # N h forecast + current observation
        # [current DAM] + [N DAM forecasts] + [current ID] + [N ID forecasts]
        self.bn_structure = [[] for _ in range(self.n_dam_prices)] + [[] for _ in range(self.n_intraday_prices)]

        # No need to connect the DAM prices since these come from data / forecasts
        # Connect intraday prices to 3 surrounding dam prices
        for i in range(self.n_intraday_prices):
            id_node = self.n_dam_prices + i
            
            # Get the DAM nodes
            dam_nodes = [i - 1, i, i + 1]

            # Delete the nodes that do not exist
            dam_nodes = [node for node in dam_nodes if node >= 0 and node < self.n_dam_prices]

            # Now connect the ID prices to the previous ID node
            id_nodes = [id_node - 1]

            # Delete the nodes that do not exist
            id_nodes = [node for node in id_nodes if node >= self.n_dam_prices and node < self.n_dam_prices + self.n_intraday_prices]

            # Add the nodes to the structure
            self.bn_structure[id_node] = dam_nodes + id_nodes

    def normalize_data(self):
        """
        Normalize the data by dividing the IDM prices by the DAM prices
        """
        if self.normalize and not self.bn_data_is_normalized:
            for h in range(self.pred_hor+1):
                self.bn_data[f'IDM{h}'] = self.bn_data[f'IDM{h}'].values.flatten() / self.bn_data[f'DAM{h}'].values.flatten()

    def normalize_input(self, idm_observations, dam_prices):
        """
        Normalize the data by dividing the IDM prices by the DAM prices
        """
        if self.normalize:
            # Only select the first DAM prices to cover the IDM observations
            dam_prices = dam_prices[:len(idm_observations)]
            idm_observations = idm_observations / dam_prices
        return idm_observations
    
    def denormalize_samples(self, samples, dam_prices):
        """
        Denormalize the samples to get the IDM prices
        """
        if self.normalize:
            samples = samples * dam_prices.reshape(1, -1)
        return samples
    
    def infer_idm(self, dam_scenario, idm_observations, n_samples=1000, is_normalized=False):
        """
        Infer the IDM prices given the DAM prices.

        Parameters
        ----------
        dam_scenario : np.array
            Array of DAM prices to condition the IDM on.
            Should be of length pred_hor + 1 (to properly condition the first IDM price).
        idm_observations : np.array
            Array of IDM prices to condition the IDM on.
            Minimum length is 1 (the IDM price of d-1), untill 1 + pred_hor
        """
        h = len(idm_observations) - 1
        # Define condition nodes
        condition_nodes = [i for i in range(self.n_dam_prices + 1 + h)] # DAM prices + IDM price of d-1 + IDM prices untill h
        
        if not is_normalized and self.normalize:
            # Normalize the input
            idm_observations = self.normalize_input(idm_observations, dam_scenario)
        
        condition_values = list(np.concatenate([dam_scenario, idm_observations]))

        # Infer IDM prices given the DAM prices and the observed IDM prices
        samples = inference(
            R=self.R,
            Nodes=condition_nodes,
            Values=condition_values,
            DATA=self.bn_data,
            SampleSize=n_samples
        )

        samples=samples[0].T # shape [n_samples, n_idm_prices]
        samples = self.apply_mask(samples)
        if self.normalize:
            # Denormalize the samples
            samples = self.denormalize_samples(samples, dam_scenario[h+1:])

        return samples
    
    def __call__(self, i, h, n_samples=100):
        """
        Get the conditional nodes for the IDM price at bn_data index i and hour h

        Parameters
        ----------
        i : int
            Index of the bn_data to get the conditional nodes for.
        h : int
            Hour to get the conditional nodes for. Minimum value is 1 (the IDM price of d-1), untill 1 + pred_hor
        """
        if h > self.pred_hor:
            raise ValueError(f'Hour h should be smaller than the prediction horizon {self.pred_hor}')
        elif h < 1:
            raise ValueError(f'Hour h should be larger than 0')
        elif i < 0 or i >= len(self.bn_data):
            raise ValueError(f'Index i should be between 0 and {len(self.bn_data)}')
        
        # Get the dam prices
        dam_prices = self.bn_data.iloc[i, :self.n_dam_prices].to_numpy()
        idm_observations = self.bn_data.iloc[i, self.n_dam_prices:self.n_dam_prices+h].to_numpy()

        # Infer the IDM prices
        return self.infer_idm(dam_prices, idm_observations, n_samples=n_samples, is_normalized=True)