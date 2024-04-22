from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
import pandas as pd
from scipy.optimize import minimize, Bounds
import scipy.stats as st
import matplotlib.pyplot as plt

class DistributionFitter():
    # Eenmalig herschreven voor het nieuwe format FC dfs (indices [fc date, obs date, quantile])
    # Nog niet getest.
    
    def __init__(self, df, colname, quantiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], mask_regression_quantile=False, fit=True, parallel=True):
        self.df = df
        self.colname = colname
        self.quantiles = np.array(quantiles)
        self.mask_regression_quantile = mask_regression_quantile
        self.quantiles=np.array(quantiles)

        self.truncated_distributions = ['uniform', 'truncatednormal']
        self.idx = pd.IndexSlice

        self.distributions = {
            'normal': st.norm, # loc, scale
            'lognormal': st.lognorm, # shape, loc, scale
            'gamma': st.gamma, # gamma, loc, scale
            'gumbel_r': st.gumbel_r, # loc, scale
            'gumbel_l': st.gumbel_l, # loc, scale
            'invweibull': st.invweibull, # c, loc, scale
            'weibull_min': st.weibull_min, # c, loc, scale
            'weibull_max': st.weibull_max, # c, loc, scale
            'exponential': st.expon, # loc, scale
            'rayleigh': st.rayleigh, # loc, scale
            'uniform': st.uniform, # loc, scale
            'truncatednormal': st.truncnorm, # a, b, loc, scale
            'pareto': st.pareto, # b, loc, scale
            'logistic': st.logistic, # loc, scale
            # 'beta': st.beta, # a, b, loc, scale
        }

        self.bounds = {
            'normal': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'lognormal': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # shape > 0, loc = free, scale > 0
            'gamma': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # gamma > 0, loc = free, scale > 0
            'gumbel_r': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'gumbel_l': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'invweibull': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # c > 0, loc = free, scale > 0
            'weibull_min': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # c > 0, loc = free, scale > 0
            'weibull_max': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # c > 0, loc = free, scale > 0
            'exponential': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'rayleigh': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'uniform': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'truncatednormal': ([-np.inf, -np.inf, -np.inf, 1e-6], [np.inf, np.inf, np.inf, np.inf]), # a = free, b > a (= free but warm start nicely?), loc = free, scale > 0
            'pareto': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # b > 0, loc = free, scale > 0
            'logistic': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'beta': ([1e-6, 1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf, np.inf]), # a > 0, b > 0, loc = free, scale > 0
        }

        if fit:
            if parallel:
                self.fit_distributions_parallel()
            else:
                self.fit_distributions()

    def get_quantiles(self, date, hour):
        # Hier kan je makkelijk aanpassingen doen voor het nieuwe format
        regression_quantiles = self.df.loc[self.idx[date, date + pd.DateOffset(hours=hour), :], [f'{self.colname}']].values.flatten()
        if self.mask_regression_quantile:
            regression_quantiles[regression_quantiles<0] = 0#np.nan
        return regression_quantiles
    
    def get_bounds(self, dist_name, regression_quantiles):
        lb, ub = self.bounds[dist_name]
        if dist_name in self.truncated_distributions:
            a = regression_quantiles[~np.isnan(regression_quantiles)].min()
            b = regression_quantiles[~np.isnan(regression_quantiles)].max()
            lb[0] = a
            lb[1] = b

            ub[0] = a
            ub[1] = b
            return  Bounds(lb, ub)
        else:
            return Bounds(lb, ub)

    def get_dist(self, dist_name):
            return self.distributions[dist_name]
        
    def argmin_KS(self, parameters, dist_name, regression_quantiles):
        dist = self.get_dist(dist_name)
        fitted_cdf = np.array([dist.cdf(q, *parameters) for q in regression_quantiles[~np.isnan(regression_quantiles)]])
        return max(abs(fitted_cdf - self.quantiles[~np.isnan(regression_quantiles)]))
    
    def argmin_SSE(self, parameters, dist_name, regression_quantiles):
        dist = self.get_dist(dist_name)
        fitted_quantiles = np.array([dist.cdf(q, *parameters) for q in self.quantiles[~np.isnan(regression_quantiles)]])[0]
        SSE = sum((regression_quantiles[~np.isnan(regression_quantiles)] - fitted_quantiles)**2)
        return SSE
    
    def warm_start(self, dist_name, regression_quantiles):
        q = regression_quantiles[~np.isnan(regression_quantiles)]

        loc = np.round(np.median(regression_quantiles[~np.isnan(regression_quantiles)]), decimals=2)
        scale = np.round(np.std(regression_quantiles[~np.isnan(regression_quantiles)]), decimals=2)
        if dist_name == 'normal':
            return np.array([loc, scale]) # loc = free, scale > 0
        elif dist_name == 'lognormal':
            return np.array([1, loc, scale]) # s > 0, loc = free, scale > 0
        elif dist_name == 'gamma':
            return np.array([1, loc, 1]) # a > 0, loc = free, scale > 0
        elif dist_name == 'beta':
            return np.array([1, 1, min(q), max(q)]) # a > 0, b > 0, loc = free, scale > 0
        elif dist_name in ['gumbel_l', 'gumbel_r']:
            return np.array([loc, scale]) # loc = free, scale > 0
        elif dist_name == 'invweibull':
            return np.array([1, loc, scale]) # c > 0, loc = free, scale > 0
        elif dist_name == 'exponential': 
            return np.array([loc, scale]) # loc = free, scale > 0
        elif dist_name == 'rayleigh':
            return np.array([loc, scale]) # loc = free, scale > 0
        elif dist_name == 'uniform':
            return np.array([min(q), max(q)]) # loc = free, scale > 0
        elif dist_name == 'truncatednormal':
            return np.array([min(q), max(q), loc, scale]) # a = free, b > a (= free but warm start nicely?), loc = free, scale > 0
        elif dist_name == 'pareto':
            return np.array([1, loc, scale]) # b > 0, loc = free, scale > 0
        elif dist_name == 'weibull_min':
            return np.array([1, loc, scale]) # c > 0, loc = free, scale > 0
        elif dist_name == 'weibull_max':
            return np.array([1, loc, scale]) # c > 0, loc = free, scale > 0
        elif dist_name == 'logistic':
            return np.array([loc, scale]) # loc = free, scale > 0
    
    def fit_distribution(self, regression_quantiles, obj='KS'):
        results = {}
        for name, dist in self.distributions.items():
            try:
                if obj == 'SSE':
                    to_minimize = partial(self.argmin_SSE, dist_name=name, regression_quantiles=regression_quantiles)
                elif obj == 'KS':
                    to_minimize = partial(self.argmin_KS, dist_name=name, regression_quantiles=regression_quantiles)
                # params = minimize(to_minimize, dist().getParameter(), options={'maxiter': 1000}).x
                params = minimize(to_minimize, self.warm_start(dist_name=name, regression_quantiles=regression_quantiles), method='Nelder-Mead', options={'maxiter': 1000}, bounds=self.get_bounds(dist_name=name, regression_quantiles=regression_quantiles)).x
                # params = minimize(to_minimize, self.warm_start(dist_name=name, regression_quantiles=regression_quantiles), options={'maxiter': 1000}, bounds=self.get_bounds(dist_name=name, regression_quantiles=regression_quantiles)).x
                results[name] = {}
                results[name]['params'] = params
                results[name]['KS'] = self.argmin_KS(params, name, regression_quantiles)
            except:
                pass

        return results
    
    def get_best_distribution(self, date, hour):
        regression_quantiles = self.get_quantiles(date, hour)
        results = self.fit_distribution(regression_quantiles)
        #try:
        best = min(results, key=lambda x: results[x]['KS'])
        return best, results[best]['params'], results[best]['KS']
        #except ValueError:
        #    return np.nan, np.nan, np.nan
        
    
    def make_df_row(self, datehour):
        date = datehour[0]
        hour = datehour[1]
        best, params, KS = self.get_best_distribution(date, hour)
        ret = [best, params, KS]
        return ret
    
    def fit_distributions_parallel(self, ret=False):
        num_cores = multiprocessing.cpu_count()
        date_hour = [(date, hour) for date in self.df.index.get_level_values(0).unique() for hour in range(1,49)]
        results = Parallel(n_jobs=num_cores)(delayed(self.make_df_row)(datehour) for datehour in tqdm(date_hour))
        self.fitted_distributions = pd.DataFrame(index=self.df.index.droplevel(2).unique(), columns=['dist', 'params', 'KS'])
        self.fitted_distributions.loc[:, :] = results
        self.fitted_distributions.sort_index(inplace=True)
        
        if ret:
            return self.fitted_distributions
    
    def fit_distributions(self, ret=False):
        index = self.df.index.droplevel(2).unique()
        self.fitted_distributions = pd.DataFrame(index=index, columns=['dist', 'params', 'KS'])
        for date in tqdm(self.df.index.get_level_values(0).unique()):
            for h in range(1,49):
                best, params, KS = self.get_best_distribution(date, h)
                self.fitted_distributions.loc[(date, date + pd.DateOffset(hours=h)), 'dist'] = best
                self.fitted_distributions.loc[(date, date + pd.DateOffset(hours=h)), 'params'] = params
                self.fitted_distributions.loc[(date, date + pd.DateOffset(hours=h)), 'KS'] = KS

        if ret:
            return self.fitted_distributions

class DAMDistributionFitter():
    def __init__(self, df, colname, quantiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], mask_regression_quantile=False, fit=True, parallel=True):
        self.df = df
        self.colname = colname
        self.quantiles = np.array(quantiles)
        self.mask_regression_quantile = mask_regression_quantile
        self.quantiles=np.array(quantiles)

        self.forecast_indices = self.df.index.get_level_values(0).unique()
        self.truncated_distributions = ['uniform', 'truncatednormal']
        self.idx = pd.IndexSlice

        self.distributions = {
            'normal': st.norm, # loc, scale
            'lognormal': st.lognorm, # shape, loc, scale
            'gamma': st.gamma, # gamma, loc, scale
            'gumbel_r': st.gumbel_r, # loc, scale
            'gumbel_l': st.gumbel_l, # loc, scale
            'invweibull': st.invweibull, # c, loc, scale
            'weibull_min': st.weibull_min, # c, loc, scale
            'weibull_max': st.weibull_max, # c, loc, scale
            'exponential': st.expon, # loc, scale
            'rayleigh': st.rayleigh, # loc, scale
            'uniform': st.uniform, # loc, scale
            'truncatednormal': st.truncnorm, # a, b, loc, scale
            'pareto': st.pareto, # b, loc, scale
            'logistic': st.logistic, # loc, scale
            # 'beta': st.beta, # a, b, loc, scale
        }

        self.bounds = {
            'normal': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'lognormal': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # shape > 0, loc = free, scale > 0
            'gamma': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # gamma > 0, loc = free, scale > 0
            'gumbel_r': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'gumbel_l': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'invweibull': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # c > 0, loc = free, scale > 0
            'weibull_min': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # c > 0, loc = free, scale > 0
            'weibull_max': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # c > 0, loc = free, scale > 0
            'exponential': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'rayleigh': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'uniform': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'truncatednormal': ([-np.inf, -np.inf, -np.inf, 1e-6], [np.inf, np.inf, np.inf, np.inf]), # a = free, b > a (= free but warm start nicely?), loc = free, scale > 0
            'pareto': ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf]), # b > 0, loc = free, scale > 0
            'logistic': ([-np.inf, 1e-6], [np.inf, np.inf]), # loc = free, scale > 0
            'beta': ([1e-6, 1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf, np.inf]), # a > 0, b > 0, loc = free, scale > 0
        }

        if fit:
            if parallel:
                self.fit_distributions_parallel()
            else:
                self.fit_distributions()

    def get_quantiles(self, date, hour, pred_hor):
        regression_quantiles = self.df.loc[self.idx[date, date + pd.DateOffset(days=pred_hor), :], f'{self.colname} H{hour}'].values.flatten()
        if self.mask_regression_quantile:
            regression_quantiles[regression_quantiles<0] = 0#np.nan
        return regression_quantiles
    
    def get_bounds(self, dist_name, regression_quantiles):
        lb, ub = self.bounds[dist_name]
        if dist_name in self.truncated_distributions:
            a = regression_quantiles[~np.isnan(regression_quantiles)].min()
            b = regression_quantiles[~np.isnan(regression_quantiles)].max()
            lb[0] = a
            lb[1] = b

            ub[0] = a
            ub[1] = b
            return  Bounds(lb, ub)
        else:
            return Bounds(lb, ub)

    def get_dist(self, dist_name):
            return self.distributions[dist_name]
        
    def argmin_KS(self, parameters, dist_name, regression_quantiles):
        dist = self.get_dist(dist_name)
        fitted_cdf = np.array([dist.cdf(q, *parameters) for q in regression_quantiles[~np.isnan(regression_quantiles)]])
        return max(abs(fitted_cdf - self.quantiles[~np.isnan(regression_quantiles)]))
    
    def argmin_SSE(self, parameters, dist_name, regression_quantiles):
        dist = self.get_dist(dist_name)
        fitted_quantiles = np.array([dist.cdf(q, *parameters) for q in self.quantiles[~np.isnan(regression_quantiles)]])[0]
        SSE = sum((regression_quantiles[~np.isnan(regression_quantiles)] - fitted_quantiles)**2)
        return SSE
    
    def warm_start(self, dist_name, regression_quantiles):
        q = regression_quantiles[~np.isnan(regression_quantiles)]

        loc = np.round(np.median(regression_quantiles[~np.isnan(regression_quantiles)]), decimals=2)
        scale = np.round(np.std(regression_quantiles[~np.isnan(regression_quantiles)]), decimals=2)
        if dist_name == 'normal':
            return np.array([loc, scale]) # loc = free, scale > 0
        elif dist_name == 'lognormal':
            return np.array([1, loc, scale]) # s > 0, loc = free, scale > 0
        elif dist_name == 'gamma':
            return np.array([1, loc, 1]) # a > 0, loc = free, scale > 0
        elif dist_name == 'beta':
            return np.array([1, 1, min(q), max(q)]) # a > 0, b > 0, loc = free, scale > 0
        elif dist_name in ['gumbel_l', 'gumbel_r']:
            return np.array([loc, scale]) # loc = free, scale > 0
        elif dist_name == 'invweibull':
            return np.array([1, loc, scale]) # c > 0, loc = free, scale > 0
        elif dist_name == 'exponential': 
            return np.array([loc, scale]) # loc = free, scale > 0
        elif dist_name == 'rayleigh':
            return np.array([loc, scale]) # loc = free, scale > 0
        elif dist_name == 'uniform':
            return np.array([min(q), max(q)]) # loc = free, scale > 0
        elif dist_name == 'truncatednormal':
            return np.array([min(q), max(q), loc, scale]) # a = free, b > a (= free but warm start nicely?), loc = free, scale > 0
        elif dist_name == 'pareto':
            return np.array([1, loc, scale]) # b > 0, loc = free, scale > 0
        elif dist_name == 'weibull_min':
            return np.array([1, loc, scale]) # c > 0, loc = free, scale > 0
        elif dist_name == 'weibull_max':
            return np.array([1, loc, scale]) # c > 0, loc = free, scale > 0
        elif dist_name == 'logistic':
            return np.array([loc, scale]) # loc = free, scale > 0
    
    def fit_distribution(self, regression_quantiles, obj='KS'):
        results = {}
        for name, dist in self.distributions.items():
            try:
                if obj == 'SSE':
                    to_minimize = partial(self.argmin_SSE, dist_name=name, regression_quantiles=regression_quantiles)
                elif obj == 'KS':

                    to_minimize = partial(self.argmin_KS, dist_name=name, regression_quantiles=regression_quantiles)
                params = minimize(to_minimize, self.warm_start(dist_name=name, regression_quantiles=regression_quantiles), method='Nelder-Mead', options={'maxiter': 1000}, bounds=self.get_bounds(dist_name=name, regression_quantiles=regression_quantiles)).x
                results[name] = {}
                results[name]['params'] = params
                results[name]['KS'] = self.argmin_KS(params, name, regression_quantiles)
            except:
                pass

        return results
    
    def get_best_distribution(self, date, hour, pred_hor):
        try:
            regression_quantiles = self.get_quantiles(date, hour, pred_hor)
            results = self.fit_distribution(regression_quantiles)
            best = min(results, key=lambda x: results[x]['KS'])
            return best, results[best]['params'], results[best]['KS']
        except:
           return np.nan, np.nan, np.nan
        
    
    def make_df_row(self, date):
        hours = [f'{self.colname} H{h}' for h in range(24)]
        cols = pd.MultiIndex.from_product([hours, ['dist', 'params', 'KS']], names=['hour', 'distribution'])
        m_ind = pd.MultiIndex.from_tuples([(date, date + pd.DateOffset(days=n)) for n in range(1,3)], names=['forecast_time', 'observation_time'])
        ret = pd.DataFrame(index=m_ind, columns=cols)

        for hour in range(24):
            for pred_horizon in range(1,3):
                best, params, KS = self.get_best_distribution(date, hour, pred_horizon)
                ret.loc[(date, date + pd.DateOffset(days=pred_horizon)), (f'{self.colname} H{hour}', 'dist')] = best
                ret.loc[(date, date + pd.DateOffset(days=pred_horizon)), (f'{self.colname} H{hour}', 'params')] = params
                ret.loc[(date, date + pd.DateOffset(days=pred_horizon)), (f'{self.colname} H{hour}', 'KS')] = KS
        return ret
    
    def fit_distributions_parallel(self, ret=False):
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.make_df_row)(date) for date in tqdm(self.forecast_indices))
        self.fitted_distributions = pd.concat(results, axis=0).sort_index()
        
        if ret:
            return self.fitted_distributions
    
    def fit_distributions(self, ret=False):
        hours = [f'{self.colname} H{h}' for h in range(24)]
        cols = pd.MultiIndex.from_product([hours, ['dist', 'params', 'KS']], names=['hour', 'distribution'])
        m_ind = pd.MultiIndex.from_tuples([(date, date + pd.DateOffset(days=n)) for date in self.forecast_indices for n in range(1,3)], names=['forecast_time', 'observation_time'])
        self.fitted_distributions = pd.DataFrame(index=m_ind, columns=cols)
        for date in tqdm(self.forecast_indices):
            for hour in range(24):
                for pred_hor in range(1, 3):
                    best, params, KS = self.get_best_distribution(date, hour, pred_hor)
                    self.fitted_distributions.loc[(date, date + pd.DateOffset(days=pred_hor)), (f'{self.colname} H{hour}', 'dist')] = best
                    self.fitted_distributions.loc[(date, date + pd.DateOffset(days=pred_hor)), (f'{self.colname} H{hour}', 'params')] = params
                    self.fitted_distributions.loc[(date, date + pd.DateOffset(days=pred_hor)), (f'{self.colname} H{hour}', 'KS')] = KS
        if ret:
            return self.fitted_distributions        

class DistributionLoader():
    """Loads distributions for a certain index of the DataFrame."""

    def __init__(self, df, colname, valmin, valmax):
        self.df = df
        self.index = self.df.index
        self.colname = colname
        self.valmin = valmin
        self.valmax = valmax
        self.pdfrange = np.arange(valmin, valmax, 0.01)
        self.idx = pd.IndexSlice

        self.distributions = {
            'normal': st.norm, # loc, scale
            'lognormal': st.lognorm, # shape, loc, scale
            'gamma': st.gamma, # gamma, loc, scale
            'gumbel_r': st.gumbel_r, # loc, scale
            'gumbel_l': st.gumbel_l, # loc, scale
            'invweibull': st.invweibull, # c, loc, scale
            'weibull_min': st.weibull_min, # c, loc, scale
            'weibull_max': st.weibull_max, # c, loc, scale
            'exponential': st.expon, # loc, scale
            'rayleigh': st.rayleigh, # loc, scale
            'uniform': st.uniform, # loc, scale
            'truncatednormal': st.truncnorm, # a, b, loc, scale
            'pareto': st.pareto, # b, loc, scale
            'logistic': st.logistic, # loc, scale
        }


    def _param_str_to_params(self, param_str):
        """Converts a parameter string to a list of parameters."""
        return [float(p) for p in param_str[1:-1].split()]
    
    def get_distribution(self, date, hour):
        """Returns the distribution for the given index and hour."""
        col0name = f'{self.colname}'
        dist_name = self.df.loc[self.idx[date, date+pd.DateOffset(hours=hour), :], 'dist'].values[0]
        params_str = self.df.loc[self.idx[date, date+pd.DateOffset(hours=hour), :], 'params']
        # print(params_str.values)
        dist_params = params_str.values[0]#self._param_str_to_params(params_str)
        return dist_name, dist_params
    
    def sample_distribution(self, index, hour):
        """Samples the distribution for the given index and hour."""
        dist_name, dist_params = self.get_distribution(index, hour)
        return self.distributions[dist_name].rvs(*dist_params, size=2000)
    
    def get_pdf(self, index, hour):
        """Returns the PDF for the given index and hour."""
        dist_name, dist_params = self.get_distribution(index, hour)
        return [self.distributions[dist_name].pdf(p, *dist_params) for p in self.pdfrange]
    
    def get_cdf(self, index, hour):
        """Returns the CDF for the given index and hour."""
        dist_name, dist_params = self.get_distribution(index, hour)
        return [self.distributions[dist_name].cdf(p, *dist_params) for p in self.pdfrange]
    
    def plot_pdf(self, index, hour, ax=None, title=None, return_distname=False):
        """Plots the PDF for the given index and hour."""
        dist_name, dist_params = self.get_distribution(index, hour)

        if not ax:
            fig, ax = plt.subplots(figsize=(4, 2))

        ax.plot(self.pdfrange, self.get_pdf(index, hour))

        # ax2 = ax.twinx()
        ax.hist(self.sample_distribution(index, hour), bins=50, density=True, alpha=0.5, color='C0')
        if not title:
            ax.set_title(f'{dist_name} PDF')
        else:
            ax.set_title(title)
        ax.set_xlim(self.valmin, self.valmax)
        if return_distname:
            return dist_name