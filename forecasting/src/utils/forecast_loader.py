import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, ConciseDateFormatter
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from itertools import product

class FCFormatter():
    def __init__(self, df, fc_vars, varname, n_leadtimes, fc_timestep, colstep=None, quantiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], parallel=True):
        self.df = df
        self.fc_vars = fc_vars
        self.varname = varname
        self.colstep = colstep
        self.n_leadtimes = n_leadtimes
        self.leadtime_index = range(1, n_leadtimes+1)
        self.fc_timestep = fc_timestep
        self.quantiles = quantiles
        self.parallel = parallel


    def format(self, inplace=True):
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()

        if self.fc_timestep.lower() == 'days':
            def add_ts(n):
                return pd.Timedelta(days=n)
        elif self.fc_timestep.lower() == 'hours':
            def add_ts(n):
                return pd.Timedelta(hours=n)
        elif self.fc_timestep.lower() == 'minutes':
            def add_ts(n):
                return pd.Timedelta(minutes=n)
        else:
            raise ValueError(f'Unknown fc_timestep: {self.fc_timestep}')
        
        if (len(self.fc_vars) > 1) and (self.colstep is not None):
            if self.colstep.lower() == 'hours':
                def add_col(n):
                    return pd.Timedelta(hours=n)
            elif self.colstep.lower() == 'minutes':
                def add_col(n):
                    return pd.Timedelta(minutes=n)
            else:
                raise ValueError(f'Unknown colstep: {self.colstep}')
            df = self.format_col_as_leadtime(add_ts=add_ts, add_col=add_col, inplace=inplace)

        else:
            df = self.format_col_sep(add_ts=add_ts, inplace=inplace)

        if not inplace:
            return df

        
    def format_col_as_leadtime(self, add_ts, add_col, inplace=True):
        # Create new dataframe while considering the columns as leadtimes to be added to the index
        indices = pd.MultiIndex.from_tuples([(i, i+add_ts(n)+add_col(c), q) for i in self.df.index for n in self.leadtime_index for c in range(len(self.fc_vars)) for q in self.quantiles], names=['forecast_time', 'obersvation_time', 'quantile'])
        new_df = pd.DataFrame(index=indices, columns=[self.varname])
        idx = pd.IndexSlice

        # if self.parallel:
        #     def get_vals(inc):
        #         i, n, c = inc
        #         return np.sort(np.array([self.df.loc[i, [f'{self.fc_vars[c]} +{n} q{q}' for q in self.quantiles]].values]), axis=1).flatten()

        #     iters = product(self.df.index, self.leadtime_index, range(len(self.fc_vars)))
        #     print(iters)          
        #     vals = Parallel(n_jobs=-1)(delayed(get_vals)(inc) for inc in tqdm(product(self.df.index, self.leadtime_index, range(len(self.fc_vars))), desc='Formatting'))

        for i in tqdm(self.df.index, desc='Formatting'):
            for n in self.leadtime_index:
                for c in range(len(self.fc_vars)):
                    # if self.parallel:
                    #     new_df.loc[idx[i, i+add_ts(n)+add_col(c), :], self.varname] = vals.pop(0)
                    # else:
                    new_df.loc[idx[i, i+add_ts(n)+add_col(c), :], self.varname] = np.sort(np.array([self.df.loc[i, [f'{self.fc_vars[c]} +{n} q{q}' for q in self.quantiles]].values]), axis=1).flatten()
        
        if inplace:
            self.df = new_df.astype(float).round(2)
        else:
            return new_df.astype(float).round(2)

    def format_col_sep(self, add_ts, inplace=True):
        # Create a new dataframe with the correct format
        indices = pd.MultiIndex.from_tuples([(i, i+add_ts(n), q) for i in self.df.index for n in self.leadtime_index for q in self.quantiles], names=['forecast_time', 'obersvation_time', 'quantile'])
        new_df = pd.DataFrame(index=indices, columns=[self.varname])
        idx = pd.IndexSlice

        if self.parallel:
            def get_vals(i):
                return np.sort(np.array([self.df.loc[i, [f'{self.fc_vars[0]} +{n} q{q}' for q in self.quantiles]].values for n in self.leadtime_index]), axis=1).flatten()
            vals = Parallel(n_jobs=-1)(delayed(get_vals)(i) for i in tqdm(self.df.index, desc='Formatting'))
            for i, v in tqdm(zip(self.df.index, vals), desc='Formatting'):
                new_df.loc[idx[i, :, :], self.varname] = v
        else:
            for i in tqdm(self.df.index):
                new_df.loc[idx[i, :, :], self.varname] = np.sort(np.array([self.df.loc[i, [f'{self.fc_vars[0]} +{n} q{q}' for q in self.quantiles]].values for n in self.leadtime_index]), axis=1).flatten()

        if inplace:
            self.df = new_df.astype(float).round(2)
        else:
            return new_df.astype(float).round(2)

class DAMForecastLoader():
    """Class to load QR forecasts"""

    def __init__(self, y_true, y_pred, forecast_label, ylims, varname='NL DAM', quantiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]):
        self.y_true = y_true
        self.y_pred = y_pred
        self.quantiles = quantiles
        self.forecast_label = forecast_label
        self.forecast_indices = self.y_pred.index
        self.ylims=ylims
        self.varname = varname
        self.forecast_hor = 2

    def plot(self, forecast_index = None, ax = None, col=None):
        idx = pd.IndexSlice
        if not forecast_index:
            forecast_index = self.forecast_indices[np.random.randint(0, high=len(self.forecast_indices)-1)][0]
        
        y_pred_slice = self.y_pred.loc[idx[forecast_index, :, :], :]
        y_pred_slice.index = y_pred_slice.index.droplevel('forecast_time')

        i_0 = y_pred_slice.index[0][0]
        i_f = y_pred_slice.index[-1][0] + pd.DateOffset(hours=23)
        plot_index = pd.date_range(i_0, i_f, freq='H')

        legend=False
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 2))
            legend=True
        
        q_ls = self.quantiles[:int(np.floor(len(self.quantiles)/2))]
        q_us = self.quantiles[int(np.ceil(len(self.quantiles)/2)):]
        for q_u, q_l in zip(q_us, q_ls):
            pi = int(np.round((1-2*q_u), decimals=2)*100)
            y_l = y_pred_slice.loc[idx[:, q_l], :].values.flatten()
            y_u = y_pred_slice.loc[idx[:, q_u], :].values.flatten()
            ax.fill_between(plot_index, y_u, y_l, alpha=0.3*q_l, step='post', color='r', label=f'{pi} PI%')
        y_u = y_pred_slice.loc[idx[:, 0.5], :].values.flatten()
        ax.step(plot_index, y_u, color='r', label='Expected value', alpha=0.6, where='post')

        ax.step(plot_index, self.y_true.loc[plot_index, :].values, color='k', label='Observation', where='post', ls='--')

        xtick_locator = AutoDateLocator()
        xtick_formatter = ConciseDateFormatter(xtick_locator)
        ax.xaxis.set_major_locator(xtick_locator)
        ax.xaxis.set_major_formatter(xtick_formatter)
        ax.set_xlim(plot_index[0], plot_index[-1])
        ax.set_ylim(*self.ylims)
        
        if legend:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
            ax.set_ylabel(self.forecast_label)

    def plot_multiple(self, n_plots=10, n_cols=2, figsize=(30, 15)):
        random_idx = sorted([np.random.randint(0, high=len(self.forecast_indices)-1) for _ in range(n_plots)])
        n_rows = n_plots // n_cols
        if n_plots % n_cols != 0:
            n_rows += 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)#, sharey=True)
        for i, idx in enumerate(random_idx):
            self.plot(forecast_index=self.forecast_indices[idx][0], ax=axes[i//n_cols, i%n_cols])

            if i%n_cols == 0:
                axes[i//n_cols, i%n_cols].set_ylabel(self.forecast_label)
            
        fig.tight_layout()

    def get_operational_forecast(self, date):
        idx = pd.IndexSlice
        df_slice = self.y_pred.loc[idx[date, :, :], :]
        df_slice.index = df_slice.index.droplevel('forecast_time')
        df_ind = pd.date_range(df_slice.index[0][0], df_slice.index[-1][0] + pd.DateOffset(hours=23), freq='H')
        new_df = pd.DataFrame(index = df_ind, columns=self.quantiles)

        for q in self.quantiles:
            new_df.loc[df_ind, q] = df_slice.loc[idx[:, q], :].values.flatten()

        return new_df
    
    def _get_quantile_values_at_leadtime(self, leadtime, quantile):
        idx = pd.IndexSlice
        df_slice = self.y_pred.loc[idx[:, :, quantile], :]
        df_slice.index = df_slice.index.droplevel('forecast_time')
        df_slice.index = df_slice.index.droplevel('quantile')
        return df_slice.iloc[np.arange(leadtime-1, len(df_slice), self.forecast_hor), :]
    
    def get_forecast_at_leadtime(self, leadtime):
        df_q = self._get_quantile_values_at_leadtime(leadtime, self.quantiles[0])
        df_ind = pd.to_datetime([i + pd.DateOffset(hours=h) for i in df_q.index for h in range(24)])
        df_new = pd.DataFrame(index = df_ind, columns=self.quantiles)
        df_new.loc[df_ind, self.quantiles[0]] = df_q.values.flatten()

        for q in self.quantiles[1:]:
            df_new.loc[df_ind, q] = self._get_quantile_values_at_leadtime(leadtime, q).values.flatten()
        return df_new
    

class FCLoader():
    """Class to load QR forecasts"""

    def __init__(self, y_true, y_pred, forecast_label, ylims, forecast_hor_len=48, min_fc_hor=1, varname='Aggregated', quantiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]):
        self.y_true = y_true
        self.y_pred = y_pred
        self.quantiles = quantiles
        self.forecast_label = forecast_label
        self.forecast_indices = self.y_pred.index
        self.ylims=ylims
        self.varname = varname
        self.forecast_hor_len = forecast_hor_len
        self.min_fc_hor = min_fc_hor

    def plot(self, forecast_index = None, ax = None, col=None):
        idx = pd.IndexSlice
        if not forecast_index:
            forecast_index = self.forecast_indices[np.random.randint(0, high=len(self.forecast_indices)-1)][0]
        
        y_pred_slice = self.get_operational_forecast(forecast_index)

        n_prev_steps=4
        i_0 = y_pred_slice.index[0] - pd.DateOffset(hours=n_prev_steps)
        i_f = y_pred_slice.index[-1] + pd.DateOffset(hours=1)
        plot_index = pd.date_range(i_0, i_f, freq='H')
        fc_index = pd.to_datetime(y_pred_slice.index.get_level_values(0).unique())

        legend=False
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 2))
            legend=True
        
        q_ls = self.quantiles[:int(np.floor(len(self.quantiles)/2))]
        q_us = self.quantiles[int(np.ceil(len(self.quantiles)/2)):]
        for q_u, q_l in zip(q_us, q_ls):
            pi = int(np.round((1-2*q_l), decimals=2)*100)
            y_l = y_pred_slice.loc[:, q_l].values.flatten()
            y_u = y_pred_slice.loc[:, q_u].values.flatten()
            ax.fill_between(fc_index, y_u, y_l, alpha=0.4*q_l, step='post', color='r', label=f'{pi}% PI')
        y_u = y_pred_slice.loc[:, 0.5].values.flatten()
        ax.step(fc_index, y_u, color='r', label='Expected value', alpha=0.6, where='post')
        
        act_slice = self.get_operational_target(forecast_index)
        ax.step(act_slice.index, act_slice.values, color='k', label='Observation', where='post', ls='--')

        # prev_slice = self.get_operational_prev(forecast_index, n_prev_steps)
        # ax.step(prev_slice.index, prev_slice.values, color='k', where='post', ls='--')

        xtick_locator = AutoDateLocator()
        xtick_formatter = ConciseDateFormatter(xtick_locator)
        ax.xaxis.set_major_locator(xtick_locator)
        ax.xaxis.set_major_formatter(xtick_formatter)
        # ax.set_xlim(plot_index[0], plot_index[-1])
        ax.set_xlim(fc_index[0], fc_index[-1])
        ax.set_ylim(*self.ylims)
        
        if legend:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
            ax.set_ylabel(self.forecast_label)

    def plot_multiple(self, n_plots=10, n_cols=2, figsize=(30, 15)):
        random_idx = sorted([np.random.randint(0, high=len(self.forecast_indices)-1) for _ in range(n_plots)])
        n_rows = n_plots // n_cols
        if n_plots % n_cols != 0:
            n_rows += 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)#, sharey=True)
        for i, idx in enumerate(random_idx):
            self.plot(forecast_index=self.forecast_indices[idx][0], ax=axes[i//n_cols, i%n_cols])

            if i%n_cols == 0:
                axes[i//n_cols, i%n_cols].set_ylabel(self.forecast_label)
            
        fig.tight_layout()

    def get_operational_forecast(self, date):
        idx = pd.IndexSlice
        df_slice = self.y_pred.loc[idx[date, :, :], :]
        df_slice.index = df_slice.index.droplevel('forecast_time')
        df_ind = df_slice.index.get_level_values(0).unique()
        new_df = pd.DataFrame(index = df_ind, columns=self.quantiles)

        for q in self.quantiles:
            new_df.loc[:, q] = df_slice.loc[idx[:, q], self.varname].values.flatten()

        return new_df.sort_index()
    
    def get_operational_prev(self, date, n_prev_hours):
        idx = pd.IndexSlice
        df_slice = pd.concat([
            self.y_true.loc[idx[:, date-pd.DateOffset(hours=h)], :].iloc[0] for h in range(n_prev_hours, 0, -1)
        ])
        df_slice.index = pd.to_datetime([date-pd.DateOffset(hours=h) for h in range(n_prev_hours, 0, -1)])
        return df_slice.sort_index()
    
    def get_operational_target(self, date):
        idx = pd.IndexSlice
        df_slice = self.y_true.loc[idx[date, :], :]
        df_slice.index = df_slice.index.droplevel('forecast_time')
        return df_slice.sort_index()
    
    def _get_quantile_values_at_leadtime(self, leadtime, quantile):
        idx = pd.IndexSlice
        df_slice = self.y_pred.loc[idx[:, :, quantile], :]
        df_slice.index = df_slice.index.droplevel('forecast_time')
        df_slice.index = df_slice.index.droplevel('quantile')
        return df_slice.iloc[np.arange(leadtime-self.min_fc_hor, len(df_slice), self.forecast_hor_len), :]
    
    def get_forecast_at_leadtime(self, leadtime):
        df_q = self._get_quantile_values_at_leadtime(leadtime, self.quantiles[0])
        df_ind = pd.to_datetime([i + pd.DateOffset(hours=h) for i in df_q.index for h in range(24)])
        df_new = pd.DataFrame(index = df_ind, columns=self.quantiles)
        df_new.loc[df_ind, self.quantiles[0]] = df_q.values.flatten()

        for q in self.quantiles[1:]:
            df_new.loc[df_ind, q] = self._get_quantile_values_at_leadtime(leadtime, q).values.flatten()
        return df_new.sort_index()
    
    def get_forecast_at_leadtimes(self, leadtime, cols='hours'):
        if cols == 'hours':
            cols = [f'H{i}' for i in range(24)]
        else:
            raise ValueError('cols must be "hours"')
        
        df_ind = pd.to_datetime([i for i in self.y_pred.index[leadtime-self.min_fc_hor::self.forecast_hor_len] if i.hour == 0])
        df_ind = pd.MultiIndex.from_product([df_ind, self.quantiles], names=['observation_time', 'quantile'])
        df_new = pd.DataFrame(index = df_ind, columns=cols)
        for h in range(24):
            df_new.loc[:, cols[h]] = self.get_forecast_at_leadtime(leadtime+h).values
        return df_new

