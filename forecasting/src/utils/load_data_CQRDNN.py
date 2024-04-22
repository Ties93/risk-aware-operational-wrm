import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader():
    """Data loader for the CQR-DNN model."""
    def __init__(self, cfg, lagrange, windowrange, fcrange):
        self.cfg=cfg
        self.datapath = Path(cfg.EXPERIMENT.DATAPATH)

        # Set the ranges for the lags, moving windows and hours.
        # Staat ook in TPE Search
        # Naar config file??
        self.lagrange = lagrange
        self.windowrange = windowrange
        self.fcrange = fcrange

        # Read the data
        self.wl = None
        self.read_water_level()

        self.wind = None
        self.read_wind()
        
        # Make target DataFrame
        self.y = pd.concat([self.shift_rename(self.wl, shift) for shift in self.fcrange], axis=1).dropna()
        self.fc_index = self.y.index

        # Make DF with all possible features
        self.X = None
        self.make_feature_df()

        self.X = self.X.loc[self.X.index.isin(self.y.index)]
        self.y = self.y.loc[self.y.index.isin(self.X.index)]


    def read_water_level(self):
        self.wl = pd.read_csv(self.datapath/'waterlevel'/'IJmuiden Noordersluis.csv', index_col=0, parse_dates=True).astype(np.float64).resample('1H').mean()
        self.wl.columns = ['WL']
        self.wl.loc[self.wl.WL > 1000, 'WL'] = np.nan	# Remove outliers
        self.wl = self.check_min_len(self.wl)

    def read_wind(self):
        self.wind = pd.read_csv(self.datapath/'KNMI data'/'wind_data_ijmuiden.csv', index_col=0, parse_dates=True).astype(np.float64)
        self.wind = self.check_min_len(self.wind)

    
    def check_min_len(self, df: pd.DataFrame, max_nan_frac=0.3):
        cols = [col for col in df.columns if 1-(len(df.loc[:,col].dropna().index)/len(df.loc[:,col].index)) < max_nan_frac]
        return df.loc[:,cols].interpolate(method='time')

    def shift_rename(self, df, shift):
        df_ = df.shift(-shift)
        df_.columns = [col+' +'+str(shift) for col in df_.columns]
        return df_

    def make_lag_wl_df(self):
        wl_ = self.wl.copy()
        data = {'WL lag'+str(l): wl_.shift(l).loc[wl_.index.isin(self.fc_index), 'WL'] for l in self.lagrange}
        return pd.DataFrame(index=self.fc_index, data=data)

    def make_lag_wind_df(self):
        wind_ = self.wind.copy()
        data_hour = {
            'hourly wind lag'+str(l): wind_.shift(l).loc[wind_.index.isin(self.fc_index), 'wind_speed_hourly'] 
            for l in self.lagrange}
        data_10min = {
            '10min wind lag'+str(l): wind_.rolling(24).mean().shift(l).loc[wind_.index.isin(self.fc_index), 'wind_speed_10min'	]
            for l in self.lagrange
            } 
        data_dir = {
            'wind direction lag'+str(l): wind_.shift(l).loc[wind_.index.isin(self.fc_index), 'wind_direction']
            for l in self.lagrange
            }
        data = {**data_hour, **data_10min, **data_dir}
        return pd.DataFrame(index=self.fc_index, data=data)

    def make_window_df(self):
        wl_ = self.wl.copy()
        wldata = {'WL window'+str(w): wl_.loc[:, 'WL'].rolling(w*24).sum() for w in self.windowrange}
        return pd.DataFrame(index=self.fc_index, data=wldata)
            
    def make_temporal_df(self):
        # Time features
        temporal_df = pd.DataFrame(index=self.fc_index)
        temporal_df.loc[:, 'DOY abs'] = temporal_df.index.dayofyear
        temporal_df.loc[:, 'DOY sin'] = np.sin(2 * np.pi * (temporal_df.index.dayofyear / 365).values)
        temporal_df.loc[:, 'DOY cos'] = np.cos(2 * np.pi * (temporal_df.index.dayofyear / 365).values)
        
        temporal_df.loc[:, 'HOD abs'] = temporal_df.index.hour
        temporal_df.loc[:, 'HOD sin'] = np.sin(2 * np.pi * (temporal_df.index.hour / 24).values)
        temporal_df.loc[:, 'HOD cos'] = np.cos(2 * np.pi * (temporal_df.index.hour / 24).values)
        
        return temporal_df

    def make_feature_df(self):
        self.X = pd.concat([
            self.make_lag_wl_df(),
            self.make_lag_wind_df(),
            self.make_window_df(),
            self.make_temporal_df()
            ], axis=1).dropna().round(decimals=2)