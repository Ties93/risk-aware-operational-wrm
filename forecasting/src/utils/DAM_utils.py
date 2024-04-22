import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from .KerasModels import *
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
from tqdm import tqdm

class DataLoader():
    """Data loader for the CQR-DNN model."""
    def __init__(self, datapath, lagrange, fcrange):
        self.datapath = Path(datapath)

        # Set the ranges for the lags, moving windows and hours.
        # Staat ook in TPE Search
        # Naar config file??
        self.lagrange = lagrange
        self.fcrange = fcrange

        # Read the data and group into daily dataframes
        self.read_DAM_data()
        self.daily_dam = self.make_daily_df(self.DAM)

        self.market = 'NL'
        self.MI_markets = [m for m in self.markets if m != self.market]

        self.read_load_data()
        self.daily_load = self.make_daily_df(self.load)

        self.read_load_fc_data()
        self.daily_load_fc = self.make_daily_df(self.load_fc)

        self.read_gas_prices()

        self.read_generation_data()

        self.read_holidays()

        # get handy dictionary with renewable generation
        self.renewables = {m: [] for m in self.markets}
        for col in self.gen.columns:
            m = col[0]
            r = col[1]
            if r not in self.renewables[m]:
                self.renewables[m].append(r)
        
        gen_columns = [f'{col[1]} {col[0]} H{h}' for col in self.gen.columns for h in range(24)]
        self.daily_gen = self.make_daily_df(self.gen)
        self.daily_gen.columns = gen_columns
        
        # Drop columns with only zeros (solar)
        self.daily_gen = self.daily_gen.loc[:, (self.daily_gen != 0).any(axis=0)]

        # Make target DataFrame
        self.make_target_DAM()
        self.fc_index = self.y.index

        # Make DF with all possible features
        self.make_feature_df()

        self.X = self.X.loc[self.X.index.isin(self.y.index)]
        self.y = self.y.loc[self.y.index.isin(self.X.index)]

    def read_DAM_data(self):
        self.DAM = pd.read_csv(self.datapath/ 'DAM_prices.csv', index_col=0, parse_dates=True).astype(np.float64).resample('1H').mean()
        self.DAM = self.fix_timezone(self.check_min_len(self.DAM))
        self.markets = self.DAM.columns
        self.DAM.columns = ['DAM '+col for col in self.DAM.columns]

    def read_load_data(self):
        self.load = pd.read_csv(self.datapath/ 'Load.csv', index_col=0, parse_dates=True).astype(np.float64).resample('1H').mean()
        self.load = self.fix_timezone(self.check_min_len(self.load))
        self.load.columns = ['load '+col for col in self.load.columns]

    def read_load_fc_data(self):
        self.load_fc = pd.read_csv(self.datapath/ 'Load_fc.csv', index_col=0, parse_dates=True).astype(np.float64).resample('1H').mean()
        self.load_fc = self.fix_timezone(self.check_min_len(self.load_fc))
        self.load_fc.columns = ['load fc '+col for col in self.load_fc.columns]

    def read_gas_prices(self):
        self.gas = pd.read_csv(self.datapath/ 'gas.csv', index_col=0, parse_dates=True).astype(np.float64)
        self.gas = self.fix_timezone(self.check_min_len(self.gas).sort_index())
        self.gas.columns = ['gas price']
        self.gas.index = pd.to_datetime([i - pd.DateOffset(hours=i.hour) for i in self.gas.index])

    def read_generation_data(self):
        self.gen = pd.read_csv(self.datapath/ 'Generation_FCDA.csv', index_col=0, parse_dates=True, header=[0,1]).astype(np.float64).resample('1H').mean()
        self.gen = self.fix_timezone(self.check_min_len(self.gen).sort_index())

    def read_holidays(self):
        self.holidays = pd.read_csv(self.datapath/ 'holidays.csv', index_col=0, parse_dates=True)
        self.holidays = self.fix_timezone(self.holidays)
        self.holidays.index = pd.to_datetime([i - pd.DateOffset(hours=i.hour) for i in self.holidays.index])


    def fix_timezone(self, df):
        try:
            df.index = df.index.tz_convert('Europe/Amsterdam')
        except:
            df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Europe/Amsterdam')
        return df
    
    def make_daily_df(self, df):
        if len(df.columns) > 1:
            daily_df = pd.concat([self.make_daily_df(df.loc[:,[col]]) for col in df.columns], axis=1)
        else:
            df_grouped = df.groupby(df.index.hour)
            index = df_grouped.get_group(0).index
            colname = df.columns[0]
            daily_df = pd.DataFrame(index=index, columns = [f'{colname} H{h}' for h in df_grouped.groups.keys()])

            for k in df_grouped.groups.keys():
                daily_df.loc[daily_df.index, f'{colname} H{k}'] = df_grouped.get_group(k).values.flatten()

        return daily_df
    
    def check_min_len(self, df: pd.DataFrame, max_nan_frac=0.3):
        cols = [col for col in df.columns if 1-(len(df.loc[:,col].dropna().index)/len(df.loc[:,col].index)) < max_nan_frac]
        return df.loc[:,cols]#.interpolate(method='time')

    def shift_rename(self, df, shift):
        df_ = df.shift(-shift)
        df_.columns = [col+' +'+str(shift) for col in df_.columns]
        return df_

    def make_target_DAM(self):
        self.y = pd.DataFrame(index=self.daily_dam.index, columns=[f'{col} +{h}' for h in range(1,3) for col in self.daily_dam.columns])
        for h in self.fcrange:
            self.y.loc[:,[f'{col} +{h}' for col in self.daily_dam.columns]] = self.daily_dam.shift(-h).values

    def make_lagged_df(self, df):
        data = {f'{col} lag{lag}': df.loc[:, col].shift(lag).values for lag in self.lagrange for col in df.columns}
        return pd.DataFrame(index=df.index, data=data)

    def make_lag_DAM(self):
        return self.make_lagged_df(self.daily_dam)

    def make_lag_load(self):
        return self.make_lagged_df(self.daily_load)
    
    def make_lag_load_fc(self):
        return self.make_lagged_df(self.daily_load_fc)
    
    def make_lag_gen(self):
        return self.make_lagged_df(self.daily_gen)
    
    def make_lag_gas(self):
        return self.make_lagged_df(self.gas)
    
    def make_future_load_fc(self):
        data = {f'{col} +{lag}': self.daily_load_fc.loc[self.daily_load_fc.index.isin(self.fc_index), col].shift(-lag).values for col in self.daily_load_fc.columns for lag in self.fcrange}
        return pd.DataFrame(index=self.fc_index, data=data)
    
    def make_future_gen_fc(self):
        data = {f'{col} +{lag}': self.daily_gen.loc[self.daily_gen.index.isin(self.fc_index), col].shift(-lag).values for col in self.daily_gen.columns for lag in self.fcrange}
        return pd.DataFrame(index=self.fc_index, data=data)
            
    def make_temporal_df(self):
        # Time features
        temporal_df = pd.DataFrame(index=self.fc_index)
        temporal_df.loc[:, 'DOY abs'] = temporal_df.index.dayofyear
        temporal_df.loc[:, 'DOY sin'] = np.sin(2 * np.pi * (temporal_df.index.dayofyear / 365).values)
        temporal_df.loc[:, 'DOY cos'] = np.cos(2 * np.pi * (temporal_df.index.dayofyear / 365).values)
        temporal_df.loc[:, 'DOW'] = temporal_df.index.dayofweek
        
        # Add holiday to DOW as new class, and to the day before!
        temporal_df.loc[temporal_df.index.isin(self.holidays.index - pd.DateOffset(days=1)), 'DOW'] = 7
        return temporal_df

    def make_feature_df(self):
        self.X = pd.concat([
            self.make_lag_DAM(),
            self.make_lag_load(),
            self.make_lag_load_fc(),
            self.make_future_load_fc(),
            self.make_lag_gen(),
            self.make_future_gen_fc(),
            self.make_lag_gas(),
            self.make_temporal_df()
            ], axis=1).round(decimals=2)

class TPESearch(KerasMixin):
    def __init__(self, name, dataloader, savepath, modeltype='MLP', gpu=True):
        self.name = name
        self.dataloader = dataloader
        self.savepath = Path(savepath)
        self.modeltype = modeltype
        self.gpu = gpu
        self.best_loss = np.inf

        KerasMixin.__init__(self)

        if self.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            self.device = '/gpu:0'
        else:
            self.device = '/cpu'

        self.X_all = self.dataloader.X
        self.y_all = self.dataloader.y

        self.quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.make_quantile_target()

        self.trialpath = self.savepath / 'Trials'
        self.trialpath.mkdir(parents=True, exist_ok=True)

        self.modelpath = self.savepath / 'Models'
        self.modelpath.mkdir(parents=True, exist_ok=True)

        self.fcpath = self.savepath / 'Forecasts'
        self.fcpath.mkdir(parents=True, exist_ok=True)

        self.split_data()
        
    def make_quantile_target(self):
        """Voor QR gebruik je dezelfde variabele meerdere keren, voor elke kwantiel een keer. 
        Vandaar deze 'hack', kan geheugen-vriendelijker maar heeft even geen prioriteit."""
        
        cols = [col + ' q' + str(q) for q in self.quantiles for col in self.y_all.columns]
        y = pd.DataFrame(index = self.y_all.index, columns=cols)
        y.loc[:,:] = np.concatenate([self.y_all.values for _ in self.quantiles], axis=1)
        self.y_all = y.copy()

    def get_features(self, trial):
        """Select features based on trials"""
        features = []

        MI_markets = {market: trial.suggest_categorical(f'MI {market}', [0, 1, 2]) for market in self.dataloader.MI_markets}
        market_features = [self.dataloader.market] + [str(k) for k in MI_markets.keys() if MI_markets[k] >= 1]
        target_markets = [self.dataloader.market] + [str(k) for k in MI_markets.keys() if MI_markets[k] == 2]
        # 0 = None, 1 = MI in features, 2 = MI in features and target

        t0 = self.dataloader.lagrange[0]
        t_f = self.dataloader.lagrange[-1]
        # DAM data
        dam_lag = trial.suggest_int('DAM lag', t0, t_f-1, step=1)
        dam_d7 = trial.suggest_categorical('DAM d-7 lag', [True, False])
        damrange = np.arange(t0, dam_lag)
        features += [f'DAM {market} H{h} lag{lag}' for market in market_features for h in range(24) for lag in damrange]
        if dam_d7:
            features += [f'DAM {market} H{h} lag{t0}' for market in market_features for h in range(24)]

        # Load data
        load_lag = trial.suggest_int('load lag', t0, t_f-1, step=1)
        load_d7 = trial.suggest_categorical('load d-7 lag', [True, False])
        load_fc = trial.suggest_categorical('load fc', [True, False])
        loadrange = np.arange(t0, load_lag)

        for market in market_features:
            load_status = trial.suggest_categorical(f'load {market} data', ['None', 'load', 'load fc'])
            if load_status == 'None':
                # no load data
                pass
            else:
                features += [f'{load_status} {market} H{h} lag{lag}' for h in range(24) for lag in loadrange]
                if load_d7:
                    features += [f'{load_status} {market} H{h} lag{self.dataloader.lagrange[-1]}' for h in range(24)]
            
            if load_fc:
                features += [f'load fc {market} H{h} +1' for h in range(24)]

        
            # Generation data - follows load lags
            historic_gen = trial.suggest_categorical(f'lagged gen {market}', [True, False])
            if historic_gen:
                for gentype in self.dataloader.renewables[market]:
                    features += [f'{gentype} {market} H{h} lag{lag}' for h in range(24) for lag in loadrange]
                    if load_d7:
                        features += [f'{gentype} {market} H{h} lag{self.dataloader.lagrange[-1]}' for h in range(24)]

            future_gen = trial.suggest_categorical(f'future gen {market}', [True, False])
            if future_gen:
                for gentype in self.dataloader.renewables[market]:
                    features += [f'{gentype} {market} H{h} +1' for h in range(24)]

        # Temporal features
        doy_type = trial.suggest_categorical('DOY', ['None', 'abs', 'cyclic'])
        if doy_type == 'abs':
            features += ['DOY abs']
        elif doy_type == 'cyclic':
            features += ['DOY sin', 'DOY cos']
        
        if trial.suggest_categorical('DOW', [True, False]):
            features += ['DOW']

        if trial.suggest_categorical('gas price', [True, False]):
            features += ['gas price lag0']
        return features, target_markets

    def select_features(self, features):
        """Select features based on trials"""
        self.X_train = self.X_train_all[features]
        self.X_val = self.X_val_all[features]
        self.X_test = self.X_test_all[features]

    def select_targets(self, target_markets):
        """Select targets based on trials"""
        damcols = [col for col in self.y_train_all.columns if col.split(' ')[1] in target_markets]
        self.y_train = self.y_train_all[damcols]
        self.y_val = self.y_val_all[damcols]
        self.y_test = self.y_test_all[damcols]

    def get_mlp(self, trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = [trial.suggest_int(f'n_units_l{n_layers}_{i}', 50, 450, 1) for i in range(1, n_layers+1)]

        dropout = trial.suggest_float('dropout', 1e-5, 0.5, log=True)
        regularization = trial.suggest_float('regularization', 1e-5, 1e-1, log=True)
        batch_normalization = trial.suggest_categorical('batch_normalization', [True, False])
        seed = trial.suggest_int('seed', 0, 1000, 1)
        self.set_seeds(seed)
        
        with tf.device(self.device):
            model = self.create_model(
                modeltype='MLP',
                n_features=self.X_train.shape[1],
                n_targets=self.y_train.shape[1],
                mlp_layers=layers,
                mlp_dropout=dropout,
                mlp_batchnorm=batch_normalization,
                regularization=regularization,
                seed=seed,
                dropout_seed=None
            )
        return model
    
    def drop_nan_data(self, x, y):
        """Equalize datasets by removing the indices they dont share"""
        x = x.loc[y.dropna().index].dropna()
        y = y.loc[x.dropna().index].dropna()
        return x, y

    def make_pred_df(self, y_pred, y_true):
        """Make a dataframe with the predictions and true values"""
        pred_df = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)
        return pred_df
    
    def split_data(self):
        """Split data in train, validation and test set"""
        train_years = [2018, 2020]
        val_years = [2019]
        test_years = [2021, 2022]

        self.X_train_all = self.X_all.loc[self.X_all.index.year.isin(train_years)]
        self.y_train_all = self.y_all.loc[self.y_all.index.year.isin(train_years)]

        self.X_val_all = self.X_all.loc[self.X_all.index.year.isin(val_years)]
        self.y_val_all = self.y_all.loc[self.y_all.index.year.isin(val_years)]

        self.X_test_all = self.X_all.loc[self.X_all.index.year.isin(test_years)]
        self.y_test_all = self.y_all.loc[self.y_all.index.year.isin(test_years)]

        self.y_train_nl = self.y_train_all[[col for col in self.y_train_all.columns if col.split(' ')[1] == self.dataloader.market]]
        self.y_val_nl = self.y_val_all[[col for col in self.y_val_all.columns if col.split(' ')[1] == self.dataloader.market]]
        self.y_test_nl = self.y_test_all[[col for col in self.y_test_all.columns if col.split(' ')[1] == self.dataloader.market]]

    def train_score_model(self, trial):
        features, targets = self.get_features(trial)
        self.select_features(features)
        self.select_targets(targets)

        scaler = StandardScaler()
        x_train, y_train = self.drop_nan_data(self.X_train, self.y_train)
        x_val, y_val = self.drop_nan_data(self.X_val, self.y_val)
        x_test, y_test = self.drop_nan_data(self.X_test, self.y_test)

        x_train_arr = scaler.fit_transform(x_train)
        x_val_arr = scaler.transform(x_val)
        x_test_arr = scaler.transform(x_test)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
        val_data = (x_val_arr.astype('float32'), y_val.values.astype('float32'))
        
        model = self.get_mlp(trial)
        with tf.device(self.device):
            model.fit(
                x_train_arr.astype('float32'),
                y_train.values.astype('float32'),
                validation_data=val_data,
                epochs=10000,
                batch_size=trial.suggest_int('batch_size', 1, 64, 1),
                verbose=0,
                callbacks=[es]
            )
        p_train = self.make_pred_df(model.predict(x_train_arr.astype('float32')), y_train)
        p_val = self.make_pred_df(model.predict(x_val_arr.astype('float32')), y_val)
        p_test = self.make_pred_df(model.predict(x_test_arr.astype('float32')), y_test)

        target_cols = [col for col in p_train.columns if self.dataloader.market in col]
        
        train_loss = float(tf.cast(self.calculate_loss(y_train, p_train, target_cols), dtype=tf.float32))
        val_loss = float(tf.cast(self.calculate_loss(y_val, p_val, target_cols), dtype=tf.float32))
        test_loss = float(tf.cast(self.calculate_loss(y_test, p_test, target_cols), dtype=tf.float32))

        # print(f'Losses: train: {train_loss:.3f}, val: {val_loss:.3f}, test: {test_loss:.3f}')
        # Make df so we can easily calculate metrics for NL forecasts only

        if val_loss < self.best_loss:
            model.save(self.modelpath.resolve())
            self.best_val_loss = val_loss
            self.best_train_loss = train_loss
            self.best_test_loss = test_loss

            # save the x and y data for the best model
            x_train.to_pickle(self.fcpath / 'x_train.pkl')
            y_train.to_pickle(self.fcpath / 'y_train.pkl')
            x_val.to_pickle(self.fcpath / 'x_val.pkl')
            y_val.to_pickle(self.fcpath / 'y_val.pkl')
            x_test.to_pickle(self.fcpath / 'x_test.pkl')
            y_test.to_pickle(self.fcpath / 'y_test.pkl')
            p_train.to_pickle(self.fcpath / 'p_train.pkl')
            p_val.to_pickle(self.fcpath / 'p_val.pkl')
            p_test.to_pickle(self.fcpath / 'p_test.pkl')
            with open(self.fcpath / 'scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

        trial.set_user_attr('val_loss', np.round(val_loss, decimals=3))
        trial.set_user_attr('train_loss', np.round(train_loss, decimals=3))
        trial.set_user_attr('test_loss', np.round(test_loss, decimals=3))

        return val_loss

    def calculate_loss(self, y_true, y_pred, market_cols):
        """Calculate the loss for a given market"""
        return self.combined_quantile_loss(tf.convert_to_tensor(y_true[market_cols].values.astype('float32')), tf.convert_to_tensor(y_pred[market_cols].values.astype('float32')))
    
    def run(self, n_trials=400, restart=False, show_progress_bar=True, n_startup_trials=25):
        """Run the hyperparameter search"""
        dbpath = str((self.savepath / f'{self.name} study').resolve())
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup_trials), storage=f'sqlite:///{dbpath}.db', study_name=self.name, load_if_exists=restart)
        if restart:
            try:
                self.best_loss = study.best_value
            except:
                pass
        
        study.optimize(self.train_score_model, n_trials=n_trials, show_progress_bar=show_progress_bar)
        return study

class OnlineTraining(KerasMixin):
    """
    Class for online training of the model with the best parameters.
    Iterating over days in the training set, retraining and moving 
    data to and from the training and validation sets.
    """

    def __init__(self, name, dataloader, savepath, study=None, modeltype='MLP', gpu=True):
        self.name = name
        self.dataloader = dataloader
        self.savepath = Path(savepath)
        self.modeltype = modeltype
        self.gpu = gpu

        self.dbpath = str((self.savepath / f'{self.name} study').resolve())

        if study is None:
            self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(n_startup_trials=25), storage=f'sqlite:///{self.dbpath}.db', study_name=self.name, load_if_exists=True)
        else:
            self.study = study
        
        self.best_trial = self.study.best_trial
        self.TPE = TPESearch(
            dataloader=self.dataloader,
            savepath=self.savepath,
            name=self.name,
            modeltype=self.modeltype,
            gpu=self.gpu
        )

        self.features, self.target_markets = self.TPE.get_features(self.best_trial)
        self.TPE.select_features(self.features)
        self.TPE.select_targets(self.target_markets)

        self.X_train, self.y_train = self.TPE.drop_nan_data(self.TPE.X_train, self.TPE.y_train)
        self.X_val, self.y_val = self.TPE.drop_nan_data(self.TPE.X_val, self.TPE.y_val)
        self.X_test, self.y_test = self.TPE.drop_nan_data(self.TPE.X_test, self.TPE.y_test)

        self.l_train_0 = self.X_train.loc[self.X_train.index.year==2018].shape[0]
        self.l_train_1 = self.X_train.loc[self.X_train.index.year==2020].shape[0]
        self.l_val = self.X_val.shape[0]
        self.l_test = self.X_test.shape[0]

        self.fcpath = self.savepath / 'Online training'
        self.fcpath.mkdir(parents=True, exist_ok=True)


        KerasMixin.__init__(self)

        if self.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            self.device = '/gpu:0'
        else:
            self.device = '/cpu'

        self.fcpath = self.savepath / 'Forecasts'
        self.fcpath.mkdir(parents=True, exist_ok=True)

    def update_datasets(self):
        x_train0 = self.X_train.iloc[:self.l_train_0]
        x_train1 = self.X_train.iloc[self.l_train_0:self.l_train_0+self.l_train_1]

        y_train0 = self.y_train.iloc[:self.l_train_0]
        y_train1 = self.y_train.iloc[self.l_train_0:self.l_train_0+self.l_train_1]

        x_val = self.X_val
        y_val = self.y_val

        x_test = self.X_test
        y_test = self.y_test

        x_train0 = pd.concat([x_train0.iloc[1:], x_val.iloc[:1]])
        x_val = pd.concat([x_val.iloc[1:], x_train1.iloc[:1]])
        x_train1 = pd.concat([x_train1.iloc[1:], x_test.iloc[:1]])
        x_test = x_test.iloc[1:]

        y_train0 = pd.concat([y_train0.iloc[1:], y_val.iloc[:1]])
        y_val = pd.concat([y_val.iloc[1:], y_train1.iloc[:1]])
        y_train1 = pd.concat([y_train1.iloc[1:], y_test.iloc[:1]])
        y_test = y_test.iloc[1:]

        self.X_train = pd.concat([x_train0, x_train1])
        self.X_val = x_val
        self.X_test = x_test

        self.y_train = pd.concat([y_train0, y_train1])
        self.y_val = y_val
        self.y_test = y_test

    def run(self):
        """
        Train the model on the training set and perform early stopping on the validation set.
        Iterating over the test set, retraining and moving data to and from the training and validation sets.
        """

        if (self.fcpath / 'p_test_online.pkl').exists():
            p_test = pd.read_pickle(self.fcpath / 'p_test_online.pkl')
        else:
            p_test = pd.DataFrame(index=self.y_test.index, columns=self.y_test.columns)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
        already_done = p_test.dropna().index

        for i, idx in enumerate(tqdm(self.y_test.index)):
            if i in already_done:
                self.update_datasets()
                continue

            scaler = StandardScaler()
            x_train_arr = scaler.fit_transform(self.X_train)
            x_val_arr = scaler.transform(self.X_val)
            x_test_arr = scaler.transform(self.X_test)

            val_data = (x_val_arr.astype('float32'), self.y_val.values.astype('float32'))
            model = self.TPE.get_mlp(self.best_trial)
            with tf.device(self.device):
                model.fit(
                    x_train_arr.astype('float32'),
                    self.y_train.values.astype('float32'),
                    validation_data=val_data,
                    epochs=10000,
                    batch_size=self.best_trial.params['batch_size'],
                    verbose=0,
                    callbacks=[es]
                )

            p_test.loc[idx] = model.predict(x_test_arr.astype('float32')[0, :].reshape(1, -1))[0]
            p_test.to_pickle(self.fcpath / 'p_test_online.pkl')
            p_test.to_csv(self.fcpath / 'p_test_online.csv')
            self.update_datasets()