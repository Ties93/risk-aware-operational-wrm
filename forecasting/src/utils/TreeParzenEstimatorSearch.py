from xmlrpc.client import Boolean
import numpy as np
from .KerasModels import *
from .KerasDataGenerator import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from tqdm import tqdm, trange
from functools import partial
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from typing import Optional, Union, List, Callable
from pathlib import Path

class SplitByYear():
    def __init__(self, train_years: List[int] = [2016, 2017], val_years: List[int] = [2018], test_years: List[int] = [2019, 2020, 2021]):
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years

        self.x_train=None
        self.x_val=None
        self.x_test=None
        self.y_train=None
        self.y_val=None
        self.y_test=None

    def split_data(self, X, y):
        self.x_train = X.loc[X.index.year.isin(self.train_years)]
        self.y_train = y.loc[y.index.year.isin(self.train_years)]
        
        self.x_val = X.loc[X.index.year.isin(self.val_years)]
        self.y_val = y.loc[y.index.year.isin(self.val_years)]
        
        self.x_test = X.loc[X.index.year.isin(self.test_years)]
        self.y_test = y.loc[y.index.year.isin(self.test_years)]

class TPESearch(KerasMixin, SplitByYear):
    def __init__(
        self, 
        name: str, 
        features: pd.DataFrame, 
        target: pd.DataFrame, 
        savepath: Path,
        hyperparameter_space: dict,
        feature_space: dict,
        select_features: Callable,
        modeltype: str='MLP',
        gpu=True
        ):

        print('Applying TPE search with Tensorflow version '+str(tf.version.VERSION))
        KerasMixin.__init__(self)
        SplitByYear.__init__(self)

        self.name = name
        self.modeltype=modeltype
        
        if gpu:
            self.device='/gpu:0'
        else:
            self.device='/cpu'

        self.hyperparameter_space = hyperparameter_space
        self.feature_space = feature_space

        self.X_all = features
        self.y_all = target

        self.select_features=select_features

        self.quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.make_quantile_target()

        self.savepath = savepath
        self.savepath.mkdir(exist_ok=True, parents=True)
        (self.savepath / 'Models').mkdir(exist_ok=True)

        self.trialpath = self.savepath / 'Trials'
        self.trialpath.mkdir(exist_ok=True)
        
        self.fcpath = self.savepath / 'Forecasts'
        self.fcpath.mkdir(exist_ok=True)
        
        self.best_loss = np.inf
        self.test_loss = np.inf
        
        self.trials = Trials()
        self.trials_df = None
        self.iteration = 0

        self.train_index = None
        self.val_index = None
        self.test_index = None

        self.N_improvements = 0
        self.best_model = None

        self.model = None
        print('TPE Search initialised with '+modeltype+' settings.')

    def reload_trials(self):
        with open(self.savepath / 'Trials' / (self.name+'.pickle'), 'rb') as pickle_file:
            self.trials = pickle.load(pickle_file)
            if self.trials.losses()[-1] == None:
                self.trials.trials.remove(self.trials.trials[-1])
            self.iteration = len(self.trials.trials)
            self.trials_df = self.trials_to_df()

    def read_val_from_trials(self, val):
        """Kleine handigheidje om fouten bij het lezen van de trials te voorkomen."""
        try:
            return val[0]
        except:
            return 0

    def trials_to_df(self):
        """Een functie om je trails file (ingewikkelde dictionary) om te zetten in een DF."""
        df = pd.DataFrame(index = range(len(self.trials.trials)), columns = self.trials.trials[0]['misc']['vals'].keys())
        df.loc[:, :] = [[self.read_val_from_trials(self.trials.trials[i]['misc']['vals'][k]) for k in self.trials.trials[i]['misc']['vals'].keys()] for i in range(len(self.trials.trials))]
        df.loc[:, 'test_loss'] = [self.trials.trials[i]['result']['test_loss'] for i in range(len(self.trials.trials))]
        df.loc[:, 'val_loss'] = [self.trials.trials[i]['result']['val_loss'] for i in range(len(self.trials.trials))]
        df.loc[:, 'train_loss'] = [self.trials.trials[i]['result']['train_loss'] for i in range(len(self.trials.trials))]
        
        return df

    def make_quantile_target(self):
        """Voor QR gebruik je dezelfde variabele meerdere keren, voor elke kwantiel een keer. 
        Vandaar deze 'hack', kan geheugen-vriendelijker maar heeft even geen prioriteit."""
        
        cols = [col + ' q' + str(q) for q in self.quantiles for col in self.y_all.columns]
        y = pd.DataFrame(index = self.y_all.index, columns=cols)
        y.loc[:,:] = np.concatenate([self.y_all.values for _ in self.quantiles], axis=1)
        self.y_all = y.copy()

    def reshape_input_for_tst(self):
        self.x_train = self.x_train.values
        self.x_train.reshape((self.x_train.shape[0], self.x_train.shape[1], 1))

        self.x_val = self.x_val.values
        self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1], 1))

        self.x_test = self.x_test.values
        self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1], 1))

    def train_score_model(self, params, save_every_new_best=False):
        """Dit is de functie waarin je model getraind en geevalueerd wordt, als functie van een hyperparameter- en feature-space instantiatie."""
        # Zet de seed
        self.seed = int(params['Seed'])
        self.set_seeds(self.seed)
        
        # Selecteer features
        if self.select_features is not None:
            input_data = self.select_features(self, params)
        else:
            if self.iteration == 0:
                print("No 'select_features' function given in the class, using all features.")
            input_data = self.X_all.copy()
        
        # Split je data in train en validatie set
        self.split_data(input_data, self.y_all)

        x_traindf = self.x_train.copy()
        x_valdf = self.x_val.copy()
        x_testdf = self.x_test.copy()

        # Schaal je variabelen, en pas early stopping toe
        # Het bepalen van de schalingsfactor doe je op de traning set.
                
        if self.modeltype=='MLP':
            self.get_mlp(params)
        elif self.modeltype=='TST':
            self.get_transformer(params)
            self.reshape_input_for_tst()
        else:
            print('Wrong modeltype.')
        
        scaler = StandardScaler().fit(self.x_train)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        val_data = (scaler.transform(self.x_val).astype('float32'), self.y_val.values.astype('float32'))

        with tf.device(self.device):
            self.model.fit(
                x=scaler.transform(self.x_train).astype('float32'), 
                y=self.y_train.values.astype('float32'), 
                batch_size = 5**int(int(params['Batch size'])), 
                epochs=10000, 
                validation_data=val_data, 
                callbacks=[es],
                verbose=0
                )
        
        # Voorspel/evalueer je train set
        p_val = self.model.predict(scaler.transform(self.x_val).astype('float32'))
        val_loss = self.model.evaluate(scaler.transform(self.x_val).astype('float32'), self.y_val.astype('float32'), verbose=0)
        
        p_test = self.model.predict(scaler.transform(self.x_test).astype('float32'))
        test_loss = self.model.evaluate(scaler.transform(self.x_test).astype('float32'), self.y_test.astype('float32'), verbose=0)
        
        # Voorspel/evalueer je validatie set
        p_train = self.model.predict(scaler.transform(self.x_train.astype('float32')))
        train_loss = self.model.evaluate(scaler.transform(self.x_train).astype('float32'), self.y_train.astype('float32'), verbose=0)
            
        # Check if val2 loss is the lowest so far, if yes -> save model, FCs and features
        if val_loss < self.best_loss:
            self.model.save(os.path.join(self.savepath, 'Models'))
            self.best_model = tf.keras.models.clone_model(self.model)
            self.best_loss = val_loss
            self.test_loss = test_loss

            self.y_train.to_csv(self.savepath / 'Forecasts' / ('y_train.csv'))
            self.y_val.to_csv(self.savepath / 'Forecasts' / ('y_val.csv'))
            self.y_test.to_csv(self.savepath / 'Forecasts' / ('y_test.csv'))
            
            if isinstance(self.y_train, pd.DataFrame):
                train_df = self.y_train.copy()
                train_df.iloc[:,:] = p_train
                val_df = self.y_val.copy()
                val_df.iloc[:,:] = p_val
                test_df = self.y_test.copy()
                test_df.iloc[:,:] = p_test
                

                if save_every_new_best:
                    pre_name = str(self.N_improvements)
                else:
                    pre_name = ''

                train_df.to_csv(self.savepath / 'Forecasts' / (pre_name+'train_fc.csv'))
                val_df.to_csv(self.savepath / 'Forecasts' / (pre_name+'val_fc.csv'))
                test_df.to_csv(self.savepath / 'Forecasts' / (pre_name+'test_fc.csv'))

                x_traindf.to_csv(self.savepath / 'Forecasts' / ('train_features.csv'))
                x_valdf.to_csv(self.savepath / 'Forecasts' / ('val_features.csv'))
                x_testdf.to_csv(self.savepath / 'Forecasts' / ('test_features.csv'))

            self.N_improvements += 1
        
        return {'loss': val_loss, 'status': STATUS_OK, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
        
    def get_mlp(self, params):
        # Maak een tuple met je model structuur
        mlp_layers = [int(params['N hidden layers'][k]) for k in params['N hidden layers'].keys() if k.startswith('Hidden nodes layer') and int(params['N hidden layers'][k]) > 0]
        
        # Maak je model!
        with tf.device(self.device):
            self.model = self.create_model(
                modeltype='MLP',
                n_features=self.x_train.shape[1],
                n_targets=self.y_train.shape[1],
                mlp_layers=mlp_layers,
                mlp_dropout=params['Dropout'],
                mlp_batchnorm=params['Batch normalization'],
                regularization=params['Regularization'],
                seed=int(params['Seed']),
                dropout_seed=None
            )
    
    def get_transformer(self, params):
        N_layers = int(params['N mlp layers']['N'])
        mlp_layers = [int(params['N mlp layers']['dim_'+str(N_layers)]) for _ in range(N_layers)]

        with tf.device(self.device):
            self.model = self.create_model(
                modeltype='TST',
                n_features=self.x_train.shape[1],
                n_targets=self.y_train.shape[1],
                mlp_layers=mlp_layers,
                mlp_dropout=params['MLP dropout'],
                mlp_batchnorm=params['Batch normalization'],
                regularization=params['Regularization'],
                seed=int(params['Seed']),
                attention_head_size=4**int(params['Attention head size']),
                n_attention_heads=4**int(params['N attention heads']),
                transformer_ff_dim=4**int(params['Transformer filter dimension']),
                num_encoder_decoder_blocks=int(params['N encoder decoder blocks']),
                encoder_dropout=int(params['Encoder dropout']),
            )

    def TreeParzenEstimatorSearch(self, evals=400, init_evals=25, minimize=True, check_restart=True, save_features=True, save_every_new_best=False):
        """De TPE Search functie is de functie die je uiteindelijk aanroept. 
        Deze functie is een schil om je model train en optimalisatie heen, en voert die steeds uit met andere features en hyperparameters."""
        
        # Check if the optimization is restarted, so it will continue where it last stopped.
        if check_restart:
            try:
                self.reload_trials()
                print('Starting search from iteration '+str(self.iteration))
            except:
                print('Starting a new search.')
        else:
            print('Starting a new search.')
        
        # Combine the feature- and hyperparameter search space
        search_space = {}
        for k in self.hyperparameter_space.keys():
            search_space[k] = self.hyperparameter_space[k]
        for k in self.feature_space.keys():
            search_space[k] = self.feature_space[k]
        
        # Start the TPE loop
        pbar=trange(self.iteration, evals, initial=self.iteration, total=evals)
        for t in pbar:
            best = fmin(
                fn=self.train_score_model,
                space=search_space,
                max_evals=self.iteration+1,
                algo=partial(tpe.suggest, n_startup_jobs=init_evals),
                trials=self.trials,
                verbose=0
            )
                
            self.trials_df = self.trials_to_df()
            self.trials_df.to_csv(self.trialpath / (self.name+'_df.csv'))
            best_ = self.trials_df.loc[self.trials_df.val_loss == self.trials_df.val_loss.min()]
            with open(self.trialpath / (self.name+'.pickle'), 'wb') as pickle_file:
                pickle.dump(self.trials, pickle_file)
            
            self.iteration += 1
            pbar.set_postfix({'Lowest val loss': np.round(self.best_loss, decimals=2), 'Corresponding test loss': np.round(self.test_loss, decimals=2)})


class TPESearchUniformDataGenerator(KerasMixin, SplitByYear):
    def __init__(
        self, 
        name: str, 
        features: pd.DataFrame, 
        target: pd.DataFrame, 
        trainval_labels,
        savepath: Path,
        hyperparameter_space: dict,
        feature_space: dict,
        select_features: Callable,
        modeltype: str='MLP',
        gpu=True,
        generator='uniform'
        ):

        print('Applying TPE search with Tensorflow version '+str(tf.version.VERSION))
        KerasMixin.__init__(self)
        SplitByYear.__init__(self)

        self.name = name
        self.modeltype=modeltype
        self.generator=generator
        self.trainval_labels=trainval_labels
        
        if gpu:
            self.device='/gpu:0'
        else:
            self.device='/cpu'

        self.hyperparameter_space = hyperparameter_space
        self.feature_space = feature_space

        self.X_all = features
        self.y_all = target

        self.select_features=select_features

        self.quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.make_quantile_target()

        self.savepath = savepath
        self.savepath.mkdir(exist_ok=True, parents=True)
        (self.savepath / 'Models').mkdir(exist_ok=True)

        self.trialpath = self.savepath / 'Trials'
        self.trialpath.mkdir(exist_ok=True)
        
        self.fcpath = self.savepath / 'Forecasts'
        self.fcpath.mkdir(exist_ok=True)
        
        self.best_loss = np.inf
        self.test_loss = np.inf
        
        self.trials = Trials()
        self.trials_df = None
        self.iteration = 0

        self.train_index = None
        self.val_index = None
        self.test_index = None

        self.N_improvements = 0
        self.best_model = None

        self.model = None
        print('TPE Search initialised with '+modeltype+' settings.')

    def reload_trials(self):
        with open(self.savepath / 'Trials' / (self.name+'.pickle'), 'rb') as pickle_file:
            self.trials = pickle.load(pickle_file)
            if self.trials.losses()[-1] == None:
                self.trials.trials.remove(self.trials.trials[-1])
            self.iteration = len(self.trials.trials)
            self.trials_df = self.trials_to_df()

    def read_val_from_trials(self, val):
        """Kleine handigheidje om fouten bij het lezen van de trials te voorkomen."""
        try:
            return val[0]
        except:
            return 0

    def trials_to_df(self):
        """Een functie om je trails file (ingewikkelde dictionary) om te zetten in een DF."""
        df = pd.DataFrame(index = range(len(self.trials.trials)), columns = self.trials.trials[0]['misc']['vals'].keys())
        df.loc[:, :] = [[self.read_val_from_trials(self.trials.trials[i]['misc']['vals'][k]) for k in self.trials.trials[i]['misc']['vals'].keys()] for i in range(len(self.trials.trials))]
        df.loc[:, 'test_loss'] = [self.trials.trials[i]['result']['test_loss'] for i in range(len(self.trials.trials))]
        df.loc[:, 'val_loss'] = [self.trials.trials[i]['result']['val_loss'] for i in range(len(self.trials.trials))]
        df.loc[:, 'train_loss'] = [self.trials.trials[i]['result']['train_loss'] for i in range(len(self.trials.trials))]
        
        return df

    def make_quantile_target(self):
        """Voor QR gebruik je dezelfde variabele meerdere keren, voor elke kwantiel een keer. 
        Vandaar deze 'hack', kan geheugen-vriendelijker maar heeft even geen prioriteit."""
        
        cols = [col + ' q' + str(q) for q in self.quantiles for col in self.y_all.columns]
        y = pd.DataFrame(index = self.y_all.index, columns=cols)
        y.loc[:,:] = np.concatenate([self.y_all.values for _ in self.quantiles], axis=1)
        self.y_all = y.copy()

    def reshape_input_for_tst(self):
        self.x_train = self.x_train.values
        self.x_train.reshape((self.x_train.shape[0], self.x_train.shape[1], 1))

        self.x_val = self.x_val.values
        self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1], 1))

        self.x_test = self.x_test.values
        self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1], 1))

    def make_train_generator(self, batch_size):
        if self.generator=='uniform':
            train_generator=UniformDataGenerator(X=self.x_train, y=self.y_train.values, labels=self.train_labels, batch_size=batch_size)
            # val_generator=DataGenerator(X=self.x_val, y=self.y_val, batch_size=batch_size)
        else:
            train_generator=DataGenerator(X=self.x_train, y=self.y_train.values, batch_size=batch_size)
            # val_generator=DataGenerator(X=self.x_val, y=self.y_val, batch_size=batch_size)
        
        return train_generator#, val_generator


    def train_score_model(self, params, save_every_new_best=False):
        """Dit is de functie waarin je model getraind en geevalueerd wordt, als functie van een hyperparameter- en feature-space instantiatie."""
        # Zet de seed
        self.seed = int(params['Seed'])
        self.set_seeds(self.seed)
        
        # Selecteer features
        if self.select_features is not None:
            input_data = self.select_features(self, params)
        else:
            if self.iteration == 0:
                print("No 'select_features' function given in the class, using all features.")
            input_data = self.X_all.copy()
        
        # Split je data in train en validatie set
        self.split_data(input_data, self.y_all)
        
        # Concat de op jaar geplitste traine n val set
        self.x_trainval = pd.concat([self.x_train, self.x_val], axis=0).sort_index()
        self.y_trainval = pd.concat([self.y_train, self.y_val], axis=0).sort_index()
        
        # Stratified split van de twee
        self.x_train, self.x_val, self.y_train, self.y_val, self.train_labels, self.val_labels = train_test_split(self.x_trainval, self.y_trainval, self.trainval_labels, train_size=0.7, shuffle=True, stratify=self.trainval_labels, random_state=1)
        
        # self.x_train = x_train
        # self.x_val = x_val
        # self.y_train = y_train
        # self.y_val = y_val
        # self.train_labels = train_labels
        # self.val_labels = val_labels
        

        x_traindf = self.x_train.copy()
        x_valdf = self.x_val.copy()
        x_testdf = self.x_test.copy()

        self.y_trainval.to_csv(self.savepath / 'Forecasts' / ('y_trainval.csv'))
        self.y_test.to_csv(self.savepath / 'Forecasts' / ('y_test.csv'))

        # Schaal je variabelen, en pas early stopping toe
        # Het bepalen van de schalingsfactor doe je op de traning set.
                
        if self.modeltype=='MLP':
            self.get_mlp(params)
        elif self.modeltype=='TST':
            self.get_transformer(params)
            self.reshape_input_for_tst()
        else:
            print('Wrong modeltype.')

        scaler = StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)

        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        val_data = (scaler.transform(self.x_val).astype('float32'), self.y_val.values.astype('float32'))
        # train_generator, val_generator = self.make_train_val_generator(batch_size=5**int(int(params['Batch size'])))
        train_generator = self.make_train_generator(batch_size=5**int(int(params['Batch size'])))

        with tf.device(self.device):
            self.model.fit(
                x=train_generator,
                validation_data=val_data, 
                steps_per_epoch=train_generator.__len__(),
                epochs=10000, 
                callbacks=[es],
                verbose=0
                )
        
        # Voorspel/evalueer je train set
        p_val = self.model.predict(scaler.transform(self.x_val).astype('float32'))
        val_loss = self.model.evaluate(scaler.transform(self.x_val).astype('float32'), self.y_val.astype('float32'), verbose=0)
        
        p_test = self.model.predict(scaler.transform(self.x_test).astype('float32'))
        test_loss = self.model.evaluate(scaler.transform(self.x_test).astype('float32'), self.y_test.astype('float32'), verbose=0)
        
        # Voorspel/evalueer je validatie set
        p_train = self.model.predict(scaler.transform(self.x_train.astype('float32')))
        train_loss = self.model.evaluate(scaler.transform(self.x_train).astype('float32'), self.y_train.astype('float32'), verbose=0)
            
        # Check if val2 loss is the lowest so far, if yes -> save model, FCs and features
        if val_loss < self.best_loss:
            self.model.save(os.path.join(self.savepath, 'Models'))
            self.best_model = tf.keras.models.clone_model(self.model)
            self.best_loss = val_loss
            self.test_loss = test_loss
            
            if isinstance(self.y_train, pd.DataFrame):
                train_df = self.y_train.copy()
                train_df.iloc[:,:] = p_train
                val_df = self.y_val.copy()
                val_df.iloc[:,:] = p_val
                test_df = self.y_test.copy()
                test_df.iloc[:,:] = p_test
                

                if save_every_new_best:
                    pre_name = str(self.N_improvements)
                else:
                    pre_name = ''

                train_df.to_csv(self.savepath / 'Forecasts' / (pre_name+'train_fc.csv'))
                val_df.to_csv(self.savepath / 'Forecasts' / (pre_name+'val_fc.csv'))
                test_df.to_csv(self.savepath / 'Forecasts' / (pre_name+'test_fc.csv'))

                x_traindf.to_csv(self.savepath / 'Forecasts' / ('train_features.csv'))
                x_valdf.to_csv(self.savepath / 'Forecasts' / ('val_features.csv'))
                x_testdf.to_csv(self.savepath / 'Forecasts' / ('test_features.csv'))

            self.N_improvements += 1
        
        return {'loss': val_loss, 'status': STATUS_OK, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
        
    def get_mlp(self, params):
        # Maak een tuple met je model structuur
        mlp_layers = [int(params['N hidden layers'][k]) for k in params['N hidden layers'].keys() if k.startswith('Hidden nodes layer') and int(params['N hidden layers'][k]) > 0]
        
        # Maak je model!
        with tf.device(self.device):
            self.model = self.create_model(
                modeltype='MLP',
                n_features=self.x_train.shape[1],
                n_targets=self.y_train.shape[1],
                mlp_layers=mlp_layers,
                mlp_dropout=params['Dropout'],
                mlp_batchnorm=params['Batch normalization'],
                regularization=params['Regularization'],
                seed=int(params['Seed']),
                dropout_seed=None
            )
    
    def get_transformer(self, params):
        N_layers = int(params['N mlp layers']['N'])
        mlp_layers = [int(params['N mlp layers']['dim_'+str(N_layers)]) for _ in range(N_layers)]

        with tf.device(self.device):
            self.model = self.create_model(
                modeltype='TST',
                n_features=self.x_train.shape[1],
                n_targets=self.y_train.shape[1],
                mlp_layers=mlp_layers,
                mlp_dropout=params['MLP dropout'],
                mlp_batchnorm=params['Batch normalization'],
                regularization=params['Regularization'],
                seed=int(params['Seed']),
                attention_head_size=4**int(params['Attention head size']),
                n_attention_heads=4**int(params['N attention heads']),
                transformer_ff_dim=4**int(params['Transformer filter dimension']),
                num_encoder_decoder_blocks=int(params['N encoder decoder blocks']),
                encoder_dropout=int(params['Encoder dropout']),
            )

    def TreeParzenEstimatorSearch(self, evals=400, init_evals=25, minimize=True, check_restart=True, save_features=True, save_every_new_best=False):
        """De TPE Search functie is de functie die je uiteindelijk aanroept. 
        Deze functie is een schil om je model train en optimalisatie heen, en voert die steeds uit met andere features en hyperparameters."""
        
        # Check if the optimization is restarted, so it will continue where it last stopped.
        if check_restart:
            try:
                self.reload_trials()
                best_ = self.trials_df.loc[self.trials_df.val_loss.idxmin(), :]
                self.best_loss = best_['val_loss']
                self.test_loss = best_['test_loss']
                print('Starting search from iteration '+str(self.iteration))
            except:
                print('Starting a new search.')
        else:
            print('Starting a new search.')
        
        # Combine the feature- and hyperparameter search space
        search_space = {}
        for k in self.hyperparameter_space.keys():
            search_space[k] = self.hyperparameter_space[k]
        for k in self.feature_space.keys():
            search_space[k] = self.feature_space[k]
        
        # Start the TPE loop
        pbar=trange(self.iteration, evals, initial=self.iteration, total=evals)
        for t in pbar:
            best = fmin(
                fn=self.train_score_model,
                space=search_space,
                max_evals=self.iteration+1,
                algo=partial(tpe.suggest, n_startup_jobs=init_evals),
                trials=self.trials,
                verbose=0
            )
                
            self.trials_df = self.trials_to_df()
            self.trials_df.to_csv(self.trialpath / (self.name+'_df.csv'))
            best_ = self.trials_df.loc[self.trials_df.val_loss == self.trials_df.val_loss.min()]
            with open(self.trialpath / (self.name+'.pickle'), 'wb') as pickle_file:
                pickle.dump(self.trials, pickle_file)
            
            self.iteration += 1
            pbar.set_postfix({'Lowest val loss': np.round(self.best_loss, decimals=2), 'Corresponding test loss': np.round(self.test_loss, decimals=2)})



            
class OnlineTraining(KerasMixin, SplitByYear):
    def __init__(
        self, 
        name: str, 
        features: pd.DataFrame, 
        target: pd.DataFrame, 
        savepath: Path,
        select_features: Callable,
        modeltype: str = 'MLP',
        gpu: Boolean = True
        ):
        
        print('Applying TPE search with Tensorflow version '+str(tf.version.VERSION))
        KerasMixin.__init__(self)
        SplitByYear.__init__(self)
        self.name = name
        self.modeltype=modeltype

        if gpu:
            self.device='/gpu:0'
        else:
            self.device='/cpu'

        self.X_all = features
        self.y_all = target
        self.select_features=select_features

        self.make_quantile_target()

        self.savepath = savepath
        self.trialpath = self.savepath / 'Trials'
        self.fcpath = self.savepath / 'Forecasts'
        
        self.trials = Trials()
        self.trials_df = None
        self.reload_trials()

        self.best_trials = None
        self.best_trials_dict = None
        self.read_best_trials()

        self.iteration = 0

        self.train_index = None
        self.val_index = None
        self.test_index = None

        self.N_improvements = 0
        self.best_model = None

        self.model = None
        self.scaler = None

        self.forecast_df = None
        print('Online training initialised.')
        print('Now add a select_features function to select features based on a feature space instantiation.')

    def reload_trials(self):
        with open(self.savepath / 'Trials' / (self.name+'.pickle'), 'rb') as pickle_file:
            self.trials = pickle.load(pickle_file)
            if self.trials.losses()[-1] == None:
                self.trials.trials.remove(self.trials.trials[-1])
            self.iteration = len(self.trials.trials)
            self.trials_df = self.trials_to_df()

    def read_val_from_trials(self, val):
        """Kleine handigheidje om fouten bij het lezen van de trials te voorkomen."""
        try:
            return val[0]
        except:
            return 0

    def trials_to_df(self):
        """Een functie om je trails file (ingewikkelde dictionary) om te zetten in een DF."""
        df = pd.DataFrame(index = range(len(self.trials.trials)), columns = self.trials.trials[0]['misc']['vals'].keys())
        df.loc[:, :] = [[self.read_val_from_trials(self.trials.trials[i]['misc']['vals'][k]) for k in self.trials.trials[i]['misc']['vals'].keys()] for i in range(len(self.trials.trials))]
        df.loc[:, 'test_loss'] = [self.trials.trials[i]['result']['test_loss'] for i in range(len(self.trials.trials))]
        df.loc[:, 'val_loss'] = [self.trials.trials[i]['result']['val_loss'] for i in range(len(self.trials.trials))]
        df.loc[:, 'train_loss'] = [self.trials.trials[i]['result']['train_loss'] for i in range(len(self.trials.trials))]
        
        return df
        
    def read_best_trials(self):
        cols = [col if not col.endswith('discharge lag') else 'discharge lag' for col in self.trials_df.columns]
        self.trials_df.columns = cols
        self.best_trials = self.trials_df.loc[self.trials_df.val_loss==self.trials_df.val_loss.min(),:]
        best_ind = self.best_trials.index[0]
        self.best_trials = self.best_trials.loc[best_ind]

        int_cols = ['precip lag', 'discharge lag', 'precip FC', 'window size', 'precip window', 'evap window', 'temp window', 'discharge window', 'DOY', 'DOW', 'HOD']
        if (self.name == 'Rijnland') | (self.name == 'HHNK'):
                int_cols += ['H fc']

        if self.modeltype == 'MLP':
            int_cols += ['Hidden nodes layer 1_1', 'Hidden nodes layer 1_2', 'Hidden nodes layer 2', 'N hidden layers', 'Batch size', 'Batch normalization', 'Seed']

        elif self.modeltype == 'TST':
            N = int(self.best_trials['N mlp layers']) +1
            int_cols += ['N mlp layers', 'dim_'+str(N), 'Attention head size', 'N attention heads', 'Transformer filter dimension', 'N encoder decoder blocks', 'Batch size', 'Batch normalization', 'Seed']
        else:
            print('Wrong modeltype?')

        self.best_trials_dict = {col: self.best_trials[col] for col in self.best_trials.index}

        for col in int_cols:
            self.best_trials_dict[col] = int(self.best_trials_dict[col])

        temps = ['none', 'abs', 'cyclic']
        for t in ['HOD', 'DOW', 'DOY']:
            self.best_trials[t] = temps[int(self.best_trials[t])]
            self.best_trials_dict[t] = temps[int(self.best_trials_dict[t])]

    def read_best_trials(self):
        cols = [col if not col.endswith('discharge lag') else 'discharge lag' for col in self.trials_df.columns]
        self.trials_df.columns = cols
        self.best_trials = self.trials_df.loc[self.trials_df.val_loss==self.trials_df.val_loss.min(),:]
        best_ind = self.best_trials.index[0]
        self.best_trials = self.best_trials.loc[best_ind]

        int_cols = ['precip lag', 'discharge lag', 'precip FC', 'window size', 'precip window', 'evap window', 'temp window', 'discharge window', 'DOY', 'DOW', 'HOD']
        if (self.name == 'Rijnland') | (self.name == 'HHNK'):
                int_cols += ['H fc']

        if self.modeltype == 'MLP':
            int_cols += ['Hidden nodes layer 1_1', 'Hidden nodes layer 1_2', 'Hidden nodes layer 2', 'N hidden layers', 'Batch size', 'Batch normalization', 'Seed']

        elif self.modeltype == 'TST':
            N = int(self.best_trials['N mlp layers']) +1
            int_cols += ['N mlp layers', 'dim_'+str(N), 'Attention head size', 'N attention heads', 'Transformer filter dimension', 'N encoder decoder blocks', 'Batch size', 'Batch normalization', 'Seed']
        else:
            print('Wrong modeltype?')

        self.best_trials_dict = {col: self.best_trials[col] for col in self.best_trials.index}

        for col in int_cols:
            self.best_trials_dict[col] = int(self.best_trials_dict[col])

        temps = ['none', 'abs', 'cyclic']
        for t in ['HOD', 'DOW', 'DOY']:
            self.best_trials[t] = temps[int(self.best_trials[t])]
            self.best_trials_dict[t] = temps[int(self.best_trials_dict[t])]

    def make_quantile_target(self):
        """Voor QR gebruik je dezelfde variabele meerdere keren, voor elke kwantiel een keer. 
        Vandaar deze 'hack', kan geheugen-vriendelijker maar heeft even geen prioriteit."""
        
        cols = [col + ' q' + str(q) for q in self.quantiles for col in self.y_all.columns]
        y = pd.DataFrame(index = self.y_all.index, columns=cols)
        y.loc[:,:] = np.concatenate([self.y_all.values for _ in self.quantiles], axis=1)
        self.y_all = y.copy()
        
    def get_mlp(self, mlp_layers, params):
        # Maak je model!
        with tf.device(self.device):
            self.model = self.create_model(
                modeltype='MLP',
                n_features=self.x_train.shape[1],
                n_targets=self.y_train.shape[1],
                mlp_layers=mlp_layers,
                mlp_dropout=params['Dropout'],
                mlp_batchnorm=params['Batch normalization'],
                regularization=params['Regularization'],
                seed=int(params['Seed']),
                dropout_seed=None
            )
    
    def get_transformer(self, mlp_layers, params):
        with tf.device(self.device):
            self.model = self.create_model(
                modeltype='TST',
                n_features=self.x_train.shape[1],
                n_targets=self.y_train.shape[1],
                mlp_layers=mlp_layers,
                mlp_dropout=params['MLP dropout'],
                mlp_batchnorm=params['Batch normalization'],
                regularization=params['Regularization'],
                seed=int(params['Seed']),
                attention_head_size=4**int(params['Attention head size']),
                n_attention_heads=4**int(params['N attention heads']),
                transformer_ff_dim=4**int(params['Transformer filter dimension']),
                num_encoder_decoder_blocks=int(params['N encoder decoder blocks']),
                encoder_dropout=int(params['Encoder dropout']),
            )

    def make_train_model(self, mlp_layers, params):
        self.set_seeds(self.seed)
        if self.modeltype=='MLP':
            self.get_mlp(mlp_layers, params)
        elif self.modeltype=='TST':
            self.get_transformer(mlp_layers, params)

        self.scaler = StandardScaler().fit(self.x_train)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        val_data = (self.scaler.transform(self.x_val).astype('float32'), self.y_val.values.astype('float32'))
        
        with tf.device(self.device):
            self.model.fit(
                x=self.scaler.transform(self.x_train).astype('float32'), 
                y=self.y_train.values.astype('float32'), 
                batch_size = 5**int(int(params['Batch size'])), 
                epochs=10000, 
                validation_data=val_data, 
                callbacks=[es],
                verbose=0
                )
    
    def update_train_val(self, retrain_step):
        # Split training set in time before and after val set
        x_train_split_max = self.x_train.loc[self.x_train.index > self.x_val.index[-1]]
        x_train_split_min = self.x_train.loc[~self.x_train.index.isin(x_train_split_max.index)]
        y_train_split_max = self.y_train.loc[self.y_train.index > self.y_val.index[-1]]
        y_train_split_min = self.y_train.loc[~self.y_train.index.isin(y_train_split_max.index)]

        # Add train data just newer than the val set to the val set
        self.x_val = pd.concat([self.x_val, x_train_split_max.iloc[:retrain_step,:]])
        self.y_val = pd.concat([self.y_val, y_train_split_max.iloc[:retrain_step,:]])

        # Add test data to the train set, and remove the first entries
        x_train_split_max = pd.concat([x_train_split_max.iloc[retrain_step:,:], self.x_test.iloc[:retrain_step,:]])
        x_train_split_min = pd.concat([x_train_split_min.iloc[retrain_step:,:], self.x_val.iloc[:retrain_step,:]])
        self.x_train = pd.concat([x_train_split_min, x_train_split_max])
        y_train_split_max = pd.concat([y_train_split_max.iloc[retrain_step:,:], self.y_test.iloc[:retrain_step,:]])
        y_train_split_min = pd.concat([y_train_split_min.iloc[retrain_step:,:], self.y_val.iloc[:retrain_step,:]])
        self.y_train = pd.concat([y_train_split_min, y_train_split_max])

        # Remove entries from the val and test set
        self.x_val = self.x_val.iloc[retrain_step:,:]
        self.x_test = self.x_test.iloc[retrain_step:,:]
        self.y_val = self.y_val.iloc[retrain_step:,:]
        self.y_test = self.y_test.iloc[retrain_step:,:]

    def train_model_online(self, retrain_step: int=24, check_restart: bool=False):
        # Zet de seed
        self.seed = int(self.best_trials_dict['Seed'])

        # Selecteer features
        if self.select_features is not None:
            input_data = self.select_features(self, self.best_trials_dict)
        else:
            if self.iteration == 0:
                print("No 'select_features' function given in the class, using all features.")
            input_data = self.X_all.copy()

        # Split je data in train en validatie set
        self.split_data(input_data, self.y_all)

        if check_restart:
            try:
                self.forecast_df = pd.read_csv(self.savepath / 'Forecasts' / 'Online_'+str(retrain_step)+'.csv', index_col=0, parse_dates=True)
                for _ in range(0, len(self.forecast_df.dropna().index), retrain_step):
                    self.update_train_val(retrain_step)
                    self.iteration += 1
            except:
                self.forecast_df = pd.DataFrame(index=self.y_test.index, columns=self.y_all.columns)
        else:
            self.forecast_df = pd.DataFrame(index=self.y_test.index, columns=self.y_all.columns)

        max_ = len(self.forecast_df.index)

        if self.modeltype=='MLP':
            # Maak een tuple met je model structuur
            if self.best_trials_dict['N hidden layers'] == 0:
                mlp_layers = (self.best_trials_dict['Hidden nodes layer 1_1'],)
            else:
                mlp_layers = (self.best_trials_dict['Hidden nodes layer 1_2'],self.best_trials_dict['Hidden nodes layer 2'])
        elif self.modeltype=='TST':
            N_layers = int((self.best_trials_dict['N mlp layers']) +1)
            mlp_layers = [int(self.best_trials_dict['dim_'+str(N_layers)]) for _ in range(N_layers)]

        pbar = tqdm(total=np.ceil(int(max_/retrain_step - self.iteration*retrain_step)))
        self.make_train_model(mlp_layers, self.best_trials_dict)
        self.forecast_df.iloc[:retrain_step, :] = self.model.predict(self.scaler.transform(self.x_test.iloc[:retrain_step,:]).astype('float32'))
        self.update_train_val(retrain_step)
        self.iteration += 1
        pbar.update(1)

        while self.iteration*retrain_step < max_:
            self.make_train_model(mlp_layers, self.best_trials_dict)
            start_idx = self.iteration*retrain_step
            end_idx = min(start_idx + retrain_step, max_)

            self.forecast_df.iloc[start_idx:end_idx, :] = self.model.predict(self.scaler.transform(self.x_test.iloc[:retrain_step,:]).astype('float32'))
            self.update_train_val(retrain_step)
            self.forecast_df.to_csv(self.savepath / 'Forecasts' / ('Online_'+str(retrain_step)+'.csv'))
            self.iteration += 1
            pbar.update(1)


class OnlineTrainingUniformDataGenerator(KerasMixin, SplitByYear):
    def __init__(
        self, 
        name: str, 
        features: pd.DataFrame, 
        target: pd.DataFrame, 
        trainval_labels,
        test_labels,
        savepath: Path,
        select_features: Callable,
        modeltype: str='MLP',
        gpu=True,
        generator='uniform'
        ):

        print('Applying TPE search with Tensorflow version '+str(tf.version.VERSION))
        KerasMixin.__init__(self)
        SplitByYear.__init__(self)

        self.name = name
        self.modeltype=modeltype
        self.generator=generator
        self.trainval_labels=trainval_labels
        self.test_labels=test_labels
        
        if gpu:
            self.device='/gpu:0'
        else:
            self.device='/cpu'

        self.X_all = features
        self.y_all = target

        self.select_features=select_features

        self.quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.make_quantile_target()

        self.savepath = savepath
        self.trialpath = self.savepath / 'Trials'
        self.fcpath = self.savepath / 'Forecasts'
        
        self.trials = Trials()
        self.trials_df = None
        self.reload_trials()
        self.best_trials = None
        self.best_trials_dict = None
        self.read_best_trials()

        self.iteration = 0

        self.train_index = None
        self.val_index = None
        self.test_index = None

        self.N_improvements = 0
        self.best_model = None

        self.model = None
        print('TPE Search initialised with '+modeltype+' settings.')

    def reload_trials(self):
        with open(self.savepath / 'Trials' / (self.name+'.pickle'), 'rb') as pickle_file:
            self.trials = pickle.load(pickle_file)
            if self.trials.losses()[-1] == None:
                self.trials.trials.remove(self.trials.trials[-1])
            self.iteration = len(self.trials.trials)
            self.trials_df = self.trials_to_df()

    def read_val_from_trials(self, val):
        """Kleine handigheidje om fouten bij het lezen van de trials te voorkomen."""
        try:
            return val[0]
        except:
            return 0

    def trials_to_df(self):
        """Een functie om je trails file (ingewikkelde dictionary) om te zetten in een DF."""
        df = pd.DataFrame(index = range(len(self.trials.trials)), columns = self.trials.trials[0]['misc']['vals'].keys())
        df.loc[:, :] = [[self.read_val_from_trials(self.trials.trials[i]['misc']['vals'][k]) for k in self.trials.trials[i]['misc']['vals'].keys()] for i in range(len(self.trials.trials))]
        df.loc[:, 'test_loss'] = [self.trials.trials[i]['result']['test_loss'] for i in range(len(self.trials.trials))]
        df.loc[:, 'val_loss'] = [self.trials.trials[i]['result']['val_loss'] for i in range(len(self.trials.trials))]
        df.loc[:, 'train_loss'] = [self.trials.trials[i]['result']['train_loss'] for i in range(len(self.trials.trials))]
        
        return df

    def read_best_trials(self):
        cols = [col if not col.endswith('discharge lag') else 'discharge lag' for col in self.trials_df.columns]
        self.trials_df.columns = cols
        self.best_trials = self.trials_df.loc[self.trials_df.val_loss==self.trials_df.val_loss.min(),:]
        best_ind = self.best_trials.index[0]
        self.best_trials = self.best_trials.loc[best_ind]

        int_cols = ['precip lag', 'discharge lag', 'precip FC', 'window size', 'precip window', 'evap window', 'temp window', 'discharge window', 'DOY', 'DOW', 'HOD']
        if (self.name == 'Rijnland') | (self.name == 'HHNK'):
                int_cols += ['H fc']

        if self.modeltype == 'MLP':
            int_cols += ['Hidden nodes layer 1_1', 'Hidden nodes layer 1_2', 'Hidden nodes layer 2', 'N hidden layers', 'Batch size', 'Batch normalization', 'Seed']

        elif self.modeltype == 'TST':
            N = int(self.best_trials['N mlp layers']) +1
            int_cols += ['N mlp layers', 'dim_'+str(N), 'Attention head size', 'N attention heads', 'Transformer filter dimension', 'N encoder decoder blocks', 'Batch size', 'Batch normalization', 'Seed']
        else:
            print('Wrong modeltype?')

        self.best_trials_dict = {col: self.best_trials[col] for col in self.best_trials.index}

        for col in int_cols:
            self.best_trials_dict[col] = int(self.best_trials_dict[col])

        temps = ['none', 'abs', 'cyclic']
        for t in ['HOD', 'DOW', 'DOY']:
            self.best_trials[t] = temps[int(self.best_trials[t])]
            self.best_trials_dict[t] = temps[int(self.best_trials_dict[t])]

    def make_quantile_target(self):
        """Voor QR gebruik je dezelfde variabele meerdere keren, voor elke kwantiel een keer. 
        Vandaar deze 'hack', kan geheugen-vriendelijker maar heeft even geen prioriteit."""
        
        cols = [col + ' q' + str(q) for q in self.quantiles for col in self.y_all.columns]
        y = pd.DataFrame(index = self.y_all.index, columns=cols)
        y.loc[:,:] = np.concatenate([self.y_all.values for _ in self.quantiles], axis=1)
        self.y_all = y.copy()

    def reshape_input_for_tst(self):
        self.x_train = self.x_train.values
        self.x_train.reshape((self.x_train.shape[0], self.x_train.shape[1], 1))

        self.x_val = self.x_val.values
        self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1], 1))

        self.x_test = self.x_test.values
        self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1], 1))

    def get_mlp(self, mlp_layers, params):
        # Maak je model!
        with tf.device(self.device):
            self.model = self.create_model(
                modeltype='MLP',
                n_features=self.x_train.shape[1],
                n_targets=self.y_train.shape[1],
                mlp_layers=mlp_layers,
                mlp_dropout=params['Dropout'],
                mlp_batchnorm=params['Batch normalization'],
                regularization=params['Regularization'],
                seed=int(params['Seed']),
                dropout_seed=None
            )
    
    def get_transformer(self, mlp_layers, params):
        with tf.device(self.device):
            self.model = self.create_model(
                modeltype='TST',
                n_features=self.x_train.shape[1],
                n_targets=self.y_train.shape[1],
                mlp_layers=mlp_layers,
                mlp_dropout=params['MLP dropout'],
                mlp_batchnorm=params['Batch normalization'],
                regularization=params['Regularization'],
                seed=int(params['Seed']),
                attention_head_size=4**int(params['Attention head size']),
                n_attention_heads=4**int(params['N attention heads']),
                transformer_ff_dim=4**int(params['Transformer filter dimension']),
                num_encoder_decoder_blocks=int(params['N encoder decoder blocks']),
                encoder_dropout=int(params['Encoder dropout']),
            )
            
    def make_train_generator(self, batch_size):
        if self.generator=='uniform':
            train_generator=UniformDataGenerator(X=self.x_train_t.values, y=self.y_train_t.values, labels=self.train_labels_t, batch_size=batch_size)
            # val_generator=DataGenerator(X=self.x_val, y=self.y_val, batch_size=batch_size)
        else:
            train_generator=DataGenerator(X=self.x_train_t.values, y=self.y_train_t.values, batch_size=batch_size)
            # val_generator=DataGenerator(X=self.x_val, y=self.y_val, batch_size=batch_size)
        
        return train_generator#, val_generator

    def make_train_model(self, mlp_layers, params):
        self.set_seeds(self.seed)
        if self.modeltype=='MLP':
            self.get_mlp(mlp_layers, params)
        elif self.modeltype=='TST':
            self.get_transformer(mlp_layers, params)

        self.scaler = StandardScaler().fit(self.x_train_t)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        val_data = (self.scaler.transform(self.x_val_t).astype('float32'), self.y_val_t.values.astype('float32'))
        
        train_generator = self.make_train_generator(batch_size=5**int(int(params['Batch size'])))
        with tf.device(self.device):
            self.model.fit(
                x=train_generator,
                epochs=10000,
                validation_data=val_data,
                steps_per_epoch=train_generator.__len__(), 
                callbacks=[es],
                verbose=0
                )
    
    def update_train_val(self, retrain_step):
        retrain_step = min(len(self.x_test.dropna()), retrain_step)
        
        # For the datagenerator we just keep all data since we make a selection based on clusters.
        self.x_trainval = pd.concat([self.x_trainval, self.x_test.iloc[:retrain_step,:]], axis=0)
        self.y_trainval = pd.concat([self.y_trainval, self.y_test.iloc[:retrain_step,:]], axis=0)
        self.trainval_labels = np.concatenate([self.trainval_labels, self.test_labels[:retrain_step]], axis=0)
        
        self.x_test = self.x_test.iloc[retrain_step:, :]
        self.y_test = self.y_test.iloc[retrain_step:, :]
        self.test_labels = self.test_labels[retrain_step:]

    def get_new_inputs(self):
        # Stratified split van de twee
        self.x_train_t, self.x_val_t, self.y_train_t, self.y_val_t, self.train_labels_t, self.val_labels_t =  train_test_split(self.x_trainval, self.y_trainval, self.trainval_labels, train_size=0.7, shuffle=True, stratify=self.trainval_labels, random_state=1)

    def train_model_online(self, retrain_step: int=24, check_restart: bool=False):
        # Zet de seed
        self.seed = int(self.best_trials_dict['Seed'])

        # Selecteer features
        if self.select_features is not None:
            input_data = self.select_features(self, self.best_trials_dict)
        else:
            if self.iteration == 0:
                print("No 'select_features' function given in the class, using all features.")
            input_data = self.X_all.copy()

        # Split je data in train en validatie set
        self.split_data(input_data, self.y_all)
        self.x_trainval = pd.concat([self.x_train, self.x_val], axis=0).sort_index()
        self.y_trainval = pd.concat([self.y_train, self.y_val], axis=0).sort_index()
        self.get_new_inputs()

        if check_restart:
            try:
                self.forecast_df = pd.read_csv(self.savepath / 'Forecasts' / 'Online_'+str(retrain_step)+'.csv', index_col=0, parse_dates=True)
                for _ in range(0, len(self.forecast_df.dropna().index), retrain_step):
                    self.update_train_val(retrain_step)
                    self.iteration += 1
            except:
                self.forecast_df = pd.DataFrame(index=self.y_test.index, columns=self.y_all.columns)
        else:
            self.forecast_df = pd.DataFrame(index=self.y_test.index, columns=self.y_all.columns)

        max_ = len(self.forecast_df.index)

        if self.modeltype=='MLP':
            # Maak een tuple met je model structuur
            if self.best_trials_dict['N hidden layers'] == 0:
                mlp_layers = (self.best_trials_dict['Hidden nodes layer 1_1'],)
            else:
                mlp_layers = (self.best_trials_dict['Hidden nodes layer 1_2'],self.best_trials_dict['Hidden nodes layer 2'])
        elif self.modeltype=='TST':
            N_layers = int((self.best_trials_dict['N mlp layers']) +1)
            mlp_layers = [int(self.best_trials_dict['dim_'+str(N_layers)]) for _ in range(N_layers)]

        pbar = tqdm(total=np.ceil(int(max_/retrain_step - self.iteration*retrain_step)))
        self.make_train_model(mlp_layers, self.best_trials_dict)
        self.forecast_df.iloc[:retrain_step, :] = self.model.predict(self.scaler.transform(self.x_test.iloc[:retrain_step,:]).astype('float32'))
        self.update_train_val(retrain_step)
        self.get_new_inputs()
        self.iteration += 1
        pbar.update(1)

        while self.iteration*retrain_step < max_:
            self.make_train_model(mlp_layers, self.best_trials_dict)
            start_idx = self.iteration*retrain_step
            end_idx = min(start_idx + retrain_step, max_)

            self.forecast_df.iloc[start_idx:end_idx, :] = self.model.predict(self.scaler.transform(self.x_test.iloc[:retrain_step,:]).astype('float32'))
            self.update_train_val(retrain_step)
            self.get_new_inputs()
            self.forecast_df.to_csv(self.savepath / 'Forecasts' / ('Online_'+str(retrain_step)+'.csv'))
            self.iteration += 1
            pbar.update(1)

