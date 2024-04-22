import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
from keras import layers
import os
import random
from typing import Optional, List

class KerasMixin():
    """
    Class to make Keras timeseries models. Supports the MLP and Transformer.
    """
    def __init__(self):
        self.quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.optimizer = keras.optimizers.Adam()
        self.loss = self.combined_quantile_loss
        self.output_activation='linear'
        self.activation_function='relu'
        self.initializer='He'

    def quantile_loss(self, quantile, y_true, y_pred):
            e = y_pred - y_true
            return keras.backend.mean( keras.backend.maximum( quantile * e, (quantile-1) * e ) )

    def combined_quantile_loss(self, y_true, y_pred):
        N_targets = y_pred.shape[1] #nr of columns in y =  nr of quantiles * nr of targets
        w_target = int(N_targets/len(self.quantiles)) #width per quantile - Y is [q0, c0, h 0 - 24, q0, c1, h0-24, ... , q1, c0, h0-24, ... , q4, c0, h0-24]
        return keras.backend.mean( 
            tf.stack([
                self.quantile_loss(
                    self.quantiles[q_i], 
                    y_true[:, q_i * w_target : (q_i+1) * w_target], 
                    y_pred[:, q_i * w_target : (q_i+1) * w_target]
                ) for q_i in range( len(self.quantiles) ) 
                ] ) 
            )


    def set_seeds(self, seed):
        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        os.environ['PYTHONHASHSEED']=str(seed)
        
        # 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed)
        
        # 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.seed(seed)
        
        # 4. Set `tensorflow` pseudo-random generator at a fixed value
        tf.random.set_seed(seed)

    def create_model(
        self,
        modeltype: str='MLP',
        n_features: int=1,
        n_targets: int=1,
        mlp_layers: List[int]=[128],
        mlp_dropout: float=0.0,
        mlp_batchnorm: bool=True,
        regularization: float=1e-4,
        seed: Optional[int]=None,
        dropout_seed: Optional[int]=None,
        attention_head_size: Optional[int]=None,
        n_attention_heads: Optional[int]=None,
        transformer_ff_dim: Optional[int]=None,
        num_encoder_decoder_blocks: Optional[int]=None,
        encoder_dropout: Optional[float]=0.0
        ):

        # Clear the Keras memory before initializing a new model.
        K.clear_session()
        
        if seed:
            self.set_seeds(seed)
            if dropout_seed == None:
                dropout_seed = seed

        if modeltype == 'MLP':
            model = self.make_mlp(
                n_features=n_features,
                n_targets=n_targets,
                mlp_layers=mlp_layers,
                mlp_dropout=mlp_dropout,
                mlp_batchnorm=mlp_batchnorm,
                regularization=regularization,
                seed=seed
            )
            model.compile(loss=self.loss, optimizer=self.optimizer)
            return model
        elif modeltype == 'TST':
            model = self.make_transformer(
                n_features=n_features,
                n_targets=n_targets,
                head_size=attention_head_size,
                num_heads=n_attention_heads,
                ff_dim=transformer_ff_dim,
                num_encoder_decoder_blocks=num_encoder_decoder_blocks,
                mlp_units=mlp_layers,
                encoder_dropout=encoder_dropout,
                mlp_dropout=mlp_dropout,
                mlp_batchnorm=mlp_batchnorm
                )
            model.compile(loss=self.loss, optimizer=self.optimizer)
            return model
        else:
            print("Model type '"+str(type)+"' not supported. Check for spelling errors or compatibility.")

        

        
    def initializers(self, seed=None):
        inits = {
            'uniform': tf.keras.initializers.RandomUniform(seed=seed),
            'He': tf.keras.initializers.he_normal(seed=seed),
            'Glorot': tf.keras.initializers.glorot_normal(seed=seed)    
            }
        return inits[self.initializer]

    def make_mlp(
        self, 
        n_features: int, 
        n_targets: int, 
        mlp_layers: List[int], 
        mlp_dropout: float, 
        mlp_batchnorm: bool,
        regularization: float,
        seed=None
        ):
        
        inputs = keras.Input(shape=(n_features,))
        x = inputs

        for dim in mlp_layers:
            x = layers.Dense(
                dim, 
                activation=self.activation_function, 
                kernel_initializer=self.initializers(seed=seed),
                bias_initializer=tf.keras.initializers.Zeros(), 
                kernel_regularizer = keras.regularizers.l2(regularization)
                )(x)
            x = layers.Dropout(mlp_dropout)(x)

            if mlp_batchnorm:
                x = layers.BatchNormalization()(x)
        
        outputs = layers.Dense(
            n_targets,
            activation=self.output_activation
            )(x)
        return keras.Model(inputs=inputs, outputs=outputs)