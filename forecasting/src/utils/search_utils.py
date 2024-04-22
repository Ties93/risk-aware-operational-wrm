from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np

def make_mlp_hyperparameter_space():
    # De nodes in de verborgen lagen
    n1_1 = hp.quniform('Hidden nodes layer 1_1', 50, 450, 1)
    n1_2 = hp.quniform('Hidden nodes layer 1_2', 50, 450, 1)
    n2 = hp.quniform('Hidden nodes layer 2', 50, 250, 1)

    # Aantal lagen en de bijbehorende nodes zetten we op als conditional search space.
    # Pas op, n1_1 en n1_2 zijn andere random variabelen, want n1 gedraagt zich erg anders als er een tweede laag is.
    nodes = hp.choice('N hidden layers', 
        [
            {
                'N': 1,
                'Hidden nodes layer 1_1': n1_1
            },
            {
                'N': 2,
                'Hidden nodes layer 1_2': n1_2,
                'Hidden nodes layer 2': n2
                
            }
        ]
                    )


    dropout = hp.uniform('Dropout', 0, 0.5)
    batch_size = hp.quniform('Batch size', 1, 4, 1) # Deze laat ik als exponent van 5 werken
    regularizing = hp.loguniform('Regularization', np.log(1e-5), np.log(0.05))
    batchnorm = hp.choice('Batch normalization', [False, True])
    seed = hp.quniform('Seed', 0, 300, 5)

    return {
        'N hidden layers': nodes,
        'Dropout': dropout,
        'Batch size': batch_size,
        'Regularization': regularizing,
        'Batch normalization': batchnorm,
        'Seed': seed
    }

def make_tst_hyperparameter_space():
    nodes = hp.choice('N mlp layers',[{'N': n, 'dim_'+str(n): hp.quniform('dim_'+str(n), 50, 450, 1)} for n in range(1, 4)])
    mlp_dropout = hp.uniform('MLP dropout', 0, 0.5)
    attention_head_size = hp.quniform('Attention head size', 2, 4, 1) # 4**x
    n_attention_heads = hp.quniform('N attention heads', 1, 3, 1) # 4**x
    transformer_ff_dim = hp.quniform('Transformer filter dimension', 1, 4, 1) # 4**x
    num_encoder_decoder_blocks = hp.quniform('N encoder decoder blocks', 1, 4, 1)
    encoder_dropout = hp.uniform('Encoder dropout', 0, 0.5)
    batch_size = hp.quniform('Batch size', 2, 3, 1) # Deze laat ik als exponent van 5 werken
    batchnorm = hp.choice('Batch normalization', [False, True])
    regularizing = hp.loguniform('Regularization', np.log(1e-5), np.log(0.05))
    seed = hp.quniform('Seed', 0, 300, 5)

    return {
        'N mlp layers': nodes,
        'MLP dropout': mlp_dropout,
        'Attention head size': attention_head_size,
        'N attention heads': n_attention_heads,
        'Transformer filter dimension': transformer_ff_dim,
        'N encoder decoder blocks': num_encoder_decoder_blocks,
        'Encoder dropout': encoder_dropout,
        'Batch size': batch_size,
        'Batch normalization': batchnorm,
        'Seed': seed,
        'Regularization': regularizing
    }

def make_feature_space(lagrange, windowrange):

    feature_space = {'WL lag': hp.quniform('WL lag', lagrange[0], lagrange[-1], 1)}
    feature_space['wind lag'] = hp.quniform('wind lag', lagrange[0], lagrange[-1], 1)
    feature_space['hourly wind data'] = hp.choice('hourly wind data', [False, True])
    feature_space['10min wind data'] = hp.choice('10min wind data', [False, True])
    feature_space['wind direction data'] = hp.choice('wind direction data', [False, True])
    feature_space['WL window size'] = hp.quniform('WL window size', windowrange[0], windowrange[-1], 1)
    feature_space['DOY'] = hp.choice('DOY', ['none', 'abs', 'cyclic'])
    feature_space['HOD'] = hp.choice('HOD', ['none', 'abs', 'cyclic'])
    return feature_space

def select_features(self, params):
    """Hier selecteer je de juiste kolommen uit de X DataFrame op basis van een dictionary met de search space instantiation."""
    
    features = []
    
    # Lagged features
    features += ['WL lag'+str(l) for l in range(0, int(params['WL lag']))]

    if params['hourly wind data']:
        features += ['hourly wind lag'+str(l) for l in range(0, int(params['wind lag']))]
    if params['10min wind data']:
        features += ['10min wind lag'+str(l) for l in range(0, int(params['wind lag']))]
    if params['wind direction data']:
        features += ['wind direction lag'+str(l) for l in range(0, int(params['wind lag']))]
    
    # Windowed features
    features += ['WL window'+str(int(params['WL window size']))]

    # Time features
    if params['DOY'] == 'abs':
        features += ['DOY abs']
    elif params['DOY'] == 'cyclic':
        features += ['DOY sin', 'DOY cos']
    if params['HOD'] == 'abs':
        features += ['HOD abs']
    elif params['HOD'] == 'cyclic':
        features += ['HOD sin', 'HOD cos']
            
    return self.X_all.loc[:, list(set(features))]