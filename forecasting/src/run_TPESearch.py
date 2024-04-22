import hydra
from omegaconf import DictConfig
from utils.load_data_CQRDNN import DataLoader
from utils.TreeParzenEstimatorSearch import TPESearch, OnlineTraining, TPESearchUniformDataGenerator
from utils.search_utils import *
from pathlib import Path
import os
from numpy import arange
import pandas as pd

def search(cfg, data, hyperparameter_space, feature_space):
    TPE = TPESearch(
        name=cfg.EXPERIMENT.EXPERIMENT_NAME,
        features=data.X,
        target=data.y,
        savepath=Path(cfg.EXPERIMENT.SAVEPATH) / cfg.EXPERIMENT.EXPERIMENT_NAME,
        modeltype=cfg.MODEL.MODELNAME,
        hyperparameter_space=hyperparameter_space,
        feature_space=feature_space,
        select_features=select_features
        )

    TPE.TreeParzenEstimatorSearch()
    
# def clustereddatasearch(cfg, data, hyperparameter_space, feature_space, labels):
#     TPE = TPESearchUniformDataGenerator(
#         name=cfg.EXPERIMENT.WATERBOARD,
#         features=data.X,
#         target=data.y,
#         savepath=Path(cfg.EXPERIMENT.SAVEPATH) / cfg.EXPERIMENT.EXPERIMENT_NAME / cfg.EXPERIMENT.WATERBOARD,
#         modeltype=cfg.MODEL.MODELNAME,
#         hyperparameter_space=hyperparameter_space,
#         feature_space=feature_space,
#         select_features=select_features,
#         trainval_labels=labels
#         )

    # TPE.TreeParzenEstimatorSearch()
    

def train_online(cfg, data, retrain_step: int):
    online = OnlineTraining(
        name=cfg.EXPERIMENT.WATERBOARD,
        features=data.X,
        target=data.y,
        savepath=Path(cfg.EXPERIMENT.SAVEPATH) / cfg.EXPERIMENT.EXPERIMENT_NAME / cfg.EXPERIMENT.WATERBOARD,
        modeltype=cfg.MODEL.MODELNAME,
        select_features=select_features
        )
    online.train_model_online(retrain_step=retrain_step)

@hydra.main(version_base=None, config_path='../conf/', config_name='config')
def app(cfg: DictConfig) -> None:
    # if (cfg.MODEL.MODELNAME != 'CQRTST'):
    #     print('Wrong model type. This is the CQRTST script, not '+str(cfg.MODEL.MODELNAME))
    #     return None
    
    print('Experiment: '+cfg.EXPERIMENT.EXPERIMENT_NAME)
    print('Model: '+cfg.MODEL.MODELNAME)
    print()
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    print('Loading data and creating features.')
    lagrange = arange(0, 72, 1) # hours
    windowrange = arange(1, 7, 1) # days
    fcrange = arange(1,49,1) #hours

    data = DataLoader(cfg, lagrange, windowrange, fcrange)
    print('Done.')

    print()

    print('Starting TPE Search for features and hyperparameters.')
    feature_space = make_feature_space(lagrange=lagrange, windowrange=windowrange)
   
    if cfg.MODEL.MODELNAME=='MLP':
       # MLP Hyperparameter space
       hyperparameter_space = make_mlp_hyperparameter_space()
    elif cfg.MODEL.MODELNAME=='TST':
       # TST Hyperparameter space
       hyperparameter_space = make_tst_hyperparameter_space()
    
    # if cfg.EXPERIMENT.EXPERIMENT_NAME.endswith('ClusteredUniform'): #Name: CQRDNN_#_ClusteredUniform
    #     k = cfg.EXPERIMENT.EXPERIMENT_NAME.split('_')[1]
    #     labels = pd.read_csv(data.datapath / f'{cfg.EXPERIMENT.WATERBOARD}_cluster_labels_trainval.csv', index_col=0, parse_dates=True)
    #     labels=labels.sort_index().loc[:, k].values
    #     clustereddatasearch(cfg, data, hyperparameter_space=hyperparameter_space, feature_space=feature_space, labels=labels)
    # else:
    search(cfg, data, hyperparameter_space=hyperparameter_space, feature_space=feature_space)
    print('TPE Search completed.')

if __name__ == '__main__':
    app()