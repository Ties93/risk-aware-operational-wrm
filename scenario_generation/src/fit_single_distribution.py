import pandas as pd
from pathlib import Path
from utils.distfit_scipy import *
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import pickle

np.seterr(over='ignore')
os.environ['NUMEXPR_MAX_THREADS'] = '32'

def get_fcpath(name, tpepath):
    if name in ['Waternet', 'HDSR', 'HHNK']:
        return tpepath / 'CQRDNN_10_280223_NotClustered' / name / 'Forecasts' / 'test_fc_sorted.csv'
    elif name == 'Rijnland':
        return tpepath / 'CQRDNN_10_270323_NotClustered' / name / 'Forecasts' / 'test_fc_sorted.csv'
    elif name == 'WL':
        return tpepath / 'CQRDNN_WL_IJmuiden' / 'Forecasts' / 'test_fc_sorted.csv'
    elif name == 'DAM':
        return tpepath / 'CQRDNN_DAM' / 'Forecasts' / 'p_test_online_formatted.pkl'
    elif name == 'DAM_old':
        return tpepath / 'CQRDNN_DAM' / 'Forecasts' / 'p_test_EUQR_formatted.pkl'
    else:
        raise ValueError(f'Invalid name: {name}')
    
def get_colname(name):
    if name in ['Waternet', 'HDSR', 'HHNK', 'Rijnland']:
        return 'Aggregated'
    elif name == 'WL':
        return 'WL'
    elif name in ['DAM', 'DAM_old']:
        return 'DAM NL'#'Price' # DAM NL
    else:
        raise ValueError(f'Invalid name: {name}')
    
def mask_negative(name):
    if name in ['Waternet', 'HDSR', 'HHNK', 'Rijnland']:
        return True
    elif name in ['WL', 'DAM', 'DAM_old']:
        return False
    else:
        raise ValueError(f'Invalid name: {name}')
    
@hydra.main(version_base=None, config_path='../conf/', config_name='config')
def main(cfg: DictConfig) -> None:
    tpepath = Path(cfg.DATAPATH) / 'Forecasts' / 'TPE Search results'
    savepath = Path(cfg.DATAPATH) / 'Forecasts' / 'Distributions'

    with open((tpepath / 'formatted_forecasts.pkl'), 'rb') as f:
        formatted_forecasts = pickle.load(f)
    
    fcpath = get_fcpath(cfg.NAME, tpepath)

    forecast_df = formatted_forecasts[cfg.NAME]
    savename = cfg.NAME + '.csv'

    if not (savepath / savename).exists():
    #     if cfg.NAME in ['DAM', 'DAM_old']:
    #         forecast_df = pd.read_pickle(fcpath).astype(float)#, index_col=[0,1], parse_dates=True).astype(float)
    #     else:
    #         forecast_df = pd.read_csv(fcpath, index_col=0, parse_dates=True).astype(float)
        colname = get_colname(cfg.NAME)
        mask_negative_values = mask_negative(cfg.NAME)
        print(f'Experiment: {cfg.NAME}')

        if (cfg.NAME == 'DAM') or (cfg.NAME == 'DAM_old'):
            fitter = DAMScipyDistributionFitter(forecast_df, colname=colname, mask_regression_quantile=mask_negative_values, parallel=True)
        else:
            fitter = ScipyDistributionFitter(forecast_df, colname=colname, mask_regression_quantile=mask_negative_values)
        fitter.fitted_distributions.to_csv(savepath / f'{savename}')

if __name__ == '__main__':
    main()
