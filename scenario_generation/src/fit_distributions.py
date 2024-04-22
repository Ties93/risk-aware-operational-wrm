import pandas as pd
from pathlib import Path
from utils.distfit import *
import hydra
from omegaconf import DictConfig
import os
# os.environ['NUMEXPR_MAX_THREADS'] = '32'


def read_forecasts(path):
    return pd.read_pickle(path / 'Forecasts' / 'test_fc_formatted.pkl').astype(float)

def fit_distributions(path, colname, mask_regression_quantile=False, cutoff=None):
    forecast_df = read_forecasts(path)
    if cutoff is not None:
        forecast_df = forecast_df.iloc[:cutoff]
    
    if colname != 'DAM NL':
        fitter = DistributionFitter(forecast_df, colname=colname, mask_regression_quantile=mask_regression_quantile, parallel=True)
    else:
        fitter = DAMDistributionFitter(forecast_df, colname=colname, mask_regression_quantile=mask_regression_quantile, parallel=True)
    return fitter.fitted_distributions

def fit_save_distribution(path, savepath, savename, colname, mask_regression_quantile=False, n_cuts=None):

    distributions = fit_distributions(path, colname=colname, mask_regression_quantile=mask_regression_quantile, cutoff=n_cuts)
    distributions.to_pickle(savepath / f'{savename}.pkl')
    distributions.to_csv(savepath / f'{savename}.csv')

@hydra.main(version_base=None, config_path='../conf/', config_name='config')
def main(cfg: DictConfig) -> None:
    cutoff = None#'month'

    if cutoff == 'month':
        n_cuts = 31*24*48*13
    elif cutoff == 'week':
        n_cuts = 7*24*48*13
    else:
        n_cuts = None
    waterboards = ['HHNK', 'HDSR', 'Rijnland', 'Waternet']
    datapath = Path(cfg.DATAPATH)
    tpepath = datapath /  'TPE Search results'
    savepath = datapath / 'Distributions'

    wbpath = tpepath / 'CQRDNN_Waterboards'
    wlpath = tpepath / 'CQRDNN_WL_IJmuiden'
    dampath = tpepath / 'CQRDNN_DAM'

    # if cutoff == 'month':
    #     savename = f'{name}_{cutoff}'
    #     n_cuts = 31*24*48*13
    # else:
    #     savename = f'{name}'
    #     n_cuts = None
    
    # for wb in waterboards:
    #     fit_save_distribution(path=wbpath / wb, savepath=savepath, savename=f'{wb}', colname='Aggregated', mask_regression_quantile=True)#, n_cuts=31*24*48*13)
    
    fit_save_distribution(path=wlpath, savepath=savepath, savename='WL', colname='Aggregated')
    # fit_save_distribution(path=dampath, savepath=savepath, savename='DAM', colname='DAM NL', n_cuts=None)
    

if __name__ == '__main__':
    main()