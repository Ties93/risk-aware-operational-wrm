import hydra
from omegaconf import DictConfig
import pandas as pd
import pandas as pd
from pathlib import Path
from utils.distfit import *
from utils.clustering import *
from utils.scenario_gen import *
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial

import os 

os.environ['NUMEXPR_MAX_THREADS'] = '64'

waterboards = ['HHNK', 'HDSR', 'Rijnland', 'Waternet']

def get_tpepath(name, tpepath):
    if name in waterboards:
        return tpepath / 'CQRDNN_Waterboards' / name / 'Forecasts'
    elif name == 'WL':
        return tpepath / 'CQRDNN_WL_IJmuiden' / 'Forecasts'
    elif name == 'DAM':
        return tpepath / 'CQRDNN_DAM' / 'Forecasts'
    else:
        raise ValueError(f'Invalid name: {name}')
    
def get_data(name, tpepath, distpath):
    path = get_tpepath(name, tpepath)
    if name == 'DAM':
        val_data = pd.concat([pd.read_pickle(path / 'y_train.pkl'), pd.read_pickle(path / 'y_val.pkl')], axis=0).astype(float).sort_index()
        val_data = val_data.loc[:, [col for col in val_data.columns if ('q0.5' in col) and ('NL' in col)]]
        val_data.columns = [f'DAM NL +{h}' for h in range(1,49)]
        distribution_df_ = pd.read_pickle(get_distpath(name, distpath))
        # distribution_df_ = pd.read_csv(get_distpath(name), index_col=[0,1], parse_dates=True, header=[0,1])
        distribution_df_.index = distribution_df_.index.droplevel(1)
        indices = np.array([i for i in range(0, len(distribution_df_.index), 2)])
        distribution_df = pd.concat([distribution_df_.iloc[indices, :], distribution_df_.iloc[indices+1, :]], axis=1)
        distribution_df.columns = pd.MultiIndex.from_product([[f'DAM NL_{i}' for i in range(1,49)], ['dist', 'params', 'KS']])
    
    else:
        if name in waterboards + ['WL']:
            f = 'y_train.csv'
        else:
            f = 'y_trainval.csv'
        val_data = pd.read_csv(path / f, index_col=0, parse_dates=True).astype(float)
        if name in waterboards:
            val_data = val_data.loc[:, [f'Aggregated +{i} q0.5' for i in range(1, 49)]]
            val_data.columns =  [f'Q +{i}' for i in range(1, 49)]
            
        elif name == 'WL':
            val_data = val_data.loc[:, [f'WL +{i} q0.5' for i in range(1, 49)]]
            val_data.columns =  [f'WL +{i}' for i in range(1, 49)]
        distribution_df = pd.read_pickle(get_distpath(name, distpath))
        #distribution_df = pd.read_csv(get_distpath(name), index_col=0, parse_dates=True, header=[0,1])
    return val_data, distribution_df
    
def get_distpath(name, distpath):
    if name in waterboards:
        return distpath / f'{name}.pkl'
    elif name == 'WL':
        return distpath / f'WL.pkl'
    elif name == 'DAM':
        return distpath / f'DAM.pkl'
    else:
        raise ValueError(f'Invalid name: {name}')
    
class ScenarioGenerator():
    def __init__(self, name, varname, val_data=None, distribution_df=None, valmin=None, valmax=None, scenario_mask=None, tpepath=None, distpath=None):
        self.name = name
        self.varname = varname

        if val_data is None or distribution_df is None:
            self.val_data, self.distribution_df = get_data(name, tpepath, distpath)
        else:
            self.val_data = val_data
            self.distribution_df = distribution_df
        self.indices = self.distribution_df.index

        
        if self.name in ['Rijnland', 'Waternet', 'HDSR', 'HHNK']:
            self.scenario_mask = (0, self.val_data.max().max())
        elif self.name == 'WL':
            self.scenario_mask = (self.val_data.min().min(), self.val_data.max().max())
        else:
            self.scenario_mask = scenario_mask


        if valmin is None:
            self.valmin = np.floor(self.val_data.min().min() * 1.1)  
            self.valmax = np.floor(self.val_data.max().max() * 1.1)
        else:
            self.valmin = valmin
            self.valmax = valmax

        self.distributionloader = DistributionLoader(self.distribution_df, self.varname, valmin=self.valmin, valmax=self.valmax)
        self.make_bn()
    
    def make_bn(self):
        self.bn = BN(self.val_data, self.distributionloader, varname=self.varname, n=len(self.val_data.columns), threshold=None)
        self.bn.make_structure()

    def sample_scenarios(self, date, n=1000, BN=True, return_df=True, inplace=True):
        if BN:
            scenarios = self.bn.sample_BN(date, n).round(2)
        else:
            scenarios = pd.DataFrame(index=range(n), columns=self.val_data.columns)
            for col in self.val_data.columns:
                dist = self.distribution_df.loc[date, (col, 'dist')]
                params = self.distribution_df.loc[date, (col, 'params')]
                self.scenarios.loc[:, col] = get_rv(dist, params).rvs(n)

        if self.scenario_mask is not None:
            scenarios[scenarios < self.scenario_mask[0]] = self.scenario_mask[0]
            scenarios[scenarios > self.scenario_mask[1]] = self.scenario_mask[1]

        if inplace:
            self.scenarios = scenarios

        if return_df:
            return scenarios
            

    def cluster_scenarios(self, subset_size, method='energy', return_df=True, inplace=True, scenarios=None):
        if inplace:
            scenarios = self.scenarios

        energy_clusterer = ReduceForward(x=scenarios.values, cdn=subset_size, dist=method, verbose=False)
        energy_clusterer.reduce_forward()
        clusters = energy_clusterer.clusters
        clusters.columns=[col for col in scenarios.columns] + ['weights']
        if inplace:
            self.clusters = clusters
        if return_df:
            return clusters
    
    def sample_cluster_single_date_(self, date, method='energy', n=1000, subset_size=5, BN=True, return_df=True, inplace=False):
        scenarios = self.sample_scenarios(date, n=n, BN=BN, return_df=True, inplace=inplace)
        clusters = self.cluster_scenarios(subset_size=subset_size, scenarios=scenarios, method=method, return_df=True, inplace=inplace)
        
        if return_df:
            return clusters.values
        
    def generate_all_scenarios(self, n_samples, subset_size, method='energy', BN=True, return_df=True, inplace=False, n_jobs=-1):
        indices = self.indices.droplevel(1).unique()
        idx = pd.IndexSlice
        all_scenarios = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([(ind, i) for ind in indices for i in range(subset_size)]), 
            columns=[col for col in self.val_data.columns] + ['weights']
            )

        # for i, date in enumerate(tqdm(indices)):
        #     print(date)
        #     res = self.sample_cluster_single_date_(date, subset_size=subset_size, method=method, n=n_samples, BN=BN, return_df=True, inplace=False)
        #     all_scenarios.loc[idx[date, :], :] = res

        results = Parallel(n_jobs=n_jobs, verbose=0, prefer='threads')(
            delayed(partial(self.sample_cluster_single_date_, n=n_samples, BN=BN, return_df=True))(date) for date in tqdm(indices)
        )
        print('Filling DataFrame')
        for i, date in enumerate(indices):
            all_scenarios.loc[idx[date, :], :] = results[i]
        
        if inplace:
            self.all_scenarios = all_scenarios

        if return_df:
            return all_scenarios
        
class DischargeGenerator():
    def __init__(self, scenario_generator_dict, cluster_individual_wbs=False):
        self.waterboards = ['HHNK', 'Rijnland', 'Waternet', 'HDSR']
        self.scenario_generator_dict = scenario_generator_dict
        self.cluster_individual_wbs = cluster_individual_wbs
    def sample_scenarios(self, date, n=1000, BN=True, return_df=True):
        self.wb_scenarios = {}
        for wb in self.waterboards:
            self.wb_scenarios[wb] = self.scenario_generator_dict[wb].sample_scenarios(date, n=n, return_df=True)
        
        if return_df:
            return self.wb_scenarios
    
    def cluster_wb_scenarios(self, subset_size, method='energy', return_df=True):
        # Cluster the individual waterboard scenarios
        ind = pd.MultiIndex.from_product([range(subset_size), self.waterboards], names=['cluster', 'waterboard'])
        cluster_cols = [col for col in self.wb_scenarios[self.waterboards[0]].columns] + ['weights']
        self.wb_clusters = pd.DataFrame(index=ind, columns=cluster_cols, dtype=float)
        self.idx = pd.IndexSlice
        
        if self.cluster_individual_wbs:
            for wb in self.waterboards:
                self.wb_clusters.loc[self.idx[:, wb], :] = self.scenario_generator_dict[wb].cluster_scenarios(subset_size, method=method, return_df=True).values
        else:
            # Combine the scenarios of the different waterboards with random sampling
            for wb in self.waterboards:
                self.wb_clusters.loc[self.idx[:, wb], :] = self.wb_scenarios[wb].sample(n=subset_size, replace=False, random_state=10).values
        
        if return_df:
            return self.wb_clusters
    
    def _sum_discharges(self, comb):
        return np.sum( np.array([
            self.wb_clusters.loc[self.idx[comb[i], wb], [col for col in self.wb_clusters.columns if col != 'weights']].values 
            for i, wb in enumerate(self.waterboards) ]), axis=0)

    def combine_clusters(self, subset_size, method='energy', return_df=True):
        combinations = list(itertools.product(*[range(len(self.wb_clusters.loc[self.idx[:, wb], :].index)) for wb in self.waterboards]))
        # res = Parallel(n_jobs=-1, verbose=1)(delayed(self._sum_discharges)(comb) for comb in combinations)
        # self.combined_scenarios = pd.DataFrame(res, columns=[col for col in self.wb_clusters.columns if col != 'weights'])
        self.combined_scenarios = pd.DataFrame(index=range(len(combinations)), columns=[col for col in self.wb_clusters.columns if col != 'weights'], dtype=float)
        for i, comb in enumerate(combinations):
            self.combined_scenarios.loc[i, :] = self._sum_discharges(comb)

        clusterer = ReduceForward(x=self.combined_scenarios.values, cdn=subset_size, dist=method, verbose=False)
        clusterer.reduce_forward()
        self.combined_clusters = clusterer.clusters
        self.combined_clusters.columns=[col for col in self.combined_scenarios.columns] + ['weights']
        if return_df:
            return self.combined_clusters
        
    # def generate_all_scenarios()
    
@hydra.main(version_base=None, config_path='../conf/', config_name='config')
def main(cfg: DictConfig) -> None:
    name = cfg.NAME

    datapath = Path(cfg.DATAPATH)
    savepath = datapath / 'Scenarios'
    distpath = datapath / 'Distributions'
    tpepath = datapath / 'TPE Search results'

    idx = pd.IndexSlice
    init_samples = cfg.init_samples
    n_scenarios = cfg.cluster_size
    n_wb_scenarios = cfg.n_wb_scenarios
    method = cfg.method
    continue_from_previous = int(cfg.continue_from_previous)
    print(f'Generating {n_scenarios} scenarios for {name} using {method} method.')

    if name == 'discharge':
        val_data = {}
        distribution_df = {}
        for wb in waterboards:
            val_data[wb], distribution_df[wb] = get_data(wb, tpepath, distpath)

        generators = {wb: ScenarioGenerator(wb, 'Aggregated', valdata=val_data[wb], distribution_df=distribution_df[wb], tpepath=tpepath, distpath=distpath) for wb in waterboards}
        generator = DischargeGenerator(generators, cluster_individual_wbs=True)
        indices = generators['HHNK'].distribution_df.index.droplevel(1).unique()
        cols = generator.combined_clusters.columns
        savename = f'{name}_{method}_{n_wb_scenarios}_{n_scenarios}'
    elif name == 'discharge_random':
        val_data = {}
        distribution_df = {}
        for wb in waterboards:
            val_data[wb], distribution_df[wb] = get_data(wb, tpepath, distpath)

        generators = {wb: ScenarioGenerator(wb, 'Aggregated', valdata=val_data[wb], distribution_df=distribution_df[wb], tpepath=tpepath, distpath=distpath) for wb in waterboards}
        generator = DischargeGenerator(generators, cluster_individual_wbs=False)
        indices = generators['HHNK'].distribution_df.index.droplevel(1).unique()
        cols = generator.combined_clusters.columns
        savename = f'{name}_r_{method}_{n_wb_scenarios}_{n_scenarios}'
    elif name == 'WL':
        val_data, distribution_df = get_data(name, tpepath, distpath)
        generator = ScenarioGenerator(name, 'WL', val_data=val_data, distribution_df=distribution_df, tpepath=tpepath, distpath=distpath)
        indices = generator.distribution_df.index.droplevel(1).unique()
        cols = [col for col in generator.val_data.columns] + ['weights']
        savename = f'{name}_{method}_{n_scenarios}'
    elif name == 'DAM':
        generator = ScenarioGenerator(name, 'DAM NL', tpepath=tpepath, distpath=distpath)
        indices = generator.distribution_df.index.droplevel(1).unique()
        cols = [col for col in generator.val_data.columns] + ['weights']
        savename = f'{name}_{method}_{n_scenarios}'

    scenario_df = generator.generate_all_scenarios(init_samples, n_scenarios, method=method, return_df=True, inplace=False, n_jobs=-1)
    scenario_df.to_csv(savepath / f'{savename}.csv')
    # scenario_df.to_pickle(savepath / f'{savename}.pkl')
    

if __name__ == '__main__':
    main()