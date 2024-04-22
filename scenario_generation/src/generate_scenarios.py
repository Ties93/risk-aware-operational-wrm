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
        forecast_times = distribution_df_.index.droplevel(1)#.unique()
        observation_times = distribution_df_.index.droplevel(0)#.unique()
        index = pd.MultiIndex.from_tuples([
            # substract 1 day from the observation time and 1h from the forecast time as little hack for the BN later on
            # this results in the same index for the BN as the 1-48h forecasts (t+1h:t+49h)
            (fctime+pd.DateOffset(hours=-1), obstime+pd.DateOffset(days=-1)+pd.DateOffset(hours=h)) for (fctime, obstime) in zip(forecast_times, observation_times) for h in range(0, 24)
        ])
        distribution_df = pd.DataFrame(
            index=index,
            columns=['dist', 'params', 'KS'],
        )
        idx=pd.IndexSlice
        for i, (fctime, obstime) in enumerate(zip(forecast_times, observation_times)):
            distribution_df.iloc[i*24:(i+1)*24, 0] = distribution_df_.loc[idx[fctime, obstime], idx[:, 'dist']].values
            distribution_df.iloc[i*24:(i+1)*24, 1] = distribution_df_.loc[idx[fctime, obstime], idx[:, 'params']].values
            distribution_df.iloc[i*24:(i+1)*24, 2] = distribution_df_.loc[idx[fctime, obstime], idx[:, 'KS']].values
    
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
    def __init__(self, name, varname, valmin=None, valmax=None, scenario_mask=None, tpepath=None, distpath=None):
        self.name = name
        self.varname = varname
        self.val_data, self.distribution_df = get_data(name, tpepath, distpath)
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

    def sample_scenarios(self, date, n=1000, BN=True, return_df=True):
        if BN:
            self.scenarios = self.bn.sample_BN(date, n).round(2)
        else:
            self.scenarios = pd.DataFrame(index=range(n), columns=self.val_data.columns)
            for col in self.val_data.columns:
                dist = self.distribution_df.loc[date, (col, 'dist')]
                params = self.distribution_df.loc[date, (col, 'params')]
                self.scenarios.loc[:, col] = get_rv(dist, params).rvs(n)

        if self.scenario_mask is not None:
            self.scenarios[self.scenarios < self.scenario_mask[0]] = self.scenario_mask[0]
            self.scenarios[self.scenarios > self.scenario_mask[1]] = self.scenario_mask[1]

        if return_df:
            return self.scenarios

    def cluster_scenarios(self, subset_size, method='energy', return_df=True):
        energy_clusterer = ReduceForward(x=self.scenarios.values, cdn=subset_size, dist=method, verbose=False)
        energy_clusterer.reduce_forward()
        self.clusters = energy_clusterer.clusters
        self.clusters.columns=[col for col in self.scenarios.columns] + ['weights']
        if return_df:
            return self.clusters
        
class DischargeGenerator():
    def __init__(self, scenario_generator_dict, cluster_individual_wbs=False):
        self.waterboards = ['HHNK', 'Rijnland', 'Waternet', 'HDSR']
        self.scenario_generator_dict = scenario_generator_dict
        self.cluster_individual_wbs = cluster_individual_wbs
        self.idx = pd.IndexSlice

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
        if self.cluster_individual_wbs:
            for wb in self.waterboards:
                self.wb_clusters.loc[self.idx[:, wb], :] = self.scenario_generator_dict[wb].cluster_scenarios(subset_size, method=method, return_df=True).values
        else:
            for wb in self.waterboards:
                self.wb_clusters.loc[self.idx[:, wb], :] = self.wb_scenarios[wb].sample(subset_size, replace=False, random_state=10).values
        
        if return_df:
            return self.wb_clusters
    
    def _sum_discharges(self, comb):
        return np.sum( np.array([
            self.wb_clusters.loc[self.idx[comb[i], wb], [col for col in self.wb_clusters.columns if col != 'weights']].values 
            for i, wb in enumerate(self.waterboards) ]), axis=0)

    def combine_clusters(self, subset_size, method='energy', return_df=True, exhaustive=False, n_init_samples=1000):
        # Make all possible combinations of scenarios
        combinations = list(itertools.product(*[range(len(self.wb_clusters.loc[self.idx[:, wb], :].index)) for wb in self.waterboards]))
        if not exhaustive:
            # Make a random subset of combinations
            combinations = [combinations[i] for i in np.random.choice(len(combinations), size=n_init_samples, replace=False)]
        
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
        generators = {wb: ScenarioGenerator(wb, 'Aggregated', tpepath=tpepath, distpath=distpath) for wb in waterboards}
        generator = DischargeGenerator(generators, cluster_individual_wbs=True)
        indices = generators['HHNK'].distribution_df.index.droplevel(1).unique()
        cols = [col for col in generators['HHNK'].val_data.columns] + ['weights']
        savename = f'{name}_{method}_{n_wb_scenarios}_{n_scenarios}'
    elif name == 'discharge_random':
        generators = {wb: ScenarioGenerator(wb, 'Aggregated', tpepath=tpepath, distpath=distpath) for wb in waterboards}
        generator = DischargeGenerator(generators, cluster_individual_wbs=True)
        indices = generators['HHNK'].distribution_df.index.droplevel(1).unique()
        cols = [col for col in generators['HHNK'].val_data.columns] + ['weights']
        savename = f'{name}_{method}_{n_wb_scenarios}_{n_scenarios}'
        idx = pd.IndexSlice
        
        data = []
        for wb in generator.waterboards:
            savename_ = f'{wb}_{method}_{n_wb_scenarios}'
            data.append(pd.read_pickle(savepath / f'{savename_}.pkl'))
    elif name == 'WL':
        generator = ScenarioGenerator(name, 'WL', tpepath=tpepath, distpath=distpath)
        indices = generator.distribution_df.index.droplevel(1).unique()
        cols = [col for col in generator.val_data.columns] + ['weights']
        savename = f'{name}_{method}_{n_scenarios}'
    elif name == 'DAM':
        generator = ScenarioGenerator(name, 'DAM NL', tpepath=tpepath, distpath=distpath)
        indices = generator.distribution_df.index.droplevel(1).unique()
        cols = [col for col in generator.val_data.columns] + ['weights']
        savename = f'{name}_{method}_{n_scenarios}'
    elif name in waterboards:
        generator = ScenarioGenerator(name, 'Aggregated', tpepath=tpepath, distpath=distpath)
        indices = generator.distribution_df.index.droplevel(1).unique()
        cols = [col for col in generator.val_data.columns] + ['weights']
        savename = f'{name}_{method}_{n_scenarios}'

    if cfg.parallel_process:
        n_splits = cfg.n_splits
        split_nr = cfg.split_nr

        split_indices = np.linspace(0, len(indices), n_splits+1, dtype=int)
        splitted_indices = []
        for i in range(len(split_indices)-1):
            i0 = split_indices[i]
            i1 = split_indices[i+1]
            splitted_indices.append(indices[i0:i1])
        
        indices = splitted_indices[split_nr]
        savename += f'_part{split_nr}'

    if continue_from_previous:
        # if (savepath / f'{savename}.csv').exists():
        if (savepath / f'{savename}.pkl').exists():
            print('Continuing from previous run.')
            # scenario_df = pd.read_csv(savepath / f'{savename}.csv', index_col=[0,1], parse_dates=True)
            scenario_df = pd.read_pickle(savepath / f'{savename}.pkl')
            indices = scenario_df[scenario_df.isna().any(axis=1)].index.droplevel(1).unique()
            # indices = scenario_df.iloc[len(scenario_df.dropna()):].index.droplevel(1).unique()
        else:
            print('No previous run found, starting from scratch.')
            scenario_df = pd.DataFrame(index=pd.MultiIndex.from_product([indices, range(n_scenarios)], names=['date', 'scenario']), columns=cols, dtype=float)
    else:
        print('Starting from scratch.')
        scenario_df = pd.DataFrame(index=pd.MultiIndex.from_product([indices, range(n_scenarios)], names=['date', 'scenario']), columns=cols, dtype=float)

    for i, date in enumerate(tqdm(indices)):
        # try:
        generator.sample_scenarios(date, n=init_samples, BN=True, return_df=False)
        if name == 'discharge':
            generator.cluster_wb_scenarios(n_wb_scenarios, method=method, return_df=False)
            generator.combine_clusters(n_scenarios, return_df=False, exhaustive=True)
            scenario_df.loc[idx[date, :], :] = generator.combined_clusters.values
        elif name == 'discharge_random':
            # generator.cluster_wb_scenarios(n_wb_scenarios, method=method, return_df=False)
            # Read the waterboard scenarios and make a custom wb_cluster df
            ind = pd.MultiIndex.from_product([range(n_wb_scenarios), generator.waterboards], names=['cluster', 'waterboard'])
            cluster_cols = [col for col in generator.wb_scenarios[generator.waterboards[0]].columns] + ['weights']
            wb_clusters = pd.DataFrame(index=ind, columns=cluster_cols, dtype=float)
            idx = pd.IndexSlice
            for j, wb in enumerate(generator.waterboards):
                wb_clusters.loc[idx[:, wb], :] = data[j].loc[idx[date,:], :].values
            generator.wb_clusters = wb_clusters.astype(float)
            generator.combine_clusters(n_scenarios, method=method, return_df=False, exhaustive=False)
            scenario_df.loc[idx[date, :], :] = generator.combined_clusters.values
            print(scenario_df.dropna())
        else:
            generator.cluster_scenarios(n_scenarios, method=method, return_df=False)
            scenario_df.loc[idx[date, :], :] = generator.clusters.values
        # except:
        #     print(f'Error occured at date {date}')
        #     continue
        if i % 100 == 0: # save every 100 iterations
            # scenario_df.to_csv(savepath / f'{savename}.csv')
            scenario_df.to_pickle(savepath / f'{savename}.pkl')

    # scenario_df.to_csv(savepath / f'{name}_{method}_{n_scenarios}.csv')
    scenario_df.to_pickle(savepath / f'{savename}.pkl')
    

if __name__ == '__main__':
    main()