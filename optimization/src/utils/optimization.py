from .scenario_gen import *
from .trees import *
from .clustering import *
from pymoo.optimize import minimize as minimize_pymoo
from pymoo.algorithms.soo.nonconvex.ga import GA

import pandas as pd
import numpy as np

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import copy

class ObservationData():
    """
    Class for generating observation data for the optimization problem.
    """
    def __init__(self, market_datapath, wl_datapath, wb_datapath, ark_datapath):
        # Load the DAM data
        self.dam = pd.read_pickle(market_datapath / 'DAM_prices.pkl').loc[:, ['NL']].dropna()
        self.dam.index = self.dam.index.tz_convert('Europe/Amsterdam')
        self.dam.columns = ['DAM']

        # Load the IDM data
        self.idm = pd.read_pickle(market_datapath / 'NL_ID.pickle').loc[:, ['ID3']].astype(float)
        self.idm.index = self.idm.index.tz_convert('Europe/Amsterdam')
        self.idm.columns = ['IDM']

        # Load the waterlevel data
        self.wl = pd.read_csv(wl_datapath / 'IJmuiden Noordersluis.csv', index_col=0, parse_dates=True)
        self.wl.index = self.wl.index.tz_convert('Europe/Amsterdam')
        self.wl.columns = ['WL']
        self.wl.replace(to_replace=self.wl.max(), value=np.nan, inplace=True)
        self.wl = self.wl / 100 # cm+NAP to m+NAP
        
        self.wl_hourly = self.wl.copy().resample('1H').mean()

        # Load the waterboard data
        self.waterboards = ['HHNK', 'HDSR', 'Waternet', 'Rijnland']
        self.waterboard_discharge = {
            'HHNK': pd.read_csv(wb_datapath / 'Q_HHNK.csv', index_col=0, parse_dates=True).astype(float).resample('1H').mean(),
            'HDSR': pd.read_csv(wb_datapath / 'Q_HDSR.csv', index_col=0, parse_dates=True).astype(float).resample('1H').mean(),
            'Waternet': pd.read_csv(wb_datapath / 'Q_WATERNET.csv', index_col=0, parse_dates=True).astype(float).resample('1H').mean(),
            'Rijnland': pd.read_csv(wb_datapath / 'Q_RIJNLAND.csv', index_col=0, parse_dates=True).astype(float).resample('1H').mean(),
        }

        # Make a new dataframe with the sum
        self.discharge = pd.concat([
            self.waterboard_discharge[wb].sum(axis=1) for wb in self.waterboards
        ], axis=1)
        self.discharge.loc[:, 'Q'] = self.discharge.sum(axis=1)
        self.discharge = self.discharge.loc[:, ['Q']]
        self.discharge.index = self.discharge.index.tz_convert('Europe/Amsterdam')

        # Load ARK discharge data measured in Maarssen
        self.ark = pd.read_pickle(ark_datapath / 'Q_ark.pkl')

        # Equalize the indices
        # self._equalize_indices()

    def get_observation_data(self, start, end):
        """
        Get the observation data for the optimization problem.
        """
        dam = self.dam.loc[start:end, :]
        idm = self.idm.loc[start:end, :]
        wl_hourly = self.wl_hourly.loc[start:end, :]
        discharge = self.discharge.loc[start:end, :]
        ark = self.ark.loc[start:end, :]

        return dam, idm, wl_hourly, discharge, ark
    
    def get_observation_data_single(self, var, start, end):
        """
        Get the observation data for a single variable.
        """
        if var == 'DAM':
            return self.dam.loc[start:end, :]
        elif var == 'IDM':
            return self.idm.loc[start:end, :]
        elif var == 'WL':
            return self.wl.loc[start:end, :]
        elif var == 'WL_hourly':
            return self.wl_hourly.loc[start:end, :]
        elif var == 'discharge':
            return self.discharge.loc[start:end, :]
        elif var == 'ARK':
            return self.ark.loc[start:end, :]
        else:
            raise ValueError('Variable not recognized.')
        
    def _equalize_indices(self, t_min=None, t_max=None):
        """
        Equalize the indices of the different observation dataframes.
        """
        if t_min is None:
            t_min = max([df.index.min() for df in [self.dam, self.idm, self.wl, self.discharge, self.ark]])
        if t_max is None:
            t_max = min([df.index.max() for df in [self.dam, self.idm, self.wl, self.discharge, self.ark]])

        self.dam = self.dam.loc[t_min:t_max, :]
        self.idm = self.idm.loc[t_min:t_max, :]
        self.wl = self.wl.loc[t_min:t_max, :]
        self.discharge = self.discharge.loc[t_min:t_max, :]
        self.ark = self.ark.loc[t_min:t_max, :]


class IDMScenarios():
    """
    Class for generating scenarios for IDM.
    """
    def __init__(self, obs_dataclass, n_init, n, method='energy', n_prices=24+1, normalize_idm=True, cluster=False):
        self.n_init = n_init
        self.n = n
        self.method = method
        self.dam = obs_dataclass.dam
        self.idm = obs_dataclass.idm
        self.n_prices = n_prices
        self.normalize_idm = normalize_idm
        self.bn_structure = None
        self.cluster = cluster

        if self.method != 'obs':
            self._prep_market_data()

    def _prep_market_data(self):
        # Group the data by day
        self.dam_grouped = pd.DataFrame(index=self.dam.loc[self.dam.index.hour==0].index, columns=[f'DAM{i}' for i in range(1, self.n_prices)])
        self.idm_grouped = pd.DataFrame(index=self.idm.loc[self.idm.index.hour==0].index, columns=[f'IDM{i}' for i in range(1, self.n_prices)])
        
        # Shift the index by one hour -> future IDM and DAM prices
        # DAM0 data is the last DAM price of the previous day
        # DAM0 = 2300 - 00:00, DAM1 = 00:00 - 01:00, DAM2 = 01:00 - 02:00, etc.
        for i in range(self.n_prices):
            self.dam_grouped[f'DAM{i}'] = self.dam.shift(-(i-1)).loc[self.dam.index.hour==0, 'DAM'].values
        for i in range(self.n_prices):
            self.idm_grouped[f'IDM{i}'] = self.idm.shift(-(i-1)).loc[self.idm.index.hour==0, 'IDM'].values
        
    def _prep_bn_data(self, t_min=None, t_max=None):
        """
        Prepare the data for the Bayesian network.
        """
        # Equalize the indices
        if t_min is None:
            t_min = max([df.index.min() for df in [self.dam_grouped, self.idm_grouped]])
        if t_max is None:
            t_max = min([df.index.max() for df in [self.dam_grouped, self.idm_grouped]])

        self.t_min = t_min
        self.t_max = t_max
        
        dam_grouped = self.dam_grouped.loc[t_min:t_max, :].copy()
        idm_grouped = self.idm_grouped.loc[t_min:t_max, :].copy()

        self.bn_data = pd.concat([dam_grouped, idm_grouped], axis=1).dropna()
        self._normalize_bn_data()
    
    def _normalize_bn_data(self, inplace=True, bn_data=None, mask=True):
        """
        Normalize the data by dividing the IDM prices by the DAM prices
        """
        if inplace:
            bn_data = self.bn_data
        else:
            if bn_data is None:
                raise ValueError('If inplace is False, bn_date must be specified.')

        if self.normalize_idm:
            for h in range(self.n_prices):
                dams = bn_data[f'DAM{h}'].values.flatten()
                dams[dams == 0] = 0.01 # Add a small number to prevent division by zero
                bn_data[f'IDM{h}'] = bn_data[f'IDM{h}'].values.flatten() / dams 
        if mask:
            # Mask all the values of the scaled IDM to remove outliers
            min_val = bn_data.loc[:, [f'IDM{h}' for h in range(self.n_prices)]].min()
            min_val = min_val.mean() - 4 * min_val.std()
            max_val = bn_data.loc[:, [f'IDM{h}' for h in range(self.n_prices)]].max()
            max_val = max_val.mean() + 4 * max_val.std()

            for h in range(self.n_prices):
                bn_data.loc[bn_data[f'IDM{h}'] < min_val, f'IDM{h}'] = min_val
                bn_data.loc[bn_data[f'IDM{h}'] > max_val, f'IDM{h}'] = max_val

        if inplace:
            self.bn_data = bn_data

    def _cluster_bn_data(self, data, method='energy'):
        """
        Cluster the data.
        """
        clusterer = ReduceForward(
            x=data,
            cdn=self.n,
            w=None,
            dist=method,
            p=1,
            parallel=False,
            verbose=False
        )
        clusterer.reduce_forward()
        return clusterer.clusters.astype(float)
                
    def update_bn(self, t_min=None, t_max=None):
        """
        Update the Bayesian network.
        """
        if self.method != 'obs':
            self._prep_bn_data(t_min=t_min, t_max=t_max)
            self.bn = IDM_BN(
                bn_data=self.bn_data,
                bn_structure=None,
                normalize_idm=True,
                bn_data_is_normalized=True,
                pred_hor=self.n_prices-1,
                mask_factor=4,
                mask_method='std',
                mask_by_hour=True
            )

    def _mask_inputs(self, dam_scenario, idm_observation):
        """
        Mask the inputs of the Bayesian network so that any values outside of the range are set to the minimum or maximum value.
        """
        # Mask the DAM scenario
        for h in range(len(dam_scenario)):
            val = dam_scenario[h] - self.bn_data[f'DAM{h}'].mean()
            min_ = self.bn_data[f'DAM{h}'].mean() - 3 * self.bn_data[f'DAM{h}'].std()
            max_ = self.bn_data[f'DAM{h}'].mean() + 3 * self.bn_data[f'DAM{h}'].std()
            if val < min_:
                dam_scenario[h] = min_#self.bn_data[f'DAM{h}'].min()
            elif val > max_:
                dam_scenario[h] = max_
            #elif dam_scenario[h] > self.bn_data[f'DAM{h}'].max():
                # dam_scenario[h] = self.bn_data[f'DAM{h}'].max()
        
        # Mask the IDM observation
        for h in range(len(idm_observation)):
            val = idm_observation[h] - self.bn_data[f'IDM{h}'].mean()
            min_ = self.bn_data[f'IDM{h}'].mean() - 3 * self.bn_data[f'IDM{h}'].std()
            max_ = self.bn_data[f'IDM{h}'].mean() + 3 * self.bn_data[f'IDM{h}'].std()
            if val < min_:
                idm_observation[h] = min_
            elif val > max_:
                idm_observation[h] = max_
            # if idm_observation[h] < self.bn_data[f'IDM{h}'].min():
            #     idm_observation[h] = self.bn_data[f'IDM{h}'].min()
            # elif idm_observation[h] > self.bn_data[f'IDM{h}'].max():
            #     idm_observation[h] = self.bn_data[f'IDM{h}'].max()

        return dam_scenario, idm_observation
                
    def __call__(self, dam_scenario, idm_observations, is_normalized=False):
        # dam_scenario, idm_observations = self._mask_inputs(dam_scenario, idm_observations)
        scens = self.bn.infer_idm(dam_scenario, idm_observations, n_samples=self.n_init, is_normalized=is_normalized)
        if (not self.cluster) or (self.n >= self.n_init):
            df = pd.DataFrame(index=range(scens.shape[0]), columns=[f'IDM{i}' for i in range(1, scens.shape[1]+1)]).astype(float)
            df.loc[:, :] = scens
            df.loc[:, 'weight'] = 1 / scens.shape[0]
            return df
        elif self.n < self.n_init:
            return self._cluster_bn_data(scens, method=self.method)
        else:
            return scens
    
class ScenarioData():
    """
    Class to keep all scenario data for the optimization problem.
    """
    def __init__(self, scenario_params, scenario_path, constrain_tree_complexity=True, complexity_reduction=0.8):
        self.scenario_params = scenario_params
        self.scenario_path = scenario_path
        self.read_wl()
        self.read_discharge()
        self.read_dam()
        self.n_root_steps = scenario_params['n_root_steps']

        # Ensure that all dataframes have the same index
        datemin = max(self.wl.index.min(), self.discharge.index.min(), self.dam.index.min())
        datemax = min(self.wl.index.max(), self.discharge.index.max(), self.dam.index.max())
        self.wl = self.wl.loc[datemin:datemax]
        self.discharge = self.discharge.loc[datemin:datemax]
        self.dam = self.dam.loc[datemin:datemax]

        self.constrain_tree_complexity = constrain_tree_complexity
        self.complexity_reduction = complexity_reduction
        self.idx = pd.IndexSlice

    def read_wl(self):
        if self.scenario_params['wl']['method'] == 'obs':
            self.wl = pd.read_pickle(self.scenario_path / 'WL_obs.pkl')    
        else:
            self.wl = pd.read_pickle(self.scenario_path / f'WL_{self.scenario_params["wl"]["method"]}_{self.scenario_params["wl"]["n"]}.pkl').dropna()
        self.wl.index = pd.MultiIndex.from_arrays([self.wl.index.droplevel(1).tz_convert('Europe/Amsterdam'), self.wl.index.droplevel(0)], names=['date', 'scenario'])
        
    def read_discharge(self):
        if self.scenario_params['discharge']['method'] == 'obs':
            self.discharge = pd.read_pickle(self.scenario_path / 'discharge_obs.pkl')
        else:
            if self.scenario_params['discharge']['n'] < 10:
                name = 'discharge'
            else:
                name = 'discharge_random'
            self.discharge = pd.read_pickle(self.scenario_path / f'{name}_{self.scenario_params["discharge"]["method"]}_{self.scenario_params["discharge"]["n_wb"]}_{self.scenario_params["discharge"]["n"]}.pkl').dropna()
        self.discharge.index = pd.MultiIndex.from_arrays([self.discharge.index.droplevel(1).tz_convert('Europe/Amsterdam'), self.discharge.index.droplevel(0)], names=['date', 'scenario'])
    
    def read_dam(self):
        if self.scenario_params['DAM']['method'] == 'obs':
            self.dam = pd.read_pickle(self.scenario_path / 'DAM_obs.pkl')
        else:
            self.dam = pd.read_pickle(self.scenario_path / f'DAM_{self.scenario_params["DAM"]["method"]}_{self.scenario_params["DAM"]["n"]}.pkl').dropna()
        self.reformat_dam_index()

    def generate_tree(self, date, varname, scenarios=None):
        if scenarios is None:
            scenarios = self.get_scenarios(date, varname)
            
        scenarios.index = scenarios.index.droplevel(0)
        problem = ScenarioTreeOptimizationProblem(
            scenarios=scenarios,
            n_leafs=len(scenarios.index),
            n_splits=10, # max nr of node-splitting locations
            constrain_complexity=self.constrain_tree_complexity,
            complexity_reduction_factor=self.complexity_reduction,
            n_root_scenario_steps=self.n_root_steps
        )
        algorithm = GA(
            pop_size=30,
            eliminate_duplicates=True,
        )
        # termination = DefaultSingleObjectiveTermination(
        #     n_max_gen=100,
        #     # max_time=30,
        #     period=20,
        # )
        termination = ("time", 10)
        # termination = ("n_gen", 100)
        res = minimize_pymoo(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            seed=1,
            verbose=False,
        )

        node_locations = list(res.X[:problem.n_splits])
        split_sizes = list(res.X[problem.n_splits:])

        node_locations, split_sizes = problem.format_node_split(node_locations, split_sizes)
        
        tc = TreeClusterer(scenarios, verbose=False)
        tc.generate_tree(
            node_locations=node_locations,
            split_sizes=split_sizes,
            method=self.scenario_params[varname]['method'],
        )
        return tc.tree
    
    def get_scenarios(self, date, varname):
        if varname == 'DAM':
            return self.dam.loc[self.idx[date, :], :]
        elif varname == 'wl':
            return self.wl.loc[self.idx[date, :], :]
        elif varname == 'discharge':
            return self.discharge.loc[self.idx[date, :], :]
        else:
            raise ValueError(f'Invalid varname {varname}')
        
    def reformat_dam_index(self):
        dates = self.dam.index.get_level_values(0).unique()
        scenario_nrs = self.dam.index.get_level_values(1).unique()
        dates = dates + pd.DateOffset(hours=1)
        dates = dates.tz_convert('Europe/Amsterdam')
        dates = pd.date_range(dates[0] - pd.DateOffset(hours=1), freq='1D', periods=len(dates))
        self.dam.index = pd.MultiIndex.from_product([dates, scenario_nrs],names=['date', 'scenario'])

class Node():
    def __init__(self, index, domain, values, parent_id, p_conditional, p_marginal=None):
        self.index = index
        self.domain = domain
        self.values = values
        self.parent_id = parent_id
        self.p_conditional = p_conditional
        self.p_marginal = p_marginal

    def __repr__(self):
        return f'Node {self.index} with domain {self.domain}, values {self.values} and probability {self.p_marginal}'
    
class NodesEdges():
    def __init__(self, tree):
        self.tree = tree
        
        self.counter = 0
        self.nodes = []
        self.edges = []

        self._get_nodes(self.tree)
    
    def __repr__(self):
        returnstring = f'Nodes and edges with nodes \n'
        for node in self.nodes:
            returnstring += f'{node} \n'
        return returnstring

    def __call__(self, t):
        """
        Return all nodes indices with t in the domain.
        """
        return [node.index for node in self.nodes if t in node.domain]
    
    def __getitem__(self, index):
        """
        Return the node with the given index.
        """
        return self.nodes[index]
        
    def _get_marginal_probabilities(self, node):
        """
        Get the marginal probabilities of the given node.
        """
        if node.parent_id is None:
            return node.p_conditional
        else:
            parent = self.nodes[node.parent_id]
            return node.p_conditional * self._get_marginal_probabilities(parent)
        
    def _correct_domain(self, domain):
        """
        Correct the domain of the node such that it starts with 1.
        """
        return [d+1 for d in domain]
    
    def _get_nodes(self, tree, parent_index=None):
        if tree.is_leaf:
            node = Node(copy.deepcopy(self.counter), self._correct_domain(tree.domain), tree.edge_values, parent_index, tree.weight)
            node.p_marginal = self._get_marginal_probabilities(node)
            self.nodes.append(node)
            self.counter += 1
        else:
            node = Node(copy.deepcopy(self.counter), self._correct_domain(tree.domain), tree.edge_values, parent_index, tree.weight)
            node.p_marginal = self._get_marginal_probabilities(node)
            self.nodes.append(node)
            parent_index = copy.deepcopy(self.counter)
            self.counter += 1
            for child in tree.children:
                self.edges.append((parent_index, self.counter))
                self._get_nodes(child, parent_index=parent_index)

class Fan():
    """
    Class to generate nodes and edges for a scenario fan to equalize the workflow compared to tree based scenarios.
    """
    def __init__(self, data_dict, probabilities, cluster_method='energy', n_root_steps=3):
        self.data_dict = data_dict
        self.n_root_steps = n_root_steps
        self.probabilities = probabilities
        self.n_nodes = len(self.probabilities) +1
        self.cluster_method = cluster_method
        self.nodes = []
        self.edges = []
        self._generate_nodes()

    def __call__(self, t):
        """
        Return all nodes indices with t in the domain.
        """
        return [node.index for node in self.nodes if t in node.domain]
    
    def __getitem__(self, index):
        """
        Return the node with the given index.
        """
        return self.nodes[index]

    def reduce_root_steps(self, data):
        clusterer = ReduceForward(
            x=data.T,
            cdn=1,
            w=None,
            dist=self.cluster_method,
            p=1,
            parallel=False,
            verbose=False
        )
        clusterer.reduce_forward()
        return clusterer.clusters.astype(float)

    def _generate_nodes(self):
        # Root node
        root_data = np.array([
            self.data_dict[i] for i in range(1, self.n_root_steps+1)
        ])
        root_data = self.reduce_root_steps(root_data)
        root_values = root_data.iloc[:, :-1].values[0]
        root_node = Node(
            index=0,
            domain=[i for i in range(1, len(root_values)+1)],
            values=root_data.iloc[:, :-1].values[0],
            p_marginal=1,
            p_conditional=1,
            parent_id=None,
        )
        self.nodes.append(root_node)

        for child_id in range(1, self.n_nodes):
            vals = [self.data_dict[i+1][child_id-1] for i in range(self.n_root_steps, [k for k in self.data_dict.keys()][-1])]
            child_node = Node(
                index=child_id,
                domain=[self.n_root_steps + i for i in range(1, len(vals)+1)],
                values=vals,
                p_marginal=self.probabilities[child_id-1],
                p_conditional=self.probabilities[child_id-1],
                parent_id=0,
            )
            self.nodes.append(child_node)
            self.edges.append((0, child_id))


class ResultNode():
    """
    Class for storing the results of the optimization problem.
    """
    def __init__(self, index, domain, parent_id, results=None, q_node=None, wl_node=None, p_conditional=None, p_marginal=None):
        self.index = index
        self.domain = domain
        self.parent_id = parent_id
        self.results = results
        self.q_node = q_node
        self.wl_node = wl_node
        self.p_conditional = p_conditional
        self.p_marginal = p_marginal

class NZKProblem():
    def __init__(self, optimization_data, optimization_settings, tree_classes=['discharge', 'wl'], pred_hor=48, logfile=None):
        self.optimization_data = optimization_data
        self.optimization_settings = optimization_settings
        self.tree_classes = tree_classes
        self.pred_hor = pred_hor
        self.lin_qh_pump_coefs = [{'a': -27.70, 'b': 269.58}, {'a': -45.72, 'b': 314.79}, {'a': -60.80, 'b': 383.10}]
        self.lin_qh_gate_coefs = [860.21, 153.34]
        self.logfile = logfile
        self.obj_scale = 1

        if 'discharge' in self.tree_classes:
            self.nodes_edges_discharge = NodesEdges(self.optimization_data['wb_discharge'][1])
        else:
            self.nodes_edges_discharge = Fan(self.optimization_data['wb_discharge'], self.optimization_data['probabilities']['wb_discharge'])
        self.n_discharge_scenarios_per_timestep = self._get_n_scenarios_per_timestep(self.nodes_edges_discharge)

        if 'wl' in self.tree_classes:
            self.nodes_edges_wl = NodesEdges(self.optimization_data['wl'][1])
        else:
            self.nodes_edges_wl = Fan(self.optimization_data['wl'], self.optimization_data['probabilities']['wl'])    
        self.n_wl_scenarios_per_timestep = self._get_n_scenarios_per_timestep(self.nodes_edges_wl)
        
        self.idx = pd.IndexSlice

        self.wl_constraint = self.optimization_settings['wl_constraint_type']
        self.obj_type = self.optimization_settings['obj_type']

        if self.wl_constraint == 'cvar':
            # CVaR constraint on upper bound of WL
            self.cvar_alpha = self.optimization_settings['cvar_alpha']
            self.var_wl = self.optimization_settings['var_wl']
            self.cvar_wl = self.optimization_settings['cvar_wl']
            self.h_max = self.optimization_settings['h_max']
        elif self.wl_constraint == 'chance':
            self.p_chance = self.optimization_settings['p_chance']
            self.h_max = self.optimization_settings['h_max'] # -0.4
        elif self.wl_constraint == 'robust':
            self.h_max = self.optimization_settings['h_max'] # -0.4
        else:
            raise ValueError(f'Invalid wl_constraint {self.wl_constraint}')


    def _get_n_scenarios_per_timestep(self, tree_class):
        n_nodes_per_timestep = [0 for _ in range(self.pred_hor)]
        for node in tree_class.nodes:
            for t in node.domain:
                n_nodes_per_timestep[t-1] += 1
        return n_nodes_per_timestep

    def _get_node(self, node_index, tree_class):
        if tree_class == 'discharge':
            return self.nodes_edges_discharge.nodes[node_index]
        elif tree_class == 'wl':
            return self.nodes_edges_wl.nodes[node_index]
        else:
            raise ValueError(f'Invalid tree_class {tree_class}')
        
    def _get_edge(self, edge_index, tree_class):
        if tree_class == 'discharge':
            return self.nodes_edges_discharge.edges[edge_index]
        elif tree_class == 'wl':
            return self.nodes_edges_wl.edges[edge_index]
        else:
            raise ValueError(f'Invalid tree_class {tree_class}')
    def _get_marginal(self, t_q_wl):
        t = t_q_wl[0]
        q = t_q_wl[1]
        wl = t_q_wl[2]

        q_node = self.nodes_edges_discharge[q]
        wl_node = self.nodes_edges_wl[wl]
        return q_node.p_marginal * wl_node.p_marginal
    
    def _get_prev_timestep(self, t_q_wl):
        """
        Get the index of the previous timestep.
        """
        t = t_q_wl[0]
        q = t_q_wl[1]
        wl = t_q_wl[2]

        if t == 1:
            return None
        
        # Check if the timestep is the first in any domain of the q or wl nodes
        if t == self.nodes_edges_discharge[q].domain[0]:
            parent_index = self.nodes_edges_discharge[q].parent_id
            q = self.nodes_edges_discharge[parent_index].index
        if t == self.nodes_edges_wl[wl].domain[0]:
            parent_index = self.nodes_edges_wl[wl].parent_id
            wl = self.nodes_edges_wl[parent_index].index
        return (t-1, q, wl)
    
    def _get_wl(self, t, wl):
        node = self.nodes_edges_wl[wl]
        t_id = node.domain.index(t)
        return node.values[t_id]
    
    def _get_q(self, t, q):
        node = self.nodes_edges_discharge[q]
        t_id = node.domain.index(t)
        return node.values[t_id]
        
        
    def make_tree_model(self):
        model = pyo.ConcreteModel()

        model.A_nzk = pyo.Param(initialize=36*10e6) # m2
        model.dt = pyo.Param(initialize=3600) # s
        
        model.discharge_nodes = pyo.RangeSet(0, len(self.nodes_edges_discharge.nodes)-1)
        model.wl_nodes = pyo.RangeSet(0, len(self.nodes_edges_wl.nodes)-1)
        model.timesteps = pyo.Set(initialize=range(1, self.pred_hor+1))
        model.lin_qhpump = pyo.RangeSet(1, len(self.lin_qh_pump_coefs))
        
        timestep_scenario_combinations = []
        for t in model.timesteps:
            for q in self.nodes_edges_discharge(t):
                for h in self.nodes_edges_wl(t):
                    timestep_scenario_combinations.append((t, q, h))
        model.timestep_scenario_combinations = pyo.Set(initialize=timestep_scenario_combinations, dimen=3)

        model.h_nzk = pyo.Var(model.timestep_scenario_combinations, domain=pyo.Reals, bounds=(-1, 1), initialize=-0.45)
        model.q_gate = pyo.Var(model.timestep_scenario_combinations, domain=pyo.NonNegativeReals, bounds=(0, 500), initialize=0)
        model.q_pump = pyo.Var(model.timestep_scenario_combinations, domain=pyo.NonNegativeReals, bounds=(0,260), initialize=0)
        model.B_gate = pyo.Var(model.timestep_scenario_combinations, domain=pyo.Binary, initialize=0)
        model.B_pump = pyo.Var(model.timestep_scenario_combinations, domain=pyo.Binary, initialize=0)
        model.dH2 = pyo.Var(model.timestep_scenario_combinations, domain=pyo.NonNegativeReals, bounds=(0,50), initialize=0)

        if self.wl_constraint == 'cvar':
            model.VaR_wl = pyo.Param(initialize=self.var_wl)
            model.CVaR_wl_con = pyo.Param(initialize=self.cvar_wl)
            model.z_wl = pyo.Var(model.timestep_scenario_combinations, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, 5))
            model.alpha_wl = pyo.Param(initialize=self.cvar_alpha)
        elif self.wl_constraint == 'chance':
            model.p_chance = pyo.Param(initialize=self.p_chance)
            model.B_wl = pyo.Var(model.timestep_scenario_combinations, domain=pyo.Binary)
        elif self.wl_constraint == 'robust':
            model.penalty_wl = pyo.Var(model.timestep_scenario_combinations, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, 5))

        def volume_balance_rule(model, t, q, wl):
            t_q_wl = (t, q, wl)
            # If its the first timestep, h[t-1] is from data
            if t == 1:
                h0 = self.optimization_data['h_nzk'][0]
            
            # Get the correct timestep and scenarios for t-1
            else:
                h0 = model.h_nzk[self._get_prev_timestep(t_q_wl)]
            
            return model.h_nzk[t_q_wl] == h0 + (model.dt / model.A_nzk) * (self._get_q(t, q) + self.optimization_data['ark_discharge'][t] - model.q_gate[t_q_wl] - model.q_pump[t_q_wl])
        model.volume_balance = pyo.Constraint(model.timestep_scenario_combinations, rule=volume_balance_rule)

        def dh_pump(model, t, q, wl):
            t_q_wl = (t, q, wl)

            if t == 1:
                # dh = self.optimization_data['wl'][0] - self.optimization_data['h_nzk'][0] # dh is based on the previous timestep NS wl
                dh = self._get_wl(t, wl) - self.optimization_data['h_nzk'][0] # dh is based on the NS WL forecast
            else:
                (t_, q_, wl_) = self._get_prev_timestep(t_q_wl)
                # dh = self._get_wl(t_, wl_) - model.h_nzk[(t_, q_, wl_)] # dh is based on the previous timestep NS wl
                dh = self._get_wl(t, wl) - model.h_nzk[(t_, q_, wl_)] # dh is based on the NS WL forecast

            return dh
            
        def pump_rule(model, t, q, wl, lin):
            t_q_wl = (t, q, wl)
            dh = dh_pump(model, t, q, wl)
            return model.q_pump[t_q_wl] / 260 <= model.B_pump[t_q_wl] * (self.lin_qh_pump_coefs[lin-1]['a'] * dh + self.lin_qh_pump_coefs[lin-1]['b']) / 260
        model.pump = pyo.Constraint(model.timestep_scenario_combinations, model.lin_qhpump, rule=pump_rule)

        def dh_gate(model, t, q, wl):
            t_q_wl = (t, q, wl)

            if t == 1:
                # dh = self.optimization_data['h_nzk'][0] - self.optimization_data['wl'][0] # dh is based on the previous timestep NS wl
                dh = self.optimization_data['h_nzk'][0] - self._get_wl(t, wl) # dh is based on the NS WL forecast
            else:
                (t_, q_, wl_) = self._get_prev_timestep(t_q_wl)
                # dh = model.h_nzk[(t_, q_, wl_)] - self._get_wl(t_, wl_) # dh is based on the previous timestep NS wl
                dh = model.h_nzk[(t_, q_, wl_)] - self._get_wl(t, wl) # dh is based on the NS WL forecast
            return dh
        
        def gate_rule(model, t, q, wl):
            t_q_wl = (t, q, wl)
            dh = dh_gate(model, t, q, wl)
            return model.q_gate[t_q_wl] / 500 <= model.B_gate[t_q_wl] * (self.lin_qh_gate_coefs[0] * dh + self.lin_qh_gate_coefs[1]) / 500
        model.gate_con = pyo.Constraint(model.timestep_scenario_combinations, rule=gate_rule)

        def binary_gate_rule_upper(model, t, q, wl):
            t_q_wl = (t, q, wl)

            # If dh < 0: B=0, if dh >= 0: B=free
            dh = dh_gate(model, t, q, wl)
            delta_ijm = 0.12
            return (dh - delta_ijm) + (1 - model.B_gate[t_q_wl]) * 10 >= 0
        model.binary_gate_con_upper = pyo.Constraint(model.timestep_scenario_combinations, rule=binary_gate_rule_upper)

        def binary_gate_rule_lower(model, t, q, wl):
            t_q_wl = (t, q, wl)

            # If dh < 0: B=0, if dh >= 0: B=free
            dh = dh_gate(model, t, q, wl)
            delta_ijm = 0.12
            return (dh - delta_ijm) - (model.B_gate[t_q_wl]) * 10 <= 0
        model.binary_gate_con_lower = pyo.Constraint(model.timestep_scenario_combinations, rule=binary_gate_rule_lower)

        def binary_pump_rule_upper(model, t, q, wl):
            t_q_wl = (t, q, wl)

            # If dh < 0: B=0, if dh >= 0: B=free
            dh = dh_pump(model, t, q, wl)
            delta_ijm = 0.01 # min 1cm head difference for pumping
            return (dh - delta_ijm) + (1 - model.B_pump[t_q_wl]) * 10 >= 0
        model.binary_pump_con_upper = pyo.Constraint(model.timestep_scenario_combinations, rule=binary_pump_rule_upper)

        def binary_pump_rule_lower(model, t, q, wl):
            t_q_wl = (t, q, wl)

            # If dh < 0: B=0, if dh >= 0: B=free
            dh = dh_pump(model, t, q, wl)
            delta_ijm = 0.01
            return (dh - delta_ijm) - (model.B_pump[t_q_wl]) * 10 <= 0
        model.binary_pump_con_lower = pyo.Constraint(model.timestep_scenario_combinations, rule=binary_pump_rule_lower)

        def bin_complementarity_rule(model, t, q, wl):
            t_q_wl = (t, q, wl)
            return model.B_gate[t_q_wl] + model.B_pump[t_q_wl] <= 1
        model.bin_complementarity = pyo.Constraint(model.timestep_scenario_combinations, rule=bin_complementarity_rule)

        def quad_head_rule(model, t, q, wl):
            t_q_wl = (t, q, wl)
            dh = dh_pump(model, t, q, wl)
            return model.dH2[t_q_wl] == dh**2
        model.quad_head = pyo.Constraint(model.timestep_scenario_combinations, rule=quad_head_rule)

        # CVar WL constraints
        def cvar_wl_rule(model, t):
            # Only on the timesteps with more than one scenario!
            # For the others we can penalize the slack variable
            if self.n_wl_scenarios_per_timestep[t-1] * self.n_discharge_scenarios_per_timestep[t-1] == 1:
                return pyo.Constraint.Skip
            return model.VaR_wl + (1 / (1 - model.alpha_wl)) * sum(model.z_wl[t_q_wl] * self._get_marginal(t_q_wl) for t_q_wl in model.timestep_scenario_combinations if t_q_wl[0] == t) <= model.CVaR_wl_con

        def z_wl_rule(model, t, q, wl):
            t_q_wl = (t, q, wl)
            if self.n_wl_scenarios_per_timestep[t-1] * self.n_discharge_scenarios_per_timestep[t-1] == 1:
                return model.z_wl[t_q_wl] >= model.h_nzk[t_q_wl] - self.h_max   
            return model.z_wl[t_q_wl] >= model.h_nzk[t_q_wl] - model.VaR_wl
        
        # Chance WL constraints
        def big_M_wl_upper(mode, t, q, wl):
            t_q_wl = (t, q, wl)
            dh = model.h_nzk[t_q_wl] - self.h_max
            M = 10
            return dh - model.B_wl[t_q_wl] * M <= 0
        
        def big_M_wl_lower(model, t, q, wl):
            t_q_wl = (t, q, wl)
            dh = model.h_nzk[t_q_wl] - self.h_max
            M = 10
            return dh + (1 - model.B_wl[t_q_wl]) * M >= 0
        
        def chance_wl_rule(model, t):
            return sum(model.B_wl[t_q_wl] * self._get_marginal(t_q_wl) for t_q_wl in model.timestep_scenario_combinations if t_q_wl[0] == t) <= model.p_chance
        
        # Robust constraint
        def robust_wl_rule(model, t, q, wl):
            t_q_wl = (t, q, wl)
            return model.h_nzk[t_q_wl] <= self.h_max + model.penalty_wl[t_q_wl]

        if self.wl_constraint == 'cvar':
            model.cvar_wl = pyo.Constraint(model.timesteps, rule=cvar_wl_rule)
            model.z_wl_con = pyo.Constraint(model.timestep_scenario_combinations, rule=z_wl_rule)
        elif self.wl_constraint == 'chance':
            model.big_M_wl_upper = pyo.Constraint(model.timestep_scenario_combinations, rule=big_M_wl_upper)
            model.big_M_wl_lower = pyo.Constraint(model.timestep_scenario_combinations, rule=big_M_wl_lower)
            model.chance_wl = pyo.Constraint(model.timesteps, rule=chance_wl_rule)
        elif self.wl_constraint == 'robust':
            model.robust_wl = pyo.Constraint(model.timestep_scenario_combinations, rule=robust_wl_rule)
        else:
            raise ValueError(f'Invalid wl_constraint {self.wl_constraint}')

        def objective(model):
            a = 0.033
            b = 0.061
            c = 11.306
            
            def pump_energy(t, q, wl):
                t_q_wl = (t, q, wl)
                dh = dh_pump(model, t, q, wl)
                return (a * model.q_pump[t_q_wl] ** 2 + b * model.dH2[t_q_wl] \
                    * model.q_pump[t_q_wl] + c * model.q_pump[t_q_wl] * dh) * model.dt / 3600 / 1000 # kWh -> MWh
            
            def intraday_energy(t_q_wl):
                t = t_q_wl[0]
                q = t_q_wl[1]
                wl = t_q_wl[2]
                return pump_energy(t, q, wl) - self.optimization_data['E_dam'][t]

            def dam_energy(t_q_wl):
                t = t_q_wl[0]
                q = t_q_wl[1]
                wl = t_q_wl[2]
                return pump_energy(t, q, wl)
            
            dam_trading_indices = [t for t in self.optimization_data['E_dam'].keys() if np.isnan(self.optimization_data['E_dam'][t])]
            id_trading_indices = [t for t in self.optimization_data['E_dam'].keys() if not np.isnan(self.optimization_data['E_dam'][t])]
            p_max_da = max( max( self.optimization_data['dam'][t_q_wl[0]][i] for t_q_wl in model.timestep_scenario_combinations if t_q_wl[0] in dam_trading_indices) for i in range(len(self.optimization_data['probabilities']['dam'])))
            p_max_id = max( max( self.optimization_data['idm'][t_q_wl[0]][i] for t_q_wl in model.timestep_scenario_combinations if t_q_wl[0] in id_trading_indices) for i in range(len(self.optimization_data['probabilities']['idm'])))
            p_max = max(p_max_da, p_max_id) # for scaling the objective
            self.obj_scale = p_max

            def idm_cost():
                # Expected cost of intraday trading    
                return sum( sum(
                    intraday_energy(t_q_wl) * self._get_marginal(t_q_wl) * self.optimization_data['idm'][t_q_wl[0]][i] \
                        for t_q_wl in model.timestep_scenario_combinations if t_q_wl[0] in id_trading_indices
                ) * self.optimization_data['probabilities']['idm'][i] for i in range(len(self.optimization_data['probabilities']['idm'])))

            def dam_cost():
                # Expected cost of DAM trading
                return sum( sum(
                    dam_energy(t_q_wl) * self._get_marginal(t_q_wl) * self.optimization_data['dam'][t_q_wl[0]][i] \
                        for t_q_wl in model.timestep_scenario_combinations if t_q_wl[0] in dam_trading_indices
                ) * self.optimization_data['probabilities']['dam'][i] for i in range(len(self.optimization_data['probabilities']['dam'])))

            def wl_penalty():
                E_max = a*260**2 * b * 5**2 * 260 + c * 260 * 5
                if self.wl_constraint == 'robust':
                    # Relax the WL constraint with penalty of max pump cost (260m3/s and 5m head) with the max observed DAM price
                    return sum(model.penalty_wl[t_q_wl] * 100 * E_max * p_max for t_q_wl in model.timestep_scenario_combinations)
                elif self.wl_constraint == 'cvar':
                    # Relax the WL constraint in the root node with penalty of max pump cost (260m3/s and 5m head) with the max observed DAM price
                    one_scenario_timesteps = [t for t in model.timesteps if self.n_wl_scenarios_per_timestep[t-1] * self.n_discharge_scenarios_per_timestep[t-1] == 1]
                    return sum(model.z_wl[t_q_wl] * 100 * E_max * p_max for t_q_wl in model.timestep_scenario_combinations if t_q_wl[0] in one_scenario_timesteps)
                else:
                    return 0

            return (idm_cost() + dam_cost() + wl_penalty()) / self.obj_scale
        model.obj = pyo.Objective(rule=objective, sense=pyo.minimize)
        
        return model
    
    def make_model(self):
        self.model = self.make_tree_model()

    def solve(self, verbose=False, mode='normal', extra_options=[], option_values=[]):
        # self.model = self.make_tree_model()
        opt = SolverFactory('gurobi')
        opt.options["NonConvex"] = 2
        opt.options["MIPGap"] = 0.01
        opt.options['MIPGapAbs'] = 1 / self.obj_scale
        opt.options['FeasibilityTol'] = 1e-6
        # if self.optimization_settings['n_scenarios'] < 25:
        if mode == 'normal':
            opt.options['TimeLimit'] = 15*60 # 15 minutes
        elif mode == 'extra_time':
            opt.options['SolutionLimit'] = 1
        # else:
            # opt.options['TimeLimit'] = 15*60*2 # 0.5 hour
        for i, opt in enumerate(extra_options):
            opt.options[opt] = option_values[i]
        try:
            self.opt_results = opt.solve(self.model, tee=verbose, logfile=self.logfile)
        except ValueError:
            # Solution aborted
            print('Time limit eached without solution, resolving until one feasible solution is found.')
            self.solve(verbose=verbose, mode='extra_time', extra_options=extra_options, option_values=option_values)
        return self.model

    def get_results(self, return_results=True):
        self.result_nodes = []
        
        self.node_id_mapping = dict()
        self.counter = 0
        
        for t_q_wl in self.model.timestep_scenario_combinations:
            t = t_q_wl[0]
            q = t_q_wl[1]
            wl = t_q_wl[2]

            if (q, wl) not in self.node_id_mapping.keys():
                self.result_nodes.append(ResultNode(
                    index=self.counter,
                    domain=[t],
                    parent_id=None,
                    q_node=self.nodes_edges_discharge[q],
                    wl_node=self.nodes_edges_wl[wl],
                    p_conditional=self.nodes_edges_discharge[q].p_conditional * self.nodes_edges_wl[wl].p_conditional,
                    p_marginal=self.nodes_edges_discharge[q].p_marginal * self.nodes_edges_wl[wl].p_marginal,
                    results={
                        'h_nzk': [self.model.h_nzk[t, q, wl].value],
                        'q_gate': [self.model.q_gate[t, q, wl].value],
                        'q_pump': [self.model.q_pump[t, q, wl].value],
                        'B_gate': [self.model.B_gate[t, q, wl].value],
                        'B_pump': [self.model.B_pump[t, q, wl].value],
                        'dH2': [self.model.dH2[t, q, wl].value],
                    }
                ))
                self.node_id_mapping[(q, wl)] = copy.deepcopy(self.counter)
                self.counter += 1
            else:
                self.result_nodes[self.node_id_mapping[(q, wl)]].domain.append(t)
                self.result_nodes[self.node_id_mapping[(q, wl)]].results['h_nzk'].append(self.model.h_nzk[t, q, wl].value)
                self.result_nodes[self.node_id_mapping[(q, wl)]].results['q_gate'].append(self.model.q_gate[t, q, wl].value)
                self.result_nodes[self.node_id_mapping[(q, wl)]].results['q_pump'].append(self.model.q_pump[t, q, wl].value)
                self.result_nodes[self.node_id_mapping[(q, wl)]].results['B_gate'].append(self.model.B_gate[t, q, wl].value)
                self.result_nodes[self.node_id_mapping[(q, wl)]].results['B_pump'].append(self.model.B_pump[t, q, wl].value)
                self.result_nodes[self.node_id_mapping[(q, wl)]].results['dH2'].append(self.model.dH2[t, q, wl].value)
        
        # Now add the parent nodes
        for node in self.result_nodes:
            q_node = self.nodes_edges_discharge[node.q_node.index]
            wl_node = self.nodes_edges_wl[node.wl_node.index]
            q_parent = node.q_node.parent_id
            wl_parent = node.wl_node.parent_id

            if (q_parent is None) and (wl_parent is None):
                # Both parents are none -> both nodes are the root
                parent_id = None
            else:
                # We are not in the root node
                if q_parent is not None:
                    # q_node is not the root q_node
                    if node.domain[0] != q_node.domain[0]:
                        # The split is not on the q_node
                        q_parent = node.q_node.index
                    if node.domain[0] != wl_node.domain[0]:
                        # The split is not on the wl_node
                        wl_parent = node.wl_node.index
                else:
                    # q_node is the root q_node, so the split is on the wl_node
                    q_parent = node.q_node.index
                if wl_parent is not None:
                    # wl_node is not the root wl_node
                    if node.domain[0] != wl_node.domain[0]:
                        # The split is not on the wl_node
                        wl_parent = node.wl_node.index
                    if node.domain[0] != q_node.domain[0]:
                        # The split is not on the q_node
                        q_parent = node.q_node.index
                        
                parent_id = self.node_id_mapping[(q_parent, wl_parent)]
            
            node.parent_id = copy.deepcopy(parent_id)

        if return_results:
            return self.result_nodes

    def _add_result_node(self, node, parent_id=None):
        q = node[0]
        wl = node[1]
        q_node = self.nodes_edges_discharge[q]
        wl_node = self.nodes_edges_wl[wl]
        domain = sorted(list(set(q_node.domain) & set(wl_node.domain)))
        node_id = copy.deepcopy(self.counter)
        self.node_id_mapping[(q_node.index, wl_node.index)] = node_id

        result_dict = {
            'h_nzk': [self.model.h_nzk[t, q_node.index, wl_node.index].value for t in domain],
            'q_gate': [self.model.q_gate[t, q_node.index, wl_node.index].value for t in domain],
            'q_pump': [self.model.q_pump[t, q_node.index, wl_node.index].value for t in domain],
            'B_gate': [self.model.B_gate[t, q_node.index, wl_node.index].value for t in domain],
            'B_pump': [self.model.B_pump[t, q_node.index, wl_node.index].value for t in domain],
            'dH2': [self.model.dH2[t, q_node.index, wl_node.index].value for t in domain],
        }
        self.results_nodes.append(ResultNode(
            index=node_id,
            domain=q_node.domain,
            parent_id=parent_id,
            q_node=q_node,
            wl_node=wl_node,
            p_conditional=q_node.p_conditional * wl_node.p_conditional,
            p_marginal=q_node.p_marginal * wl_node.p_marginal,
            results=result_dict
        ))