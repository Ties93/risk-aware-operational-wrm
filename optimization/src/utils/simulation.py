from utils.scenario_gen import *
from utils.trees import *
from utils.clustering import *
from utils.optimization import *
from pyomo.opt import SolverStatus, TerminationCondition
import pickle

class ClosedLoopSimulation():
    """
    Class for closed-loop simulation of the DR optimization problem.
    """
    def __init__(self, savepath, simulation_index, optimization_settings, observation_dataclass, scenario_dataclass, idm_scenarioclass, tree_classes=['discharge', 'wl'], simulation_data=None, save_individual_timesteps=False):
        self.obs = observation_dataclass
        self.scenario = scenario_dataclass
        self.idm_scenario = idm_scenarioclass
        self.control_horizon = 48 # hour
        self.simulation_index = simulation_index
        self.idx = pd.IndexSlice
        self.optimization_settings = optimization_settings
        self.savepath = savepath
        self.save_individual_timesteps = save_individual_timesteps
        self.refit_idm_bn_every = self.optimization_settings['refit_idm_bn_every']

        self.savepath.mkdir(exist_ok=True, parents=True)
        if self.save_individual_timesteps:
            self.savepath_timesteps = self.savepath / 'timesteps'
            self.savepath_timesteps.mkdir(exist_ok=True, parents=True)

        self.tree_shaped = tree_classes

        # Initialize simulation data
        # This will be the dataframe that contains the performed actions.
        if simulation_data is None: # Start from scratch
            self.simulation_data = pd.DataFrame(
                index=self.simulation_index,
                columns=['h_nzk', 'Q_gate', 'Q_pump', 'Q_wb', 'Q_ark', 'E_act', 'E_dam', 'h_ns', 'p_dam', 'p_idm']
            )
            
            # Start with a half full reservoir
            self.simulation_data.loc[self.simulation_data.index[0]-pd.DateOffset(hours=1), 'h_nzk'] = self.optimization_settings['start_wl']
            self.simulation_data.loc[self.simulation_data.index[0]-pd.DateOffset(hours=1), [col for col in self.simulation_data.columns if col != 'h_nzk']] = 0
            self.simulation_data = self.simulation_data.sort_index()

            # Start with initial DAM bid of 0
            startdate = self.simulation_index.min()
            idm_index = pd.date_range(startdate, startdate + pd.DateOffset(hours=23-startdate.hour), freq='H')
            self.simulation_data.loc[idm_index, 'E_dam'] = 0

        else:
            self.simulation_data = simulation_data


    def _get_idm_scenarios(self, t_now): # how does this work with daylight savings?
        # Get the current DAM prices (observed) untill the next DAM price
        
        # Start at this 23:00 the last day, because we need the DAM price of the hour before this day as well
        dam_obs_index = pd.date_range(t_now - pd.DateOffset(hours=t_now.hour + 1), freq='H', periods=25) # 25 because we need the DAM price of the hour before this day as well
        dam_obs = self.obs.get_observation_data_single('DAM', dam_obs_index[0], dam_obs_index[-1])
        # print(dam_obs_index)

        # Get the IDM scenarios untill now (observed)
        idm_obs_index = pd.date_range(t_now - pd.DateOffset(hours=t_now.hour + 1), t_now - pd.DateOffset(hours=1), freq='H')
        idm_obs = self.obs.get_observation_data_single('IDM', idm_obs_index[0], idm_obs_index[-1])
        # print(idm_obs_index)
        
        # print(dam_obs)
        # print(idm_obs)
        if self.idm_scenario.method != 'obs':
            # Get the IDM scenarios untill the DAM (forecast / generated)
            return self.idm_scenario(dam_obs.values.flatten(), idm_obs.values.flatten()) #,10)
        else:
            # If observations -> return the observation data
            idm_scen_index = pd.date_range(idm_obs_index[-1]+pd.DateOffset(hours=1), freq='H', periods=len(dam_obs_index)-len(idm_obs_index))
            return self.obs.get_observation_data_single('IDM', idm_scen_index[0], idm_scen_index[-1]).values.flatten()


    def prep_opt_data(self, t_now, inplace=True, return_data=False):
        """
        Prepare the optimization data for the optimization problem.
        This will return the data that is used to calculate the optimal actions.
        So we return:
        - the observation at t_now
        - the IDM scenarios untill the DAM
        - the DAM scenarios
        """

        # Get the observations of the external variables at t_now-1h (so one timestep )
        t_prev = t_now - pd.DateOffset(hours=1)
        dam_o, idm_o, wl_hourly_o, discharge_o, ark_o = self.obs.get_observation_data(start=t_prev, end=t_now)

        # Get the observations of the discharge at ark, this is a perfect forecast.
        ark_meas = self.obs.get_observation_data_single(var='ARK', start=t_now, end=t_now+pd.DateOffset(hours=self.control_horizon))
        
        # Get the observations of h_nzk at t_now-1h (so one timestep )
        # We need to get the h_nzk at t_now-1h, because we need to know the state of the system to decide control actions
        # So we decide the control action from t_now:t_now+control_horizon
        h_nzk_o = self.simulation_data.loc[t_prev, 'h_nzk']

        # Get the IDM scenarios untill the DAM
        idm_scenarios = self._get_idm_scenarios(t_now).astype(float)
        
        if len(idm_scenarios.shape) == 1:
            idm_scenarios_df = pd.DataFrame(index=[0], columns=range(1, idm_scenarios.shape[0]+2), dtype=float)
            idm_scenarios_df.iloc[:, :-1] = idm_scenarios
            idm_scenarios_df.iloc[:, -1] = 1
            idm_scenarios_df.columns = [i for i in range(1, idm_scenarios.shape[0]+1)] + ['weights']
            idm_scenarios = idm_scenarios_df
            idm_scenarios.fillna(1000, inplace=True)

        # Get the DAM scenarios for tomorrow -> DA bid and prep
        next_dam_timestep = t_now + pd.DateOffset(hours=24-t_now.hour)
        if next_dam_timestep.hour != 0:
            next_dam_timestep = next_dam_timestep + pd.DateOffset(hours=1) # daylight savings

        dam_scenarios = self.scenario.get_scenarios(next_dam_timestep, 'DAM').copy()
        # Select the DAM scenarios untill t_now+48h
        dam_weights = dam_scenarios.iloc[:, [-1]].copy()
        dam_scenarios = pd.concat([dam_scenarios.iloc[:, [i for i in range(self.control_horizon - (idm_scenarios.shape[1] - 1))]], dam_weights], axis=1).astype(float)

        # Get the WL scenarios for the next 48 hours
        wl_scenarios = self.scenario.get_scenarios(t_now, 'wl').copy()
        wl_scenarios.iloc[:, :-1] = wl_scenarios.iloc[:, :-1] / 100 # cm+NAP to m+NAP

        # Get the discharge scenarios for the next 48 hours
        discharge_scenarios = self.scenario.get_scenarios(t_now, 'discharge').copy()

        optimization_data = {}
        optimization_data['dam'] = {}
        optimization_data['idm'] = {}
        optimization_data['wl'] = {}
        optimization_data['wb_discharge'] = {}
        optimization_data['h_nzk'] = {}
        optimization_data['ark_discharge'] = {}
        optimization_data['probabilities'] = {}
        optimization_data['E_dam'] = {}
        
        optimization_data['h_nzk'][-1] = h_nzk_o # Both are the last calculated, so t-1h!
        optimization_data['h_nzk'][0] = h_nzk_o

        optimization_data['dam'][-1] = dam_o.values[0].round(2)
        optimization_data['dam'][0] = dam_o.values[1].round(2)
        optimization_data['idm'][-1] = idm_o.values[0].round(2)
        optimization_data['idm'][0] = idm_o.values[1].round(2)
        optimization_data['wl'][-1] = wl_hourly_o.values[0].round(3)[0]
        optimization_data['wl'][0] = wl_hourly_o.values[1].round(3)[0]
        
        optimization_data['wb_discharge'][-1] = discharge_o.values[0].round(2)
        optimization_data['wb_discharge'][0] = discharge_o.values[1].round(2)
        optimization_data['ark_discharge'][-1] = ark_o.values[0].round(2)
        optimization_data['ark_discharge'][0] = ark_o.values[1].round(2)

        n_id_timesteps = len(idm_scenarios.columns) - 1 # -1 since weights are in DF
        # Intraday trading
        for t_id in range(1,  n_id_timesteps+1): 
            optimization_data['dam'][t_id] = np.nan
            optimization_data['idm'][t_id] = idm_scenarios.iloc[:, t_id-1].values.round(2)
            optimization_data['ark_discharge'][t_id] = ark_meas.iloc[t_id-1].values.round(2)
            if 'discharge' not in self.tree_shaped:
                optimization_data['wb_discharge'][t_id] = discharge_scenarios.iloc[:, t_id-1].values.round(2)
            if 'wl' not in self.tree_shaped:
                optimization_data['wl'][t_id] = wl_scenarios.iloc[:, t_id-1].values.round(3)

            # Get the DAM bid made for the ID trading period
            optimization_data['E_dam'][t_id] = self.simulation_data.loc[t_now + pd.DateOffset(hours=t_id-1), 'E_dam']

        # DA trading
        for t in range(n_id_timesteps+1, self.control_horizon+1):
            optimization_data['dam'][t] = dam_scenarios.iloc[:, t-n_id_timesteps-1].values.round(2)
            optimization_data['idm'][t] = np.nan # NO multistage IDM yet
            optimization_data['ark_discharge'][t] = ark_meas.iloc[t-1].values.round(2)
            if 'discharge' not in self.tree_shaped:
                optimization_data['wb_discharge'][t] = discharge_scenarios.iloc[:, t-1].values.round(2)
            if 'wl' not in self.tree_shaped:
                optimization_data['wl'][t] = wl_scenarios.iloc[:, t-1].values.round(2)
            optimization_data['E_dam'][t] = np.nan
        
        optimization_data['probabilities']['dam'] = dam_scenarios.iloc[:, -1].values.round(3)
        optimization_data['probabilities']['idm'] = idm_scenarios.iloc[:, -1].values.round(3)

        if 'discharge' in self.tree_shaped:
            optimization_data['wb_discharge'][1] = self.scenario.generate_tree(t_now, 'discharge', scenarios=discharge_scenarios)
        else:
            optimization_data['probabilities']['wb_discharge'] = discharge_scenarios.iloc[:, -1].values.round(3)
        
        if 'wl' in self.tree_shaped:
            optimization_data['wl'][1] = self.scenario.generate_tree(t_now, 'wl', scenarios=wl_scenarios)
        else:
            optimization_data['probabilities']['wl'] = wl_scenarios.iloc[:, -1].values.round(3)

        if inplace:
            self.optimization_data = optimization_data
        if return_data:
            return optimization_data
    
    def optimize(self, t_now):
        """
        Optimize the control actions for the next 48 hours.
        """
        solved=False
        i=0
        # var = self.optimization_settings['var_wl']
        opt_settings = copy.deepcopy(self.optimization_settings)
        while not solved:
            self.prep_opt_data(t_now, inplace=True, return_data=False)
            logfile = str((self.savepath_timesteps / f'{t_now.strftime("%Y-%m-%d %H")}.log').resolve())
            problem = NZKProblem(
                optimization_data=self.optimization_data,
                optimization_settings=opt_settings,
                pred_hor=self.control_horizon,
                tree_classes=self.tree_shaped,
                logfile=logfile
            )
            problem.make_model()
            model = problem.solve(verbose=False, mode='normal')
            
            if problem.opt_results.solver.status == SolverStatus.ok:
                solved=True
            elif problem.opt_results.solver.termination_condition != TerminationCondition.infeasible:
                solved=True
            else:
                i+=1
                print(f'Infeasible optimization problem nr {i}, trying again with different scenarios')
            
            if i > 5:
                if i % 4 == 0:
                    opt_settings['cvar_alpha'] -= 0.01
                else:
                    opt_settings['cvar_wl'] += 0.01 # Relax the value at risk constraint
                    
            if opt_settings['cvar_wl'] > self.optimization_settings['cvar_wl'] + 0.2:
                raise ValueError('Infeasible optimization problem, tried 10 times with different scenarios')

        # try:
        results = problem.get_results(model)
        # except:
        #     # No solution was found within the time limit, so we run untill one feasible solution was found
        #     model = problem.solve(verbose=False, mode='extra_time')
        #     results = problem.get_results(model)

        if self.save_individual_timesteps:
            with open(self.savepath_timesteps / f'{t_now.strftime("%Y-%m-%d %H")}.pkl', 'wb') as f:
                pickle.dump(results, f)

        return results
    
    def _correct_q_gate(self, q, dh):
        N=7 # kokers
        B=5.9 # m
        a=1.0 # coefficient
        H=4.8 #m keelhoogte
        g=9.81 # m/s^2
        q_max = N*a*B*H* np.sqrt(2*g*dh)
        return np.round(min(q, q_max), decimals=2)
    
    def _correct_q_pump(self, q, dh):
        a = -3.976
        b = -17.7244
        c = 269.58
        q_max = a*dh**2 + b*dh + c
        return np.round(min(q, q_max), decimals=2)

    def pump_energy(self, q, dh, dt=3600):
        return (0.033*q**2 + 0.061*dh**2*q + 11.306*dh*q) * dt / 3600 / 1000 # MWh
    
    def calculate_expected_energy(self, timestep_index, results, dt=3600):
        """
        Calculate the expected energy use over all scenarios.
        Timestep index is the index of optimization timestep (so [1,48])
        """
        E_exp = 0
        for node in results:
            if timestep_index in node.domain:
                node_timestep_index = node.domain.index(timestep_index)
                q_pump = node.results['q_pump'][node_timestep_index]
                h_nzk = node.results['h_nzk'][node_timestep_index]

                wl_node_timestep_index = node.wl_node.domain.index(timestep_index)
                wl_ns = node.wl_node.values[wl_node_timestep_index]
                
                dh = max(wl_ns - h_nzk, 0)
                q_pump = self._correct_q_pump(q_pump, dh)
                E_exp += node.p_marginal * self.pump_energy(q_pump, dh, dt)
        return E_exp

    def simulate_timestep(self, t_now, results):
        """
        Simulate the next timestep.
        """
        # Get the observations of the external variables at t_now-1h (so one timestep)
        wl_ns = np.round(self.optimization_data['wl'][0], decimals=3) # accuracy in mm
        h_nzk0 = self.optimization_data['h_nzk'][0] # accuracy in mm
        q_wb = np.round(self.optimization_data['wb_discharge'][0], decimals=2)[0]
        q_ark = np.round(self.optimization_data['ark_discharge'][0], decimals=2)[0]
        dh_gate = max(h_nzk0 - wl_ns, 0)
        dh_pump = max(wl_ns - h_nzk0, 0)

        # Get the actions (Q_gate, Q_pump) from the optimization results, correct them for the physical limitations.
        q_gate = self._correct_q_gate(results[0].results['q_gate'][0], dh_gate)
        q_pump = self._correct_q_pump(results[0].results['q_pump'][0], dh_pump)
        
        # Calculate the next water level
        A_nzk = 36*10e6 # m^2
        dt = 3600 # s
        q_out = q_gate + q_pump
        q_in = q_wb + q_ark
        h_nzk1 = np.round(h_nzk0 + (q_in - q_out) * dt / A_nzk, decimals=4)
        E_act = self.pump_energy(q_pump, dh_pump, dt) # MWh

        self.simulation_data.loc[t_now, 'h_nzk'] = h_nzk1
        self.simulation_data.loc[t_now, 'Q_gate'] = q_gate
        self.simulation_data.loc[t_now, 'Q_pump'] = q_pump
        self.simulation_data.loc[t_now, 'Q_wb'] = q_wb
        self.simulation_data.loc[t_now, 'Q_ark'] = q_ark
        self.simulation_data.loc[t_now, 'E_act'] = E_act
        self.simulation_data.loc[t_now, 'h_ns'] = wl_ns
        self.simulation_data.loc[t_now, 'p_dam'] = self.optimization_data['dam'][0].flatten()[0]
        self.simulation_data.loc[t_now, 'p_idm'] = self.optimization_data['idm'][0].flatten()[0]

        if t_now.hour == 11:
            # Make a DAM bid at 11:00 AM for the next day (00:00 - 24:00)
            # We bid the expected energy use over all scenarios
            dam_start = 24 - t_now.hour 
            for t in range(dam_start, dam_start + 24):
                self.simulation_data.loc[t_now + pd.DateOffset(hours=t), 'E_dam'] = self.calculate_expected_energy(t+1, results, dt) # +1 because time starts at 1 in the optimization problem
    
    def check_refit_bn(self, t_now):
        if self.optimization_settings['distance'] != 'obs':
            if t_now - self.idm_scenario.t_max > pd.Timedelta(value=self.refit_idm_bn_every, unit='hours'):
                print('Refitting IDM BN')
                self.idm_scenario.update_bn(t_min=t_now-pd.DateOffset(days=365), t_max=t_now)

    def fit_bn_until(self, t_now):
        i=0
        while self.simulation_data.index[1] + pd.DateOffset(hours=i*self.refit_idm_bn_every) < t_now:
            i += 1
        t_refit = self.simulation_data.index[1] + pd.DateOffset(hours=(i-1)*self.refit_idm_bn_every)
        self.check_refit_bn(t_refit)


    def run_simulation(self, save_every=1):
        t0 = self.simulation_data.loc[self.simulation_data.Q_pump.isna()].index.min()
        print('Running simulation from', t0)
        self.fit_bn_until(t0)
        
        for i, date in enumerate(tqdm(self.simulation_data.loc[t0:].index)):
            if i % save_every == 0:
                # self.simulation_data.to_csv(self.savepath / 'simulation_data.csv')
                self.simulation_data.to_pickle(self.savepath / 'simulation_data.pkl')
            results = self.optimize(date)
            self.simulate_timestep(date, results)
            self.check_refit_bn(date)
        self.simulation_data.to_pickle(self.savepath / 'simulation_data.pkl')

def make_exp_name(optimization_settings):
    if optimization_settings["distance"] == 'obs':
        exp_name = 'obs'
    else:
        exp_name = optimization_settings['wl_constraint_type']
    
        if exp_name == 'cvar':
            exp_name += f'_{optimization_settings["cvar_alpha"]}'
            exp_name += f'_{abs(optimization_settings["cvar_wl"])}'
        elif exp_name == 'chance':
            exp_name += f'_{optimization_settings["p_chance"]}'
        elif exp_name == 'robust':
            pass
        else:
            raise NotImplementedError('Only implemented for cvar, chance and robust')
        exp_name += f'_{optimization_settings["distance"]}'
        exp_name += f'_{optimization_settings["n_scenarios"]}'
        if optimization_settings['tree']:
            exp_name += '_tree'
        else:
            exp_name += '_fan'
    
    return exp_name