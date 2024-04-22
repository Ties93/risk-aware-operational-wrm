from utils.simulation import *
from pathlib import Path
import hydra
from omegaconf import DictConfig
import os

@hydra.main(version_base=None, config_path='../conf/', config_name='config')
def main(cfg: DictConfig) -> None:
    os.environ['NUMEXPR_MAX_THREADS'] = str(cfg.max_threads) # maak variabel in cfg?

    datapath = Path(cfg.DATAPATH)
    # datapath = Path(Path().resolve()) / 'data'
    scenario_path = datapath / 'forecast data' / 'Scenarios' 
    market_datapath = datapath / 'market data'
    wl_datapath = datapath / 'waterlevel data'
    wb_datapath = datapath / 'waterboard data'
    ark_datapath = datapath / 'ark data'

    optimization_datapath = datapath / 'timeseries simulation'
    n_root_steps = cfg.n_root_steps
    n_scenarios = cfg.n_scenarios

    n_q_scenarios = n_scenarios
    n_dam_scenarios = n_scenarios
    n_idm_scenarios = n_scenarios
    n_wl_scenarios = n_scenarios
    
    if (n_q_scenarios <= 3):
        n_wb_scenarios = 3
    else:
        n_wb_scenarios = n_q_scenarios

    distance_metric = cfg.distance_metric
    scenario_params = {
        'DAM': {
            'n': n_dam_scenarios,
            'method': distance_metric,
        },
        'IDM':{
            'n_init': 100,
            'n': n_idm_scenarios,
            'method': distance_metric,
        },
        'wl': {
            'n': n_wl_scenarios,
            'method': distance_metric,
        },
        'discharge': {
            'n': n_q_scenarios,
            'n_wb': n_wb_scenarios,
            'method': distance_metric
        },
        'n_root_steps': n_root_steps
    }
    obs_data = ObservationData(market_datapath, wl_datapath, wb_datapath, ark_datapath)
    idm_scenarios = IDMScenarios(
        obs_dataclass=obs_data,
        n_init=scenario_params['IDM']['n_init'],
        n=scenario_params['IDM']['n'],
        method=scenario_params['IDM']['method'],
        cluster=True,
    )
    t_min = idm_scenarios.dam.index.min()
    idm_scenarios.update_bn(t_min=t_min, t_max=t_min + pd.DateOffset(days=365))

    scenario_data = ScenarioData(
        scenario_params=scenario_params,
        scenario_path=scenario_path,
        constrain_tree_complexity=True,
        complexity_reduction=0.5,
    )

    optimization_settings = {
        'wl_constraint_type': cfg.wl_constraint_type,
        'var_wl': cfg.var_wl,
        'cvar_wl': cfg.cvar_wl,
        'cvar_alpha': cfg.cvar_alpha,
        'p_chance': cfg.p_chance,
        'h_max': cfg.h_max,
        'obj_type': 'expected_value',
        'refit_idm_bn_every': 7*24, # every week
        'start_wl': -0.45, # m+NAP
        'distance': distance_metric,
        'n_scenarios': n_scenarios,
        'tree': cfg.tree_based
    }

    # month = cfg.month
    year = cfg.year
    if cfg.h_max == -0.4:
        wl_type = 'low wl'
    elif cfg.h_max == -0.3:
        wl_type = 'high wl'
    else:
        raise ValueError('h_max should be -0.4 or -0.3')

    for month in [1, 4, 7, 10]:
        t_now = obs_data.dam.loc[(obs_data.dam.index.year==year) & (obs_data.dam.index.month==month)].index.min() + pd.DateOffset(hours=11) # Start at 11am
        # t_now = obs_data.dam.index[11] + pd.DateOffset(years=1, days=0) # Misschien in maart beginnen? Of iig deels? Die was vrij nat.
        simulation_index = pd.date_range(t_now, t_now + pd.DateOffset(months=1, hours=-25), freq='H') # End at 23pm at the last day of the month
        exp_name = make_exp_name(optimization_settings)
        optimization_datapath_exp = optimization_datapath / wl_type / f'{year}-{month}'
        print('Running experiment: ', exp_name)
        if cfg.start_from_scratch:
            simulation_data = None
        else:
            try:
                print('Loading simulation data.')
                simulation_data = pd.read_pickle(optimization_datapath_exp / exp_name / 'simulation_data.pkl')
            except:
                print('No simulation data found, starting from scratch.')
                simulation_data = None
            # simulation_data.dropna(how='all')
        
        if optimization_settings['tree']:
            tree_classes = ['discharge', 'wl']
        else:
            tree_classes = []
        
        try:
            if simulation_data.Q_gate.dropna().index[-1] >= simulation_index[-1]:
                done = True # means it is done running
            else:
                done = False
        except:
            done = False
        
        if not done:
            try:
                simulator = ClosedLoopSimulation(
                    savepath=optimization_datapath_exp / exp_name,
                    optimization_settings=optimization_settings,
                    simulation_index=simulation_index,
                    observation_dataclass=obs_data,
                    scenario_dataclass=scenario_data,
                    idm_scenarioclass=idm_scenarios,
                    save_individual_timesteps=True,
                    simulation_data=simulation_data,
                    tree_classes=tree_classes
                )
                simulator.run_simulation(save_every=1)
            except KeyError:
                t_now = simulator.simulation_data.Q_gate.dropna().index[-1]
                print('Simulation stopped at date: ', t_now)
        else:
            print('Simulation already run.')

if __name__ == '__main__':
    main()