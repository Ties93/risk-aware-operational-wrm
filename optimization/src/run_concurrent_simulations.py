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

    optimization_datapath = datapath / 'optimization data'
    n_root_steps = cfg.n_root_steps
    obs_data = ObservationData(market_datapath, wl_datapath, wb_datapath, ark_datapath)
    
    for n_scenarios in [2, 3, 5, 10, 25, 50, 100]:
        
    # if (n_scenarios == False):
    #     n_q_scenarios = cfg.n_discharge_scenarios
    #     n_dam_scenarios = cfg.n_dam_scenarios
    #     n_idm_scenarios = cfg.n_idm_scenarios
    #     n_wl_scenarios = cfg.n_wl_scenarios
    # else:
        n_q_scenarios = n_scenarios
        n_dam_scenarios = n_scenarios
        n_idm_scenarios = n_scenarios
        n_wl_scenarios = n_scenarios
    
        if (n_q_scenarios <= 3):
            n_wb_scenarios = 3
        else:
            n_wb_scenarios = n_q_scenarios
    
        distance_metric = 'energy'#cfg.distance_metric
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

        for cvar_wl in [-0.395, -0.3, -0.2]:
            for cvar_alpha in [0.99, 0.95, 0.9, 0.8, 0.5]:
                optimization_settings = {
                    'wl_constraint_type': 'cvar',
                    'var_wl': cfg.var_wl,
                    'cvar_wl': cvar_wl,
                    'cvar_alpha': cvar_alpha,
                    'p_chance': 0.1,
                    'obj_type': 'expected_value',
                    'refit_idm_bn_every': 7*24, # every week
                    'start_wl': -0.45, # m+NAP
                    'distance': distance_metric,
                    'n_scenarios': n_scenarios,
                    'tree': cfg.tree_based
                }

                t_now = obs_data.dam.index[11] + pd.DateOffset(years=1, days=0) # Misschien in maart beginnen? Of iig deels? Die was vrij nat.
                simulation_index = pd.date_range(t_now, t_now + pd.DateOffset(days=30, hours=12), freq='H')
                exp_name = make_exp_name(optimization_settings)
                print('Running experiment: ', exp_name)
                if cfg.start_from_scratch:
                    simulation_data = None
                else:
                    try:
                        print('Loading simulation data.')
                        simulation_data = pd.read_pickle(optimization_datapath / exp_name / 'simulation_data.pkl')
                    except:
                        print('No simulation data found, starting from scratch.')
                        simulation_data = None
                    # simulation_data.dropna(how='all')
                
                if optimization_settings['tree']:
                    tree_classes = ['discharge', 'wl']
                else:
                    tree_classes = []
                
                try:
                    if simulation_data.dropna().index[-1] >= simulation_index[-1]:
                        done = True # means it is done running
                    else:
                        done = False
                except:
                    done = False
                
                if not done:
                    simulator = ClosedLoopSimulation(
                        savepath=optimization_datapath / exp_name,
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
                else:
                    print('Simulation already run.')

if __name__ == '__main__':
    main()