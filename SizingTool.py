from utils import *
import yaml
from time import time
import logging

def Run_Sizing_Tool(config_file_path):

    ### Load user config file a###
    logging.info(f"Reading config file {config_file_path}...")
    with open(os.path.join("Config", config_file_path), 'r') as file:
        user_config = yaml.safe_load(file)

    ### Overwrite default config with entries from user config ###
    config = initialize_config(default_config, user_config)

    ### Load parameters and variables from config ###
    Max_iterations, LifetimeSplits, x_cur, Delta_x, cur_cap_info = Load_params_from_config(config)

    ### Start main optimization loop ###
    logging.info(f"Starting HESS size optimization...")
    init_time = time()

    for i in range(int(Max_iterations)):

        logging.info(f"[It {i}] Current config: {''.join([f'{unit} (P={P}kW, E={E}kWh), ' for (unit, (P, E)) in cur_cap_info.items()])[:-2]}")

        ### Approximate HESS net operational cost for current capacity configuration based on multiple lifetime periods ###
        logging.info(f"[It {i}] Approximating HESS net operational cost for current capacity configuration...")
        NetOpCostPerLifetime = []

        for LifetimePeriod in range(LifetimeSplits):  # Loop over the three lifetime periods

            # Define storage unit models in config for given lifetime
            config = Select_Storage_Models_for_Lifetime(config, LifetimePeriod/(LifetimeSplits-1))

            # Determine storage unit loss and degradation coefficients in config for current capacities & lifetime based on simulation
            config = Parameterization(config)

            # Optimize operation schedule of current HESS config and lifetime for 12 representative days
            schedules = Schedule_optimization(config)

            # Approximate cost of the 12 representative days based on simulation
            NetOpCostPerLifetime.append(Approximate_Yearly_Cost_from_Simulation(schedules, config))


        # Average the cost of the lifetime periods
        NetOpCost_cur = np.mean(NetOpCostPerLifetime)
        logging.info(f"[It {i}] Net Operational Cost: {np.round(NetOpCost_cur, 2)} €/year")


        ### Store results for current HESS configuration ###
        Store_data(x_cur, NetOpCost_cur, i, init_time, config)

        ### Approximate HESS net operational cost for all perturbations based on multiple lifetime periods ###
        logging.info(f"[It {i}] Approximating HESS net operational cost for perturbations of current capacity config...")
        x_cur_pert = x_cur.copy()

        NetOpCost_pert = np.full(len(x_cur), 1)

        for idx in range(len(x_cur)):

            x_cur_pert[idx] = x_cur[idx] + Delta_x

            config, _, _ = Update_capacities(x_cur_pert, config)

            # Approximate HESS lifetime cost for current perturbation based on multiple lifetime periods
            NetOpCostPerLifetime_pert = []
            for LifetimePeriod in range(LifetimeSplits):  # Loop over the three lifetime periods

                # Define storage unit models in config for given lifetime
                config = Select_Storage_Models_for_Lifetime(config, LifetimePeriod/(LifetimeSplits-1))

                # Update the storage unit's loss and degradation coefficients in the config for current capacities & lifetime
                config = Parameterization(config)

                # Optimize operation schedule of current HESS config and lifetime for 12 representative days
                schedules = Schedule_optimization(config)

                # Approximate cost of the 12 representative days based on simulation
                NetOpCostPerLifetime_pert.append(Approximate_Yearly_Cost_from_Simulation(schedules, config))

            # Average the cost of the lifetime periods
            NetOpCost_pert[idx] = np.mean(NetOpCostPerLifetime_pert)

            # Restore perturbed value
            x_cur_pert[idx] = x_cur[idx]

            logging.info(f"[It {i}] Perturbation {idx+1}/{len(x_cur)} done...")

        ### Calculate total gradient ###
        logging.info(f"[It {i}] Calculating total gradient...")
        grad_x = Calculate_total_gradient(NetOpCost_cur, NetOpCost_pert, config)
        logging.info(f"[It {i}] Total gradient: {np.round(grad_x, 2)}")

        ### Determine suggested capacities based on the gradient ###
        logging.info(f"[It {i}] Determining suggested capacity update based on the gradient...")
        x_suggested = Calculate_suggested_capacities_from_gradient(x_cur, grad_x, config)

        ### Check exit conditions for suggested capacities ###
        logging.info(f"[It {i}] Checking exit conditions for suggested capacities...")
        if Check_exit_conditions(x_suggested, grad_x, config, i):
            break

        ### Overwrite current capacities with the suggested ones in different variables for next iteration ###
        config, x_cur, cur_cap_info = Update_capacities(x_suggested, config)

if __name__ == '__main__':

    ### Config file selection ###
    Config_file_path = "SizingTool_config.yaml"

    # Set up the logger
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])

    # Run the sizing tool
    Run_Sizing_Tool(Config_file_path)
