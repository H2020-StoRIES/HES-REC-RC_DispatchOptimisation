"""
This script can be used to run the parameterization standalone.

Usage:
    Select the config file in the user settings and run the script.
    In the config file, the storage unit characteristics, number of
    representative frequency time series days, and granularity of the
    parameterization can be selected.

Output:
    Plots of the simulation results and fitted function for degradation(pRt),
    delivery fraction(pRt) and losses(pRt) and stores them as json if selected
    in the settings.

Author:
    Nils Müller
    nilmu@dtu.dk
"""

import os
from matplotlib import pyplot as plt
import yaml
import json
from utils import Parameterization, numpy_array_encoder, initialize_config, default_config
from time import time
import logging
import sys


### User Settings ###
config_path = "StandaloneParameterization_config.yaml"
plot_results = True
store_results = True


### Set up the logger ###
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])


### Load storage config ###
logging.info(f"Reading config file {config_path}...")
with open(os.path.join("../Config", config_path), 'r') as file:
    user_config = yaml.safe_load(file)


### Overwrite default config with entries from user config ###
config = initialize_config(default_config, user_config)


### Run the parameterization ###
logging.info(f"Running parameterization...")
pRt_range, Loss_coef, Deg_cost_coef, Delivery_frac_coef, Deg_cost, Losses, Delivery_fraction = Parameterization(config, standalone=True)


if store_results:
    results_filename = f"{int(time())}_Parameterization_results.json"
    logging.info(f"Saving results to {results_filename}...")
    Results_dict = {"pRt_range": pRt_range,
                    "Loss_coef": Loss_coef,
                    "Deg_cost_coef": Deg_cost_coef,
                    "Delivery_frac_coef": Delivery_frac_coef,
                    "Deg_cost": Deg_cost,
                    "Losses": Losses,
                    "Delivery_fraction": Delivery_fraction}

    Results_dict = json.dumps(Results_dict, default=numpy_array_encoder, indent=4)
    with open(rf"../Results/Parameterization/{results_filename}", 'w') as file:
        file.write(Results_dict)

if plot_results:
    logging.info(f"Plotting results...")

    ### Plot degradation cost simulation results vs fitted function ###
    plt.plot(pRt_range, Deg_cost, marker='o', linestyle='None', label="Sim results")
    plt.plot(pRt_range, [Deg_cost_coef[0]*value**2 + Deg_cost_coef[1]*value + Deg_cost_coef[2] for value in pRt_range], label="Fitted curve")
    plt.xlabel("pRt")
    plt.ylabel("Hourly degradation cost (€/hour)")
    plt.legend()
    plt.show()


    ### Plot delivery fraction simulation results vs fitted function ###
    plt.plot(pRt_range, Delivery_fraction, marker='o', linestyle='None', label="Sim results")
    plt.plot(pRt_range, [Delivery_frac_coef[0]*value**2 + Delivery_frac_coef[1]*value + Delivery_frac_coef[2] for value in pRt_range], label="Fitted curve")
    plt.xlabel("pRt")
    plt.ylim(0, 1.1)
    plt.ylabel("Service delivery fraction (-)")
    plt.legend()
    plt.show()


    ### Plot losses simulation results vs fitted function ###
    available_units = [list(unit.keys())[0] for unit in config.get("Storage_Units")]
    for unit in available_units:
        plt.plot(pRt_range, Losses[unit], marker='o', linestyle='None', label="Sim results")
        plt.plot(pRt_range, [Loss_coef[unit][0]*value + Loss_coef[unit][1] for value in pRt_range], label="Fitted curve")
        plt.xlabel("pRt")
        plt.ylabel("Hourly losses (kWh/hour)")
        plt.legend()
        plt.show()
