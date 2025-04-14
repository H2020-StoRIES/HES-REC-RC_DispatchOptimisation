"""
This script can be used to run the daily schedule optimization standalone.

Usage:
    Define an optimization run (config, prices,...) in the user settings
    and run the script. Storage unit characteristic are defined in the config.
    Prices can either be manually defined or provided as csv by defining the
    path to the data file in the config.

Output:
    Shows result summary in the console and, if selected in the user settings,
    plots and/or stores detailed results.

Authors:
    Nils Müller
    nilmu@dtu.dk
    ---------
    Zahra Tajalli
    szata@dtu.dk
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler

import yaml
import logging
from time import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import Run_Daily_Schedule_Optimization, initialize_config, default_config
### User Settings ###
Config_file = "config_template.yaml"
Manual_prices = None #pd.DataFrame({"EnPrice[€/kWh]": [101, 102, 50, 103, 104], "RegPrice[€/kW]": [200, 0, 0, 0, 0]})
plot_results = True
store_results = True
plot_variables = ['pEt', 'pRt', 'pEt_A', 'pRt_A', 'et_A', 'pEt_B', 'pRt_B', 'et_B', 'pEt_C', 'pRt_C', 'et_C']
scale_plotted_data = False


### Set up the logger ###
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])


### Load config ###
logging.info(f"Reading config file {Config_file}...")
with open(os.path.join("./Config", Config_file), 'r') as file:
    user_config = yaml.safe_load(file)

# with open (os.path.join("./Config", "scenario_run_01_1.yaml"), 'r') as file:
#     scenario_config = yaml.safe_load(file)
### Overwrite default config with entries from user config ###
config = initialize_config(default_config, user_config)
# config = initialize_config(default_config, scenario_config)
with open(os.path.join("./Config", Config_file), 'r') as file:
    user_config = yaml.safe_load(file)


### Overwrite default config with entries from user config ###
config = initialize_config(default_config, user_config)


### Run Daily optimization problem ###
logging.info(f"Running daily schedule optimization with {'prices from config...' if Manual_prices is None else 'manually defined prices...'}")
Daily_results = Run_Daily_Schedule_Optimization(config=config, manual_prices=Manual_prices)


### Show result summary in console ###
logging.info(f"Result summary:")
logging.info(f"   Optimization status: {Daily_results[1]['Solver_status']}")
logging.info(f"   Total cost: {round(Daily_results[1]['Cost_upper_bound'], 2)} €")
logging.info(f"   Yearly cost: {round(Daily_results[1]['Cost_upper_bound']*365, 2)} €")


### Store detailed results ###
Prices = pd.read_csv(os.path.join("./InputData/Prices/", config["Sizing_Params"]["Price_file"])) if Manual_prices is None else Manual_prices
Daily_results_df = pd.concat([Prices, pd.DataFrame({key: Daily_results[1][key] for key in plot_variables} | {})], axis=1)

if store_results:
    results_filename = f"{int(time())}_DailyScheduleOpt_results.csv"
    logging.info(f"Saving results to {results_filename}...")
    Daily_results_df.to_csv(rf"./Results/DailyScheduleOptimization/{results_filename}", index=False)


### Plot some results ###
if plot_results:
    logging.info(f"Plotting results...")
    if scale_plotted_data:
        scaler = MinMaxScaler()
        # Daily_results_df = pd.DataFrame(scaler.fit_transform(Daily_results_df.to_numpy()), columns=Daily_results_df.columns)

    # Set up plot
    fig, (ax11, ax21, ax31, ax41, ax51) = plt.subplots(5, 1, sharex=True)
    ax11.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax11.grid()
    ax21.grid()
    ax31.grid()
    ax41.grid()
    ax51.grid()

    # Prices
    ax11.plot(Daily_results_df["EnPrice[€/kWh]"])
    ax11.plot(Daily_results_df["RegPrice[€/kW]"])
    ax11.legend(['Energy prices', 'Regulation prices'])
    ax11.set_title('Prices')

    # HESS
    ax21.plot(Daily_results_df[[col for col in ['pEt', 'pRt'] if col in plot_variables]])
    ax21.legend([col for col in ['pEt', 'pRt'] if col in plot_variables])
    ax21.set_title('HESS')

    # ESS A
    ax31.plot(Daily_results_df[[col for col in Daily_results_df.columns if "_A" in col]])
    ax31.legend([col for col in Daily_results_df.columns if "_A" in col])
    ax31.set_title('ESS A')

    # ESS B
    ax41.plot(Daily_results_df[[col for col in Daily_results_df.columns if "_B" in col]])
    ax41.legend([col for col in Daily_results_df.columns if "_B" in col])
    ax41.set_title('ESS B')

    # ESS C
    ax51.plot(Daily_results_df[[col for col in Daily_results_df.columns if "_C" in col]])
    ax51.legend([col for col in Daily_results_df.columns if "_C" in col])
    ax51.set_title('ESS C')

    plt.show()
