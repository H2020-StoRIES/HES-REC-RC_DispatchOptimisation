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
import sys
import yaml
import logging
import pandas as pd
from time import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import Run_Daily_Schedule_Optimization, initialize_config, default_config


def run_daily_schedule(
    config_filename="config_scenario_run_01_1.yaml",
    manual_prices=None,
    plot_results=False,
    store_results=False,
    plot_variables=None,
    scale_plotted_data=False
):
    if plot_variables is None:
        plot_variables = ['pBt_A', 'et_A', 'pBt_C', 'et_C', 'qBt_A', 'Qt_A', 'pGrid', 'qGrid', 'q2p', 'pBt_RC']

    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])

    logging.info(f"Reading config file {config_filename}...")
    with open(os.path.join(config_filename), 'r') as file:
        user_config = yaml.safe_load(file)

    config = initialize_config(default_config, user_config)

    logging.info(f"Running daily schedule optimization with {'prices from config...' if manual_prices is None else 'manually defined prices...'}")
    Daily_results = Run_Daily_Schedule_Optimization(config=config, manual_prices=manual_prices)

    logging.info(f"Result summary:")
    logging.info(f"   Optimization status: {Daily_results[1]['Solver_status']}")
    logging.info(f"   Total cost: {round(Daily_results[1]['Cost_upper_bound'], 2)} €")
    logging.info(f"   Yearly cost: {round(Daily_results[1]['Cost_upper_bound'] * 365, 2)} €")

    result_df = pd.DataFrame({key: Daily_results[1][key] for key in plot_variables})
    # print(result_df)

    if store_results:
        filename = f"{int(time())}_DailyScheduleOpt_results.csv"
        result_df.to_csv(os.path.join("Results", "DailyScheduleOptimization", filename), index=False)
        logging.info(f"Saved results to {filename}")

    return Daily_results, result_df


import sys

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "default.yaml"
    run_daily_schedule(config_filename=config_file)
        