import numpy as np
from scipy.optimize import curve_fit
import requests
import subprocess
from copy import deepcopy
import pandas as pd
import sys
import os
from multiprocessing import Pool, cpu_count
import logging
import csv
import time
from pyomo.environ import *
import pyomo.environ as pyo
import yaml

pd.set_option('display.max_columns', None)

### Parameters ###
aliases = {"Vanadium_Battery": "vrb", "Lithium_Battery": "li", "Super_Capacitor": "sc"} #TODO: Why we need them?

default_config = {
    'Name': 'Minimal configuration for HESS optimization',
    'Comments': 'Includes only required parameters from optimization formulation.',
    'Timeseries': {
        'Type': 'Frequency',
        'Input_File': None,
        'Scaling': 1,
        'Power_reserve_schedule': None,
        'Power_reserve_period': 60
    },
    'Execution': {
        'Real_Time': False,
        'End_Time': 100000,
        'Minimum_Resolution': 1
    },
    'Sizing_Params': {
    'Num_price_days': 12,
    'Price_day_weighting': [], 
    'UseParallel': True,
    'Max_iterations': 1000,
    'Stepsize': 0.01,
    'Tolerance': 0.001,
    'Delta_x': 0.01
},
    # 'HESS_Params': {
    #     'HESS_score': 1,
    #     'Kappa': 0.25,
    #     'Zeta': 1,
    #     'BudgetLimit': 10000,
    #     'FixedCost': 0,
    #     'Int_rate': 0.05,
    #     'Lifetime': 10
    # },
    'Electrcial_Storage_Units': [{
        'Lithium_Battery': {
            'Available_Capacity': 1,
            'Available_Power': 1,
            'Min_Limit_Energy_Capacity': 0.0001,
            'Max_Limit_Energy_Capacity': 40,
            'Min_Limit_Power': 0,
            'Max_Limit_Power': 15,
            'Eta_ch': 0.9,
            'Eta_dis': 0.9,
            'Static': 0,
            'Gamma': 1,
            'Initial_SOC': 50,
            'Maximum_SOC': 90,
            'Minimum_SOC': 10,
            'Loss_coef_a': 0.005,
            'Loss_coef_b': 0.005,
            'Deg_cost_coef_a': 0.005,
            'Deg_cost_coef_b': 0.005,
            'Deg_cost_coef_c': 0.005,
            'Deg_cost_per_kWh': 1000
        },
    
    'Super_Capacitor': {
        'Available_Capacity': 0.5,
        'Available_Power': 1,
        'Min_Limit_Energy_Capacity': 0.0001,
        'Max_Limit_Energy_Capacity': 0.5,
        'Min_Limit_Power': 0,
        'Max_Limit_Power': 15,
        'Eta_ch': 0.92,
        'Eta_dis': 0.92,
        'Static': 0,
        'Gamma': 1,
        'Initial_SOC': 50,
        'Maximum_SOC': 90,
        'Minimum_SOC': 10,
        'Loss_coef_a': 0.005,
        'Loss_coef_b': 0.005,
        'Deg_cost_coef_a': 0.005,
        'Deg_cost_coef_b': 0.005,
        'Deg_cost_coef_c': 0.005,
        'Deg_cost_per_kWh': 1000
    },
    'Pumped_Hydro': {
        'Available_Capacity': 100,
        'Available_Power': 10,
        'Min_Limit_Energy_Capacity': 0.0001,
        'Max_Limit_Energy_Capacity': 100,
        'Min_Limit_Power': 0,
        'Max_Limit_Power': 10,
        'Eta_ch': 0.85,
        'Eta_dis': 0.85,
        'Static': 0,
        'Gamma': 1,
        'Initial_SOC': 50,
        'Maximum_SOC': 90,
        'Minimum_SOC': 10,
        'Loss_coef_a': 0.002,
        'Loss_coef_b': 0.002,
        'Deg_cost_coef_a': 0.002,
        'Deg_cost_coef_b': 0.002,
        'Deg_cost_coef_c': 0.002,
        'Deg_cost_per_kWh': 100
    }
    }],
    'Thermal_Storage_Units': [{
        'PCM': {
            'Available_Capacity': 0.5,
            'Available_Power': 1,
            'Min_Limit_Energy_Capacity': 0.0001,
            'Max_Limit_Energy_Capacity': 0.5,
            'Min_Limit_Power': 0,
            'Max_Limit_Power': 15,
            'Eta_ch': 0.92,
            'Eta_dis': 0.92,
            'Static': 0,
            'Initial_SOC': 50,
            'Maximum_SOC': 90,
            'Minimum_SOC': 10
        }
    }],
    'Thermal_to_Electrical_Converters': [{
        'Ranking_Cycle': {
            'Available_Power': 1,
            'Min_Limit_Power': 0,
            'Max_Limit_Power': 15,
            'Eta_RC': 0.38
        }
    }]
}
                      


### Functions ###
def initialize_config(default_config, user_config):

    # Create a copy of the default config
    updated_config = deepcopy(default_config)

    # Overwrite values in the updated config which are defined by the user
    for config_key, config_value in user_config.items():
        if isinstance(config_value, dict):
            for dict_key, dict_value in config_value.items():
                updated_config[config_key][dict_key] = dict_value
        elif config_key == "Electrcial_Storage_Units":
            for storage_dict in config_value:
                unit_name = list(storage_dict.keys())[0]
                unit_idx = None
                for i, storage_dict_in_updated_config in enumerate(updated_config["Electrcial_Storage_Units"]):
                    if unit_name in storage_dict_in_updated_config.keys():
                        unit_idx = i
                for k2, v2 in storage_dict.items():
                    for k3, v3 in v2.items():
                        updated_config["Electrcial_Storage_Units"][unit_idx][unit_name][k3] = v3
        else:
            updated_config[config_key] = config_value

    # Remove all storage units which are not defined by the user
    user_defined_units = [list(storage_dict.keys())[0] for storage_dict in user_config["Electrcial_Storage_Units"]]
    updated_config["Electrcial_Storage_Units"] = [storage_dict_in_updated_config for storage_dict_in_updated_config in updated_config["Electrcial_Storage_Units"] if list(storage_dict_in_updated_config.keys())[0] in user_defined_units]

    return updated_config

def Hourly_li_degradation_cost(res, config):
    """
    Calculates the cost for lithium battery degradation from the simulation results.

    Args:
        res (dataframe): Simulation results of one day at constant power reserve
        config (dict): Current HESS configuration used to initialize and run the simulation

    Return:
        Hourly cost for degradation

    Assumptions:
        Simulation run of one day (24 hours)
    """

    # Get index of Lithium_Battery
    for i, storage_dict in enumerate(config["Electrcial_Storage_Units"]):
        if "Lithium_Battery" in storage_dict.keys():
            li_idx = i

    cost_per_kWh = config["Electrcial_Storage_Units"][li_idx]["Lithium_Battery"]["Deg_cost_per_kWh"]  # €/kWh
    e_cap_before = res["e_li_max"].iloc[0]  # kWh
    e_cap_after = res["e_li_max"].iloc[-1]  # kWh
    lost_e_cap = e_cap_before - e_cap_after  # kWh

    return cost_per_kWh * lost_e_cap / 24  # (€/kWh)*(kWh)/h


def Service_delivery_fraction(res):
    """
    Calculates the service delivery fraction from the simulation results.

    Args:
        res (dataframe): Simulation results of one day at constant power reserve

    Return:
        Service delivery fraction

    Assumptions:
        Simulation run of one day (24 hours)
    """

    res_w_o_SOC_res = res.iloc[:86400].copy()  # Remove SOC restoration period from sim results

    return len(res_w_o_SOC_res[res_w_o_SOC_res["p_ref"].round(1) == res_w_o_SOC_res["p_set_hess"].round(1)]) / len(res_w_o_SOC_res)


def Hourly_losses(res, unit_alias):
    """
    Calculates the losses for a particular storage unit from the simulation results.

    Args:
        res (dataframe): Simulation results of one day at constant power reserve
        unit_alias (string): Alias of storage unit

    Return:
        Losses

    Assumptions:
        - Simulation run of one day (24 hours)
        - SOC starts and ends at same value. (must be ensured by controller and a sufficient restoration time)
    """

    e_in = res[res[f"p_{unit_alias}_meas"] < 0][f"p_{unit_alias}_meas"].sum()  # Energy charged over one day (kWs)
    e_out = res[res[f"p_{unit_alias}_meas"] > 0][f"p_{unit_alias}_meas"].sum()  # Energy discharged over one day (kWs)
    #TODO: Update?

    return -(e_in + e_out) / 3600 / 24  # (kWs/d)/((s/h)*(h/d)) -> kWh/h


def Schedule_optimization(config):
    # Create config set for the selected representative price days and add the respective price file to each
    Num_price_days = config["Sizing_Params"]["Num_price_days"]
    Price_day_weighting = config["Sizing_Params"]["Price_day_weighting"]
    Sorted_and_selected_price_day_indexes = sorted(range(len(Price_day_weighting)), key=lambda i: Price_day_weighting[i], reverse=True)[:Num_price_days]

    configs = []
    for price_day_idx in Sorted_and_selected_price_day_indexes:
        cfg = deepcopy(config)
        cfg["Sizing_Params"]["Price_file"] = f"price_day{price_day_idx + 1}.csv"
        configs.append(cfg)

    # Determine optimal schedules for the selected representative days of price data
    if config["Sizing_Params"]["UseParallel"]:
        with Pool(processes=cpu_count()) as pool:
            results = list(
                pool.starmap(Run_Daily_Schedule_Optimization, [(configs[day], day) for day in range(Num_price_days)]))
    else:
        results = []
        for day in range(Num_price_days):
            results.append(Run_Daily_Schedule_Optimization(configs[day], day))

    return results


def Approximate_Yearly_Cost_from_Optimization(results, config):
    Price_day_weighting = config["Sizing_Params"]["Price_day_weighting"]
    Num_price_days = config["Sizing_Params"]["Num_price_days"]
    results = dict(results)

    Costs = []

    # Extract cost of each day and add to the Costs list
    for key in results.keys():
        Costs.append(results[key]["Cost_upper_bound"])

    # Define list which contains the yearly occurrences of the representative price days
    Sorted_and_selected_price_day_weights = sorted(Price_day_weighting, reverse=True)[:Num_price_days]
    Sorted_and_selected_price_day_weights_normalized = [w / sum(Sorted_and_selected_price_day_weights) for w in Sorted_and_selected_price_day_weights]
    Sorted_and_selected_price_day_occurrences = [int(round(w * 365)) for w in Sorted_and_selected_price_day_weights_normalized]

    # Calculate yearly cost by multiplying the cost of the representative price days with their yearly occurrences
    Yearly_cost = sum(np.array(Costs) * np.array(Sorted_and_selected_price_day_occurrences))

    return Yearly_cost


def Approximate_Yearly_Cost_from_Simulation(schedules, config):
    ### Create config set for the selected representative price days with the optimal pRt schedules and create it for each frequency day ###
    pRt_schedules = [day_schedule[1]["pRt"] for day_schedule in schedules]
    Num_freq_days = config["Sizing_Params"]["Num_freq_days"]
    Num_price_days = config["Sizing_Params"]["Num_price_days"]
    Frequency_day_weighting = config["Sizing_Params"]["Frequency_day_weighting"]
    Price_day_weighting = config["Sizing_Params"]["Price_day_weighting"]
    Sorted_and_selected_frequency_day_indexes = sorted(range(len(Frequency_day_weighting)), key=lambda i: Frequency_day_weighting[i], reverse=True)[:Num_freq_days]
    Sorted_and_selected_price_day_indexes = sorted(range(len(Price_day_weighting)), key=lambda i: Price_day_weighting[i], reverse=True)[:Num_price_days]

    configs = []

    for schedule_idx, price_day_idx in enumerate(Sorted_and_selected_price_day_indexes):
        for freq_day_idx in Sorted_and_selected_frequency_day_indexes:
            cfg = deepcopy(config)
            cfg["Timeseries"]["Input_File"] = os.path.join(os.path.dirname(__file__), rf"InputData\Frequency\frequency_day{freq_day_idx + 1}.csv")
            cfg["Sizing_Params"]["Price_file"] = f"price_day{price_day_idx + 1}.csv"
            cfg["Timeseries"]["Power_reserve_schedule"] = pRt_schedules[schedule_idx]
            configs.append(cfg.copy())

    ### Get simulation results ###
    simulation_process = subprocess.Popen(["python", "SimulationServer.py"],
                                          cwd=r"C:\Users\nnimy\PycharmProjects\lab-controller")
    response = requests.post('http://localhost:5000/HESS_simulation', json=configs)  # Send a POST request
    if response.status_code == 200:  # Check if the request was successful
        Results = [pd.DataFrame(res) for res in response.json()]  # Extract the result from the response
    else:
        print("Failed to get result from HESS simulation")
        sys.exit(1)
    simulation_process.terminate()

    ### Calculating cost from simulation results ###

    Daily_cost = []

    # Calculate the cost for each price day from weighted averaging over the different frequency time series [sum(0.X*day1, 0.X*day1), sum(0.X*day1, 0.X*day1), sum(0.X*day1, 0.X*day1)]
    Sorted_and_selected_frequency_day_weights = sorted(Frequency_day_weighting, reverse=True)[:Num_freq_days]
    Sorted_and_selected_frequency_day_weights_normalized = [w / sum(Sorted_and_selected_frequency_day_weights) for w in Sorted_and_selected_frequency_day_weights]
    Cost_list = [sum([c*Sorted_and_selected_frequency_day_weights_normalized[idx] for idx, c in enumerate(Daily_cost[i:i + Num_freq_days])]) for i in range(0, len(Daily_cost), Num_freq_days)]

    # Define list which contains the occurrence of the representative price days within one year
    Sorted_and_selected_price_day_weights = sorted(Price_day_weighting, reverse=True)[:Num_price_days]
    Sorted_and_selected_price_day_weights_normalized = [w / sum(Sorted_and_selected_price_day_weights) for w in Sorted_and_selected_price_day_weights]
    Sorted_and_selected_price_day_occurrences = [int(round(w * 365)) for w in Sorted_and_selected_price_day_weights_normalized]

    # Calculate yearly cost by multiplying the cost of the representative price days with their yearly occurrences
    Yearly_cost = sum(np.array(Cost_list) * np.array(Sorted_and_selected_price_day_occurrences))

    return Yearly_cost


def Run_Daily_Schedule_Optimization(config, day=0, manual_prices=None):
      
    pGen = [0] * TimePeriods  # Electrical generation (default to 0)
    qGen = [0] * TimePeriods  # Thermal generation (default to 0)
    pLoad = [0] * TimePeriods  # Electrical load (default to 0)
    qDemand = [0] * TimePeriods  # Thermal load (default to 0)
    EnPrice= [0] * TimePeriods
    HeatPrice= [0] * TimePeriods
    pGrid_max = [float('inf')] * TimePeriods  # Maximum electricity exchange (default to infinity)
    qGrid_max = [float('inf')] * TimePeriods  # Maximum heat exchange (default to infinity)
    eta_RC = config["Thermal_to_Electrical_Converters"][0]["Ranking_Cycle"]["Eta_RC"]  # Efficiency of Rankine Cycle
    Cost_ESS_A= [0]
    Cost_ESS_B= [0]
    Cost_ESS_C= [0]
    Cost_TES_A= [0]
    ### Time Periods ###
    TimePeriods = len(EnPrice)  # used for debugging update to 24 periods
    dt = float(1)  # duration of time period

    P_A = float(
        config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Available_Power"])  # Power Capacity
    E_A = float(
        config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Available_Capacity"])  # Energy Capacity
    try:
        E_A = E_A * (1 -
                     config["Electrcial_Storage_Units"][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Unit_Specific_Parameters"][
                         "L_total"])
    except KeyError:
        pass
    eta_ch_A = float(
        config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                          "Eta_dis"])  # efficiency loss (discharging)
    static_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Static"])  # static loss
    aEmin_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                        "Minimum_SOC"]) / 100  # minimum State of Energy (0.2 = 20%)
    aEmax_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                        "Maximum_SOC"]) / 100  # maximum State of Energy (0.9 = 90%)

        # ESS_B
    P_B = float(
        config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Available_Power"])  # Power Capacity
    E_B = float(
        config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Available_Capacity"])  # Energy Capacity
    try:
        E_B = E_B * (1 -
                     config["Electrcial_Storage_Units"][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Unit_Specific_Parameters"][
                         "L_total"])
    except KeyError:
        pass
    eta_ch_B = float(
        config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                          "Eta_dis"])  # efficiency loss (discharging)
    static_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Static"])
    aEmin_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                        "Minimum_SOC"]) / 100  # minimum State of Energy (0.2 = 20%)
    aEmax_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                        "Maximum_SOC"]) / 100  # maximum State of Energy (0.9 = 90%)


    # ESS_C
    P_C = float(
        config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Available_Power"])  # Power Capacity
    E_C = float(
        config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Available_Capacity"])  # Energy Capacity
    try:
        E_C = E_C * (1 -
                     config["Electrcial_Storage_Units"][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Unit_Specific_Parameters"][
                         "L_total"])
    except KeyError:
        pass
    eta_ch_C = float(
        config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                          "Eta_dis"])  # efficiency loss (discharging)
    static_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Static"])
    aEmin_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                        "Minimum_SOC"]) / 100  # minimum State of Energy (0.2 = 20%)
    aEmax_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                        "Maximum_SOC"]) / 100  # maximum State of Energy (0.9 = 90%)

    # TESS_A
    P_TA = float(
        config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Available_Power"])  # Power Capacity
    E_TA = float(
        config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Available_Capacity"])  # Energy Capacity
    try:
        E_TA = E_TA * (1 -
                       config["Thermal_Storage_Units"][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Unit_Specific_Parameters"][
                           "L_total"])
    except KeyError:
        pass

    
    eta_ch_TA = float(
        config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
                           "Eta_dis"])  # efficiency loss (discharging)
    static_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Static"])  # static loss
    aEmin_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
                         "Minimum_SOC"]) / 100  # minimum State of Energy (0.2 = 20%)
    aEmax_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
        "Maximum_SOC"]) / 100  # maximum State of Energy (0.9 = 90%)


    ############################################
    ### Defining the optimization model ###

    # Create model instance
    model = ConcreteModel()

    # Sets
    model.T = RangeSet(1, TimePeriods)  # Set of Time Periods
    # Set of electrical energy storage units
    model.ESS = Set(initialize=['A', 'B', 'C'])
    model.TES = set(initialize=['A'])
    # HESS Variables
    model.pEt = Var(model.T, domain=Reals)  # HESS Total Energy Consumption
    model.pRt = Var(model.T, domain=NonNegativeReals)  # HESS Total Regulation Provision

    # ESS A Variables
    model.pEt_A = Var(model.T, domain=Reals)  # Energy Consumption
    model.pBt_A = Var(model.T, domain=Reals)  # Basepoint (power)
    model.pBtch_A = Var(model.T, domain=NonNegativeReals)  # Basepoint charging
    model.pBtdis_A = Var(model.T, domain=NonNegativeReals)  # Basepoint discharging
    model.zch_A = Var(model.T, domain=Boolean)  # Binary indicating charging
    model.zdis_A = Var(model.T, domain=Boolean)  # Binary indicating discharging
    model.et_A = Var(model.T, domain=NonNegativeReals)  # State of Energy (Charge)
    model.et0_A = Var(domain=NonNegativeReals)  # Initial State of Energy (Charge)

    # ESS B Variables
    model.pEt_B = Var(model.T, domain=Reals)  # Energy Consumption
    model.pBt_B = Var(model.T, domain=Reals)  # Basepoint (power)
    model.pBtch_B = Var(model.T, domain=NonNegativeReals)  # Basepoint charging
    model.pBtdis_B = Var(model.T, domain=NonNegativeReals)  # Basepoint discharging
    model.zch_B = Var(model.T, domain=Boolean)  # Binary indicating charging
    model.zdis_B = Var(model.T, domain=Boolean)  # Binary indicating discharging
    model.et_B = Var(model.T, domain=NonNegativeReals)  # State of Energy (Charge)
    model.et0_B = Var(domain=NonNegativeReals)  # Initial State of Energy (Charge)

    # ESS C Variables
    model.pEt_C = Var(model.T, domain=Reals)  # Energy Consumption
    model.pBt_C = Var(model.T, domain=Reals)  # Basepoint (power)
    model.pBtch_C = Var(model.T, domain=NonNegativeReals)  # Basepoint charging
    model.pBtdis_C = Var(model.T, domain=NonNegativeReals)  # Basepoint discharging
    model.zch_C = Var(model.T, domain=Boolean)  # Binary indicating charging
    model.zdis_C = Var(model.T, domain=Boolean)  # Binary indicating discharging
    model.et_C = Var(model.T, domain=NonNegativeReals)  # State of Energy (Charge)
    model.et0_C = Var(domain=NonNegativeReals)  # Initial State of Energy (Charge)


    # # Thermal ESS A Variables
    model.qEt_A = Var(model.T, domain=Reals)  # Energy Consumption
    model.qBt_A = Var(model.T, domain=Reals)  # Basepoint (power)
    model.qBtch_A = Var(model.T, domain=NonNegativeReals)  # Basepoint charging
    model.qBtdis_A = Var(model.T, domain=NonNegativeReals)  # Basepoint discharging
    model.zqch_A = Var(model.T, domain=Boolean)  # Binary indicating charging
    model.zqdis_A = Var(model.T, domain=Boolean)  # Binary indicating discharging
    model.Qt_A = Var(model.T, domain=NonNegativeReals)  # State of Energy (Charge)
    model.Qt0_A = Var(domain=NonNegativeReals)  # Initial State of Energy (Charge)
    #TODO: qt0 and et0: Can be constant 
    # Rankine cycle
    model.q2p = Var(model.T, domain=NonNegativeReals)  # Thermal power covrted to electrical power
    
    # Energy imports/exports
    model.pImp = Var(model.T, domain=NonNegativeReals)  # Electricity import
    model.pExp = Var(model.T, domain=NonNegativeReals)  # Electricity export
    model.pGrid = Var(model.T, domain=NonNegativeReals)  # Grid electricity exchange
    model.qImp = Var(model.T, domain=NonNegativeReals)  # Heat import
    model.qExp = Var(model.T, domain=NonNegativeReals)  # Heat export
    model.qGrid = Var(model.T, domain=NonNegativeReals)  # Grid heat exchange
    # Basepoint power for each storage unit
    model.pBt = Var(model.ESS, model.T, domain=Reals)
    #TODO: check if we can use i instead of _A, _B, _C in all constraints

    #             )
    def obj_rule(model):
        return (
            # pGrid= pImport - pExport TODO: Why tt-1?
            sum(EnPrice[tt - 1] * model.pGrid[tt] *dt for tt in model.T)
            # qGrid= qImport - qExport
            + sum(HeatPrice[tt - 1] * model.qGrid[tt]*dt for tt in model.T)
            
            + sum(
                Cost_ESS_A*(model.pBtch_A[tt]+model.pBtdis_A[tt]) 
                + Cost_ESS_B*(model.pBtch_B[tt]+model.pBtdis_B[tt])
                + Cost_ESS_C*(model.pBtch_C[tt]+model.pBtdis_C[tt])
                + Cost_TES_A* (model.qBtch_A[tt] +model.qBtdis_A[tt])
                for tt in model.T
            )
        )
    model.obj = Objective(rule=obj_rule, sense=minimize)


    # Constraints

    #TODO: Check conventions
    

    def Electricity_balance_rule(model, tt):
        return (
            pGen[tt] + model.pGrid[tt] ==
            model.pB_A[tt] + model.pBt_B[tt] + model.pBt_C[tt] +
            pLoad[tt] 
        )
    model.Electricity_balance = Constraint(model.T, rule=Electricity_balance_rule)
    
    def Thermal_balance_rule(model, tt):
        return (
            qGen[tt] + model.qGrid[tt] ==
            model.qB_A[tt] + qDemand[tt] 
        )
    model.Thermal_balance = Constraint(model.T, rule=Thermal_balance_rule)

    # ESS Add bound on charging variable (pBtch_t <= zch_i[tt] * P_i)
    def ESS_pBtch_A_bound_rule(model, tt):
        return model.pBtch_A[tt] <= model.zch_A[tt] * P_A

    model.ESS_pBtch_A_bound = Constraint(model.T, rule=ESS_pBtch_A_bound_rule)

    def ESS_pBtch_B_bound_rule(model, tt):
        return model.pBtch_B[tt] <= model.zch_B[tt] * P_B

    model.ESS_pBtch_B_bound = Constraint(model.T, rule=ESS_pBtch_B_bound_rule)

    def ESS_pBtch_C_bound_rule(model, tt):
        return model.pBtch_C[tt] <= model.zch_C[tt] * P_C

    model.ESS_pBtch_C_bound = Constraint(model.T, rule=ESS_pBtch_C_bound_rule)

    # ESS Add bound on discharging variable (pBtdis_t <= zdis_i[tt] * P_i)
    def ESS_pBtdis_A_bound_rule(model, tt):
        return model.pBtdis_A[tt] <= model.zdis_A[tt] * P_A

    model.ESS_pBtdis_A_bound = Constraint(model.T, rule=ESS_pBtdis_A_bound_rule)

    def ESS_pBtdis_B_bound_rule(model, tt):
        return model.pBtdis_B[tt] <= model.zdis_B[tt] * P_B

    model.ESS_pBtdis_B_bound = Constraint(model.T, rule=ESS_pBtdis_B_bound_rule)

    def ESS_pBtdis_C_bound_rule(model, tt):
        return model.pBtdis_C[tt] <= model.zdis_C[tt] * P_C

    model.ESS_pBtdis_C_bound = Constraint(model.T, rule=ESS_pBtdis_C_bound_rule)

    # ESS Avoid charging and discharging simultaneously (zch_i[tt] + zdis_i[tt] <= 1)
    def ESS_zchdis_A_rule(model, tt):
        return model.zch_A[tt] + model.zdis_A[tt] <= 1

    model.ESS_zchdis_A = Constraint(model.T, rule=ESS_zchdis_A_rule)

    def ESS_zchdis_B_rule(model, tt):
        return model.zch_B[tt] + model.zdis_B[tt] <= 1

    model.ESS_zchdis_B = Constraint(model.T, rule=ESS_zchdis_B_rule)

    def ESS_zchdis_C_rule(model, tt):
        return model.zch_C[tt] + model.zdis_C[tt] <= 1

    model.ESS_zchdis_C = Constraint(model.T, rule=ESS_zchdis_C_rule)

    #### ======================= ####

    # ESS State of Charge (Energy)
    # et_i[tt] = et_i[tt-1]*(1-static_i)+eta_ch_i*pBtch_i[tt]*dt-(1/eta_dis_i)*pBtdis_i[tt]*dt-Loss_i*pRt_i[tt]*dt
    # Constraints are separately for hour 1 and hours 2..,T
    def ESS_SoE_A_def_rule(model, tt):
        if tt == 1:
            return Constraint.Skip
        return model.et_A[tt] == (1 - static_A) * model.et_A[tt - 1] \
            + eta_ch_A * model.pBtch_A[tt] * dt \
            - (1 / eta_dis_A) * model.pBtdis_A[tt] * dt \
            # - Loss_A * model.pRt_A[tt]*dt

    model.ESS_SoE_A_def = Constraint(model.T, rule=ESS_SoE_A_def_rule)

    def ESS_SoE_B_def_rule(model, tt):
        if tt == 1:
            return Constraint.Skip
        return model.et_B[tt] == (1 - static_B) * model.et_B[tt - 1] \
            + eta_ch_B * model.pBtch_B[tt] * dt \
            - (1 / eta_dis_B) * model.pBtdis_B[tt] * dt \
            # - Loss_B * model.pRt_B[tt]*dt

    model.ESS_SoE_B_def = Constraint(model.T, rule=ESS_SoE_B_def_rule)

    def ESS_SoE_C_def_rule(model, tt):
        if tt == 1:
            return Constraint.Skip
        return model.et_C[tt] == (1 - static_C) * model.et_C[tt - 1] \
            + eta_ch_C * model.pBtch_C[tt] * dt \
            - (1 / eta_dis_C) * model.pBtdis_C[tt] * dt \
            # - Loss_C * model.pRt_C[tt]*dt

    model.ESS_SoE_C_def = Constraint(model.T, rule=ESS_SoE_C_def_rule)

    # Constraint for first hour
    def ESS_SoE_1_A_def_rule(model):
        return model.et_A[1] == (1 - static_A) * model.et0_A \
            + eta_ch_A * model.pBtch_A[1] * dt \
            - (1 / eta_dis_A) * model.pBtdis_A[1] * dt \
            # - Loss_A * model.pRt_A[1]*dt

    model.ESS_SoE_1_A_def = Constraint(rule=ESS_SoE_1_A_def_rule)

    def ESS_SoE_1_B_def_rule(model):
        return model.et_B[1] == (1 - static_B) * model.et0_B \
            + eta_ch_B * model.pBtch_B[1] * dt \
            - (1 / eta_dis_B) * model.pBtdis_B[1] * dt \
            # - Loss_B * model.pRt_B[1]*dt

    model.ESS_SoE_1_B_def = Constraint(rule=ESS_SoE_1_B_def_rule)

    def ESS_SoE_1_C_def_rule(model):
        return model.et_C[1] == (1 - static_C) * model.et0_C \
            + eta_ch_C * model.pBtch_C[1] * dt \
            - (1 / eta_dis_C) * model.pBtdis_C[1] * dt \
            # - Loss_C * model.pRt_C[1]*dt

    model.ESS_SoE_1_C_def = Constraint(rule=ESS_SoE_1_C_def_rule)

    # Set Initial SoE equal to Final SoE (avoid discharging the battery at the end of day) (et0_i = et_i[TimePeriods])
    def InitSoE_A_rule(model):
        return model.et0_A == model.et_A[TimePeriods]

    model.InitSoE_A = Constraint(rule=InitSoE_A_rule)

    def InitSoE_B_rule(model):
        return model.et0_B == model.et_B[TimePeriods]

    model.InitSoE_B = Constraint(rule=InitSoE_B_rule)

    def InitSoE_C_rule(model):
        return model.et0_C == model.et_C[TimePeriods]

    model.InitSoE_C = Constraint(rule=InitSoE_C_rule)

    # ESS Minimum State of Energy (et_i[tt] >= aEmin_i * E_i)
    def SoE_A_min_limit_rule(model, tt):
        return model.et_A[tt] >= aEmin_A * E_A

    model.SoE_A_min_limit = Constraint(model.T, rule=SoE_A_min_limit_rule)

    def SoE_B_min_limit_rule(model, tt):
        return model.et_B[tt] >= aEmin_B * E_B

    model.SoE_B_min_limit = Constraint(model.T, rule=SoE_B_min_limit_rule)

    def SoE_C_min_limit_rule(model, tt):
        return model.et_C[tt] >= aEmin_C * E_C

    model.SoE_C_min_limit = Constraint(model.T, rule=SoE_C_min_limit_rule)

    # ESS Maximum State of Energy (et_i[tt] <= aEmax_i * E_i)
    def SoE_A_max_limit_rule(model, tt):
        return model.et_A[tt] <= aEmax_A * E_A

    model.SoE_A_max_limit = Constraint(model.T, rule=SoE_A_max_limit_rule)

    def SoE_B_max_limit_rule(model, tt):
        return model.et_B[tt] <= aEmax_B * E_B

    model.SoE_B_max_limit = Constraint(model.T, rule=SoE_B_max_limit_rule)

    def SoE_C_max_limit_rule(model, tt):
        return model.et_C[tt] <= aEmax_C * E_C

    model.SoE_C_max_limit = Constraint(model.T, rule=SoE_C_max_limit_rule)

    
    def ESS_qSoE_A_def_rule(model, tt):
        if tt == 1:
            return Constraint.Skip
        return model.Qt_A[tt] == (1 - static_TA) * model.Qt_A[tt - 1] \
            + eta_ch_TA * model.qBtch_A[tt] * dt \
            - (1 / eta_dis_TA) * model.qBtdis_A[tt] * dt \
            # - Loss_TA * model.qRt_A[tt]*dt
    model.ESS_qSoE_A_def = Constraint(model.T, rule=ESS_qSoE_A_def_rule)
    # Constraints for first hour
    # Qt_i[1] = qt0_i*(1-static_i)+eta_ch_i*qBtch_i[1]*dt-(1/eta_dis_i)*qBtdis_i[1]*dt-Loss_i*qRt_i[1]*dt
    def ESS_qSoE_1_A_def_rule(model):
        return model.Qt_A[1] == (1 - static_TA) * model.Qt0_A \
            + eta_ch_TA * model.qBtch_A[1] * dt \
            - (1 / eta_dis_TA) * model.qBtdis_A[1] * dt \
            # - Loss_TA * model.qRt_A[1]*dt
    model.ESS_qSoE_1_A_def = Constraint(rule=ESS_qSoE_1_A_def_rule)
    def InitSoE_A_rule(model):
        return model.Qt0_A == model.Qt_A[TimePeriods]
    model.InitSoE_A = Constraint(rule=InitSoE_A_rule)
    def SoE_A_min_limit_rule(model, tt):
        return model.Qt_A[tt] >= aEmin_TA * E_TA
    model.SoE_A_min_limit = Constraint(model.T, rule=SoE_A_min_limit_rule)
    def SoE_A_max_limit_rule(model, tt):
        return model.Qt_A[tt] <= aEmax_TA * E_TA
    model.SoE_A_max_limit = Constraint(model.T, rule=SoE_A_max_limit_rule)
   
    def Rankine_Cycle_rule(model, tt):
        return model.pBt_RC[tt] == model.q2p[tt] * dt * eta_RC
    model.Rankine_Cycle = Constraint(model.T, rule=Rankine_Cycle_rule)
    #TODO: Rankine cycle input is thermal waste! q2p = qGen- qLoad
    
    def Elec_Import_Limit_rule(model, tt):
        return model.pGrid[tt] <= pGrid_max[tt]
    model.Elec_Import_Limit = Constraint(model.T, rule=Elec_Import_Limit_rule)

    def Elec_Export_Limit_rule(model, tt):
        return -pGrid_max[tt] <=  model.pGrid[tt]
    model.Elec_Export_Limit = Constraint(model.T, rule=Elec_Export_Limit_rule)

    def Heat_Import_Limit_rule(model, tt):
        return model.qGrid[tt] <= qGrid_max[tt]
    model.Heat_Import_Limit = Constraint(model.T, rule=Heat_Import_Limit_rule)

    def Heat_Export_Limit_rule(model, tt):
        return -qGrid_max[tt] <= model.qGrid[tt]
    model.Heat_Export_Limit = Constraint(model.T, rule=Heat_Export_Limit_rule)
    #TODO: check regulation limits for Rankine Cycle
    ### Define solver, create model instance and solve ###
    opt = pyo.SolverFactory("gurobi")
    instance = model.create_instance()
    results = opt.solve(instance)

    ### Extract results ###
    result_dict = {"Solver_status": str(results.solver.status),
                   "Cost_upper_bound": results.problem.lower_bound,
                   "Cost_lower_bound": results.problem.upper_bound,
                   }

    for dict_key in model.component_map(ctype=pyo.Var).keys():
        result_dict[dict_key] = list(getattr(instance, dict_key).extract_values().values())

    return f"Day_{day}", result_dict