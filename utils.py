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
            'Gamma': 1,
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


# def Select_Storage_Models_for_Lifetime(config, hess_lifetime):
#     ### Select storage unit models for given lifetime ###
#     Electrcial_Storage_Units = [list(unit.keys())[0] for unit in config.get("Electrcial_Storage_Units")]

#     if hess_lifetime == 0:
#         for idx, unit in enumerate(Electrcial_Storage_Units):
#             if unit == "Lithium_Battery":  # Currently only degradation for Lithium storage assumed
#                 config["Electrcial_Storage_Units"][idx]["Lithium_Battery"]["Implementation"] = "simulators.NMCBatteryFirstLife"
#                 config["Electrcial_Storage_Units"][idx]["Lithium_Battery"]["Unit_Specific_Parameters"]["L_total"] = 0
#     else:
#         for idx, unit in enumerate(Electrcial_Storage_Units):
#             if unit == "Lithium_Battery":  # Currently only degradation for Lithium storage assumed
#                 config["Electrcial_Storage_Units"][idx]["Lithium_Battery"]["Implementation"] = "simulators.NMCBatterySecondLife"
#                 config["Electrcial_Storage_Units"][idx]["Lithium_Battery"]["Unit_Specific_Parameters"][
#                     "L_total"] = 0.2 / 1 * hess_lifetime

#     return config


def Parameterization(config, standalone=False):
    """
    Function to update the coefficients which represent hourly degradation cost, losses and fraction of successful
    service delivery as a function of the power reserve in the optimization problem. Losses, degradation and service
    delivery fraction are calculated for frequency data of one day (per second resolution) and for different power
    reserve values which are assumed to be constant throughout the day.

    Args:
        config (dict): Current HESS configuration used to initialize and run the simulation
        standalone (boolean): If True the function returns coefficients and sim results for plotting, else the updated config

    Returns:
        Coefficients and simulation results for losses, degradation cost and service delivery

    Assumptions:
        - Simulation run of one day (24 hours)
        - SOC starts and ends at same value. (must be ensured by controller and a sufficient restoration time)
    """
    #start_time = time.time()
    ### Define config set ###
    configs = []
    Num_freq_days = config["Sizing_Params"]["Num_freq_days"]
    num_periods = int(1440 / config["Timeseries"]["Power_reserve_period"])
    # granularity = config["Sizing_Params"]["Parameterization_granularity"]
    Frequency_day_weighting = config["Sizing_Params"]["Frequency_day_weighting"]
    Sorted_and_selected_frequency_day_indexes = sorted(range(len(Frequency_day_weighting)), key=lambda i: Frequency_day_weighting[i], reverse=True)[:Num_freq_days]

    # Define power reserve range for determining the degradation and loss coefficients
    # pRt_range = np.linspace(0.1, sum([d[list(d.keys())[0]]["Available_Power"] for d in config['Electrcial_Storage_Units']]), granularity)

    # for r in pRt_range:
    #     for freq_day_idx in Sorted_and_selected_frequency_day_indexes:
    #         cfg = deepcopy(config)
    #         cfg["Timeseries"]["Input_File"] = os.path.join(os.path.dirname(__file__), rf"InputData\Frequency\frequency_day{freq_day_idx + 1}.csv")
    #         cfg["Timeseries"]["Power_reserve_schedule"] = num_periods * [r]
    #         configs.append(cfg.copy())


    #print(f"time for config set: {round(time.time()-start_time)}")
    config_time = time.time()
    ### Get simulation results ###
    simulation_process = subprocess.Popen(["python", "SimulationServer.py"], cwd=r"C:\Users\nnimy\PycharmProjects\lab-controller")
    response = requests.post('http://localhost:5000/HESS_simulation', json=configs)  # Send a POST request
    if response.status_code == 200:  # Check if the request was successful
        Results = [pd.DataFrame(res) for res in response.json()]  # Extract the result from the response
    else:
        print("Failed to get result from HESS simulation")
        sys.exit(1)
    simulation_process.terminate()

    #print(f"time for sim: {round(time.time() - config_time)}")
    sim_time = time.time()
    ### Calculate loss, degradation cost and service delivery metrics from simulation results ###
    Electrcial_Storage_Units = [list(unit.keys())[0] for unit in config.get("Electrcial_Storage_Units")]

    # Calculate occurrences based on the weights for the number of frequency time series days
    Sorted_and_selected_frequency_day_weights = sorted(Frequency_day_weighting, reverse=True)[:Num_freq_days]
    Sorted_and_selected_frequency_day_weights_normalized = [w/sum(Sorted_and_selected_frequency_day_weights) for w in Sorted_and_selected_frequency_day_weights]
    Sorted_and_selected_frequency_day_occurrences = [int(round(w * 365)) for w in Sorted_and_selected_frequency_day_weights_normalized]

    # Losses
    losses = {}
    for unit in Electrcial_Storage_Units:
        # Determine losses for each simulation run
        losses_unit = [Hourly_losses(Results_of_one_pRt, aliases[unit]) for Results_of_one_pRt in Results]
        # Repeat each loss value based on the occurrence of the underlying frequency time series
        losses[unit] = [loss for loss, count in zip(losses_unit, Sorted_and_selected_frequency_day_occurrences*len(pRt_range)) for _ in range(count)]

    # Degradation cost
    if "Lithium_Battery" in Electrcial_Storage_Units:
        # Determine degradation cost for each simulation run
        deg_cost = [Hourly_li_degradation_cost(Results_of_one_pRt, config) for Results_of_one_pRt in Results]
        # Repeat each loss value based on the occurrence of the underlying frequency time series
        deg_cost = [dc for dc, count in zip(deg_cost, Sorted_and_selected_frequency_day_occurrences * len(pRt_range)) for _ in range(count)]
    else:
        deg_cost = [0] * len(Results)

    # Service delivery fraction
    # Determine degradation cost for each simulation run
    delivery_fraction = [Service_delivery_fraction(Results_of_one_pRt) for Results_of_one_pRt in Results]
    # Repeat each loss value based on the occurrence of the underlying frequency time series
    delivery_fraction = [df for df, count in zip(delivery_fraction, Sorted_and_selected_frequency_day_occurrences * len(pRt_range)) for _ in range(count)]

    ### Fit functions to get coefficients for losses, degradation cost and service delivery fraction ###

    # Repeat each pRt value by the number of frequency time series days
    # pRt_range = [r for r in pRt_range for _ in range(Num_freq_days)]
    # # Repeat each pRt value by the occurrence of each frequency time series day
    # pRt_range = [pRt for pRt, count in zip(pRt_range, Sorted_and_selected_frequency_day_occurrences * len(pRt_range)) for _ in range(count)]

    # Losses
    # loss_coefficients = {}
    # for unit in Electrcial_Storage_Units:
    #     loss_coefficients[unit] = curve_fit(lambda pRt_range, a, b: a * pRt_range + b, pRt_range, losses[unit])[0]

    # # Degradation cost
    # deg_cost_coefficients = \
    # curve_fit(lambda pRt_range, a, b, c: a * pRt_range ** 2 + b * pRt_range + c, pRt_range, deg_cost,
    #           bounds=((0, -np.inf, -np.inf), (np.inf, np.inf, np.inf)))[0]

    # # Service delivery fraction
    # delivery_fraction_coefficients = \
    # curve_fit(lambda pRt_range, a, b, c: a * pRt_range ** 2 + b * pRt_range + c, pRt_range, delivery_fraction,
    #           bounds=((0, -np.inf, -np.inf), (np.inf, np.inf, np.inf)))[0]

    # # Update config file accordingly
    # for idx, unit in enumerate(Electrcial_Storage_Units):
    #     config["Electrcial_Storage_Units"][idx][unit]["Loss_coef_a"] = loss_coefficients[unit][0]
    #     config["Electrcial_Storage_Units"][idx][unit]["Loss_coef_b"] = loss_coefficients[unit][1]

    #     if unit == "Lithium_Battery":
    #         config["Electrcial_Storage_Units"][idx]["Lithium_Battery"]["Deg_cost_coef_a"] = deg_cost_coefficients[0]
    #         config["Electrcial_Storage_Units"][idx]["Lithium_Battery"]["Deg_cost_coef_b"] = deg_cost_coefficients[1]
    #         config["Electrcial_Storage_Units"][idx]["Lithium_Battery"]["Deg_cost_coef_c"] = deg_cost_coefficients[2]

    #print(f"time for coef: {round(time.time() - sim_time)}")

    # if not standalone:
    #     return config
    # else:

    #     return pRt_range, loss_coefficients, deg_cost_coefficients, delivery_fraction_coefficients, deg_cost, losses, delivery_fraction


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

    
# def Calculate_total_gradient(netOpCost_cur, netOpCost_pert, config):
#     c_cap = np.array([float(d[list(d.keys())[0]][k]) for k in ["PowerCapCost", "EnergyCapCost"] for d in
#                       config['Electrcial_Storage_Units']])  ### Get power, energy capacity and fixed cost from the config ###
#     int_rate, lifetime = float(config["HESS_Params"]["Int_rate"]), float(config["HESS_Params"]["Lifetime"])
#     invAnnuFactor = int_rate * ((int_rate + 1) ** lifetime) / (
#                 (int_rate + 1) ** lifetime - 1)  # Annualized Investment Cost

#     grad_x1 = invAnnuFactor * c_cap
#     Delta_x = config["Sizing_Params"]["Delta_x"]

#     # Calculate grad x2
#     grad_x2 = (netOpCost_pert - netOpCost_cur) / Delta_x

#     # Add the two gradient components
#     grad_x = grad_x1 + grad_x2

#     return grad_x


# def Calculate_suggested_capacities_from_gradient(x_cur, grad_x, config):
#     Stepsize = float(config["Sizing_Params"]["Stepsize"])
#     scaling = np.array(
#         [d[list(d.keys())[0]][k] for k in ["Available_Power", "Available_Capacity"] for d in config['Electrcial_Storage_Units']])

#     x_min = np.array([d[list(d.keys())[0]][k] for k in ["Min_Limit_Power", "Min_Limit_Energy_Capacity"] for d in
#                       config['Electrcial_Storage_Units']])
#     x_max = np.array([d[list(d.keys())[0]][k] for k in ["Max_Limit_Power", "Max_Limit_Energy_Capacity"] for d in
#                       config['Electrcial_Storage_Units']])

#     # Calculate directions from gradients
#     d_cur = np.where(grad_x != 0, -grad_x / np.abs(grad_x), 0)

#     # Update capacities
#     x_new = x_cur + 10 * Stepsize * scaling * d_cur

#     # Set values which are outside the allowed range to min/max
#     x_new = np.maximum(x_min, np.minimum(x_max, x_new))

#     return x_new


# def Check_exit_conditions(x_new, grad_x, config, i):
#     BudgetLimit = float(config["HESS_Params"]["BudgetLimit"])  # Budget Limit
#     FixedCost = float(config["HESS_Params"]["FixedCost"])  # Fixed cost of HESS
#     c_cap = np.array([float(d[list(d.keys())[0]][k]) for k in ["PowerCapCost", "EnergyCapCost"] for d in
#                       config['Electrcial_Storage_Units']])  ### Get power, energy capacity and fixed cost from the config ###
#     int_rate, lifetime = float(config["HESS_Params"]["Int_rate"]), float(config["HESS_Params"]["Lifetime"])
#     invAnnuFactor = int_rate * ((int_rate + 1) ** lifetime) / (
#                 (int_rate + 1) ** lifetime - 1)  # Annualized Investment Cost
#     grad_tolerance = float(config["Sizing_Params"]["Tolerance"])
#     x_min = np.array([d[list(d.keys())[0]][k] for k in ["Min_Limit_Power", "Min_Limit_Energy_Capacity"] for d in
#                       config['Electrcial_Storage_Units']])
#     x_max = np.array([d[list(d.keys())[0]][k] for k in ["Max_Limit_Power", "Max_Limit_Energy_Capacity"] for d in
#                       config['Electrcial_Storage_Units']])

#     # Check if gradient within accepted tolerance
#     if np.linalg.norm(grad_x) <= grad_tolerance:
#         logging.info(f"[It {i}] Gradient within tolerance. Exiting.")
#         return True

#     # Check budget limit
#     AIC_new = (np.dot(c_cap, x_new) + FixedCost) * invAnnuFactor

#     if AIC_new >= BudgetLimit:
#         logging.info(f"[It {i}] Budget limit exceeded. Exiting.")
#         return True

#     # Check capacity limit
#     if np.all(~((x_new > x_min) & (x_new < x_max))):  # if all x_new values are NOT (~) in the range then break
#         logging.info(f"[It {i}] All power and energy capacities reached limit. Exiting.")
#         return True

#     else:
#         logging.info(f"[It {i}] No exit condition met. Entering next iteration.")
#         return False


# def Update_capacities(x, config):
#     for j in range(int(len(x) / 2)):
#         config['Electrcial_Storage_Units'][j][list(config['Electrcial_Storage_Units'][j].keys())[0]]["Available_Power"] = x[j]
#         config['Electrcial_Storage_Units'][j][list(config['Electrcial_Storage_Units'][j].keys())[0]]["Available_Capacity"] = x[j + 3]

#     cur_cap_info = {unit_name: [unit_info['Available_Power'], unit_info['Available_Capacity']] for d in
#                     config['Electrcial_Storage_Units'] for unit_name, unit_info in d.items()}

#     return config, x.copy(), cur_cap_info


# def Store_data(x_cur, netopcost_cur, i, init_time, config):
#     Electrcial_Storage_Units = [list(d.keys())[0] for d in config["Electrcial_Storage_Units"]]
#     c_cap = np.array([float(d[list(d.keys())[0]][k]) for k in ["PowerCapCost", "EnergyCapCost"] for d in
#                       config['Electrcial_Storage_Units']])  ### Get power, energy capacity and fixed cost from the config ###
#     FixedCost = float(config["HESS_Params"]["FixedCost"])  # Fixed cost of HESS
#     int_rate, lifetime = float(config["HESS_Params"]["Int_rate"]), float(config["HESS_Params"]["Lifetime"])
#     invAnnuFactor = int_rate * ((int_rate + 1) ** lifetime) / (
#                 (int_rate + 1) ** lifetime - 1)  # Annualized Investment Cost
#     Iter_dict = {[f"{a}_{aliases[b]}" for a in ["P", "E"] for b in Electrcial_Storage_Units][index]: [x_cur[index]] for index in
#                  range(len(x_cur))}
#     for idx, key in enumerate(Iter_dict.keys()):
#         Iter_dict[key] = x_cur[idx]

#     data_dict = {"Iter": i,
#                  "NetOpCost": netopcost_cur,
#                  "AIC": (np.dot(c_cap, x_cur) + FixedCost) * invAnnuFactor,
#                  "HESS_obj": (np.dot(c_cap, x_cur) + FixedCost) * invAnnuFactor + netopcost_cur
#                  }

#     data_dict.update(Iter_dict)

#     with open(f"Results/SizingTool/{int(init_time)}_iter_results.csv", "a", newline="") as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=data_dict.keys())
#         if os.stat(f"Results/SizingTool/{int(init_time)}_iter_results.csv").st_size == 0:
#             writer.writeheader()
#         writer.writerow(data_dict)


# def numpy_array_encoder(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()  # Convert numpy array to list
#     raise TypeError("Numpy array is not JSON serializable")


# def Load_params_from_config(config):
#     x_cur = np.array([d[list(d.keys())[0]][k] for k in ["Available_Power", "Available_Capacity"] for d in config['Electrcial_Storage_Units']])
#     Max_iterations = float(config["Sizing_Params"]["Max_iterations"])
#     Delta_x = config["Sizing_Params"]["Delta_x"]
#     LifetimeSplits = int(config["Sizing_Params"]["LifetimeSplits"])

#     # Extracting required information for each battery type
#     cur_cap_info = {unit_name: [unit_info['Available_Power'], unit_info['Available_Capacity']] for d in config['Electrcial_Storage_Units'] for unit_name, unit_info in d.items()}

#     return Max_iterations, LifetimeSplits, x_cur, Delta_x, cur_cap_info


# Function for calculating weighted cost scaled to one year
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

    for idx in range(len(Results)):
        day_Sim_res = Results[idx].iloc[:86400]  # Cut off the battery restoration period after 24 hours
        day_config = configs[idx]
        day_prices = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "InputData/Prices/",
                                              day_config["Sizing_Params"]["Price_file"]))
        day_pRt_schedule = day_config["Timeseries"]["Power_reserve_schedule"]

        Energy_cost = -day_prices["EnPrice[€/kWh]"].values * day_Sim_res["p_set_hess"].groupby(
            day_Sim_res.index // 3600).sum().values / 3600
        Regulation_cost = -day_prices["RegPrice[€/kW]"].values * np.array(day_pRt_schedule)
        Penalty_cost = 0
        Degradation_cost = Hourly_li_degradation_cost(day_Sim_res, day_config) * 24

        Daily_cost.append(sum(Energy_cost) + sum(Regulation_cost) + Penalty_cost + Degradation_cost)

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
    ### Load price data of the current day ###
    # if manual_prices is None:
    #     prices = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "InputData/Prices/",
    #                                       config["Sizing_Params"]["Price_file"]))
    #     EnPrice = prices["EnPrice[€/kWh]"].values
    #     # RegPrice = prices["RegPrice[€/kW]"].values
    # else:
    #     EnPrice = manual_prices["EnPrice[€/kWh]"].values
    #     # RegPrice = manual_prices["RegPrice[€/kW]"].values

    # Deg_cost_coef_a = [Deg_cost_coef_a_A, Deg_cost_coef_a_B, Deg_cost_coef_a_C]  # Assign degradation cost coefficients for each ESS
    # Deg_cost_coef_b = [Deg_cost_coef_b_A, Deg_cost_coef_b_B, Deg_cost_coef_b_C]  # Assign degradation cost coefficients for each ESS
    # Deg_cost_coef_c = [Deg_cost_coef_c_A, Deg_cost_coef_c_B, Deg_cost_coef_c_C]  # Assign degradation cost coefficients for each ESS
    # Define missing parameters
  
    pGen = [0] * TimePeriods  # Electrical generation (default to 0)
    qGen = [0] * TimePeriods  # Thermal generation (default to 0)
    pLoad = [0] * TimePeriods  # Electrical load (default to 0)
    qDemand = [0] * TimePeriods  # Thermal load (default to 0)
    EnPrice= [0] * TimePeriods
    HeatPrice= [0] * TimePeriods
    pImp_max = [float('inf')] * TimePeriods  # Maximum electricity import (default to infinity)
    pExp_max = [float('inf')] * TimePeriods  # Maximum electricity export (default to infinity)
    qImp_max = [float('inf')] * TimePeriods  # Maximum heat import (default to infinity)
    qExp_max = [float('inf')] * TimePeriods  # Maximum heat export (default to infinity)
    eta_RC = config["Thermal_to_Electrical_Converters"][0]["Ranking_Cycle"]["Eta_RC"]  # Efficiency of Rankine Cycle
    
    ### Time Periods ###
    TimePeriods = len(EnPrice)  # used for debugging update to 24 periods
    dt = float(1)  # duration of time period

    ### Defining Service Parameter ###
    # kappa = float(config['HESS_Params']["Kappa"])  # max duration of the regulation reserve in hours (15 min)
    # zeta = float(
    #     config['HESS_Params']["Zeta"])  # coefficient for conservative sustained duration (if > 1 less conservative)

    # ### Defining the Storage Parameter ###
    # # HESS
    # HESS_score = float(
    #     config['HESS_Params']['HESS_score'])  # performance score for frequency regulation (should be by hour)

    # ESS_A
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
    Loss_coef_a_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                              "Loss_coef_a"])  # Losses due to regulation provision
    Loss_coef_b_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Loss_coef_b"])
    eta_ch_A = float(
        config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                          "Eta_dis"])  # efficiency loss (discharging)
    static_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Static"])  # static loss
    aEmin_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                        "Minimum_SOC"]) / 100  # minimum State of Energy (0.2 = 20%)
    aEmax_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                        "Maximum_SOC"]) / 100  # maximum State of Energy (0.9 = 90%)
    gamma_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                        "Gamma"])  # min{E_i/P_i, k}/k, where k = 0.25 (15 min duration), gamma = 1
    try:
        Deg_cost_coef_a_A = float(
            config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Deg_cost_coef_a"])
        Deg_cost_coef_b_A = float(
            config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Deg_cost_coef_b"])
        Deg_cost_coef_c_A = float(
            config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Deg_cost_coef_c"])

    except KeyError:
        Deg_cost_coef_a_A = 0
        Deg_cost_coef_b_A = 0
        Deg_cost_coef_c_A = 0

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
    Loss_coef_a_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                              "Loss_coef_a"])  # Losses due to regulation provision
    Loss_coef_b_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Loss_coef_b"])
    eta_ch_B = float(
        config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                          "Eta_dis"])  # efficiency loss (discharging)
    static_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Static"])
    aEmin_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                        "Minimum_SOC"]) / 100  # minimum State of Energy (0.2 = 20%)
    aEmax_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                        "Maximum_SOC"]) / 100  # maximum State of Energy (0.9 = 90%)
    gamma_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                        "Gamma"])  # min{E_i/P_i, k}/k, where k = 0.25 (15 min duration) gamma = 0.1/0.25
    try:
        Deg_cost_coef_a_B = float(
            config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Deg_cost_coef_a"])
        Deg_cost_coef_b_B = float(
            config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Deg_cost_coef_b"])
        Deg_cost_coef_c_B = float(
            config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Deg_cost_coef_c"])

    except KeyError:
        Deg_cost_coef_a_B = 0
        Deg_cost_coef_b_B = 0
        Deg_cost_coef_c_B = 0

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
    Loss_coef_a_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                              "Loss_coef_a"])  # Losses due to regulation provision
    Loss_coef_b_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Loss_coef_b"])
    eta_ch_C = float(
        config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                          "Eta_dis"])  # efficiency loss (discharging)
    static_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Static"])
    aEmin_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                        "Minimum_SOC"]) / 100  # minimum State of Energy (0.2 = 20%)
    aEmax_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                        "Maximum_SOC"]) / 100  # maximum State of Energy (0.9 = 90%)
    gamma_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                        "Gamma"])  # min{E_i/P_i, k}/k, where k = 0.25 (15 min duration) gamma = 0.1/0.25
    try:
        Deg_cost_coef_a_C = float(
            config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Deg_cost_coef_a"])
        Deg_cost_coef_b_C = float(
            config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Deg_cost_coef_b"])
        Deg_cost_coef_c_C = float(
            config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Deg_cost_coef_c"])

    except KeyError:
        Deg_cost_coef_a_C = 0
        Deg_cost_coef_b_C = 0
        Deg_cost_coef_c_C = 0
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

    
    Loss_coef_a_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
                               "Loss_coef_a"])  # Losses due to regulation provision
    Loss_coef_b_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Loss_coef_b"])
    eta_ch_TA = float(
        config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
                           "Eta_dis"])  # efficiency loss (discharging)
    static_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Static"])  # static loss
    aEmin_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
                         "Minimum_SOC"]) / 100  # minimum State of Energy (0.2 = 20%)
    aEmax_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
                         "Maximum_SOC"]) / 100  # maximum State of Energy (0.9 = 90%)
    gamma_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
                         "Gamma"])  # min{E_i/P_i, k}/k, where k = 0.25 (15 min duration), gamma = 1
    try:
        Deg_cost_coef_a_TA = float(
            config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Deg_cost_coef_a"])
        Deg_cost_coef_b_TA = float(
            config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Deg_cost_coef_b"])
        Deg_cost_coef_c_TA = float(
            config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Deg_cost_coef_c"])

    except KeyError:
        Deg_cost_coef_a_TA = 0
        Deg_cost_coef_b_TA = 0
        Deg_cost_coef_c_TA = 0

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
    model.qImp = Var(model.T, domain=NonNegativeReals)  # Heat import
    model.qExp = Var(model.T, domain=NonNegativeReals)  # Heat export
    # Basepoint power for each storage unit
    model.pBt = Var(model.ESS, model.T, domain=Reals)
    #TODO: check if we can use i instead of _A, _B, _C in all constraints

    # Objective function
    # def obj_rule(model):
    #     return (sum(EnPrice[tt - 1] * model.pEt[tt] for tt in model.T) \
    #             - sum(RegPrice[tt - 1] * model.pRt[tt] * HESS_score for tt in model.T) \
    #             + sum(Deg_cost_coef_a_A * model.pRt[tt] ** 2 + Deg_cost_coef_b_A * model.pRt[tt] + Deg_cost_coef_c_A for
    #                   tt in model.T) \
    #             + sum(Deg_cost_coef_a_B * model.pRt[tt] ** 2 + Deg_cost_coef_b_B * model.pRt[tt] + Deg_cost_coef_c_B for
    #                   tt in model.T) \
    #             + sum(Deg_cost_coef_a_C * model.pRt[tt] ** 2 + Deg_cost_coef_b_C * model.pRt[tt] + Deg_cost_coef_c_C for
    #                   tt in model.T)
    #             + sum(Deg_cost_coef_a_TA * model.qRt_A[tt] ** 2 + Deg_cost_coef_b_TA * model.qRt_A[tt] + Deg_cost_coef_c_TA for
    #                   tt in model.T) #TODO: check this cost fundtion for thermal ess 
    #             )
    def obj_rule(model):
        return (
            # Cost of imported electricity TODO: Why tt-1?
            sum(EnPrice[tt - 1] * model.pImp[tt] for tt in model.T)
            # Revenue from exported electricity
            - sum(EnPrice[tt - 1] * model.pExp[tt] for tt in model.T)
            # Cost of imported heat
            + sum(HeatPrice[tt - 1] * model.qImp[tt] for tt in model.T)
            # Revenue from exported heat
            - sum(HeatPrice[tt - 1] * model.qExp[tt] for tt in model.T)
            # Electrical storage degradation cost
            + sum(
                # Deg_cost_coef_a[i] * (model.pBt[i, tt] * dt) ** 2
                # + Deg_cost_coef_b[i] * abs(model.pBt[i, tt]) * dt
                # + Deg_cost_coef_c[i]
                # for i in model.ESS
                # for tt in model.T
            )
        )
    model.obj = Objective(rule=obj_rule, sense=minimize)


    # Constraints

    # HESS Total Energy Consumption (The pEt[tt] = sum_{i=A,B,C} pEt_i[tt])
    #TODO: Check conventions
    # def HESS_pEt_def_rule(model, tt):
    #     return model.pEt[tt] == model.pEt_A[tt] + model.pEt_B[tt] + model.pEt_C[tt] + model.pEt_RC [tt] + Electrical_gen + model.Electrical_Grid[tt]

    # model.HESS_pEr_def = Constraint(model.T, rule=HESS_pEt_def_rule)

    def Electricity_balance_rule(model, tt):
        return (
            pGen[tt] + model.pImp[tt] ==
            model.pB_A[tt] + model.pBt_B[tt] + model.pBt_C[tt] +
            pLoad[tt] + model.pExp[tt]
        )
    model.Electricity_balance = Constraint(model.T, rule=Electricity_balance_rule)
    
    # # ESS Energy Consumption (pEt_i[tt] = pBt_i[tt] * dt + Loss_i[tt] * pRt_i[tt] *dt) Assume dt = 1 (hour)
    # def ESS_pEt_A_def_rule(model, tt):
    #     return model.pEt_A[tt] == model.pBt_A[tt] * dt + Loss_coef_b_A
    #     # Loss coefficients are for pRt not pRt_A
    #     # dt not required for losses since included in the coefficients

    # model.ESS_pEt_A_def = Constraint(model.T, rule=ESS_pEt_A_def_rule)

    # def ESS_pEt_B_def_rule(model, tt):
    #     return model.pEt_B[tt] == model.pBt_B[tt] * dt + Loss_coef_a_B * model.pRt[tt] + Loss_coef_b_B

    # model.ESS_pEt_B_def = Constraint(model.T, rule=ESS_pEt_B_def_rule)

    # def ESS_pEt_C_def_rule(model, tt):
    #     return model.pEt_C[tt] == model.pBt_C[tt] * dt + Loss_coef_a_C * model.pRt[tt] + Loss_coef_b_C

    # model.ESS_pEt_C_def = Constraint(model.T, rule=ESS_pEt_C_def_rule)

    # HESS Total Frequency Regulation Provision (pRt[tt] = sum_{i=A,B} pRt_i[tt])
    # def HESS_pRt_def_rule(model, tt):
    #     return model.pRt[tt] == model.pRt_A[tt] + model.pRt_B[tt] + model.pRt_C[tt]

    # model.HESS_pRt_def = Constraint(model.T, rule=HESS_pRt_def_rule)
    def Thermal_balance_rule(model, tt):
        return (
            qGen[tt] + model.qImp[tt] ==
            model.qB_A[tt] + qDemand[tt] + model.qExp[tt]
        )
    model.Thermal_balance = Constraint(model.T, rule=Thermal_balance_rule)

    # # ESS Power Limit (RegUp) (pBt_i[tt] + pRt_i[tt] <= P_i)
    # def ESS_P_up_A_limit_rule(model, tt):
    #     return model.pBt_A[tt] + model.pRt_A[tt] <= P_A

    # model.ESS_P_up_A_limit = Constraint(model.T, rule=ESS_P_up_A_limit_rule)

    # def ESS_P_up_B_limit_rule(model, tt):
    #     return model.pBt_B[tt] + model.pRt_B[tt] <= P_B

    # model.ESS_P_up_B_limit = Constraint(model.T, rule=ESS_P_up_B_limit_rule)

    # def ESS_P_up_C_limit_rule(model, tt):
    #     return model.pBt_C[tt] + model.pRt_C[tt] <= P_C

    # model.ESS_P_up_C_limit = Constraint(model.T, rule=ESS_P_up_C_limit_rule)

    # # ESS Power Limit (RegDown) (pBt_i[tt] - pRt_i[tt] >= - P_i)
    # def ESS_P_down_A_limit_rule(model, tt):
    #     return model.pBt_A[tt] - model.pRt_A[tt] >= - P_A

    # model.ESS_P_down_A_limit = Constraint(model.T, rule=ESS_P_down_A_limit_rule)

    # def ESS_P_down_B_limit_rule(model, tt):
    #     return model.pBt_B[tt] - model.pRt_B[tt] >= - P_B

    # model.ESS_P_down_B_limit = Constraint(model.T, rule=ESS_P_down_B_limit_rule)

    # def ESS_P_down_C_limit_rule(model, tt):
    #     return model.pBt_C[tt] - model.pRt_C[tt] >= - P_C

    # model.ESS_P_down_C_limit = Constraint(model.T, rule=ESS_P_down_C_limit_rule)

    # ESS Basepoint Definition (Charging and Discharging) (pBt_i[tt] = pBtch_i[tt] - pBtdis_i[tt])
    def ESS_pBt_A_def_rule(model, tt):
        return model.pBt_A[tt] == model.pBtch_A[tt] - model.pBtdis_A[tt]

    model.ESS_pBt_A_def = Constraint(model.T, rule=ESS_pBt_A_def_rule)

    def ESS_pBt_B_def_rule(model, tt):
        return model.pBt_B[tt] == model.pBtch_B[tt] - model.pBtdis_B[tt]

    model.ESS_pBt_B_def = Constraint(model.T, rule=ESS_pBt_B_def_rule)

    def ESS_pBt_C_def_rule(model, tt):
        return model.pBt_C[tt] == model.pBtch_C[tt] - model.pBtdis_C[tt]

    model.ESS_pBt_C_def = Constraint(model.T, rule=ESS_pBt_C_def_rule)

    # Constraints with binary variables

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

    # # Bound on Regulation Reserve Capability (Duration) (pRt_i[tt] <= gamma_i * P_i)
    # def pRt_A_bound_rule(model, tt):
    #     return model.pRt_A[tt] <= gamma_A * P_A

    # model.pRt_A_bound = Constraint(model.T, rule=pRt_A_bound_rule)

    # def pRt_B_bound_rule(model, tt):
    #     return model.pRt_B[tt] <= gamma_B * P_B

    # model.pRt_B_bound = Constraint(model.T, rule=pRt_B_bound_rule)

    # def pRt_C_bound_rule(model, tt):
    #     return model.pRt_C[tt] <= gamma_C * P_C

    # model.pRt_C_bound = Constraint(model.T, rule=pRt_C_bound_rule)

    # Bound on Regulation Reserve and max SoE (0.5(et_i[tt]+et_i[tt-1])+gamma_i*kappa*pRt_i[tt]/zeta_i <= aEmax_i *E_i)
    # def pRt_A_maxSoE_bound_rule(model, tt):
    #     if tt == 1:
    #         return Constraint.Skip
    #     return 0.5 * (model.et_A[tt] + model.et_A[tt - 1]) \
    #         + gamma_A * kappa * model.pRt_A[tt] / zeta <= aEmax_A * E_A

    # model.pRt_A_maxSoE_bound = Constraint(model.T, rule=pRt_A_maxSoE_bound_rule)

    # def pRt_B_maxSoE_bound_rule(model, tt):
    #     if tt == 1:
    #         return Constraint.Skip
    #     return 0.5 * (model.et_B[tt] + model.et_B[tt - 1]) \
    #         + gamma_B * kappa * model.pRt_B[tt] / zeta <= aEmax_B * E_B

    # model.pRt_B_maxSoE_bound = Constraint(model.T, rule=pRt_B_maxSoE_bound_rule)

    # def pRt_C_maxSoE_bound_rule(model, tt):
    #     if tt == 1:
    #         return Constraint.Skip
    #     return 0.5 * (model.et_C[tt] + model.et_C[tt - 1]) \
    #         + gamma_C * kappa * model.pRt_C[tt] / zeta <= aEmax_C * E_C

    # model.pRt_C_maxSoE_bound = Constraint(model.T, rule=pRt_C_maxSoE_bound_rule)

    # Constraint for first hour
    # def pRt_A_maxSoE_1_bound_rule(model):
    #     return 0.5 * (model.et_A[1] + model.et0_A) \
    #         + gamma_A * kappa * model.pRt_A[1] / zeta <= aEmax_A * E_A

    # model.pRt_A_maxSoE_1_bound = Constraint(rule=pRt_A_maxSoE_1_bound_rule)

    # def pRt_B_maxSoE_1_bound_rule(model):
    #     return 0.5 * (model.et_B[1] + model.et0_B) \
    #         + gamma_B * kappa * model.pRt_B[1] / zeta <= aEmax_B * E_B

    # model.pRt_B_maxSoE_1_bound = Constraint(rule=pRt_B_maxSoE_1_bound_rule)

    # def pRt_C_maxSoE_1_bound_rule(model):
    #     return 0.5 * (model.et_C[1] + model.et0_C) \
    #         + gamma_C * kappa * model.pRt_C[1] / zeta <= aEmax_C * E_C

    # model.pRt_C_maxSoE_1_bound = Constraint(rule=pRt_C_maxSoE_1_bound_rule)

    # # Bound on Regulation Reserve and min SoE (0.5(et_i[tt]+et_i[tt-1])-gamma_i*kappa*pRt_i[tt]/zeta_i >= aEmin_i *E_i)
    # def pRt_A_minSoE_bound_rule(model, tt):
    #     if tt == 1:
    #         return Constraint.Skip
    #     return 0.5 * (model.et_A[tt] + model.et_A[tt - 1]) \
    #         - gamma_A * kappa * model.pRt_A[tt] / zeta >= aEmin_A * E_A

    # model.pRt_A_minSoE_bound = Constraint(model.T, rule=pRt_A_minSoE_bound_rule)

    # def pRt_B_minSoE_bound_rule(model, tt):
    #     if tt == 1:
    #         return Constraint.Skip
    #     return 0.5 * (model.et_B[tt] + model.et_B[tt - 1]) \
    #         - gamma_B * kappa * model.pRt_B[tt] / zeta >= aEmin_B * E_B

    # model.pRt_B_minSoE_bound = Constraint(model.T, rule=pRt_B_minSoE_bound_rule)

    # def pRt_C_minSoE_bound_rule(model, tt):
    #     if tt == 1:
    #         return Constraint.Skip
    #     return 0.5 * (model.et_C[tt] + model.et_C[tt - 1]) \
    #         - gamma_C * kappa * model.pRt_C[tt] / zeta >= aEmin_C * E_C

    # model.pRt_C_minSoE_bound = Constraint(model.T, rule=pRt_C_minSoE_bound_rule)

    # # Constraint for first hour
    # def pRt_A_minSoE_1_bound_rule(model):
    #     return 0.5 * (model.et_A[1] + model.et0_A) \
    #         - gamma_A * kappa * model.pRt_A[1] / zeta >= aEmin_A * E_A

    # model.pRt_A_minSoE_1_bound = Constraint(rule=pRt_A_minSoE_1_bound_rule)

    # def pRt_B_minSoE_1_bound_rule(model):
    #     return 0.5 * (model.et_B[1] + model.et0_B) \
    #         - gamma_B * kappa * model.pRt_B[1] / zeta >= aEmin_B * E_B

    # model.pRt_B_minSoE_1_bound = Constraint(rule=pRt_B_minSoE_1_bound_rule)

    # def pRt_C_minSoE_1_bound_rule(model):
    #     return 0.5 * (model.et_C[1] + model.et0_C) \
    #         - gamma_C * kappa * model.pRt_C[1] / zeta >= aEmin_C * E_C

    # model.pRt_C_minSoE_1_bound = Constraint(rule=pRt_C_minSoE_1_bound_rule)


# Constraints for Thermal Storage
    # def ESS_qEt_A_def_rule(model, tt):
    #     return model.qEt_A[tt] == model.qBt_A[tt] * dt + Loss_coef_a_TA * model.qRt[tt] + Loss_coef_b_TA
    # model.ESS_qEt_A_def = Constraint(model.T, rule=ESS_qEt_A_def_rule)
    # def HESS_qEt_def_rule(model, tt):
    #     return model.qEt[tt] == model.qBt_A[tt] *dt  + model.q2p[tt]  * dt + qDemand[tt] + qGen[tt] + model.Thermal_grid[tt]
    # model.HESS_qEt_def = Constraint(model.T, rule=HESS_qEt_def_rule)
    # def ESS_q_up_A_limit_rule(model, tt):
    #     return model.qBt_A[tt] + model.qRt_A[tt] <= P_TA
    # model.ESS_q_up_A_limit = Constraint(model.T, rule=ESS_q_up_A_limit_rule)
    # def ESS_q_down_A_limit_rule(model, tt):
    #     return model.qBt_A[tt] - model.qRt_A[tt] >= - P_TA
    # model.ESS_q_down_A_limit = Constraint(model.T, rule=ESS_q_down_A_limit_rule)
    def ESS_qBt_A_def_rule(model, tt):
        return model.qBt_A[tt] == model.qBtch_A[tt] - model.qBtdis_A[tt]
    model.ESS_qBt_A_def = Constraint(model.T, rule=ESS_qBt_A_def_rule)
    def ESS_qBtch_A_bound_rule(model, tt):
        return model.qBtch_A[tt] <= model.zqch_A[tt] * P_TA
    model.ESS_qBtch_A_bound = Constraint(model.T, rule=ESS_qBtch_A_bound_rule)
    def ESS_qBtdis_A_bound_rule(model, tt):
        return model.qBtdis_A[tt] <= model.zqdis_A[tt] * P_TA
    model.ESS_qBtdis_A_bound = Constraint(model.T, rule=ESS_qBtdis_A_bound_rule)
    # Avoid charging and discharging simultaneously
    def ESS_qzchdis_A_rule(model, tt):
        return model.zqch_A[tt] + model.zqdis_A[tt] <= 1
    model.ESS_qzchdis_A = Constraint(model.T, rule=ESS_qzchdis_A_rule)
    # ESS State of Energy (Charge)
    # Qt_i[tt] = Qt_i[tt-1]*(1-static_i)+eta_ch_i*qBtch_i[tt]*dt-(1/eta_dis_i)*qBtdis_i[tt]*dt-Loss_i*qRt_i[tt]*dt
    # Constraints are separately for hour 1 and hours 2..,T
    
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
    # def qRt_A_bound_rule(model, tt):
    #     return model.qRt_A[tt] <= gamma_TA * P_TA
    # model.qRt_A_bound = Constraint(model.T, rule=qRt_A_bound_rule)
    # def qRt_A_maxSoE_bound_rule(model, tt):
    #     if tt == 1:
    #         return Constraint.Skip
    #     return 0.5 * (model.Qt_A[tt] + model.Qt_A[tt - 1]) \
    #         + gamma_TA * kappa * model.qRt_A[tt] / zeta <= aEmax_TA * E_TA
    # model.qRt_A_maxSoE_bound = Constraint(model.T, rule=qRt_A_maxSoE_bound_rule)
    # def qRt_A_minSoE_bound_rule(model, tt):
    #     if tt == 1:
    #         return Constraint.Skip
    #     return 0.5 * (model.Qt_A[tt] + model.Qt_A[tt - 1]) \
    #         - gamma_TA * kappa * model.qRt_A[tt] / zeta >= aEmin_TA * E_TA
    # model.qRt_A_minSoE_bound = Constraint(model.T, rule=qRt_A_minSoE_bound_rule)
    # def qRt_A_maxSoE_1_bound_rule(model):
    #     return 0.5 * (model.Qt_A[1] + model.Qt0_A) \
    #         + gamma_TA * kappa * model.qRt_A[1] / zeta <= aEmax_TA * E_TA
    # model.qRt_A_maxSoE_1_bound = Constraint(rule=qRt_A_maxSoE_1_bound_rule)
    # def qRt_A_minSoE_1_bound_rule(model):
    #     return 0.5 * (model.Qt_A[1] + model.Qt0_A) \
    #         - gamma_TA * kappa * model.qRt_A[1] / zeta >= aEmin_TA * E_TA
    # model.qRt_A_minSoE_1_bound = Constraint(rule=qRt_A_minSoE_1_bound_rule)

    # Constraints for Thermal Electrical conversion
    # Rankine Cycle

    def Rankine_Cycle_rule(model, tt):
        return model.pBt_RC[tt] == model.q2p[tt] * eta_RC
    model.Rankine_Cycle = Constraint(model.T, rule=Rankine_Cycle_rule)
    # def Rankine_Cycle_pEt_rule(model, tt):
    #     return model.pEt_RC[tt] == model.pBt_RC[tt] * dt
    # model.Rankine_Cycle_pEt = Constraint(model.T, rule=Rankine_Cycle_pEt_rule)
    # def Rankine_Cycle_q2E_rule(model, tt):
    #     return model.q2E[tt] == model.q2p[tt]  * dt
    # model.Rankine_Cycle_q2E = Constraint(model.T, rule=Rankine_Cycle_q2E_rule)
    
    def Elec_Import_Limit_rule(model, tt):
        return model.pImp[tt] <= pImp_max[tt]
    model.Elec_Import_Limit = Constraint(model.T, rule=Elec_Import_Limit_rule)

    def Elec_Export_Limit_rule(model, tt):
        return model.pExp[tt] <= pExp_max[tt]
    model.Elec_Export_Limit = Constraint(model.T, rule=Elec_Export_Limit_rule)

    def Heat_Import_Limit_rule(model, tt):
        return model.qImp[tt] <= qImp_max[tt]
    model.Heat_Import_Limit = Constraint(model.T, rule=Heat_Import_Limit_rule)

    def Heat_Export_Limit_rule(model, tt):
        return model.qExp[tt] <= qExp_max[tt]
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

    

