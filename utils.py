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
default_config = {'Name': 'Minimal configuration for HESS optimization',
                'Comments': 'Includes only required parameters from optimization formulation.',
                'Timeseries': {
                    'Input_File': None,
                    'Scaling': 1,
                    'Power_reserve_schedule': None,
                    'Power_reserve_period': 60
                    
                },
                'Execution': {'Real_Time': False,
                                'Minimum_Resolution': 1,
                                'End_Time': 86400,
                                'TimePeriods': 24,
                                'dt': 1},
                'Sizing_Params':{'Price_file': "price_day12.csv"},
                'Electrcial_Storage_Units': [{'Lithium_Battery': {
                                                'Available_Capacity_Per_ESS': 2575.125,
                                                'Available_Power': 450,
                                                'Min_Limit_Energy_Capacity': 0.0001,
                                                'Max_Limit_Energy_Capacity': 40,
                                                'Eta_ch': 0.9,
                                                'Eta_dis': 0.9,
                                                'Static': 0,
                                                'Gamma': 1,
                                                'Initial_SOC': 50,
                                                'Maximum_SOC': 90,
                                                'Minimum_SOC': 10,
                                                'Cost_ESS': 0.04
                                            }},
                
                                            {'Super_Capacitor': {
                                                'Available_Capacity_Per_ESS': 0.5,
                                                'Available_Power': 1,
                                                'Min_Limit_Energy_Capacity': 0.0001,
                                                'Max_Limit_Energy_Capacity': 0.5,
                                                'Eta_ch': 0.92,
                                                'Eta_dis': 0.92,
                                                'Static': 0,
                                                'Gamma': 1,
                                                'Initial_SOC': 50,
                                                'Maximum_SOC': 90,
                                                'Minimum_SOC': 10,
                                                'Cost_ESS': 0.04
                                            }},
                                            {'Pumped_Hydro': {
                                                'Available_Capacity_Per_ESS': 100,
                                                'Available_Power': 10,
                                                'Min_Limit_Energy_Capacity': 0.0001,
                                                'Max_Limit_Energy_Capacity': 100,
                                                'Eta_ch': 0.85,
                                                'Eta_dis': 0.85,
                                                'Static': 0,
                                                'Initial_SOC': 50,
                                                'Maximum_SOC': 90,
                                                'Minimum_SOC': 10,
                                                'Cost_ESS': 0.04
                                            }
                                            }],
                'Thermal_Storage_Units': [{'PCM': {
                                                    'Available_Capacity_Per_ESS': 0.5,
                                                    'Available_Power': 1,
                                                    'Min_Limit_Energy_Capacity': 0.0001,
                                                    'Max_Limit_Energy_Capacity': 0.5,
                                                    'Eta_ch': 0.92,
                                                    'Eta_dis': 0.92,
                                                    'Static': 0,
                                                    'Initial_SOC': 50,
                                                    'Maximum_SOC': 90,
                                                    'Minimum_SOC': 10,
                                                    'Cost_ESS': 0.04
                                                }
                                            }],
                'Thermal_to_Electrical_Converters': {'Ranking_Cycle': {
                                                    'Available_Power': 1,
                                                    'Eta_RC': 0.38,
                                                    'Cost_RC': 0.04
                                                }},
                'Generation': {
                    'PV': {
                        'Cost_operational': 100,
                        'Capital_cost_per_module': 0
                    },
                    'WT': {
                        'Cost_operational': 100,
                        'Capital_cost_per_module': 0
                    },
                    'CSP': {
                        'Cost_operational': 100,
                        'Capital_cost_per_module': 0
                    },
                    'TPS': {
                        'Cost_operational': 100,
                        'Capital_cost_per_module': 0
                    }
                },
                
                'General': {'pLoad': 100,
                            'qDemand': 100,
                            'pGrid_max': 100000,
                            'qGrid_max': 100000,
                            'HeatPrice': None,
                            'EnPrice': None, 
                            'pGen': None,
                            'qGen': None
                }}


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
            for El_storage_dict in config_value:
                unit_name = list(El_storage_dict.keys())[0]
                unit_idx = None
                for i, El_storage_dict_in_updated_config in enumerate(updated_config["Electrcial_Storage_Units"]):
                    if unit_name in El_storage_dict_in_updated_config.keys():
                        unit_idx = i
                for k2, v2 in El_storage_dict.items():
                    for k3, v3 in v2.items():
                        updated_config["Electrcial_Storage_Units"][unit_idx][unit_name][k3] = v3
        elif config_key == "Thermal_Storage_Units":
            for Th_storage_dict in config_value:
                unit_name = list(Th_storage_dict.keys())[0]
                unit_idx = None
                for i, Th_storage_dict_in_updated_config in enumerate(updated_config["Thermal_Storage_Units"]):
                    if unit_name in Th_storage_dict_in_updated_config.keys():
                        unit_idx = i
                for k2, v2 in Th_storage_dict.items():
                    for k3, v3 in v2.items():
                        updated_config["Thermal_Storage_Units"][unit_idx][unit_name][k3] = v3
        elif config_key == "Generation":
            for Gen_dict in config_value:
                unit_name = list(Gen_dict.keys())[0]
                if unit_name in updated_config["Generation"]:
                    for k2, v2 in Gen_dict.items():
                        for k3, v3 in v2.items():
                            updated_config["Generation"][unit_name][k3] = v3
        else:
            updated_config[config_key] = config_value

    # Remove all storage units which are not defined by the user
    user_defined_units = [list(El_storage_dict.keys())[0] for El_storage_dict in user_config["Electrcial_Storage_Units"]]
    updated_config["Electrcial_Storage_Units"] = [El_storage_dict_in_updated_config for El_storage_dict_in_updated_config in updated_config["Electrcial_Storage_Units"] if list(El_storage_dict_in_updated_config.keys())[0] in user_defined_units]

    return updated_config

def Run_Daily_Schedule_Optimization(config, day=0):
    TimePeriods = int(config['Execution']['TimePeriods'])
    dt = float(config['Execution']['dt'])
    pGen = config['General']['pGen']     # Electrical generation
    pGen= [val/1000 for val in pGen]
    qGen =config['General']['qGen']     # Thermal generation
    qGen= [val/1000 for val in qGen]
    pLoad = config['General']['pLoad']    # Electrical load
    pLoad= [val/1000 for val in pLoad]
    qDemand = config['General']['qDemand']  # Thermal load
    qDemand= [val/1000 for val in qDemand]
    EnPrice = [row[1] for row in config['General']['EnPrice'] ]  # Electricity price
    HeatPrice =  config['General']['HeatPrice']   # Heat price
    pGrid_max = config['General']['pGrid_max'] # Max electrical grid exchange
    qGrid_max = config['General']['qGrid_max']  # Max thermal grid exchange
    w= config['General']['w']  #objective function weight
    CI_th= config['General']['CI_th'] # CO2 intensity from thermal import
    CI_el= config['General']['CI_el'] # CO2 intensity from electrical import
    CO2Price= config['General']['CO2Price'] # CO2 price
    print('PGen',pGen)
    print('PLoad',pLoad)
    
    eta_RC = float(
        config['Thermal_to_Electrical_Converters'][0][list(config['Thermal_to_Electrical_Converters'][0].keys())[0]]["Eta_RC"])  # Efficiency of Rankine Cycle

    P_A = float(
        config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Available_Power"]*config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Ness"])  # Power Capacity
    E_A = float(
        config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Available_Capacity_Per_ESS"]*config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Ness"])  # Energy Capacity
    print('P_A',P_A)
    print('E_A',E_A)
    
    eta_ch_A = float(
        config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                          "Eta_dis"])  # efficiency loss (discharging)
    static_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Static"])  # static loss
    aEmin_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                        "Minimum_SOC"])   # minimum State of Energy (0.2 = 20%)
    aEmax_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]][
                        "Maximum_SOC"])   # maximum State of Energy (0.9 = 90%)
    Et0_A= float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Initial_SOC"]) # Initial SoC 
    Cost_ESS_A = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Cost_ESS"])
        # ESS_B
    P_B = float(
        config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Available_Power"]*config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Ness"])  # Power Capacity
    E_B = float(
        config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Available_Capacity_Per_ESS"]*config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Ness"])

    eta_ch_B = float(
        config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                          "Eta_dis"])  # efficiency loss (discharging)
    static_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Static"])
    aEmin_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                        "Minimum_SOC"])   # minimum State of Energy (0.2 = 20%)
    aEmax_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]][
                        "Maximum_SOC"])  # maximum State of Energy (0.9 = 90%)

    Et0_B= float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Initial_SOC"]) # Initial SoC 
    Cost_ESS_B = float(config['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["Cost_ESS"])

    # ESS_C
    P_C = float(
        config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Available_Power"]*config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Ness"])  # Power Capacity
    E_C = float(
        config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Available_Capacity_Per_ESS"]*config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Ness"])
    
    eta_ch_C = float(
        config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                          "Eta_dis"])  # efficiency loss (discharging)
    static_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Static"])
    aEmin_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                        "Minimum_SOC"])   # minimum State of Energy (0.2 = 20%)
    aEmax_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]][
                        "Maximum_SOC"])   # maximum State of Energy (0.9 = 90%)
    Et0_C= float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Initial_SOC"]) # Initial SoC 
    Cost_ESS_C = float(config['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["Cost_ESS"])

    # TESS_A
    P_TA = float(
        config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Available_Power"]*config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Ness"])  # Power Capacity
    E_TA = float(
        config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Available_Capacity_Per_ESS"]*config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Ness"])
    eta_ch_TA = float(
        config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Eta_ch"])  # efficiency loss (charging)
    eta_dis_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
                           "Eta_dis"])  # efficiency loss (discharging)
    static_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Static"])  # static loss
    aEmin_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
                         "Minimum_SOC"])   # minimum State of Energy (0.2 = 20%)
    aEmax_TA = float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]][
        "Maximum_SOC"])   # maximum State of Energy (0.9 = 90%)

    Qt0_A= float(config['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["Initial_SOC"]) # Initial SoC 
    Cost_ESS_TA = float(config['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["Cost_ESS"])
    Cost_RC= float(config['Thermal_to_Electrical_Converters'][0][list(config['Thermal_to_Electrical_Converters'][0].keys())[0]]["Cost_RC"])
    ############################################
    ### Defining the optimization model ###

    # Create model instance
    model = ConcreteModel()

    # Sets
    model.T = RangeSet(1, TimePeriods)  # Set of Time Periods
    # Set of electrical energy storage units
    model.ESS = Set(initialize=['A', 'B', 'C'])
    # model.TES = set(initialize=['A'])
    # ESS A Variables
    model.pBt_A = Var(model.T, domain=Reals)  # Basepoint (power)
    model.pBtch_A = Var(model.T, domain=NonNegativeReals)  # Basepoint charging
    model.pBtdis_A = Var(model.T, domain=NonNegativeReals)  # Basepoint discharging
    model.zch_A = Var(model.T, domain=Boolean)  # Binary indicating charging
    model.zdis_A = Var(model.T, domain=Boolean)  # Binary indicating discharging
    model.et_A = Var(model.T, domain=NonNegativeReals)  # State of Energy (Charge)
    # model.et0_A = Var(domain=NonNegativeReals)  # Initial State of Energy (Charge)

    # ESS B Variables
    model.pBt_B = Var(model.T, domain=Reals)  # Basepoint (power)
    model.pBtch_B = Var(model.T, domain=NonNegativeReals)  # Basepoint charging
    model.pBtdis_B = Var(model.T, domain=NonNegativeReals)  # Basepoint discharging
    model.zch_B = Var(model.T, domain=Boolean)  # Binary indicating charging
    model.zdis_B = Var(model.T, domain=Boolean)  # Binary indicating discharging
    model.et_B = Var(model.T, domain=NonNegativeReals)  # State of Energy (Charge)
    # model.et0_B = Var(domain=NonNegativeReals)  # Initial State of Energy (Charge)

    # ESS C Variables
    model.pBt_C = Var(model.T, domain=Reals)  # Basepoint (power)
    model.pBtch_C = Var(model.T, domain=NonNegativeReals)  # Basepoint charging
    model.pBtdis_C = Var(model.T, domain=NonNegativeReals)  # Basepoint discharging
    model.zch_C = Var(model.T, domain=Boolean)  # Binary indicating charging
    model.zdis_C = Var(model.T, domain=Boolean)  # Binary indicating discharging
    model.et_C = Var(model.T, domain=NonNegativeReals)  # State of Energy (Charge)
    # model.et0_C = Var(domain=NonNegativeReals)  # Initial State of Energy (Charge)
    #RC
    model.pBt_RC = Var(model.T,domain=NonNegativeReals)  # RC output power
    model.q2p = Var(model.T, domain=NonNegativeReals)  # Thermal power covrted to electrical power
    
    # # Thermal ESS A Variables
    model.qBt_A = Var(model.T, domain=Reals)  # Basepoint (power)
    model.qBtch_A = Var(model.T, domain=NonNegativeReals)  # Basepoint charging
    model.qBtdis_A = Var(model.T, domain=NonNegativeReals)  # Basepoint discharging
    model.zqch_A = Var(model.T, domain=Boolean)  # Binary indicating charging
    model.zqdis_A = Var(model.T, domain=Boolean)  # Binary indicating discharging
    model.Qt_A = Var(model.T, domain=NonNegativeReals)  # State of Energy (Charge)
    # model.Qt0_A = Var(domain=NonNegativeReals)  # Initial State of Energy (Charge)
    
    
    # Energy imports/exports
    model.pGrid = Var(model.T, domain=Reals)  # Grid electricity exchange
    model.qGrid = Var(model.T, domain=Reals)  # Grid heat exchange

    # Define import/export variables for CO2 calculation
    model.pGrid_import = Var(model.T, domain=NonNegativeReals)   # Electricity import
    model.pGrid_export = Var(model.T, domain=NonNegativeReals)   # Electricity export
    model.qGrid_import = Var(model.T, domain=NonNegativeReals)   # Heat import
    model.qGrid_export = Var(model.T, domain=NonNegativeReals)   # Heat export

    # Constraints to link pGrid/qGrid with import/export variables
    def pGrid_import_export_rule(model, tt):
        return model.pGrid[tt] == model.pGrid_import[tt] - model.pGrid_export[tt]
    model.pGrid_import_export = Constraint(model.T, rule=pGrid_import_export_rule)

    def qGrid_import_export_rule(model, tt):
        return model.qGrid[tt] == model.qGrid_import[tt] - model.qGrid_export[tt]
    model.qGrid_import_export = Constraint(model.T, rule=qGrid_import_export_rule)
    
    # Basepoint power for each storage unit
    # model.pBt = Var(model.ESS, model.T, domain=Reals)

    #             )
    def obj_rule(model):
        # Objective function: weighted sum of cost and CO2 emissions
        return w * (
            # convention: pGrid = pImport - pExport 
            sum(EnPrice[tt - 1] * model.pGrid[tt] for tt in model.T)
            # qGrid = qImport - qExport
            + sum(HeatPrice * model.qGrid[tt] for tt in model.T)
            + sum(
            Cost_ESS_A * (model.pBtch_A[tt] + model.pBtdis_A[tt])
            + Cost_ESS_B * (model.pBtch_B[tt] + model.pBtdis_B[tt])
            + Cost_ESS_C * (model.pBtch_C[tt] + model.pBtdis_C[tt])
            + Cost_ESS_TA * (model.qBtch_A[tt] + model.qBtdis_A[tt])
            + Cost_RC * model.pBt_RC[tt]
            for tt in model.T
            )
        ) + (1 - w) * CO2Price * (
            # sum(CI_th * model.qGrid_export[tt] for tt in model.T)+
            sum(CI_el * model.pGrid_import[tt] for tt in model.T)+
            # sum(CI_el * model.pGrid_export[tt] for tt in model.T)+
            sum(CI_th * model.qGrid_import[tt] for tt in model.T)
        )
    model.obj = Objective(rule=obj_rule, sense=minimize)

    
    print('EnPrice:', EnPrice,
          'HeatPrice:',HeatPrice,
          'Cost_ESS_A:',Cost_ESS_A,
          'Cost_RC:',Cost_RC,
          'pLoad:', pLoad,
          'qDemand:', qDemand,
          'qGen:',qGen,
          'PNet:', [val - val1 for val, val1 in zip(pGen, pLoad)],
          'qNet:',[val - val1 for val, val1 in zip(qGen, qDemand)])
    # Constraints

    
    def Electricity_balance_rule(model, tt):
        return (
            pGen[tt-1] + model.pGrid[tt] ==
            model.pBt_A[tt] 
            + model.pBt_B[tt] 
            + model.pBt_C[tt] +
            pLoad[tt-1] 
            + model.pBt_RC[tt]
        )
    print('net load:',[pl - pg for pl, pg in zip(pLoad, pGen)])
    model.Electricity_balance = Constraint(model.T, rule=Electricity_balance_rule)
    
    def Thermal_balance_rule(model, tt):
        return (
            qGen[tt-1] + model.qGrid[tt] ==
            model.qBt_A[tt] +
            qDemand[tt-1] + model.q2p[tt]
        )
    model.Thermal_balance = Constraint(model.T, rule=Thermal_balance_rule)
    def qGrid_import_export_rule(model, tt):
        return model.qGrid[tt] == model.qGrid_import[tt] - model.qGrid_export[tt]
    model.qGrid_import_export = Constraint(model.T, rule=qGrid_import_export_rule)
    def pGrid_import_export_rule(model, tt):
        return model.pGrid[tt] == model.pGrid_import[tt] - model.pGrid_export[tt]
    model.pGrid_import_export = Constraint(model.T, rule=pGrid_import_export_rule)

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
    def TES_qBt_A_def_rule(model, tt):
        return model.qBt_A[tt] == model.qBtch_A[tt] - model.qBtdis_A[tt]
    model.TES_qBt_A_def = Constraint(model.T, rule=TES_qBt_A_def_rule)

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
    def TES_zqBtch_A_bound_rule(model, tt):
        return model.qBtch_A[tt] <= model.zqch_A[tt] * P_TA

    model.TES_zqBtch_A_bound = Constraint(model.T, rule=TES_zqBtch_A_bound_rule)
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
    def TES_zqchdis_A_rule(model, tt):
        return model.zqch_A[tt] + model.zqdis_A[tt] <= 1

    model.TES_zqchdis_A = Constraint(model.T, rule=TES_zqchdis_A_rule)
    #### ======================= ####

#     # ESS State of Charge (Energy)
#     # et_i[tt] = et_i[tt-1]*(1-static_i)+eta_ch_i*pBtch_i[tt]*dt-(1/eta_dis_i)*pBtdis_i[tt]*dt-Loss_i*pRt_i[tt]*dt
#     # Constraints are separately for hour 1 and hours 2..,T
    def ESS_SoE_A_def_rule(model, tt):
        if tt == 1:
            return Constraint.Skip
        return model.et_A[tt] == (1 - static_A) * model.et_A[tt - 1] \
            + eta_ch_A * model.pBtch_A[tt-1] * dt \
            - ( 1/eta_dis_A) * model.pBtdis_A[tt-1] * dt \
            # - Loss_A * model.pRt_A[tt]*dt

    model.ESS_SoE_A_def = Constraint(model.T, rule=ESS_SoE_A_def_rule)

    def ESS_SoE_B_def_rule(model, tt):
        if tt == 1:
            return Constraint.Skip
        return model.et_B[tt] == (1 - static_B) * model.et_B[tt - 1] \
            + eta_ch_B * model.pBtch_B[tt-1] * dt \
            - (1/eta_dis_B) * model.pBtdis_B[tt-1] * dt \
            # - Loss_B * model.pRt_B[tt]*dt

    model.ESS_SoE_B_def = Constraint(model.T, rule=ESS_SoE_B_def_rule)

    def ESS_SoE_C_def_rule(model, tt):
        if tt == 1:
            return Constraint.Skip
        return model.et_C[tt] == (1 - static_C) * model.et_C[tt - 1] \
            + eta_ch_C * model.pBtch_C[tt-1] * dt \
            - (1/eta_dis_C) * model.pBtdis_C[tt-1] * dt \
            # - Loss_C * model.pRt_C[tt]*dt

    model.ESS_SoE_C_def = Constraint(model.T, rule=ESS_SoE_C_def_rule)

    # Constraint for first hour
    def ESS_SoE_1_A_def_rule(model):
        return model.et_A[1] == (1 - static_A) * Et0_A*E_A \
            # + eta_ch_A * model.pBtch_A[1] * dt \
            # - ( 1/eta_dis_A) * model.pBtdis_A[1] * dt \
            # # - Loss_A * model.pRt_A[1]*dt

    model.ESS_SoE_1_A_def = Constraint(rule=ESS_SoE_1_A_def_rule)

    def ESS_SoE_1_B_def_rule(model):
        return model.et_B[1] == (1 - static_B) * Et0_B*E_B \
    #         + eta_ch_B * model.pBtch_B[1] * dt \
    #         - ( 1/eta_dis_B) * model.pBtdis_B[1] * dt \
    #         # - Loss_B * model.pRt_B[1]*dt

    model.ESS_SoE_1_B_def = Constraint(rule=ESS_SoE_1_B_def_rule)
    print(Et0_A,Et0_C,Qt0_A,E_A,E_C,E_TA)
    def ESS_SoE_1_C_def_rule(model):
        return model.et_C[1] == (1 - static_C) * Et0_C*E_C  \
            # + eta_ch_C * model.pBtch_C[1] * dt \
            # - (1/eta_dis_C) * model.pBtdis_C[1] * dt \
            # # - Loss_C * model.pRt_C[1]*dt

    model.ESS_SoE_1_C_def = Constraint(rule=ESS_SoE_1_C_def_rule)
    def ESS_qSoE_A_def_rule(model, tt):
        if tt == 1:
            return Constraint.Skip
        return model.Qt_A[tt] == (1 - static_TA) * model.Qt_A[tt - 1] \
            + eta_ch_TA * model.qBtch_A[tt-1] * dt \
            - (1/eta_dis_TA) * model.qBtdis_A[tt-1] * dt \
            # - Loss_TA * model.qRt_A[tt]*dt
    model.ESS_qSoE_A_def = Constraint(model.T, rule=ESS_qSoE_A_def_rule)
    # Constraints for first hour
    # Qt_i[1] = qt0_i*(1-static_i)+eta_ch_i*qBtch_i[1]*dt-(1/eta_dis_i)*qBtdis_i[1]*dt-Loss_i*qRt_i[1]*dt
    def ESS_qSoE_1_A_def_rule(model):
        return model.Qt_A[1] ==  Qt0_A *E_TA
        # return model.Qt_A[1] == (1 - static_TA) * Qt0_A *E_TA\
        #     + eta_ch_TA * model.qBtch_A[1] * dt \
        #     - (1/eta_dis_TA) * model.qBtdis_A[1] * dt \
        #     # - Loss_TA * model.qRt_A[1]*dt
    model.ESS_qSoE_1_A_def = Constraint(rule=ESS_qSoE_1_A_def_rule)

#     # Set Initial SoE equal to Final SoE (avoid discharging the battery at the end of day) (et0_i = et_i[TimePeriods])
#     # TODO: Temporary
    def InitSoE_A_rule(model):
        return Et0_A*E_A == model.et_A[TimePeriods]*(1 - static_A) \
            + eta_ch_A * model.pBtch_A[TimePeriods] * dt \
            - ( 1/eta_dis_A) * model.pBtdis_A[TimePeriods] * dt 
    model.InitSoE_A = Constraint(rule=InitSoE_A_rule)

    def InitSoE_B_rule(model):
        return Et0_B*E_B  == model.et_B[TimePeriods]*(1 - static_B) \
            + eta_ch_B * model.pBtch_B[TimePeriods] * dt \
            - ( 1/eta_dis_B) * model.pBtdis_B[TimePeriods] * dt

    model.InitSoE_B = Constraint(rule=InitSoE_B_rule)

    def InitSoE_C_rule(model):
        return Et0_C*E_C  == model.et_C[TimePeriods]*(1 - static_C) \
            + eta_ch_C * model.pBtch_C[TimePeriods] * dt \
            - ( 1/eta_dis_C) * model.pBtdis_C[TimePeriods] * dt 

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

    
    
    def InitSoE_A_th_rule(model):
        return Qt0_A *E_TA == model.Qt_A[TimePeriods]*(1 - static_TA) \
            + eta_ch_TA * model.qBtch_A[TimePeriods] * dt \
            - ( 1/eta_dis_TA) * model.qBtdis_A[TimePeriods] * dt        
    model.InitSoE_th_A = Constraint(rule=InitSoE_A_th_rule)
    def SoE_A_min_th_limit_rule(model, tt):
        return model.Qt_A[tt] >= aEmin_TA * E_TA
    model.SoE_A_min_th_limit = Constraint(model.T, rule=SoE_A_min_th_limit_rule)
    def SoE_A_max_th_limit_rule(model, tt):
        return model.Qt_A[tt] <= aEmax_TA * E_TA
    model.SoE_A_max_th_limit = Constraint(model.T, rule=SoE_A_max_th_limit_rule)
    def Rankine_Cycle_q2p_rule(model, tt):
        if qGen[tt-1]>=qDemand[tt-1]:
            return model.q2p[tt] == qGen[tt-1]-qDemand[tt-1] - model.qBt_A[tt]
        else:
            return model.q2p[tt] == 0
    model.Rankine_Cycle_q2p = Constraint(model.T, rule=Rankine_Cycle_q2p_rule)
    def Rankine_Cycle_rule(model, tt):
        return model.pBt_RC[tt] == model.q2p[tt] * dt * eta_RC
    model.Rankine_Cycle = Constraint(model.T, rule=Rankine_Cycle_rule)

    
    def Elec_Import_Limit_rule(model, tt):
        return model.pGrid[tt] <= pGrid_max
    model.Elec_Import_Limit = Constraint(model.T, rule=Elec_Import_Limit_rule)

    def Elec_Export_Limit_rule(model, tt):
        return -pGrid_max <=  model.pGrid[tt]
    model.Elec_Export_Limit = Constraint(model.T, rule=Elec_Export_Limit_rule)

    def Heat_Import_Limit_rule(model, tt):
        return model.qGrid[tt] <= qGrid_max
    model.Heat_Import_Limit = Constraint(model.T, rule=Heat_Import_Limit_rule)

    def Heat_Export_Limit_rule(model, tt):
        return -qGrid_max <= model.qGrid[tt]
    model.Heat_Export_Limit = Constraint(model.T, rule=Heat_Export_Limit_rule)

    opt = pyo.SolverFactory("gurobi")
    instance = model.create_instance()
    results = opt.solve(instance)
    opt.options["OutputFlag"] = 1
    
    
    ### Extract results ###
    result_dict = {"Solver_status": str(results.solver.status),
                   "Cost_upper_bound": results.problem.lower_bound,
                   "Cost_lower_bound": results.problem.upper_bound
                   }
    # Extract variable values from the model instance
    pBtch_A_values = [instance.pBtch_A[tt].value for tt in range(1, TimePeriods + 1)]
    pBtdis_A_values = [instance.pBtdis_A[tt].value for tt in range(1, TimePeriods + 1)]
    pBtch_B_values = [instance.pBtch_B[tt].value for tt in range(1, TimePeriods + 1)]
    pBtdis_B_values = [instance.pBtdis_B[tt].value for tt in range(1, TimePeriods + 1)]
    pBtch_C_values = [instance.pBtch_C[tt].value for tt in range(1, TimePeriods + 1)]
    pBtdis_C_values = [instance.pBtdis_C[tt].value for tt in range(1, TimePeriods + 1)]
    qBtch_A_values = [instance.qBtch_A[tt].value for tt in range(1, TimePeriods + 1)]
    qBtdis_A_values = [instance.qBtdis_A[tt].value for tt in range(1, TimePeriods + 1)]
    pBt_RC_values = [instance.pBt_RC[tt].value for tt in range(1, TimePeriods + 1)]

    # Calculate operation cost based on the extracted values
    Cost_operation_ESS = sum(
        Cost_ESS_A * (pBtch_A_values[tt - 1] + pBtdis_A_values[tt - 1]) +
        Cost_ESS_B * (pBtch_B_values[tt - 1] + pBtdis_B_values[tt - 1]) +
        Cost_ESS_C * (pBtch_C_values[tt - 1] + pBtdis_C_values[tt - 1]) +
        Cost_ESS_TA * (qBtch_A_values[tt - 1] + qBtdis_A_values[tt - 1]) +
        Cost_RC * pBt_RC_values[tt - 1]
        for tt in range(1, TimePeriods + 1)
    )
    result_dict["Cost_operation_ESS"] = Cost_operation_ESS

    for dict_key in model.component_map(ctype=pyo.Var).keys():
        result_dict[dict_key] = list(getattr(instance, dict_key).extract_values().values())

    return f"Day_{day}", result_dict
def Result_Update (config, result_dict, day=0):
    pBt_A= result_dict['pBt_A']
    pBt_B= result_dict['pBt_B']
    pBt_C= result_dict['pBt_C']
    q2p= result_dict['q2p']
    pBt_RC= result_dict['pBt_RC']
    qBt_A= result_dict['qBt_A']
    config ['Electrcial_Storage_Units'][0][list(config['Electrcial_Storage_Units'][0].keys())[0]]["pBt"]=  [-val*1000 for val in pBt_A]
    config ['Electrcial_Storage_Units'][1][list(config['Electrcial_Storage_Units'][1].keys())[0]]["pBt"]= [-val*1000 for val in pBt_B]
    config ['Electrcial_Storage_Units'][2][list(config['Electrcial_Storage_Units'][2].keys())[0]]["pBt"]= [-val*1000 for val in pBt_C]
    config ['Thermal_Storage_Units'][0][list(config['Thermal_Storage_Units'][0].keys())[0]]["qBt"]= [-val*1000 for val in qBt_A]
    config ['Thermal_to_Electrical_Converters'][0][list(config['Thermal_to_Electrical_Converters'][0].keys())[0]]["q2p"]= [-val*1000 for val in q2p]
    config['General']['cost_obj'] = result_dict['Cost_upper_bound']
    config['General']['Cost_operation_ESS'] = result_dict['Cost_operation_ESS']
    return config