variable_map_ESS = {
    "HP" : "A", #TODO: check if we can use i instead of _A, _B, _C
    "BAT" : "B",
    "SC": "C",
    "PCM": "A",
    "Ness": "Available_Capacity",
    "PNominal": "Available_Power",
    "SoCi": "Initial_SOC",
    "Price": "EnPrice"
    # "epzProfile_val": "EnPrice and HeatPrice (alternative profile)",

}
variable_map_General = {
    "Price": "EnPrice",
    "Price_gas": "HeatPrice",
    "TypeOfESS": "-",
    "P_baseElectricProfile_val": "-",
    "P_baseThermalProfile_val": "-",
    "nuProfile_val": "eta_RC",
    "pImp_max": "?",
    "pExp_max":"?"
    # "epzProfile_val": "EnPrice and HeatPrice (alternative profile)",

}

Outputs = [
    # i: A, B, C
    ("pEt_i", "SOC_i"),
    ("pBt_i", "P_ess_i"),
    ("pBtch_i", "-"),
    ("pBtdis_i", "-"),
    ("zch_i", "-"),
    ("zdis_i", "-"),
    
    # Thermal ESS A
    #j: A or PCM
    ("qEt_A", "SOC_j"),
    ("qBt_A", "P_ess_j"),
    ("qBtch_A", "-"),
    ("qBtdis_A", "-"),
    ("zqch_A", "-"),
    ("zqdis_A", "-"),
    
    # Rankine Cycle
    ("q2p", "p2rk"),
    
    # Energy imports/exports
    ("pImp", "Pe_grid<0"), #TODO: Check if we need to change this part for operation
    ("pExp", "Pe_grid>0"),
    ("qImp", "Pt_grid<0"),
    ("qExp", "Pt_grid>0"),

    
    # Basepoint power
    ("pBt", "P_ess" )# TODO: Check this
] 