# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:52:56 2025

@author: c01712ey
"""

weight_variables = ["filteff", "kfac", "xsec", "mcWeight", "ScaleFactor_PILEUP", 
                    "ScaleFactor_ELE", "ScaleFactor_MUON", "ScaleFactor_LepTRIGGER"]

# Calculate the total weight for an event by multiplying all the important weights
def calculate_weight(events, luminosity):
    total_weight = luminosity * 1000 / events["sum_of_weights"] # * 1000 to go from fb-1 to pb-1
    for variable in weight_variables:
        total_weight = total_weight * abs(events[variable])
    return total_weight