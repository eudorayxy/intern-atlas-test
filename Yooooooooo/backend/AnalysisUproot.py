# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:48:15 2025

@author: c01712ey
"""

# import vector
# import awkward as ak # for handling complex and nested data structures efficiently

# Define what variables are important to our analysis
variables = ["lep_n", "lep_pt", "lep_eta", "lep_phi", "lep_e", 
             #"lep_ptvarcone30", "lep_topoetcone20", 
             "lep_type", "lep_charge",
             "lep_isLooseID", "lep_isMediumID", "lep_isTightID",
             "lep_isLooseIso", "lep_isTightIso",
             "trigE", "trigM", "lep_isTrigMatched"]

# Note: first lepton in each event is [:, 0], 2nd lepton is [:, 1] etc
# Functions return bool. True means we should remove the event

# Function to cut on the number of leptons in each event
def cut_lep_n(lep_n, user_input):
    return (lep_n == user_input)

# Function to cut on the lepton type (based on type of first two lep_type)
# lep_type is a number signifying the lepton type (electron (11) or muon (13))
def cut_lep_type(lep_type, user_input):
    sum_lep_type = lep_type[:, 0] + lep_type[:, 1] # Sum of first two leptons' type in the event 
    return (sum_lep_type == user_input)

# Function to cut on the lepton charge (based on charge of first two lep_charge)
def cut_lep_charge(lep_charge, user_input):
    product_lep_charge = lep_charge[:, 0] * lep_charge[:, 1] # Product of first two leptons' charge in the event
    return (product_lep_charge == user_input)

# Function to cut on the lepton transverse momentum
def cut_lep_pt(lep_pt, index, lower_limit):
    return (lep_pt[:, index] > lower_limit) # Accept events with lepton pt higher than lower limit

# Function to cut on the isolation pt (based on first two leptons)
def cut_lep_ptvarcone30(lep_ptvarcone30, upper_limit):
    # Accept events with lep_ptvarcone30 in the range
    return (lep_ptvarcone30[:, 0] < upper_limit) & (lep_ptvarcone30[:, 1] < upper_limit)

# Function to cut on the isolation et (based on first two leptons)
def cut_lep_topoetcone20(lep_topoetcone20, upper_limit):
    # Accept events with lep_topoetcone20 in the range
    return (lep_topoetcone20[:, 0] < upper_limit) & (lep_topoetcone20[:, 1] < upper_limit)

# Function to accept events with at least one lepton is triggering
def cut_trig_match(lep_trigmatch): 
    return ak.sum(lep_trigmatch, axis=1) >= 1

# Function to accept events that has been selected by any of the single electron OR muon triggers
def cut_trig(trigE, trigM):
    return trigE | trigM

# # Function to filter events based on the identification and isolation criteria of all leptons in each event
# def ID_iso_cut(electron_isID, muon_isID, electron_isIso, muon_isIso, lep_type, lep_n): 
#     return (ak.sum(((lep_type == 13) & muon_isID & muon_isIso) | 
#                    ((lep_type == 11) & electron_isID & electron_isIso), axis=1) == lep_n)

# Function to keep events that have all leptons passed the identification and isolation criteria
def ID_iso_cut(lep_isID, lep_isIso, lep_n): 
    return (ak.sum(lep_isID & lep_isIso, axis=1) == lep_n)

# Function to calculate the invariant mass using four momentum (pt, eta, phi, energy)    
def calculate_inv_mass(lep_pt, lep_eta, lep_phi, lep_e):
    four_momentum = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_e})
    invariant_mass = (four_momentum[:, 0] + four_momentum[:, 1]).M

    return invariant_mass

def selection_cut(data):
    # Keep events that pass electron / muon trigger 
    data = data[cut_trig(data['trigE'], data['trigM'])]
    # Keep events where at least one lepton is triggering
    data = data[cut_trig_match(data['lep_isTrigMatched'])] 
    
    print("after trig " + str(len(data)))

#             # Record transverse momenta
#             data['leading_lep_pt'] = data['lep_pt'][:,0]
#             data['sub_leading_lep_pt'] = data['lep_pt'][:,1]
#             data['third_leading_lep_pt'] = data['lep_pt'][:,2]
#             data['last_lep_pt'] = data['lep_pt'][:,3]
            
#             # Cuts on transverse momentum
#             data = data[data['leading_lep_pt'] > 20]
#             data = data[data['sub_leading_lep_pt'] > 15]
#             data = data[data['third_leading_lep_pt'] > 10]
            
    # Lepton cuts
    lep_n = data['lep_n']
    data = data[cut_lep_n(lep_n, 2)]
    print("after lep_n cut " + str(len(data)))
          
    lep_type = data['lep_type']
    data = data[cut_lep_type(lep_type, 26)]
    print("after type cut " + str(len(data)))
    
    lep_charge = data['lep_charge']
    data = data[cut_lep_charge(lep_charge, -1)]
    print("after charge cut " + str(len(data)))
    
    # Invariant Mass
    data['mass'] = calculate_inv_mass(data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_e'])

