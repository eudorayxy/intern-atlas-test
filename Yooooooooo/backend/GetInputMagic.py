# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 09:58:18 2025

@author: c01712ey
"""
import atlasopenmagic as atom
atom.set_release('2025e-13tev-beta')
from DataSetsMagic import dataSets, validSkims


def get_samples_magic(skim_input, keys_input, string_codes_input, colors_input):
    
    if skim_input not in validSkims:
        raise ValueError(f"'{skim_input}' is not a valid skim. Valid options are: {validSkims}")
    
    # Check if all input lists are the same length
    if not (len(keys_input) == len(string_codes_input) == len(colors_input)):
        raise ValueError("Input lists 'keys', 'string_codes', and 'colors' "
                         "must have the same length.")
        
    data_samples = {}  # Dictionary to hold Data sample URLs
    mc_samples = {}    # Dictionary to hold MC sample URLs
    mc_defs = {}       # Dictionary to define MC datasets (keys, process codes, colors)
    
    if keys_input: # Input lists are not empty

        for key, string_codes, color in zip(keys_input, string_codes_input, colors_input):
            
            if 'Data' in key:
                # Build a dict of Data samples URLS
                data_samples = atom.build_data_dataset(skim_input, protocol='https')
                data_samples[key] = data_samples.pop('Data') # Can customise the key to the data sample
            else:    
                # Split the input string that uses '+' to combine the string code of
                # different phyiscs processes
                # Strip / remove any white spaces
                physics_processes = [string_code.strip() for string_code in string_codes.split('+')] 
                # string_codes = 'H+ZZllll+ttbar'
                # e.g. physics_process = ['H','ZZllll', 'ttbar']

                dataset_id_list = [] # Hold the dataset ids

                for i in physics_processes:
                    if i in dataSets:    
                        dataset_id_list.extend(dataSets[i])
                    else:
                        raise KeyError(f'The string code: {i} not found.')

                mc_defs[key] = {
                    'dids': dataset_id_list,
                    'color': color
                }
    
        if mc_defs: # mc samples present
            # Build a dict of MC samples URLs
            mc_samples = atom.build_mc_dataset(mc_defs, skim=skim_input, protocol='https')
    
    # Combine dict of MC and data samples
    samples = {**data_samples, **mc_samples} # ** unpacks dictionary
    
    return samples


