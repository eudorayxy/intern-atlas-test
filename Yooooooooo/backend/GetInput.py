# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 17:44:37 2025

@author: c01712ey
"""
from DataSets import mc_dict, data_dict

def get_samples(keys_input, string_codes_input):
    
    # Check if all input lists are the same length
    if not (len(keys_input) == len(string_codes_input)):
        raise ValueError("Input lists 'keys' and 'string_codes' must have the same length.")
        
    samples = {}
    
    if keys_input: # Input lists are not empty
        for key, string_codes in zip(keys_input, string_codes_input):
            # Split the input string that uses '+' to combine the string code of
            # different phyiscs processes
            # Strip / remove any white spaces
            physics_processes = [string_code.strip() for string_code in string_codes.split('+')] 
            # string_codes = 'H+ZZllll+ttbar'
            # e.g. physics_process = ['H','ZZllll', 'ttbar']

            dataset_file_list = [] # Hold the dataset file
            for i in physics_processes:
                if i in mc_dict:    
                    dataset_file_list.extend([f'./backend/datasets/{fname}' for fname in mc_dict[i]])
                elif i in data_dict:
                    dataset_file_list.extend([f'./backend/datasets/{fname}' for fname in data_dict[i]])
                else:
                    raise KeyError(f'The string code: {i} not found.')

            samples[key] = {
                'files': dataset_file_list
            }
    else:
        print("Input lists are empty.")
            
    return samples


# def get_samples(keys_input, string_codes_input, colors_input):
    
#     # Check if all input lists are the same length
#     if not (len(keys_input) == len(string_codes_input) == len(colors_input)):
#         raise ValueError("Input lists 'keys', 'string_codes', and 'colors' "
#                          "must have the same length.")
        
#     samples = {}
    
#     if keys_input: # Input lists are not empty
#         for key, string_codes, color in zip(keys_input, string_codes_input, colors_input):
#             # Split the input string that uses '+' to combine the string code of
#             # different phyiscs processes
#             # Strip / remove any white spaces
#             physics_processes = [string_code.strip() for string_code in string_codes.split('+')] 
#             # string_codes = 'H+ZZllll+ttbar'
#             # e.g. physics_process = ['H','ZZllll', 'ttbar']

#             dataset_file_list = [] # Hold the dataset file
#             for i in physics_processes:
#                 if i in mc_dict:    
#                     dataset_file_list.extend([f'./backend/datasets/{fname}' for fname in mc_dict[i]])
#                 elif i in data_dict:
#                     dataset_file_list.extend([f'./backend/datasets/{fname}' for fname in data_dict[i]])
#                 else:
#                     raise KeyError(f'The string code: {i} not found.')

#             samples[key] = {
#                 'files': dataset_file_list,
#                 'color': color
#             }
#     else:
#         print("Input lists are empty.")
            
#     return samples