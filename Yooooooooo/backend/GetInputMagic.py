# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 09:58:18 2025

@author: c01712ey
"""
import atlasopenmagic as atom
import os
import requests
atom.set_release('2025e-13tev-beta')
from DataSetsMagic import validSkims, DIDS_ALL


# keys_input = ['Data 2to4lep', 'Signal $Z→ee$', 'Signal $Z→μμ$']
# string_codes_input = ['Data', 'Zee', 'Zmumu']

# dict_input = {
#      'Data 2to4lep' : 'Data',
#      'Signal $Z→ee$' : 'Zee',
#      'Signal $Z→μμ$' : 'Zmumu'
# }

def get_samples_magic(skim_input, dict_input):

    if skim_input not in DIDS_ALL:
        raise ValueError(f"'{skim_input}' is not a valid skim_input. Valid options are: {list(DIDS_ALL.keys())}")

    dataSets = DIDS_ALL[skim_input]
        
    samples_defs = {}
    
    if dict_input: # Input dict not empty

        for key, string_codes in dict_input.items():

            if not isinstance(string_codes, str):
                raise TypeError('The value of the input dict must be a str.')
            
            if 'Data' in string_codes:
                samples_defs[key] = {'dids': ["data"]} 
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
                        raise ValueError(f'The string code: {i} was not found.')

                samples_defs[key] = {'dids': dataset_id_list}
                
    samples = atom.build_dataset(samples_defs, skim=skim_input, protocol='https')
    # Download root files if not found
    filepath_dict = validate_files(samples, skim_input)
    
    return filepath_dict

# def get_samples_magic(skim_input, keys_input, string_codes_input):

#     if skim_input not in DIDS_ALL:
#         raise ValueError(f"'{skim_input}' is not a valid skim_input. Valid options are: {list(DIDS_ALL.keys())}")

#     dataSets = DIDS_ALL[skim_input]
    
#     # Check if all input lists are the same length
#     if not (len(keys_input) == len(string_codes_input)):
#         raise ValueError("Input lists 'keys_input' and 'string_codes_input' must have the same length.")
        
#     samples_defs = {}
    
#     if keys_input: # Input lists are not empty

#         for key, string_codes in zip(keys_input, string_codes_input):
            
#             if 'Data' in string_codes:
#                 samples_defs[key] = {'dids': ["data"]} 
#             else:    
#                 # Split the input string that uses '+' to combine the string code of
#                 # different phyiscs processes
#                 # Strip / remove any white spaces
#                 physics_processes = [string_code.strip() for string_code in string_codes.split('+')] 
#                 # string_codes = 'H+ZZllll+ttbar'
#                 # e.g. physics_process = ['H','ZZllll', 'ttbar']

#                 dataset_id_list = [] # Hold the dataset ids

#                 for i in physics_processes:
#                     if i in dataSets:    
#                         dataset_id_list.extend(dataSets[i])
#                     else:
#                         raise KeyError(f'The string code: {i} not found.')

#                 samples_defs[key] = {'dids': dataset_id_list}
                
#     samples = atom.build_dataset(samples_defs, skim=skim_input, protocol='https')
#     filepath_dict = validate_files(samples, skim_input)
    
#     return filepath_dict

def validate_files(samples, skim):
    sample_filepath = {}
    for key, value in samples.items():
        file_path_list = []
        for val in value['list']: 
            fileString = val.split("/")[-1] # file name to open

            # Validate / Download to the correct folder
            if 'mc' in fileString:
                folder = f'backend/datasets/{skim}/MC'
            else:
                folder = f'backend/datasets/{skim}/Data'

            os.makedirs(folder, exist_ok=True)
            
            file_path = f'{folder}/{fileString}'
            
            # Download the file, use a local copy
            if os.path.exists(file_path):
                print(f"File {fileString} already exists in {folder}. Skipping download.")
            else:
                print(f"Downloading {fileString} to {folder} ...")
                with requests.get(val, stream=True) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            file_path_list.append(file_path)
        sample_filepath[key] = file_path_list
    return sample_filepath
