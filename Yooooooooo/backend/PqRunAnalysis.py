# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:54:49 2025

@author: c01712ey
"""
import os
import re
import uproot
import time
import polars as pl
import awkward as ak # for handling complex and nested data structures efficiently
import datetime
from zoneinfo import ZoneInfo
from EventWeights import weight_variables, calculate_weight
from ValidateVariables import validate_variables

# Validate input variables that are to read from the database
# Update the variable list with weight-related variables for MC
def validate_read_variables(samples, variables):
    # Check if the real data is present
    has_Data = any('Data' in key for key in samples)
    # Check if MC data is present
    has_mc = any('Data' not in key for key in samples)

    validated = validate_variables(variables) # Validate input variables
    
    # For real data, variables to read from database are simply the input variables
    data_read_variables = validated if has_Data else None
    # For MC, weight variables and sum of weights need to be read too
    mc_read_variables = validated + weight_variables + ["sum_of_weights"] if has_mc else None

    return data_read_variables, mc_read_variables

# Validate variables to be saved. Remove any duplicated entry.
# Print message if input variable is not found in the database. The variable will only
# be saved if it is derived and saved in the cut_function
def validate_save_variables(save_variables_input, read_variables):
    if read_variables == None:
        raise ValueError('No valid input variables that can be read.')
        
    save_variables = []    
    for i in save_variables_input:
        # Print message to let the user know that the variable needs to be computed
        if i not in read_variables:
            print(f"Reminder: variable '{i}' will only be saved if it is computed and saved in the cut_function.")

        # Skip duplicated variables
        if i not in save_variables:
            save_variables.append(i)
        else:
            print(f"Variable '{i}' skipped: duplicated entry.")
    return save_variables       

# Calculate the number of events after selection cut. Return the sum of weights for the MC
def calculate_num_events_after(data, filestring):
    if 'mc' in filestring: # Number of events is given by the weighted count for MC
        return round(sum(data['totalWeight']), 3) # Sum of weights
    else: # The real data
        return len(data) 
    
def process_samples(fraction, luminosity, cut_function, sample,
                    read_variables, save_variables, sample_directory):
    
    sample_key, sample_value = sample
    events_passed_selection = 0

    chunk_count = 0 # Count number of chunks in all filestrings

    # Loop over each file
    for filestring in sample_value: 
        
        print("\t" + filestring + ":") 

        # Open file
        tree = uproot.open(filestring + ": analysis")

        # Loop over data in the tree - each data is a dictionary of Awkward Arrays
        for data in tree.iterate(read_variables, # Read these variables
                                 library="ak", # Return data as awkward arrays
                                 # Process up to a fraction of total number of events
                                 entry_stop=tree.num_entries * fraction):

            # Number of events in this chunk
            number_of_events_before = len(data) 
                    
            data = cut_function(data)
            
            is_Data = 'Data' in sample_key

            # Store Monte Carlo weights in the data
            if not is_Data: # Only calculates weights if the data is MC
                data['totalWeight'] = calculate_weight(data, luminosity)
                if 'totalWeight' not in save_variables:
                    save_variables.append('totalWeight')
            else:
                if 'totalWeight' in save_variables:
                    save_variables.remove('totalWeight')
                
            data_kept = {}
            for i in save_variables:
                if i in data.fields:
                    data_kept[i] = data[i]
                else:
                    print(f'Variable "{i}" not found in data.')
            
            # Write to disk immediately
            ak.to_parquet(ak.Array(data_kept), f"{sample_directory}/chunk_{chunk_count}.parquet")
            chunk_count += 1

            # Calculate the number of events after selection cuts
            number_of_events_after = calculate_num_events_after(data, filestring)

            # Add all events that passed the selection cut for each chunck
            events_passed_selection += number_of_events_after
            
            # Print number of events in this chunck before and after
            print("\t\t nIn: " + str(number_of_events_before) + ","
                  "\t nOut: \t"+ str(number_of_events_after) + "")
            
    return events_passed_selection

def analysis_parquet(luminosity, fraction, samples, cut_function, variables,
                     save_variables_input, txt_filename, output_directory=None):

    if not samples:
        return {} # Empty samples - no analysis needed

    # Write to a txt file the time start of analysis, the luminosity and fraction of
    # data used
    now = datetime.datetime.now(ZoneInfo("Europe/London"))
    strf = now.strftime("%Y-%m-%d %H:%M") # Set time format

    # Make a directory if provided in the input but doesn't exist
    os.makedirs(os.path.dirname(txt_filename), exist_ok=True)
    # Write to file current time, luminosity and fraction
    with open(txt_filename, "a") as f:
        f.write('\n' + strf + '\n')
        f.write('Luminosity: ' + str(luminosity) + '\n')
        f.write('Fraction: ' + str(fraction) + '\n')

    # Initialise the number of events after selection cut
    tot_num_events_after = 0
    
    data_read_variables, mc_read_variables = validate_read_variables(samples, variables)

    # Validate the variables to be saved
    save_variables = validate_save_variables(save_variables_input, data_read_variables)
    # The save_variables_input does not change, so save_variables does not change
    
    if not output_directory:
        # Create a folder to save the data
        # Store the information about the luminosity and the fraction in the folder name
        output_directory = f"output/lumi{luminosity}_frac{fraction}_"
        # Store the name of the saved variables in the folder name
        for variable in save_variables:
            output_directory += f'{variable}'
        # Use current time to create a unique folder name    
        strf = now.strftime("%Y%m%d%H%M") # Set time format
        output_directory += f'{strf}'

    # Write to the txt file what variables will be saved and what folder holds the data files
    with open(txt_filename, "a") as f:
        f.write('Data for variables: ' + ', '.join(save_variables) + ' will be saved.\n')
        f.write(f'Directory: {output_directory}\n')
    
    # Loop over samples
    for sample_key, sample_value in samples.items():
        
        # Create a directory to save data for each chunck of data based on the sample key
        sample_directory = output_directory + f"/{sample_key}"
        os.makedirs(sample_directory)

        if 'Data' in sample_key:
            read_variables = data_read_variables
        else:
            read_variables = mc_read_variables
        
        # Print which sample is being processed
        print('Processing ' + sample_key + ' samples') 

        with open(txt_filename, "a") as f:
            f.write('Sample: ' + sample_key + '\n')

        events_passed_selection = process_samples(fraction, luminosity, cut_function,
                        (sample_key, sample_value),
                        read_variables, save_variables, sample_directory)
        
        tot_num_events_after += events_passed_selection
        
     # To investigate the memory limit
        with open(txt_filename, "a") as f:
            f.write('Total number of stored events (cumulative): ' + str(tot_num_events_after) + '\n')


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


# def concatenate_chunks_new(directory, input_variable):

#     sample_data = []
#     sample_dict = {}
    
#     files = sorted(glob.glob(f'{directory}/*.parquet'))

#     if not files:
#         raise FileNotFoundError(f"No .parquet files found in directory: {directory}")

#     match = re.match(r'([a-zA-Z_]\w*)(?:\[(\d+)\])?$', input_variable)
#     if match:
#         variable_name = match.group(1)
#         index = int(match.group(2)) if match.group(2) else None
#     else:
#         raise ValueError("Invalid input variable format.")

#     has_subarrays = False            
    
#     for file in files:
        
#         array = ak.from_parquet(file, columns=variable_name)
    
#         # if variable_name not in array.fields:
#         #     # input_variable can be lep_pt0, lep_pt (default is index 0), lep_pt1, mass
#         #     raise ValueError(f"Invalid variable '{input_variable}' not found. Available variable(s):"
#         #                      f"{array.fields}")

#         try:
#             data = array[variable_name]
#         except Exception as e:
#             print(f'Exception occured while attempting to read the variable {variable_name} from {file} : {e}')

#         if index != None:
#             try:
#                 sample_data.append(data[:, index])
#             except Exception as e:
#                 print(f"Failed to access '{variable_name}[{index}]': {e}")
#                 raise
#         else:
#             type_str = str(ak.type(data))
#             if "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str):
#                 has_subarrays = True
#                 max_num = ak.max(ak.num(data, axis=1))
#                 for i in range(max_num):
#                     new_key = f'{variable_name}[{i}]'
#                     if new_key not in sample_dict:
#                         sample_dict[new_key] = []
#                     sample_dict[new_key].append(data[:, i])
#             else:
#                 sample_data.append(data)
            
#     if has_subarrays:
#         for i in sample_dict:
#             sample_dict[i] = ak.concatenate(sample_dict[i])
#         return sample_dict
#     else:   
#         return ak.concatenate(sample_data)
    
# def get_data_new(samples, output_directory, variable_list):
    
#     time_start = time.time()
    
#     all_data = {}
    
#     for sample_key in samples:
        
#         directory = f"{output_directory}/{sample_key}"

#         all_data[sample_key] = {}

#         # data = concatenate_chunks_new(directory, variable_list)
#         # for key in data_dict:
#         #     all_data[sample_key][key] = data_dict[key]
#         for variable in variable_list:
#             data = concatenate_chunks_new(directory, variable)
#             # data could be [[1,2], [2,3]..] or [1,2,3]
#             if isinstance(data, dict):
#                 for i in data:
#                     all_data[sample_key][i] = data[i]
#             else:
#                 all_data[sample_key][variable] = data

#         if 'Data' not in sample_key:
#             all_data[sample_key]['totalWeight'] = concatenate_chunks_new(directory, 'totalWeight')
            
#     elapsed_time = time.time() - time_start 
#     print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
                
#     return all_data

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
    
# def concatenate_chunks(directory, variable):
#     sample_data = []

#     for file in sorted(glob.glob(f'{directory}/*.parquet')):
#         array = ak.from_parquet(file)
#         if variable not in array.fields:
#             raise KeyError(f"Variable '{variable}' not found. Available varaible(s):"
#                            f"{array.fields}")
#         sample_data.append(array[variable])
        
#     results = ak.concatenate(sample_data)
    
#     return results

# def get_data_pq(samples, output_directory, variable_list, index=0, scalar_variable=True):
    
#     time_start = time.time()
    
#     all_data = {}
    
#     for sample_key in samples:
        
#         directory = f"{output_directory}/{sample_key}"
        
#         for variable in variable_list:
#             data = concatenate_chunks(directory, variable)
#             if not scalar_variable:
#                 all_data[sample_key] = {f'{variable}[{index}]' : data[:, index]}
#             else: 
#                 all_data[sample_key] = {variable : data}

#         if 'Data' not in sample_key:
#             all_data[sample_key]['totalWeight'] = concatenate_chunks(directory, 'totalWeight')
            
#     elapsed_time = time.time() - time_start 
#     print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
                
#     return all_data

