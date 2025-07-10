# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:54:49 2025

@author: c01712ey
"""
import os
import re
import uproot
import time
import glob
import awkward as ak # for handling complex and nested data structures efficiently
import datetime
from zoneinfo import ZoneInfo
from EventWeights import weight_variables, calculate_weight
from ValidateVariables import validate_variables


def calculate_num_events_after(data, filestring):
    # Calculate the number of events after selection cuts
    if not 'data' in filestring: # Number of events is given by the weighted count for MC
        return round(sum(data['totalWeight']), 3) # Sum of weights
    else: # The real data
        return len(data) 

# def validate_read_variables(sample_key, variables):
#     validated = validate_variables(variables)
#     if 'Data' in sample_key:
#         # Variables to read are simply the input variables, but any repeated entries will be removed
#         return validated
#     # For MC, weight variables and sum of weights need to be read too
#     return validated + weight_variables + ["sum_of_weights"]  

def validate_read_variables(samples, variables):
    has_Data = any('Data' in key for key in samples)
    has_mc = any('Data' not in key for key in samples)

    validated = validate_variables(variables)
    
    data_read_variables = validated if has_Data else None
    mc_read_variables = validated + weight_variables + ["sum_of_weights"] if has_mc else None

    return data_read_variables, mc_read_variables
    
def validate_save_variables(save_variables_input, read_variables):
    if read_variables == None:
        raise ValueError('No variables can be read.')
        
    save_variables = []    
    for i in save_variables_input:
        # Print message to let the user know that the variable needs to be computed
        # User needs to save the variable in the selection_cut function
        if i not in read_variables:
            print(f"Reminder: variable '{i}' will only be saved if it is computed and saved in the selection_cut function.")

        # Skip duplicated variables
        if i not in save_variables:
            save_variables.append(i)
        else:
            print(f"Variable '{i}' skipped: duplicated entry.")
    return save_variables       
    
def process_samples(fraction, luminosity, selection_cut, sample,
                    read_variables, save_variables, output_dir):
    
    sample_key, sample_value = sample
    events_passed_selection = 0

    chunk_count = 0 # Count number of chunks in all filestrings

    # Loop over each file
    for filestring in sample_value['files']: 
        
        print("\t" + filestring + ":") 

        # Open file
        tree = uproot.open(filestring + ": analysis")

        # Start the clock
        time_start = time.perf_counter()

        
        # Loop over data in the tree - each data is a dictionary of Awkward Arrays
        for data in tree.iterate(read_variables, # Read these variables
                                 library="ak", # Return data as awkward arrays
                                 # Process up to a fraction of total number of events
                                 entry_stop=tree.num_entries * fraction):

            # Number of events in this chunk
            number_of_events_before = len(data) 
                    
            data = selection_cut(data)
            
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
            ak.to_parquet(ak.Array(data_kept), f"{output_dir}/chunk_{chunk_count}.parquet")
            chunk_count += 1

            # Calculate the number of events after selection cuts
            number_of_events_after = calculate_num_events_after(data, filestring)

            # Add all events that passed the selection cut for each chunck
            events_passed_selection += number_of_events_after
            
            # Print number of events in this chunck before and after
            print("\t\t nIn: " + str(number_of_events_before) + ","
                  "\t nOut: \t"+ str(number_of_events_after) + "")
            
    return events_passed_selection

def analysis_parquet(luminosity, fraction, samples, selection_cut, variables,
             save_variables_input, filename):

    if not samples:
        return {} # Empty samples - no analysis needed
        
    now = datetime.datetime.now(ZoneInfo("Europe/London"))
    strf = now.strftime("%Y-%m-%d %H:%M")
    with open(filename, "a") as f:
        f.write('\n' + strf + '\n')
        f.write('Luminosity: ' + str(luminosity) + '\n')
        f.write('Fraction: ' + str(fraction) + '\n')
        
    tot_num_events_after = 0
    
    data_read_variables, mc_read_variables = validate_read_variables(samples, variables)

    # Validate the variables to be saved
    save_variables = validate_save_variables(save_variables_input, data_read_variables)
    # The save_variables_input does not change, so save_variables does not change
    
    output_dir = f"output/lumi{luminosity}_frac{fraction}_"

    for variable in save_variables:
        output_dir += f'{variable}'
        
    strf = now.strftime("%Y%m%d%H%M")
    output_dir += f'{strf}'

    # Write variables to be saved and output folder name
    with open(filename, "a") as f:
        f.write('Data for variables: ' + ', '.join(save_variables) + ' will be saved.\n')
        f.write(f'Directory: {output_dir}\n')
    
    # Loop over samples
    for sample_key, sample_value in samples.items():
        
        # Directory to save data for each chunck of data
        output_folder = output_dir + f"/{sample_key}"
        os.makedirs(output_folder)

        if 'Data' in sample_key:
            read_variables = data_read_variables
        else:
            read_variables = mc_read_variables
        

        # Print which sample is being processed
        print('Processing ' + sample_key + ' samples') 

        with open(filename, "a") as f:
            f.write('Sample: ' + sample_key + '\n')

        events_passed_selection = process_samples(fraction, luminosity, selection_cut,
                        (sample_key, sample_value),
                        read_variables, save_variables, output_folder)
        
        tot_num_events_after += events_passed_selection
        
     # To investigate the memory limit
        with open(filename, "a") as f:
            f.write('Total number of stored events (cumulative): ' + str(tot_num_events_after) + '\n')


def concatenate_chunks_new(directory, input_variable):
    sample_data = []

    for file in sorted(glob.glob(f'{directory}/*.parquet')):
        array = ak.from_parquet(file)

        match = re.match(r'([a-zA-Z_]\w*)(?:\[(\d+)\])?$', input_variable)
        if match:
            variable_name = match.group(1)
            index = int(match.group(2)) if match.group(2) else None
        else:
            raise ValueError("Invalid input variable format.")
        
        if variable_name not in array.fields:
            # input_variable can be lep_pt0, lep_pt (default is index 0), lep_pt1, mass
            raise KeyError(f"Variable '{input_variable}' not found. Available varaible(s):"
                           f"{array.fields}")

        data = array[variable_name]
        
        if index != None:
            try:
                sample_data.append(data[:, index])
            except Exception as e:
                raise IndexError(f"Field '{variable_name}' is likely not jagged or index {index} is invalid.") from e
        else:
            # Scalar variable — no slicing
            sample_data.append(data)
    
    return ak.concatenate(sample_data)
    
def get_data_new(samples, output_directory, variable_list):
    
    time_start = time.time()
    
    all_data = {}
    
    for sample_key in samples:
        
        directory = f"{output_directory}/{sample_key}"

        all_data[sample_key] = {}
        for variable in variable_list:
            data = concatenate_chunks_new(directory, variable)
            all_data[sample_key][variable] = data

        if 'Data' not in sample_key:
            all_data[sample_key]['totalWeight'] = concatenate_chunks_new(directory, 'totalWeight')
            
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
                
    return all_data

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
    
def concatenate_chunks(directory, variable):
    sample_data = []

    for file in sorted(glob.glob(f'{directory}/*.parquet')):
        array = ak.from_parquet(file)
        if variable not in array.fields:
            raise KeyError(f"Variable '{variable}' not found. Available varaible(s):"
                           f"{array.fields}")
        sample_data.append(array[variable])
        
    results = ak.concatenate(sample_data)
    
    return results

def get_data_pq(samples, output_directory, variable_list, index=0, scalar_variable=True):
    
    time_start = time.time()
    
    all_data = {}
    
    for sample_key in samples:
        
        directory = f"{output_directory}/{sample_key}"
        
        for variable in variable_list:
            data = concatenate_chunks(directory, variable)
            if not scalar_variable:
                all_data[sample_key] = {f'{variable}[{index}]' : data[:, index]}
            else: 
                all_data[sample_key] = {variable : data}

        if 'Data' not in sample_key:
            all_data[sample_key]['totalWeight'] = concatenate_chunks(directory, 'totalWeight')
            
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
                
    return all_data

