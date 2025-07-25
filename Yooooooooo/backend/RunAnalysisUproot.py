# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:54:49 2025

@author: c01712ey
"""
import os
import re
import uproot
import time
import datetime
from zoneinfo import ZoneInfo
import pickle
import numpy as np
import awkward as ak # for handling complex and nested data structures efficiently
from EventWeights import weight_variables, calculate_weight
from ValidateVariables import validate_variables
from PklReaderWriter import pkl_writer, pkl_reader


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
def calc_sum_of_weights(data):
    if 'totalWeight' in data.fields: # Number of events is given by the weighted count for MC
        return round(ak.sum(data['totalWeight']), 3) # Sum of weights
    else: # The real data
        return len(data)

def process_sample(fraction, luminosity, cut_function, sample_key, sample_value,
                    read_variables, save_variables,
                       write_txt, txt_filename):

    if 'Data' not in sample_key: # Only calculates weights if the data is MC
        if 'totalWeight' not in save_variables:
            save_variables.append('totalWeight' )
    else: # For real data, no need to calculate and save 'totalWeight'
        # Remove the variable 'totalWeight' from the save_variables list
        save_variables = [var for var in save_variables if var != 'totalWeight']
    
     # Initialise the number of events before and after selection cut
    tot_num_events_before = 0
    tot_num_events_after = 0
    # Initialise the total time taken for the analysis
    tot_time_elapsed = 0
    
    # Define empty list to hold data for different filestrings but same sample
    frames = [] 
    # Loop over each file
    for filestring in sample_value: 
        
        print("\t" + filestring + ":") 

        # Open file
        tree = uproot.open(filestring + ": analysis")

        sample_data = [] # Store data for all chunks in this filestring

        # Start the clock
        time_start = time.time()

        # Loop over data in the tree - each data is a dictionary of Awkward Arrays
        for data in tree.iterate(read_variables, # Read these variables
                                 library="ak", # Return data as awkward arrays
                                 # Process up to a fraction of total number of events
                                 entry_stop=tree.num_entries * fraction):

            # Number of events in this chunk
            number_of_events_before = len(data) 
            tot_num_events_before += number_of_events_before

            # Apply selection cut
            data = cut_function(data)
            
            # Store Monte Carlo weights
            if 'Data' not in sample_key:
                data['totalWeight'] = calculate_weight(data, luminosity)
                    
            # Calculate the number of events after selection cuts
            number_of_events_after = calc_sum_of_weights(data)
           
            # Only save the data for variables in save_variables list
            data_kept = {}
            for i in save_variables:
                if i in data.fields:
                    # Save the data in the new dict data_kept
                    data_kept[i] = data[i]
                else: # Print message if the variable is not found in data
                    print(f'Variable "{i}" not found in data - cannot be saved.')
                    
            # Add the data for this chunk to sample_data that holds data
            # for all chunks in this filestring
            sample_data.append(ak.zip(data_kept, depth_limit=1))
            # Cumulative events (for all data) to see the memory limit and
            # write to the txt file
            tot_num_events_after += len(data)
        
            # Time taken to process up to this chunk (This is cumulative for each filestring)
            time_elapsed = time.time() - time_start

            # Print number of events in this chunck before and after selection cut
            # and print the cumulative elapsed time
            print("\t\t nIn: " + str(number_of_events_before) + ","
                  "\t nOut: \t"+ str(number_of_events_after) + ""
                  "\t in " + str(round(time_elapsed, 1)) + "s") # Print the time elapsed
        # End of for loop through chunks of entries in one filestring of the data sample 
        # Stack chunks of data in the same filestring along the first axis and
        # add to 'frames' that holds all the data for one filestring
        frames.append(ak.concatenate(sample_data)) 
    
    # Update with the time taken for each data sample
    tot_time_elapsed += time.time() - time_start

    if write_txt:
        with open(txt_filename, "a") as f:
            f.write(f'Sample: {sample_key}\n')
            f.write(f'Total number of input: {tot_num_events_before}\n')
            f.write(f'Total number of output: {tot_num_events_after}\n')
            f.write(f'Total elapsed time: {round(tot_time_elapsed, 1)}s\n\n')
    return frames
    

def analysis(luminosity, fraction, samples, cut_function, read_variables_input,
             save_variables_input, write_txt=False, txt_filename=None,
             output_pkl_filename=None, write_pickle=True,
             return_output=True):

    if not samples:
        return ak.Array([]) # Empty samples no analysis needed

    # Write to a txt file the time start of analysis, the luminosity and fraction of
    # data used
    if write_txt:
        now = datetime.datetime.now(ZoneInfo("Europe/London"))
        if txt_filename:
            # Make a directory if provided in the input but doesn't exist
            os.makedirs(os.path.dirname(txt_filename), exist_ok=True)
        else:
            strf = now.strftime("%Y%m%d")
            txt_filename = f'txt/analysis{strf}'
            
        # Write to file current time, luminosity and fraction
        with open(txt_filename, "a") as f:
            f.write('----------------------------------------------------------------\n')
            f.write(f'{now.strftime("%Y-%m-%d %H:%M")}\n')
            f.write(f'Luminosity: {luminosity}\nFraction: {fraction}\n')

    # Initiliase a dict to hold the data
    all_data = {}

    # Validate the input read_variables_input to read from the database
    data_read_variables, mc_read_variables = validate_read_variables(samples, read_variables_input)

    # Validate the variables to be saved
    save_variables = validate_save_variables(save_variables_input, data_read_variables)
    # The save_variables_input does not change, so save_variables does not change

    # Write to the txt file what variables will be saved
    if write_txt:
        with open(txt_filename, "a") as f:
            f.write('Data for variables: ' + ', '.join(save_variables) + ' will be saved.\n\n')
    
    # Loop over samples
    for sample_key, sample_value in samples.items():
        
        analysis_results = {} # Initialise a dictionary to hold data for this sample

        if 'Data' in sample_key:
            read_variables = data_read_variables
        else:
            read_variables = mc_read_variables

        # Print which sample is being processed
        print('Processing ' + sample_key + ' samples') 
        
        frames = process_sample(fraction, luminosity, cut_function,
                                    sample_key, sample_value, read_variables,
                                    save_variables,
                                     write_txt, txt_filename)  

        # End of for loop through all filestrings for one data sample
        array = ak.concatenate(frames) # Combine data from all filestrings
        for field in array.fields:
            analysis_results[field] = array[field]

        all_data[sample_key] = ak.zip(analysis_results, depth_limit=1)

    # End of for loop through all data samples
    if write_pickle:
        output_pkl_filename = pkl_writer(all_data, output_pkl_filename)
        if write_txt:
            with open(txt_filename, "a") as f:
                f.write(f'Output pickle filename: {output_pkl_filename}\n')             
    if return_output:
        return all_data

def parse_input_variable(variable_list):
    parsed_variables = np.zeros((0, 3))
    for input_var in variable_list:
        if '[' in input_var and ']' in input_var:
            try:
                base_var = input_var.split('[')[0]
                idx_pos_start = input_var.find('[') + 1 # Index starting position in the str
                idx_pos_end = input_var.find(']')
                index = int(input_var[idx_pos_start : idx_pos_end])
            except Exception as e:
                raise ValueError(f'Invalid input variable format : {input_var}. '
                                 f'Expect "variable" or "variable[int]".\nError: {e}')
        elif '[' in input_var or ']' in input_var:
            raise ValueError('Expect input variable to be "variable" or "variable[int]".'
                             'Perhaps you forgot a "[" or "]"?')
        else:
            base_var = input_var
            index = None
        parsed_variables = np.vstack([parsed_variables,
                                      (input_var, base_var, index)])
    return parsed_variables

def extract_data(sample_data, parsed_variables, selected_data, sample_key):
    for input_var, base_var, index in parsed_variables:
        if base_var not in sample_data.fields:
            raise ValueError(f"Variable '{base_var}' not found. Failed to access '{input_var}'. Available variable(s): "
                           f"{sample_data.fields}")

        variable_data = sample_data[base_var]
        if len(variable_data) == 0:
            break
            
        # Datatype in str
        type_str = str(ak.type(variable_data))
        # Boolean if the array is nested
        is_nested = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)

        if is_nested: # Variable array is nested
            max_num = ak.max(ak.num(variable_data))
            variable_data = ak.pad_none(variable_data, max_num, axis=-1)
            if index != None: # Index given in input
                if index > max_num:
                    raise IndexError(f'Invalid index for input variable "{input_var}". '
                                     f'The maximum number of particle is {max_num}.')
                selected_data[sample_key][input_var] = variable_data[:, index]
            else: # Index not given in input
               for i in range(max_num):
                    new_key = f'{base_var}[{i}]'
                    selected_data[sample_key][new_key] = variable_data[:, i] 
        else: # Variable array is not nested
            if index != None: # Index given in input
                raise ValueError(f'{base_var} is not is_nested. Failed to access "{input_var}".')
            else: # Index not given in input
                selected_data[sample_key][input_var] = variable_data
    
def get_data(variable_list, data=None, read_pkl_filename=None, cut_function=None):
    if not data and not read_pkl_filename:
        raise ValueError('Either a valid data dict or an output filename must be provided as input.')
    if data and read_pkl_filename:
        print('Both data and read_pkl_filename are provided. Data will be extracted from read_pkl_filename if no exception occured.')
    if read_pkl_filename:
        data = pkl_reader(read_pkl_filename)

    if not isinstance(variable_list, (list, tuple)):
        raise TypeError('Expect variable_list to be a list/tuple of str.')

    time_start = time.time()
    selected_data = {}
    
    for sample_key in data:
        selected_data[sample_key] = {}

        sample_data = data[sample_key]
        
        if cut_function is not None:
            try:
                sample_data = cut_function(sample_data)
            except Exception as e:
                print(f'Exception: {e}\nReminder: Input "{cut_function}" must take one argument (data) and return it.')
                raise

        parsed_variables = parse_input_variable(variable_list)
        extract_data(sample_data, parsed_variables, selected_data, sample_key)

        if 'Data' not in sample_key:
            totalWeight = sample_data['totalWeight']
            if len(totalWeight) != 0:
                selected_data[sample_key]['totalWeight'] = totalWeight

        #selected_data[sample_key] = ak.zip(selected_data[sample_key], depth_limit=1)
            
    time_elapsed = time.time() - time_start 
    print("Elapsed time = " + str(round(time_elapsed, 1)) + "s") # Print the time elapsed
    return selected_data

    

