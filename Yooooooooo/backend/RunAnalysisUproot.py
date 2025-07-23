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

def process_filestring(fraction, luminosity, cut_function, sample,
                    read_variables, save_variables, filename,
                    tot_num_events_before, tot_num_events_after, tot_elapsed_time):
    
    sample_key, sample_value = sample
    
    events_passed_selection = 0 # This variable holds the number of events / sum of weights for this sample
    
    # Define empty list to hold data for different filestrings but same sample
    frames = [] 
    # Loop over each file
    for filestring in sample_value: 
        
        print("\t" + filestring + ":") 

        # Open file
        tree = uproot.open(filestring + ": analysis")

        sample_data = [] # Store data for all chunks in this filestring

        # Start the clock
        time_start = time.perf_counter()

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
            
            # Store Monte Carlo weights in the data
            if 'Data' not in sample_key: # Only calculates weights if the data is MC
                data['totalWeight'] = calculate_weight(data, luminosity)
                # Update to save the variable 'totalWeight'
                if 'totalWeight' not in save_variables:
                    save_variables.append('totalWeight')
            else: # For real data, no need to calculate and save 'totalWeight'
                # Remove the variable 'totalWeight' from the save_variables list
                if 'totalWeight' in save_variables:
                    save_variables.remove('totalWeight')
                    
            # Calculate the number of events after selection cuts
            number_of_events_after = calculate_num_events_after(data, filestring)
            # Update with the number of events that passed the selection cut for each chunck
            events_passed_selection += number_of_events_after
           
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
            sample_data.append(ak.Array(data_kept))
            
            # Cumulative events (for all data) to see the memory limit and
            # write to the txt file
            tot_num_events_after += len(data)
            with open(filename, "a") as f:
                f.write(f'{tot_num_events_after}, ')
        
            # Time taken to process up to this chunk (This is cumulative for each filestring)
            time_elapsed = time.perf_counter() - time_start

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
    tot_elapsed_time += time.perf_counter() - time_start
            
    return events_passed_selection, frames, tot_num_events_before, tot_num_events_after, tot_elapsed_time
    

def analysis(luminosity, fraction, samples, cut_function, variables,
             save_variables_input, filename, output_filename=None, save_output=True, return_output=False):

    if not samples:
        return {} # Empty samples no analysis needed

    # Write to a txt file the time start of analysis, the luminosity and fraction of
    # data used
    now = datetime.datetime.now(ZoneInfo("Europe/London"))
    strf = now.strftime("%Y-%m-%d %H:%M") # Set time format
    
    # Make a directory if provided in the input but doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Write to file current time, luminosity and fraction
    with open(filename, "a") as f:
        f.write('\n' + strf + ' UTC\n')
        f.write('Luminosity: ' + str(luminosity) + '\n')
        f.write('Fraction: ' + str(fraction) + '\n')

    # Initialise the number of events before and after selection cut
    tot_num_events_before = 0
    tot_num_events_after = 0
    # Initialise the total time taken for the analysis
    tot_elapsed_time = 0

    # Initiliase a dict to hold the data
    all_data = {}

    # Validate the input variables to read from the database
    data_read_variables, mc_read_variables = validate_read_variables(samples, variables)

    # Validate the variables to be saved
    save_variables = validate_save_variables(save_variables_input, data_read_variables)
    # The save_variables_input does not change, so save_variables does not change

    # Write to the txt file what variables will be saved
    with open(filename, "a") as f:
        f.write('Data for variables: ' + ', '.join(save_variables) + ' will be saved.\n')
    
    # Loop over samples
    for sample_key, sample_value in samples.items():
        
        analysis_results = {} # Initialise a dictionary to hold data for this sample
        with open(filename, "a") as f:
            f.write('Sample: ' + sample_key + '\n')
            f.write('Number of stored events (cumulative) : ')

        if 'Data' in sample_key:
            read_variables = data_read_variables
        else:
            read_variables = mc_read_variables

        # Print which sample is being processed
        print('Processing ' + sample_key + ' samples') 
        
        (events_passed_selection, frames,
         tot_num_events_before,
         tot_num_events_after,
         tot_elapsed_time) = process_filestring(fraction, luminosity, cut_function,
                                                (sample_key, sample_value), read_variables, save_variables,
                                                filename, tot_num_events_before, tot_num_events_after,
                                                tot_elapsed_time)  

        # End of for loop through all filestrings for one data sample
        array = ak.concatenate(frames) # Combine data from all filestrings
        for field in array.fields:
            analysis_results[field] = array[field]
            
        analysis_results['events'] = events_passed_selection

        all_data[sample_key] = analysis_results

        # To investigate the memory limit
        with open(filename, "a") as f:
            f.write('\nTotal number of events before (cumulative): ' + str(tot_num_events_before) + '\n')
            f.write('Total number of events after (cumulative): ' + str(tot_num_events_after) + '\n')
            f.write('Total elapsed time (cumulative) : ' + str(round(tot_elapsed_time, 1)) + '\n')
            f.write('\n')

    # End of for loop through all data samples
    if save_output:
        if not output_filename:
            os.makedirs('output_pkl', exist_ok=True)
            output_filename = f'output_pkl/lumi{luminosity}_frac{fraction}_'
            for var in save_variables:
                output_filename += f'{var}'
            # Use current time to create a unique filename    
            strf = now.strftime("%Y%m%d%H%M") # Set time format
            output_filename += f'{strf}'
        else:
            # Ensure folder exists if output_filename is provided manually
            # Extract the directory component of a path
            output_dir = os.path.dirname(output_filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_filename}.pkl", "wb") as f:
            pickle.dump(all_data, f)
        with open(filename, "a") as f:
            f.write(f'Output pickle filename: {output_filename}\n')

        
    # End of if statement of save_output
                        
    if return_output:
        return all_data


def get_data(variable_list, data=None, read_pkl_filename=None, cut_function=None):
    if not data and not read_pkl_filename:
        raise ValueError('Either a valid data dict or an output filename must be provided as input.')

    if data and read_pkl_filename:
        print('Both data and read_pkl_filename are provided. Data will be extracted from read_pkl_filename if no exception occured.')

    if not isinstance(variable_list, (list, tuple)):
        raise TypeError('Expect variable_list to be a list/tuple of str.')

    if read_pkl_filename:
        try:
            with open(f"{read_pkl_filename}", "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f'Exception occured. Filename {read_pkl_filename} was not found.')
            raise
        except Exception as e:
            print(f'Exception occured : {e}')
            raise

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
    
    time_start = time.time()
    selected_data = {}
    
    for sample_key in data:
        
        selected_data[sample_key] = {}

        if cut_function is not None:
            try:
                data[sample_key] = cut_function(data[sample_key])
            except Exception as e:
                print(f'Input "{cut_function}" must be a function that takes and returns the same parameter. Error: {e}')
                raise
        
        for input_var, base_var, index in parsed_variables:

            if base_var not in data[sample_key]:
                raise ValueError(f"Variable '{base_var}' not found. Failed to access '{input_var}'. Available variable(s): "
                               f"{list(data[sample_key].keys())}")
    
            variable_data = data[sample_key][base_var]
            type_str = str(ak.type(variable_data))
            slicable = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)
            
            if index != None: # Index given in input
                if slicable:
                    try:
                        selected_data[sample_key][input_var] = variable_data[:, index]
                    except Exception as e:
                        print(f'Exception occured : {e}')
                        raise
                else:
                    raise TypeError(f'Error: {base_var} is not slicable. Failed to access "{input_var}".')
            else: # No index given in input
                if slicable:
                    max_num = ak.max(ak.num(variable_data, axis=1))
                    for i in range(max_num):
                        new_key = f'{base_var}[{i}]'
                        selected_data[sample_key][new_key] = variable_data[:, i]
                else:
                    # Scalar variable — no slicing
                    selected_data[sample_key][input_var] = variable_data

        if 'Data' not in sample_key:
            selected_data[sample_key]['totalWeight'] = data[sample_key]['totalWeight']
            
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
    return selected_data

    

