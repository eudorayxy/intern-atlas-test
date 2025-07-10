# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:54:49 2025

@author: c01712ey
"""
import uproot
import time
import awkward as ak # for handling complex and nested data structures efficiently
from EventWeights import weight_variables, calculate_weight
from ValidateVariables import validate_variables


def calculate_num_events_after(data, filestring):
    # Calculate the number of events after selection cuts
    if not 'data' in filestring: # Number of events is given by the weighted count for MC
        return round(sum(data['totalWeight']), 3) # Sum of weights
    else: # The real data
        return len(data) 

def validate_read_variables(sample_key, variables):
    validated = validate_variables(variables)
    if 'Data' in sample_key:
        # Variables to read are simply the input variables, but any repeated entries will be removed
        return validated
    # For MC, weight variables and sum of weights need to be read too
    return validated + weight_variables + ["sum_of_weights"]  
    
def validate_save_variables(save_variables_input, read_variables):
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
    

def analysis(luminosity, fraction, samples, selection_cut, variables,
             save_variables_input, filename):

    strf = time.strftime("%D %T", time.localtime(time.time()))
    with open(filename, "a") as f:
        f.write('\n' + strf + ' UTC\n')
        f.write('Luminosity: ' + str(luminosity) + '\n')
        f.write('Fraction: ' + str(fraction) + '\n')
    
    tot_num_events_before = 0
    tot_num_events_after = 0
    tot_elapsed_time = 0
    
    if not samples:
        return {} # Empty samples no analysis needed

    all_data = {}

    has_Data = any('Data' in key for key in samples)
    has_mc = any('Data' not in key for key in samples)

    data_read_variables = validate_read_variables('Data', variables) if has_Data else None
    mc_read_variables = validate_read_variables('', variables) if has_mc else None

    # Validate the variables to be saved
    save_variables = validate_save_variables(save_variables_input, data_read_variables)

    with open(filename, "a") as f:
        f.write('Data for variables: ' + ', '.join(save_variables) + ' will be saved.\n')
    # The save_variables_input does not change, so save_variables does not change
    
    # Loop over samples
    for sample_key, sample_value in samples.items():

        events_passed_selection = 0

        with open(filename, "a") as f:
            f.write('Sample: ' + sample_key + '\n')
            f.write('Number of stored events (cumulative) : ')
        
        is_Data = 'Data' in sample_key
        
        if is_Data:
            read_variables = data_read_variables
        else:
            read_variables = mc_read_variables
        
        analysis_results = {}

        # Print which sample is being processed
        print('Processing ' + sample_key + ' samples') 

        # Define empty list to hold data
        frames = [] 

        # Loop over each file
        for filestring in sample_value['files']: 
            
            print("\t" + filestring + ":") 

            # Open file
            tree = uproot.open(filestring + ": analysis")

            sample_data = [] # Store all data for the sample in this filestring

            # Start the clock
            time_start = time.perf_counter()

            chunk_count = 0 # Count number of chunks in one filestring
            
            # Loop over data in the tree - each data is a dictionary of Awkward Arrays
            for data in tree.iterate(read_variables, # Read these variables
                                     library="ak", # Return data as awkward arrays
                                     # Process up to a fraction of total number of events
                                     entry_stop=tree.num_entries * fraction):

                # Number of events in this chunk
                number_of_events_before = len(data) 
                tot_num_events_before += number_of_events_before
                        
                data = selection_cut(data)

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
                        print(f'Variable "{i}" not found in data - cannot be saved.')
                
                sample_data.append(ak.Array(data_kept)) 

                # Calculate the number of events after selection cuts
                number_of_events_after = calculate_num_events_after(data, filestring)
                
                # Cumulative events (for all data) to see the memory limit
                tot_num_events_after += len(data)

                with open(filename, "a") as f:
                    f.write(f'{tot_num_events_after}, ')

                # Add all events that passed the selection cut for each chunck
                events_passed_selection += number_of_events_after

                time_elapsed = time.perf_counter() - time_start # Time taken to process up to this chunk

                # Print number of events in this chunck before and after
                print("\t\t nIn: " + str(number_of_events_before) + ","
                      "\t nOut: \t"+ str(number_of_events_after) + ""
                      "\t in " + str(round(time_elapsed, 1)) + "s") # Print the time elapsed


            # Time taken for each data sample
            tot_elapsed_time += time.perf_counter() - time_start

            # End of for loop through chunks of entries in one filestring of the data sample 
            # Stack chunks of data in the same filestring along the first axis and
            # add to 'frames' that holds all the data for one filestring
            frames.append(ak.concatenate(sample_data)) 

        # End of for loop through all filestrings for one data sample
        analysis_results['results'] = ak.concatenate(frames)
        analysis_results['events'] = events_passed_selection

        all_data[sample_key] = analysis_results

        # To investigate the memory limit
        with open(filename, "a") as f:
            f.write('\nTotal number of events before (cumulative): ' + str(tot_num_events_before) + '\n')
            f.write('Total number of events after (cumulative): ' + str(tot_num_events_after) + '\n')
            f.write('Total elapsed time (cumulative) : ' + str(round(tot_elapsed_time, 1)) + '\n')
            f.write('\n')
    # End of for loop through all data samples
        
    return all_data

def get_data(all_data, variable, index=0, scalar_variable=True):
    
    time_start = time.time()
    
    selected_data = {}
    
    for key, value in all_data:

        if variable not in value['results'].fields:
            raise KeyError(f"Variable '{variable}' not found in sample '{key}'. Available varaible(s):"
                           f"{value['results'].fields}")

        if scalar_variable:
            selected_data[key] = {variable : value['results'][variable]}
        else: 
            selected_data[key] = {variable + f'_{index}' : value['results'][variable][:, index]}

        if 'Data' not in key:
            selected_data[key]['totalWeight'] = value['results']['totalWeight']
            
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
                
    return selected_data

    

