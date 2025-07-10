# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:54:49 2025

@author: c01712ey
"""
import os
import uproot
import time
import glob
import psutil
import awkward as ak # for handling complex and nested data structures efficiently
import datetime
from zoneinfo import ZoneInfo
# Enables parallel execution for faster processing of large datasets 
from concurrent.futures import ProcessPoolExecutor, as_completed
from EventWeights import weight_variables, calculate_weight
from ValidateVariables import validate_variables


def calculate_num_events_after(data, filestring):
    # Calculate the number of events after selection cuts
    if not 'data' in filestring: # Number of events is given by the weighted count for MC
        return round(sum(data['totalWeight']), 3) # Sum of weights
    else: # The real data
        return len(data) 

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
            print(f"Reminder: variable '{i}' will only be saved if it is "
                  "computed and saved in the selection_cut function.")

        # Skip duplicated variables
        if i not in save_variables:
            save_variables.append(i)
        else:
            print(f"Variable '{i}' skipped: duplicated entry.")
    return save_variables     


def should_parallel_process(file_path, max_parallel=4):
    file_size_gb = os.path.getsize(file_path) / 1e9  # in GB
    total_memory_gb = psutil.virtual_memory().total / 1e9

    # Allow parallel only if file is big enough and there is enough memory
    if file_size_gb > 2.0 and total_memory_gb > 8:
        return min(max_parallel, os.cpu_count())
    else:
        return 1  # process serially
    

def analysis_new(luminosity, fraction, variables, save_variables_input,
                 samples, filename, selection_cut):
    if not samples:
        return {} # Empty samples - no analysis needed
        
    # Write date and time, luminosity and fraction to txt file
    now = datetime.datetime.now(ZoneInfo("Europe/London"))
    strf = now.strftime("%Y-%m-%d %H:%M")
    with open(filename, "a") as f:
        f.write('\n' + strf + '\n')
        f.write('Luminosity: ' + str(luminosity) + '\n')
        f.write('Fraction: ' + str(fraction) + '\n')

    data_read_variables, mc_read_variables = validate_read_variables(samples, variables)

    # Validate the variables to be saved
    save_variables = validate_save_variables(save_variables_input, data_read_variables)
    # The save_variables_input does not change, so save_variables does not change
    
    # Initialise the directory name where the files will be saved
    output_dir = f"output/lumi{luminosity}_frac{fraction}_"

    # Update the directory name with variables to be saved
    for variable in save_variables:
        output_dir += f'{variable}'
        
    # Update the directory name with current time and date    
    strf = now.strftime("%Y%m%d%H%M")
    output_dir += f'{strf}'

    # Write variables to be saved and output folder name
    with open(filename, "a") as f:
        f.write('Data for variables: ' + ', '.join(save_variables) + ' will be saved.\n')
        f.write(f'Directory: {output_dir}\n')
    
    # Initialise number of saved events after analysis
    tot_num_events_after = 0 
    
    # Loop over samples
    for sample_key, sample_value in samples.items():
        
        # Directory to save data for each chunck of data
        output_folder = output_dir + f"/{sample_key}"
        # Create directory
        os.makedirs(output_folder, exist_ok=True)

        if 'Data' in sample_key:
            read_variables = data_read_variables
        else:
            read_variables = mc_read_variables

        # Print which sample is being processed
        print('Processing ' + sample_key + ' samples') 

        # Write sample key to the txt file
        with open(filename, "a") as f:
            f.write('Sample: ' + sample_key + '\n')
            
        chunk_count = 0
        for filestring in sample_value['files']:
            
            # Get number of concurrent worker processes
            parallel_splits = should_parallel_process(filestring)
            print(f'parallel_splits = {parallel_splits}')
            
            futures = {}

            print("\t" + filestring + ":") 
            
            with ProcessPoolExecutor() as executor:
                for i in range(parallel_splits):
                    future = executor.submit(process_file, fraction, luminosity, selection_cut,
                                             filestring, sample_key, read_variables,
                                             save_variables, output_folder, i, parallel_splits, chunk_count)
                    futures[future] = i
                    
            for future in as_completed(futures):
                try:
                    events_passed_selection, chunk_count = future.result()
                    tot_num_events_after += events_passed_selection
                except Exception as e:
                    print(f"Error in loop {futures[future]}: {e}")
                    continue
       
                
     # To investigate the memory limit
        with open(filename, "a") as f:
            f.write('Total number of stored events (cumulative): ' + str(tot_num_events_after) + '\n')


# Process one filestring
def process_file(fraction, luminosity, selection_cut, filestring, sample_key,
                 read_variables, save_variables, output_dir, loop, parallel_splits, chunk_count):
    
    events_passed_selection = 0
    
    tree = uproot.open(filestring + ": analysis")

    # Start the clock
    time_start = time.perf_counter()
    
    # Loop over data in the tree - each data is a dictionary of Awkward Arrays
    for data in tree.iterate(read_variables, # Read these variables
                             library="ak", # Return data as awkward arrays
                             entry_start=int(tree.num_entries*fraction/parallel_splits*loop),
                             # Process up to a fraction of total number of events
                             entry_stop=int(tree.num_entries*fraction/parallel_splits*(loop+1))):

        # Number of events in this chunk
        number_of_events_before = len(data) 
                
        # Apply selection cut
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
        ak.to_parquet(ak.Array(data_kept), f"{output_dir}/loop_{loop}chunk_{chunk_count}.parquet")
        chunk_count += 1

        # Calculate the number of events after selection cuts
        number_of_events_after = calculate_num_events_after(data, filestring)

        # Add all events that passed the selection cut for each chunck
        events_passed_selection += number_of_events_after

        time_elapsed = time.perf_counter() - time_start # Time taken to process up to this chunk
        
        
        # Print number of events in this chunck before and after
        print("\t\t nIn: " + str(number_of_events_before) + ","
              "\t nOut: \t"+ str(number_of_events_after) + ""
              "\t in " + str(round(time_elapsed, 1)) + "s") # Print the time elapsed
        
    return events_passed_selection, chunk_count
    
    
# def process_samples(fraction, luminosity, selection_cut, sample,
#                     read_variables, save_variables, output_dir, loop, parallel_splits):
    
#     sample_key, sample_value = sample
#     events_passed_selection = 0
    
#     # Loop over each file
#     for filestring in sample_value['files']: 
        
#         print("\t" + filestring + ":") 

#         # Open file
#         tree = uproot.open(filestring + ": analysis")

#         # Start the clock
#         time_start = time.perf_counter()

#         chunk_count = 0 # Count number of chunks in one filestring
    
        
#         # Loop over data in the tree - each data is a dictionary of Awkward Arrays
#         for data in tree.iterate(read_variables, # Read these variables
#                                  library="ak", # Return data as awkward arrays
#                                  entry_start=int(tree.num_entries*fraction/parallel_splits*loop),
#                                  # Process up to a fraction of total number of events
#                                  entry_stop=int(tree.num_entries*fraction/parallel_splits*(loop+1))):

#             # Number of events in this chunk
#             number_of_events_before = len(data) 
                    
#             data = selection_cut(data)
            
#             is_Data = 'Data' in sample_key

#             # Store Monte Carlo weights in the data
#             if not is_Data: # Only calculates weights if the data is MC
#                 data['totalWeight'] = calculate_weight(data, luminosity)
#                 if 'totalWeight' not in save_variables:
#                     save_variables.append('totalWeight')
#             else:
#                 if 'totalWeight' in save_variables:
#                     save_variables.remove('totalWeight')
                
#             data_kept = {}
#             for i in save_variables:
#                 if i in data.fields:
#                     data_kept[i] = data[i]
#                 else:
#                     print(f'Variable "{i}" not found in data.')
            
            
#             # Write to disk immediately
#             ak.to_parquet(ak.Array(data_kept), f"{output_dir}/chunk_{chunk_count}.parquet")
#             chunk_count += 1

#             # Calculate the number of events after selection cuts
#             number_of_events_after = calculate_num_events_after(data, filestring)

#             # Add all events that passed the selection cut for each chunck
#             events_passed_selection += number_of_events_after

#             time_elapsed = time.perf_counter() - time_start # Time taken to process up to this chunk
            
            
#             # Print number of events in this chunck before and after
#             print("\t\t nIn: " + str(number_of_events_before) + ","
#                   "\t nOut: \t"+ str(number_of_events_after) + ""
#                   "\t in " + str(round(time_elapsed, 1)) + "s") # Print the time elapsed

#     # ----------------------------------------------------------------------------
#     # ----------------------------------------------------------------------------
#     # ----------------------------------------------------------------------------
#     # change below to modify the function passed to submit
#     futures = {}
#     with ProcessPoolExecutor() as executor:
#         for i in range(parallel_splits):
#             future = executor.submit(process_samples, fraction, luminosity, selection_cut,
#                                         (sample_key, sample_value),
#                                         read_variables, save_variables, output_folder, i, parallel_splits)
#             futures[future] = i
#     for future in as_completed(futures):
#         try:
#             events_passed_selection, _ = future.result()
#             tot_num_events_after += events_passed_selection
#         except Exception as e:
#             print(f"Error in loop {futures[future]}: {e}")
#             continue            
#     return events_passed_selection 
    
# def should_parallel_process(file_path, max_parallel=4):
#     file_size_gb = os.path.getsize(file_path) / 1e9  # in GB
#     total_memory_gb = psutil.virtual_memory().total / 1e9

#     # Allow parallel only if file is big enough and there is enough memory
#     if file_size_gb > 2.0 and total_memory_gb > 8:
#         return min(max_parallel, os.cpu_count())
#     else:
#         return 1  # process serially
        
# def analysis_parquet_parallel(luminosity, fraction, samples, selection_cut, variables,
#              save_variables_input, filename):

#     if not samples:
#         return {} # Empty samples - no analysis needed
        
#     now = datetime.datetime.now(ZoneInfo("Europe/London"))
#     strf = now.strftime("%Y-%m-%d %H:%M")
#     with open(filename, "a") as f:
#         f.write('\n' + strf + '\n')
#         f.write('Luminosity: ' + str(luminosity) + '\n')
#         f.write('Fraction: ' + str(fraction) + '\n')
        
#     tot_num_events_after = 0

#     has_Data = any('Data' in key for key in samples)
#     has_mc = any('Data' not in key for key in samples)

#     data_read_variables = validate_read_variables('Data', variables) if has_Data else None
#     mc_read_variables = validate_read_variables('', variables) if has_mc else None

#     # Validate the variables to be saved
#     save_variables = validate_save_variables(save_variables_input, data_read_variables)
#     # The save_variables_input does not change, so save_variables does not change
    
#     with open(filename, "a") as f:
#         f.write('Data for variables: ' + ', '.join(save_variables) + ' will be saved.\n')
#     # The save_variables_input does not change, so save_variables does not change
    
#     output_dir = f"output/lumi{luminosity}_frac{fraction}_"

#     for variable in save_variables:
#         output_dir += f'{variable}'
        
#     strf = now.strftime("%Y%m%d%H%M")
#     output_dir += f'{strf}'
    
#     # Loop over samples
#     for sample_key, sample_value in samples.items():
        
#         # Directory to save data for each chunck of data
#         output_folder = output_dir + f"/{sample_key}"
#         os.makedirs(output_folder, exist_ok=True)

#         if 'Data' in sample_key:
#             read_variables = data_read_variables
#         else:
#             read_variables = mc_read_variables
        

#         # Print which sample is being processed
#         print('Processing ' + sample_key + ' samples') 

#         with open(filename, "a") as f:
#             f.write('Sample: ' + sample_key + '\n')

#         events_passed_selection, elapsed_time_per_sample = process_samples(fraction, luminosity, selection_cut,
#                         (sample_key, sample_value),
#                         read_variables, save_variables, output_folder)

       
                
#      # To investigate the memory limit
#         with open(filename, "a") as f:
#             f.write('Total number of stored events (cumulative): ' + str(tot_num_events_after) + '\n')

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

def get_data_pq(samples, output_directory, variable, index=0, scalar_variable=True):
    
    time_start = time.time()
    
    all_data = {}
    
    for sample_key in samples:
        
        directory = f"{output_directory}/{sample_key}"
        data = concatenate_chunks(directory, variable)

        if not scalar_variable:
            all_data[sample_key] = {variable + f'_{index}' : data[:, index]}
        else: 
            all_data[sample_key] = {variable : data}

        if 'Data' not in sample_key:
            all_data[sample_key]['totalWeight'] = concatenate_chunks(directory, 'totalWeight')
            
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
                
    return all_data
        