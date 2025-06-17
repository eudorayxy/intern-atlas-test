# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:54:49 2025

@author: c01712ey
"""
import uproot
import time
import awkward as ak # for handling complex and nested data structures efficiently
from EventWeights import weight_variables, calculate_weight


def calculate_num_events_after(data, filestring):
    # Calculate the number of events after selection cuts
    if not 'data' in filestring: # Number of events is given by the weighted count for MC
        return round(sum(data['totalWeight']), 3) # Sum of weights in this batch
    else: # The real data
        return len(data) 
    
def validate_keep_variables_input(keep_variables_input, read_variables):
    keep_variables = []
    for i in keep_variables_input:
        if i not in read_variables:
            print(f"Warning: variable '{i}' is not in the TTree. It will only be saved "
                  "if it is computed after the analysis.")
        if i in keep_variables:
            print(f"Variable '{i}' skipped: duplicated entry.")
        else:
            keep_variables.append(i)
    return keep_variables       
    

def analysis(luminosity, fraction, samples, selection_cut, variables, keep_variables_input):
    
    read_variables = variables + weight_variables + ["sum_of_weights"]
    
    keep_variables = validate_keep_variables_input(keep_variables_input, read_variables)
    
    if keep_variables_input: # There is at least one variable to be analysed and saved
    
        # Define empty dictionary to hold awkward arrays
        all_data = {} 

        # Loop over samples
        for sample_key, sample_value in samples.items():
            data_and_colour = {}

            # Print which sample is being processed
            print('Processing ' + sample_key + ' samples') 

            # Define empty list to hold data
            frames = [] 

            # Loop over each file
            for filestring in sample_value['files']: 

                # start the clock
                time_start = time.time()
                print("\t" + filestring + ":") 

                # Open file
                tree = uproot.open(filestring + ": analysis")

                sample_data = [] # Store all data for the sample in this filestring

                # Loop over data in the tree - each data is a dictionary of Awkward Arrays
                for data in tree.iterate(read_variables, # Read these variables
                                         library="ak", # Return data as awkward arrays
                                         # Process up to a fraction of total number of events
                                         entry_stop=tree.num_entries * fraction):

                    # Number of events in this batch
                    number_of_events_before = len(data) 

                    data = selection_cut(data)

                    # Store Monte Carlo weights in the data
                    if 'Data' not in sample_key: # Only calculates weights if the data is MC
                        print("Data not in sample_key")
                        data['totalWeight'] = calculate_weight(data, luminosity)
                        if 'totalWeight' not in keep_variables:
                            keep_variables.append('totalWeight')
                    else:
                        if 'totalWeight' in keep_variables:
                            keep_variables.remove('totalWeight')
                        
                    data_kept = {}
                    for i in keep_variables:
                        if i in data.fields:
                            data_kept[i] = data[i]
                        else:
                            print(f'Variable "{i}" not found in data.')
                    
                    sample_data.append(ak.Array(data_kept))      

                    # Calculate the number of events after selection cuts
                    number_of_events_after = calculate_num_events_after(data,
                                                                        filestring)

                    time_elapsed = time.time() - time_start # time taken to process

                    # Print number of events in this batch before and after
                    print("\t\t nIn: " + str(number_of_events_before) + ","
                          "\t nOut: \t"+ str(number_of_events_after) + ""
                          "\t in " + str(round(time_elapsed, 1)) + "s") # Print the time elapsed

                # End of for loop through batches in one filestring of the data sample 
                # Stack batches of data in the same filestring along the first axis and
                # add to 'frames' that holds all the data for one filestring
                frames.append(ak.concatenate(sample_data)) 

            # End of for loop through all filestrings for one data sample
            data_and_colour['results'] = ak.concatenate(frames)
            data_and_colour['color'] = sample_value['color']

            all_data[sample_key] = data_and_colour
    
        # End of for loop through all data samples
    # End of if keep_variables
    return all_data