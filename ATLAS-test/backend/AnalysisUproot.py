import atlasopenmagic as atom
import os
import requests
import re
import uproot
import time
import awkward as ak # for handling complex and nested data structures efficiently
import datetime
from zoneinfo import ZoneInfo
from .EventWeights import WEIGHT_VAR, calculate_weight
atom.set_release('2025e-13tev-beta')
from .DataSetsMagic import validSkims, Dids_dict

def validate_files(samples, skim, sample_path):
    print(sample_path)
    filepath_dict = {}
    for key, value in samples.items():
        file_path_list = []
        for val in value['list']: 
            fileString = val.split("/")[-1] # file name to open

            # Validate / Download to the correct folder
            if 'mc' in fileString:
                folder = f'{sample_path}/{skim}/MC'
            else:
                folder = f'{sample_path}/{skim}/Data'

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
        filepath_dict[key] = file_path_list
    return filepath_dict


# string_code_dict = {
#      'Data 2to4lep' : 'Data',
#      'Signal $Z→ee$' : 'Zee',
#      'Signal $Z→μμ$' : 'Zmumu'
# }

def get_samples_magic(skim, string_code_dict, sample_path):
    if skim not in validSkims:
        raise ValueError(f"'{skim}' is not a valid skim. Valid options are: {validSkims}")

    samples_defs = {}
    
    if string_code_dict: # Input dict not empty

        for key, string_codes in string_code_dict.items():

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
                    if i in Dids_dict:    
                        dataset_id_list.extend(Dids_dict[i])
                    else:
                        raise ValueError(f'The string code: {i} was not found.')

                samples_defs[key] = {'dids': dataset_id_list}
                
    samples = atom.build_dataset(samples_defs, skim=skim, protocol='https')
    # Download root files if not found
    filepath_dict = validate_files(samples, skim, sample_path)
    
    return filepath_dict

# Calculate the number of events after selection cut. Return the sum of weights for the MC
def calc_sum_of_weights(data):
    if 'totalWeight' in data.fields: # Number of events is given by the weighted count for MC
        return round(ak.sum(data['totalWeight']), 3) # Sum of weights
    else: # The real data
        return len(data)

def process_sample(fraction, luminosity, skim, cut_function, sample_key, filepath_list, read_variables, save_variables, write_txt, txt_filename, write_parquet, output_directory, return_output):
    
    is_Data = 'Data' in sample_key

    # Initialise the number of events before and after selection cut
    total_num_events_before = 0
    total_num_events_after = 0

    # Define empty list to hold data from different files but same sample
    sample_data = [] 
    chunk_count = 0 # Count number of chunks in all filestrings
    # Loop over each file
    for filestring in filepath_list: 
        
        print(f"\t{filestring} :") 
        
        # Open file
        tree = uproot.open(filestring + ": analysis")
        # for var in read_variables:
        #     if var not in tree.arrays().fields:
        #         raise ValueError(f'"{var}" not found in the sample file. Available fields: {tree.arrays().fields}')
        
        file_data = [] # Store data for all chunks in this filestring
        # Start the clock
        time_start = time.time()

        keep_fields = []
        keep_fields.extend(save_variables)
        # Loop over data in the tree - each data is a dictionary of Awkward Arrays
        for data in tree.iterate(read_variables, # Read these variables
                                 library="ak", # Return data as awkward arrays
                                 # Process up to a fraction of total number of events
                                 entry_stop=tree.num_entries * fraction):

            # Number of events in this chunk
            number_of_events_before = len(data)
            total_num_events_before += number_of_events_before

            # Apply selection cut
            if cut_function is not None:
                try:
                    data = cut_function(data)
                except Exception as e:
                    print(f'cut_function is a function that takes one argument and returns it.\nException occurred : {e}\n')
                    raise

            # Store Monte Carlo weights
            if 'Data' not in sample_key:
                data['totalWeight'] = calculate_weight(data, luminosity, skim)

            for field in data.fields:
                if field not in read_variables:
                    # Computed field
                    keep_fields.append(field)

            # Calculate the number of events after selection cuts
            number_of_events_after = calc_sum_of_weights(data)


            for i in keep_fields:
                if i not in data.fields:
                    print(f'Variable "{i}" not found in data - cannot be written to disk.')
            
            delete_fields = [field for field in data.fields if field not in keep_fields]
           
            if delete_fields:
                for i in delete_fields:
                    data = ak.without_field(data, i)
                    
            if return_output:
                # Add the data for this chunk to file_data
                file_data.append(data)

            # Add all events that passed the selection cut for each chunck
            total_num_events_after += number_of_events_after
            
            # Write to disk
            if write_parquet:
                if len(data) != 0:
                    sample_out_dir = f'{output_directory}/{sample_key}'
                    ak.to_parquet(data, f"{sample_out_dir}/chunk_{chunk_count}.parquet")
                    chunk_count += 1

            # Time taken to process up to this chunk (This is cumulative for each filestring)
            time_elapsed = time.time() - time_start
            
            # Print number of events in this chunck before and after and time elapsed
            print(f"\t\t nIn: {number_of_events_before},"
                  f"\t nOut: \t{number_of_events_after}\t in {round(time_elapsed, 1)} s")
        # End of for loop through chunks of entries in one file of the data sample 
        if return_output:
            # Stack chunks of data in the same file along the first axis and
            # add to 'sample_data' that holds all the data for one filestring
            sample_data.append(ak.concatenate(file_data))
            
    if write_txt:
        with open(txt_filename, "a") as f:
            f.write(f'\nSample: {sample_key}\n')
            f.write(f'Total number of input: {total_num_events_before}\n')
            f.write(f'Total number of output: {total_num_events_after}\n')
    if return_output:
        return sample_data
    else:
        return []

def remove_duplicated_entry(variable_list):
    validated = []
    for var in variable_list:
        if var not in validated:
            validated.append(var)
    return validated
    
# Validate input variables that are to read from the database
# Update the variable list with weight-related variables for MC
def validate_read_variables(samples, read_variables, skim):
    # Check if the real data is present
    has_Data = any('Data' in key for key in samples)
    # Check if MC data is present
    has_mc = any('Data' not in key for key in samples)

    validated = remove_duplicated_entry(read_variables)
    
    # For real data, variables to read from database are simply the input variables
    data_read_variables = validated if has_Data else None
    # For MC, weight variables and sum of weights need to be read too
    mc_read_variables = validated + WEIGHT_VAR[skim] + ["sum_of_weights"] if has_mc else None

    return data_read_variables, mc_read_variables

def analysis_uproot(skim, string_code_dict, luminosity, fraction, read_variables, save_variables,
                    cut_function=None, sample_path='../backend/datasets',
                    write_parquet=False, output_directory=None,
                    write_txt=False, txt_filename=None, return_output=True):
    
    time_start = time.time()

    samples = get_samples_magic(skim, string_code_dict, sample_path)

    if not samples:
        return {} # Empty samples - no analysis needed

    # Write to a txt file the time start of analysis, the luminosity and fraction of the
    # data used
    if write_txt:
        now = datetime.datetime.now(ZoneInfo("Europe/London"))
        if not txt_filename:
            strf = now.strftime("%y%m%d")
            txt_filename = f'txt/analysis_uproot{strf}'
            
        # Make a directory if provided in the input but doesn't exist
        os.makedirs(os.path.dirname(txt_filename), exist_ok=True)    
        # Write to file current time, luminosity and fraction
        with open(txt_filename, "a") as f:
            f.write('----------------------------------------------------------------\n')
            f.write(f'{now.strftime("%Y-%m-%d %H:%M")}\n')
            f.write(f'Luminosity: {luminosity}\nFraction: {fraction}\n')

    # Write to the txt file what variables will be saved and what folder holds the data files
    if write_txt:
        with open(txt_filename, "a") as f:
            f.write(f"Input save_variables: {', '.join(save_variables)} will be saved.\n")

    if write_parquet:
        if not output_directory:
            # Create a folder to save the data
            # Store the information about the luminosity and the fraction in the folder name
            output_directory = f"output/lumi{luminosity}_frac{fraction}_"
            # Use current time to create a unique folder name    
            strf = now.strftime("%y%m%d%H%M") # Set time format
            output_directory += f'{strf}'
        os.makedirs(output_directory, exist_ok=True)
        print(f'\nWrite data to output_directory: {output_directory}\n')
        if write_txt:
            with open(txt_filename, "a") as f:
                f.write(f'Output_directory: {output_directory}\n')

    # Initiliase a dict to hold the data
    all_data = {}
    
    # Remove duplicated entry in read_variables and save_variables
    data_read_variables, mc_read_variables = validate_read_variables(samples, read_variables, skim)
    save_variables = remove_duplicated_entry(save_variables)
    
    # Loop over samples
    for sample_key, filepath_list in samples.items():

        if write_parquet:
            os.makedirs(f'{output_directory}/{sample_key}')
    
        if 'Data' in sample_key:
            read_var = data_read_variables
        else:
            read_var = mc_read_variables
        
        # Print which sample is being processed
        print(f'Processing "{sample_key}" samples') 
        
        sample_data = process_sample(fraction, luminosity, skim, cut_function, sample_key, filepath_list, read_var, save_variables, write_txt, txt_filename, write_parquet, output_directory, return_output)

        if return_output:
            if sample_data:
                if len(sample_data) > 1:
                    sample_data = ak.concatenate(sample_data)
                else:
                    sample_data = sample_data[0]
                    
                all_data[sample_key] = sample_data
    
    # Print how much time this function takes
    time_elapsed = time.time() - time_start
    print(f'\n\nElapsed time: {round(time_elapsed, 1)}s')
    if return_output:
        return all_data


