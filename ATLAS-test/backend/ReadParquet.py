import os
import glob
import time
import datetime
from zoneinfo import ZoneInfo
import re
import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict

def get_valid_var(file):
    # Read schema from the parquet file
    parquet_file = pq.ParquetFile(file)
    schema = parquet_file.schema.to_arrow_schema()
    var_list = [field.name for field in schema]
    return var_list

def parse_var(input_var, parsed_variables):
    if '[' in input_var and ']' in input_var:
        try:
            base_var = input_var.split('[')[0]
            idx_pos_start = input_var.find('[') + 1 # Index starting position in the str
            idx_pos_end = input_var.find(']')
            index = int(input_var[idx_pos_start : idx_pos_end])
        except Exception as e:
            print(f'Invalid input variable format : {input_var}. '
                  f'Expect "variable" or "variable[int]".\nError: {e}')
            raise
    elif '[' in input_var or ']' in input_var:
        raise ValueError('Expect input variable to be "variable" or "variable[int]".'
                         'Perhaps you forgot a "[" or "]"?')
    else:
        base_var = input_var
        index = None
    return np.vstack([parsed_variables,
                    (input_var, base_var, index)])

    
def concatenate_chunks(files, parsed_variables, cut_function, write_parquet, sample_out_dir, return_output):

    sample_dict = defaultdict(list)
    chunk_count = 0
    
    for file in files:
        columns_to_read = list(parsed_variables[:, 1])
        # Read schema from the parquet file
        parquet_file = pq.ParquetFile(file)
        all_columns = parquet_file.schema.names
        
        # Add 'totalWeight' if present and not already in the list
        if 'totalWeight' in all_columns and 'totalWeight' not in columns_to_read:
            columns_to_read.append('totalWeight')
            parsed_variables = np.vstack((parsed_variables, ('totalWeight', 'totalWeight', None)))

        chunk_dict = defaultdict(list)
        
        array = ak.from_parquet(file, columns=columns_to_read)

        if cut_function is not None:
            try:
                array = cut_function(array)
            except Exception as e:
                print(f'Exception: {e}\nReminder: Input "{cut_function}" must take one argument and return it.')
                raise
        
        if len(array) == 0:
            continue

        valid_var = get_valid_var(file)
        for field in array.fields:
            if field not in parsed_variables[:, 1]:
                valid_var = valid_var + [field]  

        for input_var, base_var, index in parsed_variables:
            if base_var not in valid_var:
                raise ValueError(f"Variable '{base_var}' not found. Failed to access '{input_var}'. Available variable(s): "
                               f"{valid_var}")
            data = array[base_var]
                
            if write_parquet:
                chunk_dict[base_var].append(data)

            if return_output:
                type_str = str(ak.type(data))
                is_nested = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)
                if index is None:
                    sample_dict[input_var].append(data)
                else:
                    if is_nested: # Variable array is nested
                        num = ak.num(data)
                        max_num = ak.max(num)
                        index = int(index)
                        if index >= max_num:
                            raise IndexError(f'Invalid index for input variable "{input_var}". '
                                             f'Input index should be less than {max_num}.')
                        if not ak.all(num >= index + 1):
                            data = ak.pad_none(data, index + 1, axis=-1)
                        sample_dict[input_var].append(data[:, index])
                    else:
                        raise ValueError(f'{base_var} is not is_nested. Failed to access "{input_var}".')
            # End of if-else statement
        # End of loop through variables
                
            # type_str = str(ak.type(data))
            # is_nested = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)

            # if is_nested: # Variable array is nested
            #     num = ak.num(data)
            #     max_num = ak.max(num)
            #     if index != None: # Index given in input
            #         index = int(index)
            #         if index >= max_num:
            #             raise IndexError(f'Invalid index for input variable "{input_var}". '
            #                              f'Input index should be less than {max_num}.')
            #         if not ak.all(num >= index + 1):
            #             data = ak.pad_none(data, index + 1, axis=-1)
            #         sample_dict[input_var].append(data[:, index])
            #     else: # Index not given in input
            #         if not ak.all(num >= max_num):
            #             data = ak.pad_none(data, max_num, axis=-1)
            #         for i in range(max_num):
            #             new_key = f'{base_var}[{i}]'
            #             sample_dict[new_key].append(data[:, i])
            # else: # Variable array is not nested
            #     if index != None: # Index given in input
            #         raise ValueError(f'{base_var} is not is_nested. Failed to access "{input_var}".')
            #     else: # Index not given in input
            #         sample_dict[input_var].append(data)
            # End of if-else statement
        # End of loop through variables
        if chunk_dict:
            chunk_dict = {key : ak.concatenate(value) for key, value in chunk_dict.items()}
            ak.to_parquet(ak.zip(chunk_dict, depth_limit=1), f'{sample_out_dir}/chunk{chunk_count}.parquet')
            chunk_count += 1
            
    # End of loop through all parquet files
    if return_output:
        if len(files) > 1:
            sample_dict = {key : ak.concatenate(value) for key, value in sample_dict.items()}
        if sample_dict:
            sample_dict = ak.zip(sample_dict, depth_limit=1)
        
        return sample_dict
    else:
        return None

def read_parquet(read_directory, subdirectory_names, read_variables, cut_function=None, write_parquet=False, output_directory=None, return_output=True):
    
    time_start = time.time()
    
    all_data = {}

    if isinstance(read_variables, str):
        raise TypeError(f'read_variables must be a list. Got a string: {read_variables}')
    
    parsed_variables = np.zeros((0, 3))
    for input_var in read_variables:
        parsed_variables = parse_var(input_var, parsed_variables)

    if write_parquet:
        if not output_directory:
            # Use current time to create a unique folder name
            now = datetime.datetime.now(ZoneInfo("Europe/London"))
            strf = now.strftime("%Y%m%d%H%M") # Set time format
            output_directory = f'output/read_parquet{strf}'
            print(f'Created output_directory: {output_directory}')
            
    
    for sample_key in subdirectory_names:
        
        sample_directory = f"{read_directory}/{sample_key}"
        if not os.path.isdir(sample_directory):
            raise FileNotFoundError(f"Folder '{sample_directory}' does not exist")

        base_var_list = parsed_variables[:, 1]
       
        if write_parquet:
            sample_out_dir = f'{output_directory}/{sample_key}'
            os.makedirs(sample_out_dir)
        else:
            sample_out_dir = None

        files = sorted(glob.glob(f'{sample_directory}/*.parquet'))

        if not files:
            print(f"No parquet files found in directory: {sample_directory}") 

        all_data[sample_key] = concatenate_chunks(files, parsed_variables, cut_function, write_parquet, sample_out_dir, return_output)
        
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

    if return_output:
        return all_data