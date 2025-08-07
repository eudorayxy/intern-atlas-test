import os
import glob
import time
import datetime
from zoneinfo import ZoneInfo
import re
import awkward as ak
import numpy as np
import pyarrow.parquet as pq

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

def concatenate_chunks(files, parsed_variables, cut_function, write_parquet, sample_out_dir, max_num_rows, return_output):

    sample_data_list = []
    chunk_count = 0
    num_rows_read = 0
    
    for file in files:
        columns_to_read = list(parsed_variables[:, 1])
        # Read schema from the parquet file
        parquet_file = pq.ParquetFile(file)
        all_columns = parquet_file.schema.names
        
        # Add 'totalWeight' if present and not already in the list
        if 'totalWeight' in all_columns and 'totalWeight' not in columns_to_read:
            columns_to_read.append('totalWeight')
            parsed_variables = np.vstack((parsed_variables, ('totalWeight', 'totalWeight', None)))
        elif 'totalWeight' not in all_columns and 'totalWeight' in columns_to_read:
            columns_to_read = [var for var in columns_to_read if var != 'totalWeight']
            rows_to_delete = np.any(parsed_variables == 'totalWeight', axis=1)
            parsed_variables = parsed_variables[~rows_to_delete]

        
        chunk_data_list = []
        num_row_groups = parquet_file.num_row_groups
       
        for group in range(num_row_groups):
            if num_rows_read >= max_num_rows:
                break
            arr = ak.from_parquet(file, columns=parsed_variables[:, 1], row_groups={group})
            if (num_rows_read + len(arr)) > max_num_rows:
                rows_to_read = max_num_rows - num_rows_read
                arr = arr[:rows_to_read]
                num_rows_read = max_num_rows
            else:
                num_rows_read += len(arr)
    
            if cut_function is not None:
                try:
                    arr = cut_function(arr)
                except Exception as e:
                    print(f'cut_function is a function that takes one argument and returns it.\nException occurred : {e}\n')
                    raise
                    
            if len(arr) == 0:
                continue

            if write_parquet:
                chunk_data_list.append(arr)

            for field in arr.fields:
                if field not in parsed_variables[:, 1]:
                    parsed_variables = np.vstack([parsed_variables,
                    (field, field, None)])
                        
            for input_var, base_var, index in parsed_variables:
                if base_var not in arr.fields:
                        raise ValueError(f"Variable '{base_var}' not found. Failed to access '{input_var}'. Available variable(s): "
                                   f"{arr.fields}")
                if base_var != input_var:
                    data = arr[base_var]
                    type_str = str(ak.type(data))
                    is_nested = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)
            
                    if is_nested: # Variable array is nested
                        num = ak.num(data)
                        max_num = ak.max(num)
                        index = int(index)
                        if index >= max_num:
                            raise IndexError(f'Invalid index for input variable "{input_var}". '
                                             f'Input index should be less than {max_num}.')
                        if not ak.all(num >= index + 1):
                            data = ak.pad_none(data, index + 1, axis=-1)
                        arr = ak.with_field(arr, data[:, index], input_var)
                    else:
                        raise ValueError(f'{base_var} is not is_nested. Failed to access "{input_var}".')
            # End of loop through variables
            unwanted_var = [field for field in arr.fields if field not in parsed_variables[:, 0]]
            if unwanted_var:
                for i in unwanted_var:
                    arr = ak.without_field(arr, i)

            if return_output:
                sample_data_list.append(arr)
        # End of loop through row groups in one file
        if chunk_data_list:
            if len(chunk_data_list) > 1:
                chunk_data_ak = ak.concatenate(chunk_data_list)
            else:
                chunk_data_ak = chunk_data_list[0]
            
            ak.to_parquet(chunk_data_ak, f'{sample_out_dir}/chunk{chunk_count}.parquet')
            chunk_count += 1
            
        if num_rows_read >= max_num_rows:
            break
    # End of loop through all parquet files
    if return_output:
        if sample_data_list:
            if len(sample_data_list) > 1:
                return ak.concatenate(sample_data_list)
            else:
                return sample_data_list[0]
    else:
        return None


def read_parquet(read_directory, subdirectory_names, fraction, parsed_variables, cut_function, write_parquet, output_directory, return_output):
    all_data = {}
    if subdirectory_names is None:
        subdirectory_names = [name for name in os.listdir(read_directory) if os.path.isdir(os.path.join(read_directory, name))]
        
    for sample_key in subdirectory_names:
        sample_directory = f"{read_directory}/{sample_key}"
        if not os.path.isdir(sample_directory):
            raise FileNotFoundError(f"Folder '{sample_directory}' does not exist")
            
        files = sorted(glob.glob(f'{sample_directory}/*.parquet'))
        if not files:
            print(f"No parquet files found in directory: {sample_directory}") 

        tot_num_rows = 0
        for file in files:
            tot_num_rows += pq.ParquetFile(file).metadata.num_rows
        max_num_rows = round(tot_num_rows * fraction)

        sample_key = f'{sample_key} x{fraction}'
        sample_key = sample_key.replace('.', '_')
        
        if write_parquet:
            sample_out_dir = f'{output_directory}/{sample_key}'
            os.makedirs(sample_out_dir)
        else:
            sample_out_dir = None

        all_data[sample_key] = concatenate_chunks(files, parsed_variables, cut_function, write_parquet, sample_out_dir, max_num_rows, return_output)
        
    if return_output:
        return all_data
        

def analysis_read_parquet(read_variables, read_directory, subdirectory_names=None, fraction=1, cut_function=None, write_parquet=False, output_directory=None, return_output=True):

    if isinstance(read_variables, str):
        raise TypeError(f'read_variables must be a list. Got a string: {read_variables}')

    time_start = time.time()
    
    parsed_variables = np.zeros((0, 3))
    for input_var in read_variables:
        parsed_variables = parse_var(input_var, parsed_variables)

    if write_parquet:
        if not output_directory:
            # Use current time to create a unique folder name
            now = datetime.datetime.now(ZoneInfo("Europe/London"))
            strf = now.strftime("%y%m%d%H%M") # Set time format
            output_directory = f'output/analysis_read_parquet{strf}'
        print(f'Write data to output_directory: {output_directory}')

    
    all_data = read_parquet(read_directory, subdirectory_names, fraction, parsed_variables, cut_function, write_parquet, output_directory, return_output)
        
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
    return all_data