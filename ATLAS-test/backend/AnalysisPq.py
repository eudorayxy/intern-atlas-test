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
from .ParquetDict import PARQUET_DICT, DATA_SKIMS

def count_num_events(string_code):
    if string_code in PARQUET_DICT:
        read_directory = PARQUET_DICT[string_code]
        if not os.path.isdir(read_directory):
            raise FileNotFoundError(f"Folder '{read_directory}' does not exist")

        files = sorted(glob.glob(f'{read_directory}/*.parquet'))
        tot_num_rows = 0
        for file in files:
            tot_num_rows += pq.ParquetFile(file).metadata.num_rows
        return tot_num_rows
    else:
        print(f'{string_code} not found.')
        raise ValueError

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

    sample_dict = defaultdict(list)
    chunk_count = 0
    num_rows_read = 0
    
    for file in files:
        
        chunk_dict = defaultdict(list)
        pq_file = pq.ParquetFile(file)
        num_row_groups = pq_file.num_row_groups
       
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
                    print(f'Exception: {e}\nReminder: Input "{cut_function}" must take one argument and return it.')
                    raise
                    
            if len(arr) == 0:
                continue

            for field in arr.fields:
                if field not in parsed_variables[:, 1]:
                    parsed_variables = np.vstack([parsed_variables,
                    (field, field, None)])
                
            for input_var, base_var, index in parsed_variables:
                if base_var not in arr.fields:
                    raise ValueError(f"Variable '{base_var}' not found. Failed to access '{input_var}'. Available variable(s): "
                                   f"{arr.fields}")
                data = arr[base_var]
                
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
        # End of loop through row groups in one file
        if chunk_dict:
            chunk_dict = {key : ak.concatenate(value) for key, value in chunk_dict.items()}
            ak.to_parquet(ak.zip(chunk_dict, depth_limit=1), f'{sample_out_dir}/chunk{chunk_count}.parquet')
            chunk_count += 1
            
        if num_rows_read >= max_num_rows:
            break
    # End of loop through all parquet files
    if return_output:
        if len(files) > 1:
            sample_dict = {key : ak.concatenate(value) for key, value in sample_dict.items()}
        if sample_dict:
            sample_dict = ak.zip(sample_dict, depth_limit=1)
        
        return sample_dict
    else:
        return None

def analysis_pq(string_code_list, read_variables, fraction=1, cut_function=None, write_parquet=False, output_directory=None, return_output=True):
    
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
            output_directory = f'output/analysis_pq{strf}'
            print(f'Created output_directory: {output_directory}')
    
    for str_code in string_code_list:
        str_code = str(str_code)
        
        files = []
        mc_present = False
        if str_code in PARQUET_DICT:
            sample_directory = PARQUET_DICT[str_code]
            pq_files = sorted(glob.glob(f'{sample_directory}/*.parquet'))
            if not pq_files:
                raise FileNotFoundError(f"No .parquet files found with the string code '{str_code}'") 
            files.extend(pq_files)
            
            base_idx = str_code.find('_')
            if base_idx == -1:
                base_str = str_code
            else:
                base_str = str_code[:base_idx]
            if base_str not in DATA_SKIMS:
                mc_present = True
        else:
            physics_processes = [code.strip() for code in str_code.split('+')]
            for i in physics_processes:
                if i in PARQUET_DICT:
                    sample_directory =  PARQUET_DICT[i]
                    pq_files = sorted(glob.glob(f'{sample_directory}/*.parquet'))
                    if not pq_files:
                        raise FileNotFoundError(f"No .parquet files found with the string code '{i}'")
                    files.extend(pq_files)

                    base_idx = i.find('_')
                    if base_idx == -1:
                        base_str = i
                    else:
                        base_str = i[:base_idx]
                    if not mc_present and base_str not in DATA_SKIMS:
                        mc_present = True
                else:
                    raise ValueError(f'Invalid string code: {i}. Available string codes: {list(PARQUET_DICT.keys())}')

        base_var_list = parsed_variables[:, 1]
        if not mc_present and 'totalWeight' in base_var_list:
            rows_to_delete = np.any(parsed_variables == 'totalWeight', axis=1)
            parsed_variables = parsed_variables[~rows_to_delete]
        elif mc_present and 'totalWeight' not in base_var_list:
            parsed_variables = np.vstack([parsed_variables,
                                          ('totalWeight', 'totalWeight', None)])

        sample_key = f'{str_code}_{fraction}'
        sample_key = sample_key.replace('.', '_')
        sample_key = sample_key.replace('+', '_')
       
        if write_parquet:
            sample_out_dir = f'{output_directory}/{sample_key}'
            os.makedirs(sample_out_dir)
        else:
            sample_out_dir = None

        max_num_rows = round(count_num_events(str_code) * fraction)

        all_data[sample_key] = concatenate_chunks(files, parsed_variables, cut_function, write_parquet, sample_out_dir, max_num_rows, return_output)
        
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

    if return_output:
        return all_data



# def concatenate_chunks(files, parsed_variables, cut_function, write_parquet, sample_out_dir, max_num_rows):

#     sample_dict = defaultdict(list)
#     chunk_count = 0
    
#     for file in files:
#         chunk_dict = {}
        
#         array = ak.from_parquet(file, columns=parsed_variables[:, 1])

#         if cut_function is not None:
#             try:
#                 array = cut_function(array)
#             except Exception as e:
#                 print(f'Exception: {e}\nReminder: Input "{cut_function}" must take one argument and return it.')
#                 raise
        
#         for input_var, base_var, index in parsed_variables:
#             if base_var not in array.fields:
#                 raise ValueError(f"Variable '{base_var}' not found. Failed to access '{input_var}'. Available variable(s): "
#                                f"{array.fields}")
#             data = array[base_var]
#             if len(data) == 0:
#                 break
                
#             if write_parquet:
#                 chunk_dict[base_var] = data
                
#             # type_str = str(ak.type(data))
#             # is_nested = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)

#             # if is_nested: # Variable array is nested
#             #     num = ak.num(data)
#             #     max_num = ak.max(num)
#             #     if index != None: # Index given in input
#             #         index = int(index)
#             #         if index >= max_num:
#             #             raise IndexError(f'Invalid index for input variable "{input_var}". '
#             #                              f'Input index should be less than {max_num}.')
#             #         if not ak.all(num >= index + 1):
#             #             data = ak.pad_none(data, index + 1, axis=-1)
#             #         sample_dict[input_var].append(data[:, index])
#             #     else: # Index not given in input
#             #         if not ak.all(num >= max_num):
#             #             data = ak.pad_none(data, max_num, axis=-1)
#             #         for i in range(max_num):
#             #             new_key = f'{base_var}[{i}]'
#             #             sample_dict[new_key].append(data[:, i])
#             # else: # Variable array is not nested
#             #     if index != None: # Index given in input
#             #         raise ValueError(f'{base_var} is not is_nested. Failed to access "{input_var}".')
#             #     else: # Index not given in input
#             #         sample_dict[input_var].append(data)
#             sample_dict[input_var].append(data)
#             # End of if-else statement
#         # End of loop through variables
#         if chunk_dict:
#             ak.to_parquet(ak.zip(chunk_dict, depth_limit=1), f'{sample_out_dir}/chunk{chunk_count}.parquet')
#             chunk_count += 1
            
#     # End of loop through all parquet files
#     if len(files) > 1:
#         sample_dict = {key : ak.concatenate(value) for key, value in sample_dict.items()}
#         if sample_dict:
#             sample_dict = ak.zip(sample_dict)
    
#     return sample_dict


# def analysis_pq(string_code_list, read_variables, cut_function=None, write_parquet=False, output_directory=None):
    
#     time_start = time.time()
    
#     all_data = {}
    
#     parsed_variables = np.zeros((0, 3))
#     for input_var in read_variables:
#         if '[' in input_var and ']' in input_var:
#             try:
#                 base_var = input_var.split('[')[0]
#                 idx_pos_start = input_var.find('[') + 1 # Index starting position in the str
#                 idx_pos_end = input_var.find(']')
#                 index = int(input_var[idx_pos_start : idx_pos_end])
#             except Exception as e:
#                 print(f'Invalid input variable format : {input_var}. '
#                       f'Expect "variable" or "variable[int]".\nError: {e}')
#                 raise
#         elif '[' in input_var or ']' in input_var:
#             raise ValueError('Expect input variable to be "variable" or "variable[int]".'
#                              'Perhaps you forgot a "[" or "]"?')
#         else:
#             base_var = input_var
#             index = None
#         parsed_variables = np.vstack([parsed_variables,
#                                       (input_var, base_var, index)])

#     if write_parquet:
#         if not output_directory:
#             # Use current time to create a unique folder name
#             now = datetime.datetime.now(ZoneInfo("Europe/London"))
#             strf = now.strftime("%Y%m%d%H%M") # Set time format
#             output_directory = f'output/read_parquet{strf}'
    
#     for str_code in string_code_list:
#         str_code = str(str_code)
        
#         files = []
#         mc_present = False
#         if str_code in PARQUET_DICT:
#             sample_directory = PARQUET_DICT[str_code]
#             pq_files = sorted(glob.glob(f'{sample_directory}/*.parquet'))
#             if not pq_files:
#                 raise FileNotFoundError(f"No .parquet files found with the string code '{str_code}'") 
#             files.extend(pq_files)
            
#             base_idx = str_code.find('_')
#             if base_idx == -1:
#                 base_str = str_code
#             else:
#                 base_str = str_code[:base_idx]
#             if base_str not in DATA_SKIMS:
#                 mc_present = True
#         else:
#             physics_processes = [code.strip() for code in str_code.split('+')]
#             for i in physics_processes:
#                 if i in PARQUET_DICT:
#                     sample_directory =  PARQUET_DICT[i]
#                     pq_files = sorted(glob.glob(f'{sample_directory}/*.parquet'))
#                     if not pq_files:
#                         raise FileNotFoundError(f"No .parquet files found with the string code '{i}'")
#                     files.extend(pq_files)

#                     base_idx = i.find('_')
#                     if base_idx == -1:
#                         base_str = i
#                     else:
#                         base_str = i[:base_idx]
#                     if not mc_present and base_str not in DATA_SKIMS:
#                         mc_present = True
#                 else:
#                     raise ValueError(f'Invalid string code: {i}. Available string codes: {list(PARQUET_DICT.keys())}')

#         base_var_list = parsed_variables[:, 1]
#         if not mc_present and 'totalWeight' in base_var_list:
#             rows_to_delete = np.any(parsed_variables == 'totalWeight', axis=1)
#             parsed_variables = parsed_variables[~rows_to_delete]
#         elif mc_present and 'totalWeight' not in base_var_list:
#             parsed_variables = np.vstack([parsed_variables,
#                                           ('totalWeight', 'totalWeight', None)])
            
#         sample_key = str_code.replace('+', '_')
       
#         if write_parquet:
#             sample_out_dir = f'{output_directory}/{sample_key}'
#             os.makedirs(sample_out_dir)
#         else:
#             sample_out_dir = None

#         all_data[sample_key] = concatenate_chunks(files, parsed_variables, cut_function, write_parquet, sample_out_dir, max_num_rows)
        
#     elapsed_time = time.time() - time_start 
#     print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

#     return all_data
