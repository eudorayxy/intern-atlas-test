# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:31:44 2025

@author: eudor
"""
import os
import time
import datetime
from zoneinfo import ZoneInfo
import glob as glob
import polars as pl
from collections import defaultdict
import numpy as np


def process_chunk(chunk, parsed_variables, schema, sample_dict):
    for input_var, base_var, index in parsed_variables:
        # Check if the variable has sublists    
        data_type = schema.get(base_var)
        var_has_sublists = isinstance(data_type, pl.List)
        
        # Extract values from dataframe
        base_var_values = chunk[base_var]

        # Append the values to sample_dict based on whether index was given
        # in input. If the variable has sublists and input index given, append
        # variable[index]; if the variable has sublists but no index given by user,
        # append all available variable[index] or none if index oob.
        if index is not None: # Index given in input
            if var_has_sublists: # Variable has sublists
                values = base_var_values.list.get(index, null_on_oob=True)
                sample_dict[input_var].append(values.to_numpy())
            else: # Variable does not have sublists
                raise TypeError(f'{base_var} is not a list. Failed to access {input_var}.')
        else: # Index not given in input
            if var_has_sublists: # Variable has sublists
                # Find the maximum length of sublists
                max_len = base_var_values.list.len().max()
                
                for i in range(max_len):
                    new_key = f'{base_var}[{i}]'
                    values = base_var_values.list.get(i, null_on_oob=True)
                    sample_dict[new_key].append(values.to_numpy())
            else: # Variable does not have sublists
                sample_dict[input_var].append(base_var_values.to_numpy())
        # End of if-else statement (Index given in input or not)
    # End of loop through variables

def process_chunk_parquet(chunk_count, chunk, schema, filtered_directory, parsed_variables):
    chunk_dict = {}
    for input_var, base_var, index in parsed_variables:
        # Check if the variable has sublists    
        data_type = schema.get(base_var)
        var_has_sublists = isinstance(data_type, pl.List)
        
        # Extract values from dataframe
        base_var_values = chunk[base_var]

        # Append the values to sample_dict based on whether index was given
        # in input. If the variable has sublists and input index given, append
        # variable[index]; if the variable has sublists but no index given by user,
        # append all available variable[index] or none if index oob.
        if index is not None: # Index given in input
            if var_has_sublists: # Variable has sublists
                values = base_var_values.list.get(index, null_on_oob=True)
                chunk_dict[input_var] = values.to_numpy()
            else: # Variable does not have sublists
                raise TypeError(f'{base_var} is not a list. Failed to access {input_var}.')
        else: # Index not given in input
            if var_has_sublists: # Variable has sublists
                # Find the maximum length of sublists
                max_len = base_var_values.list.len().max()
                
                for i in range(max_len):
                    new_key = f'{base_var}[{i}]'
                    values = base_var_values.list.get(i, null_on_oob=True)
                    chunk_dict[new_key] = values.to_numpy()
            else: # Variable does not have sublists
                chunk_dict[input_var] = base_var_values.to_numpy()
        # End of if-else statement (Index given in input or not)
    # End of loop through variables
    chunk_dataframe = pl.DataFrame(chunk_dict)
    chunk_dataframe.write_parquet(f'{filtered_directory}/filtered_chunk{chunk_count}.parquet')
    chunk_count += 1
    return chunk_count
    
    

def read_parquet_polars(samples, read_directory, variable_list,
                        filter_expression=None, chunk_size=100000,
                        write_parquet=False, return_output=True, 
                        output_directory=None):
    
    time_start = time.time()
    
    all_data = {}

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
        # Add a row to parsed_variables
        parsed_variables = np.vstack([parsed_variables,
                                      (input_var, base_var, index)])
    # End of loop through input variables to parse the input str

    if write_parquet:
        if not output_directory:
            output_directory = 'output/filtered/'
            # Use current time to create a unique folder name
            now = datetime.datetime.now(ZoneInfo("Europe/London"))
            strf = now.strftime("%Y%m%d%H%M") # Set time format
            output_directory += f'{strf}'
            
    for sample_key in samples:

        # Get parquet files
        sample_directory = f"{read_directory}/{sample_key}"
        files = sorted(glob.glob(f'{sample_directory}/*.parquet'))    
        if not files:
            raise FileNotFoundError(f"No .parquet files found in directory: {sample_directory}")

        # Collect schema using the first parquet file and validate variables
        schema = pl.scan_parquet(files[0]).collect_schema()

        base_var_array = parsed_variables[:, 1] 
            
        if 'Data' not in sample_key and 'totalWeight' not in base_var_array:
            parsed_variables = np.vstack([parsed_variables,
                                          ('totalWeight', 'totalWeight', None)])
        elif 'Data' in sample_key and 'totalWeight' in base_var_array:
            rows_to_delete = np.any(parsed_variables == 'totalWeight', axis=1)
            parsed_variables = parsed_variables[~rows_to_delete]

        for _, base_var, _ in parsed_variables:    
            if base_var not in schema:
                raise ValueError(f"Variable {base_var} not found. Available: {list(schema.keys())}")
            
        sample_dict = defaultdict(list)
        
        if write_parquet:
            chunk_count = 0
            filtered_directory = f'{output_directory}/{sample_key}'
            os.makedirs(filtered_directory)
            
        # Loop through all parquet files for one sample key
        for file in files:
            # Lazy read (no data yet)
            lazyframe = pl.scan_parquet(file)

            # Apply filter
            if filter_expression is not None:
                lazyframe = lazyframe.filter(filter_expression)

            lazyframe = lazyframe.select(parsed_variables[:, 1])
            
            # Chunk-iterate over DataFrame
            for chunk in lazyframe.collect(engine='streaming').iter_slices(chunk_size):
                if return_output:
                    process_chunk(chunk, parsed_variables, schema, sample_dict)
                if write_parquet:
                    process_chunk_parquet(chunk_count, chunk, schema,
                                          filtered_directory, parsed_variables)
                    
        if return_output:
            if len(files) > 1:
                sample_dict = {key : np.concatenate(value) for key, value in sample_dict.items()}
            all_data[sample_key] = sample_dict
    # End of for loop through all samples
            
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
     
    if return_output:           
        return all_data


# def get_data_polars(samples, read_directory, variable_list, filter_expression=None):
#     time_start = time.time()
    
#     all_data = {}
    
#     for sample_key in samples:

#         # Get parquet files
#         directory = f"{read_directory}/{sample_key}"
#         files = sorted(glob.glob(f'{directory}/*.parquet'))    
#         if not files:
#             raise FileNotFoundError(f"No .parquet files found in directory: {directory}")
            
#         sample_dict = {}
#         # Loop through all parquet files for one sample key
#         for file in files:
#             # Lazy read (no data yet)
#             lazyframe = pl.scan_parquet(file)
#             schema = lazyframe.collect_schema()

#             # User gives a list of variables
#             # Input can be lep_pt or lep_pt[0]
#             # The code needs to get 'lep_pt' and 0 from input 'lep_pt[0]' -- use re
#             # The code needs to collect all input variable names to validate with schema
#             # If user wants to do further selection, feed a filter_func to get_data func

#             select_expression = []
            
#             if 'Data' not in sample_key:
#                 select_expression.append(pl.col('totalWeight'))
                
#             for var in variable_list:
#                 match = re.match(r'([a-zA-Z_]\w*)(?:\[(\d+)\])?$', var)
#                 if match:
#                     variable_name = match.group(1)
#                     index = int(match.group(2)) if match.group(2) else None
#                 else:
#                     raise ValueError(f"Invalid input variable format : {var}.")

#                 if variable_name not in schema:
#                     raise ValueError(f'Input variable : {var} not found. Available variable: '
#                                      f'{list(schema.keys())}')
                    
#                 data_type = schema.get(variable_name)
#                 var_is_list = isinstance(data_type, pl.List)

#                 if index is not None:
#                     if var_is_list:
#                         var_alias = f'{variable_name}[{index}]'
#                         select_expression.append(pl.col(variable_name).list.get(index, null_on_oob=True).alias(var_alias))
#                     else:
#                         raise TypeError(f'{variable_name} is not a list.'
#                                         f'Failed to access {variable_name}[{index}]')
#                 else:
#                     if var_is_list:
#                         max_len = (
#                             lazyframe.select(pl.col(variable_name).list.len().max())
#                                 .collect().item()
#                             )
#                         for i in range(max_len):
#                             var_alias = f'{variable_name}[{i}]'
#                             select_expression.append(
#                                 pl.col(variable_name).list.get(i, null_on_oob=True).alias(var_alias)
#                             )
#                     else:
#                         select_expression.append(pl.col(variable_name))
#                 # End of loop through variable_list
                        
#             if select_expression:
#                 if filter_expression is not None:
#                     lazyframe = lazyframe.filter(filter_expression).select(select_expression)
#                 else:
#                     lazyframe = lazyframe.select(select_expression)
#                 dataframe = lazyframe.collect()
#                 for column in dataframe.schema:
#                     if column not in sample_dict:
#                         sample_dict[column] = []
#                     sample_dict[column].extend(dataframe[column].to_list())

#         for column in sample_dict:
#             sample_dict[column] = np.array(sample_dict[column])
                    
#         # End of loop through parquet files for one sample
#         all_data[sample_key] = sample_dict
#         # if dataframe_list:
#         #     all_data[sample_key] = pl.concat(dataframe_list, how='vertical')
#     # End of loop through all samples
                
#     elapsed_time = time.time() - time_start 
#     print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
                
#     return all_data