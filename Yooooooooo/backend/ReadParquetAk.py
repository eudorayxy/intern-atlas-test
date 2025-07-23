import glob
import time
import re
import awkward as ak
import numpy as np
from collections import defaultdict

def concatenate_chunks(sample_directory, parsed_variables, cut_function):

    files = sorted(glob.glob(f'{sample_directory}/*.parquet'))

    if not files:
        raise FileNotFoundError(f"No .parquet files found in directory: {sample_directory}") 

    sample_dict = defaultdict(list)
    
    for file in files:
        
        array = ak.from_parquet(file, columns=parsed_variables[:, 1])

        array = cut_function(array)
        
        for input_var, base_var, index in parsed_variables:
            try:
                data = array[base_var]
            except Exception as e:
                print(f'Exception occured while attempting to read the variable "{input_var}" from {file} : {e}')
                raise

            type_str = str(ak.type(data))
            slicable = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)

            if index is not None:
                field = ak.pad_none(data, index + 1, axis=1)
                if slicable:
                     try:
                         sample_dict[input_var].append(field[:, index])
                     except Exception as e:
                        print(f"Failed to access '{ininput_var}': {e}")
                        raise
                else:
                    raise TypeError(f"Error: {base_var} is not slicable. Failed to access '{input_var}'")
            else: # No valid index provided in input
                if slicable:
                    max_num = ak.max(ak.num(data, axis=1))
                    for i in range(max_num):
                        padded = ak.pad_none(data, i + 1, axis=1)
                        new_key = f'{base_var}[{i}]'
                        sample_dict[new_key].append(padded[:, i])
                else:
                    sample_dict[input_var].append(data)
            # End of if-else statement
        # End of loop through variables
    # End of loop through all parquet files
    if len(files) > 1:
        sample_dict = {key : ak.concatenate(value) for key, value in sample_dict.items()}
    return sample_dict

def read_parquet_ak(samples, read_directory, variable_list, cut_function):
    
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
        parsed_variables = np.vstack([parsed_variables,
                                      (input_var, base_var, index)])
    
    for sample_key in samples:
        
        sample_directory = f"{read_directory}/{sample_key}"

        base_var_list = parsed_variables[:, 1]
        
        if 'Data' in sample_key and 'totalWeight' in base_var_list:
            rows_to_delete = np.any(parsed_variables == 'totalWeight', axis=1)
            parsed_variables = parsed_variables[~rows_to_delete]
        elif 'Data' not in sample_key and 'totalWeight' not in base_var_list:
            parsed_variables = np.vstack([parsed_variables,
                                          ('totalWeight', 'totalWeight', None)])

        all_data[sample_key] = concatenate_chunks(sample_directory, parsed_variables, cut_function)
        
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
                
    return all_data