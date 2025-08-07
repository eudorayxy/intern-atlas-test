import os
import glob
import pyarrow.parquet as pq
from .ParquetDict import PARQUET_DICT, STR_CODE_COMBO


def get_valid_variables(string_code):
    if string_code in STR_CODE_COMBO:
        string_code = STR_CODE_COMBO[string_code]
        physics_processes = [code.strip() for code in string_code.split('+')]
        string_code = physics_processes[0]
        
    if string_code in PARQUET_DICT:
        read_directory = PARQUET_DICT[string_code]
        if not os.path.isdir(read_directory):
            raise FileNotFoundError(f"Folder '{read_directory}' does not exist")

        files = sorted(glob.glob(f'{read_directory}/*.parquet'))
        print(f'Variables validated using {files[0]}')

        # Read schema from the parquet file
        parquet_file = pq.ParquetFile(files[0])
        schema = parquet_file.schema.to_arrow_schema()
        var_list = [field.name for field in schema]
        return var_list
    else:
        print(f'{string_code} not found.')
        raise ValueError

def validate_read_variables(string_code_list, read_variables):
    validated = []
    
    for string_code in string_code_list:
        valid_var_list = get_valid_variables(string_code)
        
        for variable_input in read_variables:
            if variable_input not in valid_var_list:
                print(f"Skipping '{variable_input}' - invalid input for string code '{string_code}'")
            elif variable_input in validated:
                continue
            else:
                validated.append(variable_input)
    return validated
