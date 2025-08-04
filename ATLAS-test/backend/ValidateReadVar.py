import os
import glob
import pyarrow.parquet as pq
from .ParquetDict import PARQUET_DICT


def get_valid_variables(string_code):
    if string_code in PARQUET_DICT:
        read_directory = PARQUET_DICT[string_code]
        if not os.path.isdir(read_directory):
            raise FileNotFoundError(f"Folder '{read_directory}' does not exist")

        files = sorted(glob.glob(f'{read_directory}/*.parquet'))

        # Read schema from the parquet file
        parquet_file = pq.ParquetFile(files[0])
        schema = parquet_file.schema.to_arrow_schema()
        var_list = [field.name for field in schema]
        return var_list
    else:
        print(f'{string_code} not found.')
        raise ValueError

def validate_read_variables(string_code, read_variables):
    valid_var_list = get_valid_variables(string_code)
    
    validated = []
    for variable_input in read_variables:
        if variable_input not in valid_var_list:
            print(f"Skipping '{variable_input}' - invalid input")
        elif variable_input in validated:
            print(f"Skipping '{variable_input}' - duplicated entry")
        else:
            validated.append(variable_input)
    return validated
