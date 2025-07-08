# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:22:55 2025

@author: Mikhail
"""

import numpy as np
import os
import csv
import pandas as pd
from CSPointCloud import OUTPUT_PATH_PCLAS, OUTPUT_PATH_COMPILED

def compile_csv_results(dir_path: str = OUTPUT_PATH_PCLAS, out_csv: str = 'out.csv') -> list[dict]:
    """
    Compiles all information from generated output TXT files into one CSV file.
    
    All of the parameter and output information from the TXT files are parsed
    and compiled as one single CSV table to make the reconstruction results
    more easily extractable for plotting or other data analysis purposes 
    (especially when performing a large amount of point cloud reconstructions).
    
    Parameters
    ----------
    dir_path : str, optional
        The path to the directory containing the TXT files to be parsed.
        (Path is specified by JSON file under `OUTPUT_PATH_PCLAS` by default).
    out_csv : str, optional 
        The name of the output CSV file ('out.csv' by default).
    
    Returns
    -------
    compiled_data : list[dict]
        A list of dictionaries where each list item is a parsed output 
        TXT file. Each dictionary item is a key:value pair that describes a
        parsed value from the TXT file.
    
    Notes
    -----
    The output CSV file is stored in the directory specified by the JSON
    file under `OUTPUT_PATH_COMPILED`.
    
    Examples
    --------
    #TODO
    """
    
    compiled_data = [] # Data from all parsed .txt files
    metadata = dict() # Data in each .txt file
    fieldnames = ['Name', 'Points', 'CS Ratio', 'Measurement', 'Basis', 'Sparsity', 'Reconstruction', 'Solver Time', '2-Norm Error', 'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'Chamfer Distance', 'Hausdorff Distance', 'Earth Mover\'s Distance']
    
    # Scan through directory for all .txt files and parse them
    for filename in os.listdir(os.path.normpath(dir_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(dir_path, filename)    
            with open(file_path, 'r') as file:
                # Remove whitespace from .txt file and add non-whitespace strings to lines list
                lines = [line.strip() for line in file if line.strip()]
                
                # First line metadata (reconstruction parameters)
                # Take each parameter in the first line separated by '|' and remove whitespace
                metadata_parts = [part.strip() for part in lines[0].split('|')]
                metadata = {fieldnames[0]:           metadata_parts[2].split(' PC', 1)[0],   # Name
                            fieldnames[1]:         metadata_parts[1].split(' ', 1)[0],       # Points
                            fieldnames[2]:       metadata_parts[0].split('%',1)[0],          # CS Ratio
                            fieldnames[3]:    metadata_parts[3].split(' Measurement',1)[0],  # Measurement
                            fieldnames[4]:          metadata_parts[4],                       # Basis
                            fieldnames[5]:       metadata_parts[5].split('%',1)[0],          # Sparsity
                            fieldnames[6]: metadata_parts[6].split(' Reconstruction', 1)[0]} # Reconstruction
          
                # Put solvertime before reconstruction errors
                keysolvertime, valuesolvertime = lines[6].split(':', 1)
                # Remove whitespace
                keysolvertime = keysolvertime.strip()
                valuesolvertime = valuesolvertime.strip()
                # Add solvertime into dict and strip [s] unit
                metadata[keysolvertime] = valuesolvertime.split(' [s]', 1)[0]
                
                # Remaining lines of metadata (output data i.e. reconstruction errors, solver time, etc)
                for line in lines[1:9]:
                    if ':' in line:
                        # Key = string before colon and value = string after colon
                        key, value = line.split(':', 1)
                        # Remove whitespace
                        key = key.strip()
                        value = value.strip()
                        metadata[key] = value
                        
                compiled_data.append(metadata)
        
        # Write to CSV
        with open(os.path.join(OUTPUT_PATH_COMPILED, out_csv), 'w', newline='') as csvfile:
            # Map dictionary onto output rows
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in compiled_data:
                writer.writerow(data)
        
    return compiled_data

def extract_csv(csv_name: str, pd_query: str, x_axis: str, y_axis: str) -> tuple:
    """
    Extracts columns from compiled CSV results file to be used for plotting.
    
    This function converts the CSV file (from `OUTPUT_PATH_COMPILED`) into a 
    Pandas dataframe and uses the Pandas library to select/filter data.

    Parameters
    ----------
    csv_name : str
        Name of CSV file containing compiled reconstruction results.
    pd_query : str
        String for querying Pandas dataframe.
        (i.e. 'Column1 == x and Column2 > y and Column3 == "keyword"')
        (Note: If key (column name) in query string has a space, 
         backticks (``) are required by Pandas).
    x_axis : str
        Selection of parameter that would represent the independent value (or
        x-axis) for plotting.
        (i.e. 'Sparsity', 'CS Ratio').
    y_axis : str
        Selection of parameter that would represent the dependent value (or
        y-axis) for plotting.
        (i.e. 'Mean Absolute Error', 'Mean Squared Error', 
         'Root Mean Squared Error', 'Chamfer Distance', 'Hausdorf Distance', 
         'Earth Mover\'s Distance', 'Solver Time').        

    Returns
    -------
    tuple
        A tuple of two items: a NumPy array of extracted values selected by 
        `x_axis` and a NumPy array of extracted values selected by `y_values`.
        (Example: (sparsity_values, MAE_values)).
        
    Examples    
    --------
    sparsities, solvertimes = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis == "DCT" and `CS Ratio` >= 25', x_axis='Sparsity', y_axis='Solver Time')
    """
    
    df = pd.read_csv(os.path.join(OUTPUT_PATH_COMPILED, csv_name))
    filtered_df = df.query(pd_query)
    sorted_df = filtered_df.sort_values(by=x_axis, ascending=True, ignore_index=True)
    
    # Extract values into two separate NumPy arrays for plotting x and y
    y_vals = np.array(sorted_df[y_axis].tolist())
    x_vals = np.array(sorted_df[x_axis].tolist())
    
    return x_vals, y_vals