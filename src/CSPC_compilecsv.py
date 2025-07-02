# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:22:55 2025

@author: Mikhail
"""

import os
import csv     

def compile_csv_results(dir_path: str = 'C:\\Users\\misha\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\reconstruction\\NYCOpenData\\thresholding\\ColumbusCircle\\points5000', out_csv: str= 'C:\\Users\\misha\\Downloads\out.csv') -> list[dict]:
    """
    Compiles all information from generated output TXT files into one CSV file.
    
    All of the parameter and output information from the TXT files are parsed
    and compiled as one single CSV table to make the reconstruction results
    more easily extractable for plotting or other data analysis purposes 
    (especially when performing a large amount of point cloud reconstructions.)
    
    Parameters
    ----------
    dir_path : str
        The path to the directory containing the output TXT files.
    out_csv : str 
        Path to directory that will contain the output CSV file.
    
    Returns
    -------
    compiled_data : List[Dict]
        A list of dictionaries where each list item is a parsed output 
        TXT file. Each dictionary item is a key:value pair that describes a
        parsed value from the TXT file.
    
    Notes
    -----
    #TODO
    
    Examples
    --------
    #TODO
    """
    
    compiled_data = [] # Data from all parsed .txt files
    metadata = dict() # Data in each .txt file
    fieldnames = ['Name', 'Points', 'CS Ratio', 'Measurement', 'Basis', 'Sparsity', 'Reconstruction', 'Solver time', '2-Norm Error', 'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'Chamfer Distance', 'Hausdorff Distance', 'Earth Mover\'s Distance']
    
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
        with open(out_csv, 'w', newline='') as csvfile:
            # Map dictionary onto output rows
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in compiled_data:
                writer.writerow(data)
        
    return compiled_data