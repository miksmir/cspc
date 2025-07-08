
import pandas as pd
from os.path import join
from CSPointCloud import OUTPUT_PATH_COMPILED

csv_name = 'ColumbusCircle_5000points.csv'
pd_query = 'Basis == "Haar DWT" and `CS Ratio` == 25 and Sparsity >= 10'
error_type = 'Mean Absolute Error'
#cs_ratios = * 0.01


def extract_errors(csv_name: str, pd_query: str, error_type: str) -> tuple:
    df = pd.read_csv(join(OUTPUT_PATH_COMPILED, csv_name))
    filtered_df = df.query(pd_query)
    sorted_df = filtered_df.sort_values(by='Sparsity', ascending=True, ignore_index=True)
    
    # Extract values into two separate lists for plotting x and y
    error_vals = sorted_df[error_type].tolist()
    sparsity_vals = sorted_df['Sparsity'].tolist()
    
    return sparsity_vals, error_vals


def extract_errors_basis(csv_name: str, pd_query: str, error_type: str) -> tuple:
    df = pd.read_csv(join(OUTPUT_PATH_COMPILED, csv_name))
    filtered_df = df.query(pd_query)
    sorted_df = filtered_df.sort_values(by='CS Ratio', ascending=True, ignore_index=True)
    
    # Extract values into two separate lists for plotting x and y
    error_vals = sorted_df[error_type].tolist()
    sparsity_vals = sorted_df['CS Ratio'].tolist()
    
    return sparsity_vals, error_vals



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
         'Earth Mover\'s Distance', 'Solver time').        

    Returns
    -------
    tuple
        A tuple of two items: a list of extracted values selected by `x_axis`
        and a list of extracted values selected by `y_values`.
        (Example: (sparsity_values, MAE_values)).
        
    Examples    
    --------
    sparsities, solvertimes = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis == "DCT" and `CS Ratio` >= 25', x_axis='Sparsity', y_axis='Solver time')
    """
    
    df = pd.read_csv(join(OUTPUT_PATH_COMPILED, csv_name))
    filtered_df = df.query(pd_query)
    sorted_df = filtered_df.sort_values(by=x_axis, ascending=True, ignore_index=True)
    
    # Extract values into two separate lists for plotting x and y
    y_vals = sorted_df[y_axis].tolist()
    x_vals = sorted_df[x_axis].tolist()
    
    return x_vals, y_vals

output = extract_errors(csv_name, pd_query, error_type)
output2 = extract_errors_basis(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis == "Haar DWT" and `CS Ratio` >= 25 and Sparsity == 90', error_type=error_type)
output3 = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis == "DCT" and Sparsity == 90', y_axis='Solver time', x_axis='CS Ratio')