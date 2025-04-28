"""
Utility functions for handling file system operations and reading Excel files.
"""
import os
import pandas as pd
import warnings

def get_excel_files(base_dir):
    """
    Recursively finds all .xlsx files in the specified directory and subdirectories.

    Args:
        base_dir (str): The path to the base directory to search.

    Yields:
        str: The path to each found .xlsx file, with forward slashes.
    """
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.xlsx'):
                yield os.path.join(root, file).replace("\\", "/")


import warnings
import re

def extract_year_from_path(path):
    """
    Extracts the year from the file path, assuming the year is a 4-digit number
    typically found as the directory name directly containing the file,
    or potentially elsewhere in the path.

    """
    parts = path.split("/")

   
    if len(parts) >= 2 and parts[-2].isdigit() and len(parts[-2]) == 4:
        return parts[-2]

    match = re.search(r'/(\d{4})/', path)
    if match:
        return match.group(1)

   
    if len(parts) > 3 and parts[3].isdigit() and len(parts[3]) == 4:
         warnings.warn(f"Extracted year using fallback index 3 for path: {path}", UserWarning)
         return parts[3]

    warnings.warn(f"Could not reliably extract year from path: {path}. Structure might be unexpected.", UserWarning)
    return "unknown"


def get_sheet_names(path, year):
    """
    Returns a list of sheet names from an Excel file.
    Reads all sheets only if the year is '2015', otherwise reads the first sheet.

    """
    try:
        if year == '2015':
            return pd.ExcelFile(path).sheet_names
        else:
            return [None] # [None] tells read_excel to read the first sheet
    except Exception as e:
        warnings.warn(f"Error reading sheet names from file {path}: {e}", UserWarning)
        return []


def process_excel_file(path, year, sheet=None):
    """
    Reads data from a specified sheet (or the first sheet) of an Excel file into a DataFrame.

    """
    try:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
       
        if isinstance(df, dict):
            if df:
                
                warnings.warn(f"read_excel returned a dict for {path}, sheet {sheet}. Using first sheet.", UserWarning)
                return list(df.values())[0]
            else:
                return pd.DataFrame()
        return df
    except Exception as e:
        warnings.warn(f"Error processing Excel file {path}, sheet {sheet}: {e}", UserWarning)
        return pd.DataFrame()