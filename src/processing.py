"""
Core processing logic for converting raw Excel data into a structured DataFrame.
"""
import pandas as pd
import re
import warnings

# Import functions and constants from other modules in the src package
from .config import FINAL_COLUMN_ORDER
from .file_utils import (
    get_excel_files, extract_year_from_path, get_sheet_names, process_excel_file
)
from .data_extraction import (
    get_first_nonempty_row, extract_antibiotics, find_country_header_row,
    find_first_data_header, get_valid_n_columns, get_resistance_columns,
    extract_country_data, process_country_agnostic_data
)
from .config import ABX_ALIASES # Import ABX_ALIASES specifically if needed here

def process_dataframe(df, year):
    """
    Processes a single DataFrame (representing one Excel sheet) and extracts structured data.
    Determines if the table contains country-specific or aggregated data and calls the
    appropriate extraction function.

    """
    result = []
    iterator = 0

    if df.empty:
        return result

    first_line = get_first_nonempty_row(df)
    if not first_line:
        warnings.warn(f"Sheet for year {year} seems empty or no data found in first rows.", UserWarning)
        return result

    microorganism_match = re.split(r"[.,]", first_line, maxsplit=1)
    microorganism = microorganism_match[0].strip() if microorganism_match else first_line.strip()

    abx_list = extract_antibiotics(first_line)
    abx_str = ", ".join(abx_list)

    country_row_idx = find_country_header_row(df)

    if country_row_idx is None:
        result = process_country_agnostic_data(
            df, microorganism, abx_str, year
        )
    else:
        header_row = df.iloc[country_row_idx].fillna('').astype(str)

        country_col_idx = next(
            (idx for idx, val in enumerate(header_row) if val.lower() == 'country'),
            None
        )
        if country_col_idx is None:
             warnings.warn(f"Header 'Country' found in row {country_row_idx}, but column index not identified in {year} for {microorganism}. Skipping sheet.", UserWarning)
             return result

        resistance_col_dict = get_resistance_columns(header_row, year)

       
        n_col_dict = get_valid_n_columns(header_row, year, df, country_row_idx)

        if not n_col_dict:
             warnings.warn(f"Could not determine N/Labs columns for {year}, {microorganism}. Skipping sheet.", UserWarning)
             return result

        for target_year, col_info in n_col_dict.items():
            sheet_year_data = extract_country_data(
                df=df,
                start_row=country_row_idx,
                country_col_idx=country_col_idx,
                col_info=col_info, 
                microorganism=microorganism,
                abx_str=abx_str,
                year=target_year, 
                resistance_col_dict=resistance_col_dict,
                iterator=iterator
            )
            result.extend(sheet_year_data)
            iterator += 1

    return result


def build_microorganism_antibiotic_table(base_dir):
    """
    Main orchestrator function. Finds all Excel files in the base directory,
    processes each sheet using process_dataframe, and consolidates the results
    into a final, standardized Pandas DataFrame.

    """
    all_data = []
    print(f"Starting data extraction from: {base_dir}")

    # Use generator to find files efficiently
    excel_files = list(get_excel_files(base_dir))
    print(f"Found {len(excel_files)} Excel files to process.")

    processed_files = 0
    for path in excel_files:
        year = extract_year_from_path(path)
        if year == "unknown":
            warnings.warn(f"Could not extract year from path: {path}. Skipping file.", UserWarning)
            continue

        sheets_to_process = get_sheet_names(path, year)

        for sheet in sheets_to_process:
            processed_files += 1

            sheet_name_for_log = sheet if sheet is not None else "First Sheet"
            try:
                df_sheet = process_excel_file(path, year, sheet)
                sheet_data = process_dataframe(df_sheet, year)

                if sheet_data:
                    all_data.extend(sheet_data)

            except Exception as e:
                warnings.warn(f"Error processing file {path}, sheet '{sheet_name_for_log}': {e}", UserWarning)

    print(f"Finished processing files. Total records extracted: {len(all_data)}")

    if not all_data:
        print("Warning: No data was extracted. Returning an empty DataFrame with standard columns.")
        return pd.DataFrame(columns=FINAL_COLUMN_ORDER)

    final_df = pd.DataFrame(all_data)

    for col in FINAL_COLUMN_ORDER:
        if col not in final_df.columns:
            final_df[col] = pd.NA 

    return final_df[FINAL_COLUMN_ORDER].copy()