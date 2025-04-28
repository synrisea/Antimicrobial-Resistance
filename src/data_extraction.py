"""
Functions for extracting specific data points and structures from DataFrames.
"""
import re
import pandas as pd
from collections import defaultdict
import warnings

# Import constants from config module within the same package
from .config import (
    ANTIBIOTICS, ABX_ALIASES, NORMALIZED_COUNTRIES,
    N_PATTERNS, N_PATTERNS_LOWER, RESISTANCE_PATTERNS, RESISTANCE_PATTERNS_LOWER
)


def get_first_nonempty_row(df):
    """
    Finds the first row in the DataFrame containing at least one non-empty value
    and returns its contents joined into a single string.

    """
    for _, row in df.iterrows():
        # Drop NA values, convert remaining to string, put into list
        values = row.dropna().astype(str).tolist()
        if values: # If the list is not empty
            return " ".join(values)
    return None


def extract_antibiotics(text):
    """
    Extracts standardized antibiotic names from a text string using predefined lists and aliases.
    Uses whole-word, case-insensitive matching.

    """
    if not isinstance(text, str):
        return []

    found_standardized = set()
    text_lower = text.lower()

    for abx in ANTIBIOTICS:
        abx_lower = abx.lower()
        # Escape potential regex special characters in antibiotic name and match whole word
        pattern = r'\b{}\b'.format(re.escape(abx_lower))
        if re.search(pattern, text_lower):
            # Get the standard name (could be from alias or the original name)
            standard_name = ABX_ALIASES.get(abx, abx)
            found_standardized.add(standard_name) # Add the potentially non-lowercase standard name

    # Return sorted list of unique standard names found
    return sorted(list(found_standardized))


def find_country_header_row(df):
    """
    Finds the index of the first row containing a cell with exactly 'Country' (case-insensitive).

    """
    for i in range(len(df)):
        row = df.iloc[i]
        if row.fillna('').astype(str).str.contains("(?i)^country$", regex=True).any():
            return i
    return None


def find_first_data_header(df):
    """
    Finds the index of the first row that looks like a data header,
    by checking for the presence of any N_PATTERNS.

    """
    for i in range(len(df)):
        row_values = df.iloc[i].fillna('').astype(str).str.strip().str.lower()
        if any(val in N_PATTERNS_LOWER for val in row_values):
            return i
    return None


def get_valid_n_columns(header_row, year, df=None, header_row_idx=None):
    """
    Identifies indices for 'N' (isolates) and 'Number of laboratories' columns.
    Includes special logic for year 2013 structure and attempts to differentiate
    'N' from 'Labs' by checking the row below the header.

    """
    n_cols = []
    lab_cols = []

    header_values = header_row.fillna('').astype(str)

    for idx, val in enumerate(header_values):
        val_clean_lower = val.strip().lower()
        if not val_clean_lower:
            continue

        if "laboratories" in val_clean_lower:
            lab_cols.append(idx)
            continue

        if val_clean_lower in N_PATTERNS_LOWER:
            is_actually_lab_col = False
            if df is not None and header_row_idx is not None and header_row_idx + 1 < len(df):
                try:
                    next_row_val_raw = df.iloc[header_row_idx + 1, idx]
                    if pd.notna(next_row_val_raw):
                        next_row_val = str(next_row_val_raw).strip().lower()
                        # If the cell below says 'laboratories', assume this header was for labs
                        if next_row_val == "laboratories":
                            lab_cols.append(idx)
                            is_actually_lab_col = True
                except IndexError:
                    pass 
                except Exception as e:
                    warnings.warn(f"Error checking row below header at index {idx} for disambiguation: {e}", UserWarning)


            if not is_actually_lab_col:
                n_cols.append(idx)

    if year == '2013':
        if len(n_cols) >= 4:
            years = ['2010', '2011', '2012', '2013']
            return {y: {"n": [col], "labs": lab_cols} for y, col in zip(years, n_cols[:4])}
        else:
            n_idx_2013 = n_cols[0] if n_cols else 2
            if not n_cols:
                 warnings.warn(f"Year 2013: Found < 4 N columns and no specific N column. Falling back to index 2. This is risky.", UserWarning)
            return {year: {"n": [n_idx_2013], "labs": lab_cols}}

    elif year in {'2014', '2015'}:
        if n_cols:
            if len(n_cols) > 1:
                 warnings.warn(f"Year {year}: Found multiple N columns ({n_cols}). Using the last one ({n_cols[-1]}).", UserWarning)
            return {year: {"n": [n_cols[-1]], "labs": lab_cols}}
        else:
             warnings.warn(f"Year {year}: No N columns identified.", UserWarning)
             return {year: {"n": [], "labs": lab_cols}}
    else:
        warnings.warn(f"Year {year}: Using default logic - associating all found N columns ({n_cols}) and Labs columns ({lab_cols}) with this year.", UserWarning)
        return {year: {"n": n_cols, "labs": lab_cols}}


def get_resistance_columns(header_row, year):
    """
    Finds indices of columns related to resistance (%R, %IR, %ESBL, etc.).

    """
    resistance_cols = []
    ir_indices = set()

    header_values = header_row.fillna('').astype(str)

    for idx, val in enumerate(header_values):
        val_clean_lower = val.strip().lower()
        if val_clean_lower in RESISTANCE_PATTERNS_LOWER:
            resistance_cols.append(idx)
            if val_clean_lower.startswith("%ir"):
                ir_indices.add(idx)

    return {
        "resistance": resistance_cols,
        "ir_indices": sorted(list(ir_indices)) 
    }


def extract_country_data(df, start_row, country_col_idx, col_info, microorganism,
                         abx_str, year, resistance_col_dict, iterator):
    """
    Extracts country-specific data rows from a DataFrame.
    Handles 'resistance pattern' column to refine antibiotics for the row.
    Uses year-specific logic (via iterator/last value) for resistance values.

    """
    rows = []
    header = df.iloc[start_row].fillna('').astype(str)

    resistance_pattern_col = None
    for idx, val in enumerate(header):
        if "resistance pattern" in val.lower():
            resistance_pattern_col = idx
            break

    for i in range(start_row + 1, len(df)):
        row = df.iloc[i]

        if row.isna().all():
            continue

        country_cell_raw = None
        if country_col_idx is not None and country_col_idx < len(row):
            country_cell_raw = row.iloc[country_col_idx]

        if pd.isna(country_cell_raw) or not isinstance(country_cell_raw, str):
            continue 
        country_cell = country_cell_raw.strip()
        if not country_cell or country_cell.lower() not in NORMALIZED_COUNTRIES:
            continue 

        current_abx = abx_str
        pattern_value = ""
        is_fully_susceptible_pattern = False

        if resistance_pattern_col is not None and resistance_pattern_col < len(row):
            pattern_raw = row.iloc[resistance_pattern_col]
            if pd.notna(pattern_raw):
                pattern_value = str(pattern_raw).strip().lower()
                if "fully susceptible" in pattern_value:
                     is_fully_susceptible_pattern = True 

        if pattern_value and not is_fully_susceptible_pattern:
            matched_abx = []
            if abx_str: 
                for abx_from_header in abx_str.split(", "):
                    abx_header_norm = ABX_ALIASES.get(abx_from_header.strip(), abx_from_header.strip()).lower()
                    if abx_header_norm in pattern_value:
                        matched_abx.append(abx_from_header.strip()) # Add the original name

            if matched_abx:
                current_abx = ", ".join(sorted(list(set(matched_abx))))
            else:
                continue
        
        resistance_values = defaultdict(list)
        resistance_indices = resistance_col_dict.get("resistance", [])
        for r_idx in resistance_indices:
            if r_idx < len(row):
                col_name = header.iloc[r_idx].strip()
                if col_name: 
                     resistance_values[col_name].append(row.iloc[r_idx])

        n_value_final = None
        labs_value_final = None

        n_indices = col_info.get("n", [])
        lab_indices = col_info.get("labs", [])

        if n_indices:
            n_idx = n_indices[0]
            if n_idx < len(row):
                 n_value_raw = row.iloc[n_idx]
                 if isinstance(n_value_raw, str):
                     n_value_part = n_value_raw.split('/')[0].strip()
                     n_value_final = pd.to_numeric(n_value_part, errors='coerce')
                 elif pd.notna(n_value_raw):
                     n_value_final = pd.to_numeric(n_value_raw, errors='coerce')

        if lab_indices:
            lab_idx = lab_indices[0]
            if lab_idx < len(row):
                 labs_value_raw = row.iloc[lab_idx]
                 if isinstance(labs_value_raw, str):
                     labs_value_part = labs_value_raw.split('/')[0].strip()
                     labs_value_final = pd.to_numeric(labs_value_part, errors='coerce')
                 elif pd.notna(labs_value_raw):
                     labs_value_final = pd.to_numeric(labs_value_raw, errors='coerce')

        if pd.isna(n_value_final):
            continue

        r_value = None
        ir_value = None 
        esbl_value = None

        def get_value_by_iterator(metric_name, default=None):
            values_list = resistance_values.get(metric_name)
            if values_list and iterator < len(values_list):
                value = values_list[iterator]
                if isinstance(value,str):
                    return value
                else:
                    return pd.to_numeric(value, errors='coerce')
            return default 

        def get_last_value(metric_name, default=None):
             values_list = resistance_values.get(metric_name)
             if values_list:
                value = values_list[-1]
                if isinstance(value,str):
                    return value
                else:
                    return pd.to_numeric(value, errors='coerce')
             return default

        if year in ['2014', '2015']:
            r_value = get_last_value("%R")
            ir_value = get_last_value("%IR")
            esbl_value = get_last_value("% ESBL")
        else:
            r_value = get_value_by_iterator("%R")
            ir_value = get_value_by_iterator("%IR")
            esbl_value = get_value_by_iterator("% ESBL")

        rows.append({
            "Microorganism": microorganism,
            "Antibiotics": current_abx,
            "Year": year,
            "Country": country_cell, 
            "%ESBL": esbl_value,
            "%R": r_value,
            "Non-susceptible": ir_value, 
            "N": n_value_final,
            "Fully susceptible": None,
            "Number of laboratories": labs_value_final 
        })

    return rows


def process_country_agnostic_data(df, microorganism, abx_str, year):
    """
    Processes DataFrames without a 'Country' column, often containing aggregated data
    (like 'EU Total') and rows distinguishing 'fully susceptible' patterns.

    """
    result = []

    header_row_idx = find_first_data_header(df)
    if header_row_idx is None:
        warnings.warn(f"Could not find data header row for {microorganism} in year {year} (agnostic). Skipping sheet.", UserWarning)
        return result

    header_row = df.iloc[header_row_idx].fillna('').astype(str).str.strip()

    resistance_pattern_col = None
    for idx, val in enumerate(header_row):
        if "resistance pattern" in val.lower():
            resistance_pattern_col = idx
            break
    if resistance_pattern_col is None:
         warnings.warn(f"'resistance pattern' column not found for {microorganism} in year {year} (agnostic). Antibiotic specifics might be lost.", UserWarning)

    n_patterns_strict_lower = N_PATTERNS_LOWER - {'number of laboratories'}
    n_cols = [idx for idx, val in enumerate(header_row) if val.lower() in n_patterns_strict_lower]
    resistance_cols = [idx for idx, val in enumerate(header_row) if val.lower() in RESISTANCE_PATTERNS_LOWER]
    lab_cols = [idx for idx, val in enumerate(header_row) if "laboratories" in val.lower()]

    for row_idx in range(header_row_idx + 1, len(df)):
        row = df.iloc[row_idx]

        if row.isna().all():
            continue

        is_fully_susceptible = False
        pattern_value = ""
        if resistance_pattern_col is not None and resistance_pattern_col < len(row):
            pattern_raw = row.iloc[resistance_pattern_col]
            if pd.notna(pattern_raw):
                pattern_value = str(pattern_raw).strip().lower()
                if "fully susceptible" in pattern_value:
                    is_fully_susceptible = True

        current_abx = abx_str
        if pattern_value and not is_fully_susceptible:
             matched_abx_from_pattern = extract_antibiotics(pattern_value)
             if matched_abx_from_pattern:
                 current_abx = ", ".join(matched_abx_from_pattern)
             else:     
                  continue
    
        resistance_values = {}
        for r_idx in resistance_cols:
            if r_idx < len(row) and pd.notna(row.iloc[r_idx]):
                col_name = header_row.iloc[r_idx]
                resistance_values[col_name] = row.iloc[r_idx]

        fully_susceptible_value = None
        r_value = None

        percent_total = resistance_values.get("% of total*", resistance_values.get("% of total**"))

        if is_fully_susceptible:
             fully_susceptible_value = percent_total
        else:
             r_value = percent_total if percent_total is not None else resistance_values.get("%R")

        n_value_final = None
        if n_cols:
             n_idx = n_cols[0]
             if n_idx < len(row) and pd.notna(row.iloc[n_idx]):
                  n_value_raw = row.iloc[n_idx]
                  if isinstance(n_value_raw, str):
                      n_value_part = n_value_raw.split('/')[0].strip()
                      n_value_final = pd.to_numeric(n_value_part, errors='coerce')
                  else: 
                      n_value_final = pd.to_numeric(n_value_raw, errors='coerce')
        if pd.isna(n_value_final):
            continue

        labs_value_final = None
        if lab_cols:
             lab_idx = lab_cols[0]
             if lab_idx < len(row) and pd.notna(row.iloc[lab_idx]):
                 labs_value_raw = row.iloc[lab_idx]
                 if isinstance(labs_value_raw, str):
                      labs_value_part = labs_value_raw.split('/')[0].strip()
                      labs_value_final = pd.to_numeric(labs_value_part, errors='coerce')
                 else:
                      labs_value_final = pd.to_numeric(labs_value_raw, errors='coerce')

        record = {
            "Microorganism": microorganism,
            "Antibiotics": current_abx,
            "Year": year,
            "Country": "EU Total", 
            "N": n_value_final,
            "Fully susceptible": fully_susceptible_value,
            "%R": r_value,
            "%ESBL": resistance_values.get("% ESBL"),
            "Non-susceptible":resistance_values.get("%IR"),
            "Number of laboratories": labs_value_final
        }
        result.append(record)

    return result