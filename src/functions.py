import pandas as pd
import os
import warnings
import re
warnings.simplefilter("ignore", UserWarning)

# Function to check file validity
def checkFiles(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    
    if not file_path.lower().endswith(('.xlsx', '.xls')):
        raise ValueError(f"Файл {file_path} не является Excel-файлом.")

    try:
        excel_file = pd.ExcelFile(file_path)
        if not excel_file.sheet_names:
            print(f"{file_path} - does not contain any sheets")

        for sheet in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            if df.empty:
                print(f'{file_path} - {sheet} - пуст')
    except Exception as e:
        raise RuntimeError(f"Ошибка при открытии Excel-файла {file_path}: {str(e)}")


# Function to calculate the number of tested isolates
def calculate_total_n(file_paths):
    search_patterns = ['Number of.1','Number of isolates', 'Number of \ntested isolates', 'Number of tested isolates',
                     'Number of \n3GCREC included in analysis/ total number of 3GCREC',
                     'Number of\n3GCRKP\nincluded in\nanalysis/total\nnumber of\n3GCRKP',
                     'N','N.1','N.2','N.3']
    total_N = 0
    for file_path in file_paths:
        excel_file = pd.ExcelFile(file_path)

        for sheet in excel_file.sheet_names:
            std_header = 2

            df = pd.read_excel(file_path, header=std_header, sheet_name=sheet)
            while not df.empty and df.columns[0].startswith("Unnamed"):
                std_header += 1
                df = pd.read_excel(file_path, header=std_header, sheet_name=sheet)
            for search_pattern in search_patterns:
                for col in df.columns:
                    if col != search_pattern:
                        continue

                    if col in ['Number of \n3GCREC included in analysis/ total number of 3GCREC',
                            'Number of\n3GCRKP\nincluded in\nanalysis/total\nnumber of\n3GCRKP']:
                        df[col] = df[col].astype(str).str.extract(r'(\d+)')[0]
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif col in ['Number of isolates', 'Number of \ntested isolates', 'Number of tested isolates', 'Number of.1']:
                        df_filtered = df[~df.iloc[:, 0].str.startswith('Total', na=False)]
                        df[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                    else:
                        if (file_path.startswith("data\\raw\\2014") or file_path.startswith("data\\raw\\2015")) and col != "N.3":
                            break
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    sum_result = df[col].sum(skipna=True)

                    print(file_path, " - ", sum_result)
                    total_N += sum_result
    print(f"Total number of tested isolates throughout 2010-2015 : {total_N}")

# Function to get the resistance percentages in a single list

def get_resistance_percentages(file_paths):
    search_patterns = [
    "% ESBL", "%R ", "%R .1", "%R .2", "%R .3",
    "%IR ", "%IR .1", "%IR .2", "%IR .3",
    "% of total*", "% of total**"
    ]

    resistances = []

    for file_path in file_paths:
        excel_file = pd.ExcelFile(file_path)

        for sheet in excel_file.sheet_names:
            std_header = 2
            df = pd.read_excel(file_path, header=std_header, sheet_name=sheet)

            while not df.empty and df.columns[0].startswith("Unnamed"):
                std_header += 1
                df = pd.read_excel(file_path, header=std_header, sheet_name=sheet)
        
            for search_pattern in search_patterns:
                for col_idx, col in enumerate(df.columns):
                    if col == search_pattern:

                        left_col_idx = col_idx - 1
                        if left_col_idx < 0:  
                            continue
                        
                        if col in ["% of total*", "% of total**"]:
                            df_filtered = df[~df.iloc[:, 0].astype(str).str.startswith('Total', na=False)]
                            df_filtered.loc[df_filtered[col].astype(str).str.strip() == '<0.1', col] = 0.05
                            numeric_values = pd.to_numeric(df_filtered[col], errors='coerce').dropna()
                        elif col in ["%R ", "%R .1", "%R .2", "%R .3", "%IR ", "%IR .1", "%IR .2", "%IR .3"]:
                            if (file_path.startswith("data\raw\2014") or file_path.startswith("data\raw\2015")) and col not in ["%R .3", "%IR .3"]:
                                continue
                            numeric_values = pd.to_numeric(df[col], errors='coerce').dropna()
                        else:
                            numeric_values = pd.to_numeric(df[col], errors='coerce').dropna()
                        
                        left_column = df.iloc[:, left_col_idx]
                        for i, value in enumerate(df[col]):
                            if pd.notna(value): 
                                count = pd.to_numeric(left_column.iloc[i], errors="coerce")
                                if pd.notna(count) and count > 0:
                                    resistances.extend([value] * int(count))


    resistances_filtered = []
    for value in resistances:
        if isinstance(value, str):
            if '\xa0' in value:
                try:
                    value = float(value.replace('\xa0', '').strip())
                    resistances_filtered.append(value)
                except ValueError:
                    continue
        else:
            resistances_filtered.append(value)

    return resistances_filtered



def get_year_from_path(file_path):
    for year in ["2013", "2014", "2015"]:
        if f"data\\raw\\{year}" in file_path:
            return int(year)
    return None

def get_N_per_year(file_paths):
    search_patterns = {
        'Number of.1', 'Number of isolates', 'Number of \ntested isolates', 
        'Number of tested isolates', 'Number of \n3GCREC included in analysis/ total number of 3GCREC',
        'Number of\n3GCRKP\nincluded in\nanalysis/total\nnumber of\n3GCRKP',
        'N', 'N.1', 'N.2', 'N.3'
    }

    N_per_year = {year: [] for year in range(2010, 2016)}

    year_column_map = {
        'N': 2010, 
        'N.1': 2011, 
        'N.2': 2012, 
        'N.3': 2013
    }

    for file_path in file_paths:
        excel_file = pd.ExcelFile(file_path)

        for sheet in excel_file.sheet_names:
            std_header = 2
            df = pd.read_excel(file_path, header=std_header, sheet_name=sheet)

            while not df.empty and df.columns[0].startswith("Unnamed"):
                std_header += 1
                df = pd.read_excel(file_path, header=std_header, sheet_name=sheet)

            year = get_year_from_path(file_path)
            if year is None:
                continue

            for col in df.columns:
                if col not in search_patterns:
                    continue  

                if col in ('N', 'N.1', 'N.2', 'N.3'):
                    if year == 2013 and col in year_column_map:
                        N_per_year[year_column_map[col]].extend(df[col].dropna().tolist())
                    elif year in {2014, 2015} and col == "N.3":
                        N_per_year[year].extend(df[col].dropna().tolist())
                else: 
                    if "3GCREC" in col or "3GCRKP" in col:
                        df[col] = df[col].astype(str).str.extract(r'(\d+)')[0]
                    
                    if col in {'Number of isolates', 'Number of \ntested isolates', 'Number of tested isolates', 'Number of.1'}:
                        df = df[~df.iloc[:, 0].str.startswith('Total', na=False)]
                    
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    N_per_year[year].extend(df[col].dropna().tolist())

    N_per_year_cleaned = {
    year: [value for value in values if isinstance(value, (int, float))]
    for year, values in N_per_year.items()
    }

    sum_per_year = {year: sum(values) for year, values in N_per_year_cleaned.items()}
    return sum_per_year

