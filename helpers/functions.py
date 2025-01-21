import pandas as pd
import re

def read_excel_files(file_paths):
    """Reads all sheets from a list of Excel files."""
    all_data = {}
    for file_path in file_paths:
        all_data[file_path] = pd.read_excel(file_path, sheet_name=None, header=None)
    return all_data

def clean_and_filter_sheet(df_sheet, file_path, sheet_name):
    """Cleans and filters a sheet based on specific criteria."""
    df_sheet = df_sheet.dropna(how='all')

    # Find header row by looking for the "Country" keyword
    header_index = df_sheet.apply(lambda row: row.astype(str).str.contains("Country").any(), axis=1).idxmax()
    if not header_index.any():
        return None

    df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=header_index)

    # Rename columns
    df_sheet = df_sheet.rename(columns={
        'N': 'N_2012',
        '%R ': '%R_2012',
        '(95% CI)': '(95% CI)_2012',
        'N.1': 'N_2013',
        '%R .1': '%R_2013',
        '(95% CI).1': '(95% CI)_2013',
        'N.2': 'N_2014',
        '%R .2': '%R_2014',
        '(95% CI).2': '(95% CI)_2014',
        'N.3': 'N_2015',
        '%R .3': '%R_2015',
        '(95% CI).3': '(95% CI)_2015'
    })

    # Convert numeric columns
    for column in ['N_2012', 'N_2013', 'N_2014', 'N_2015']:
        if column in df_sheet.columns:
            df_sheet[column] = pd.to_numeric(df_sheet[column], errors='coerce').fillna(0)

    # Filter rows
    filtered_df = df_sheet[~df_sheet['Country'].astype(str).str.contains(
        r'EU/EEA\s*\(population-\s*weighted\s*mean\)', flags=re.IGNORECASE, na=False)]
    
    return filtered_df

def calculate_totals(df, columns):
    """Calculates the total values for specific columns."""
    return df[columns].sum()

def plot_totals(summary_df):
    """Plots the totals as a bar chart."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df['Year'], summary_df['Total Patients'], color='#53377A', edgecolor='black', align='edge', linewidth=0.5, width=1)

    plt.title('Tested Isolates by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of isolates')
    plt.xticks(summary_df['Year'])
    plt.tight_layout()
    plt.show()
