import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_excel_files(file_paths):
    """Reads all sheets from a list of Excel files."""
    all_data = {}
    for file_path in file_paths:
        all_data[file_path] = pd.read_excel(file_path, sheet_name=None, header=None)
    return all_data

def clean_and_filter_sheet(df_sheet, file_path, sheet_name):
    """Cleans and filters a sheet based on specific criteria."""
    df_sheet = df_sheet.dropna(how='all')

    header_index = df_sheet.apply(lambda row: row.astype(str).str.contains("Country").any(), axis=1).idxmax()
    if header_index == 0 and "Country" not in df_sheet.iloc[0].astype(str).values:
        return None


    df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=header_index)

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

    for column in ['N_2012', 'N_2013', 'N_2014', 'N_2015']:
        if column in df_sheet.columns:
            df_sheet[column] = pd.to_numeric(df_sheet[column], errors='coerce').fillna(0)

    filtered_df = df_sheet[~df_sheet['Country'].astype(str).str.contains(
        r'EU/EEA\s*\(population-\s*weighted\s*mean\)', flags=re.IGNORECASE, na=False)]
    
    return filtered_df


def clean_and_extract_resistance(df_sheet, file_path, sheet_name):
    """Cleans the sheet and extracts resistance percentage data."""
    df_sheet = df_sheet.dropna(how='all')

    header_index = df_sheet.apply(lambda row: row.astype(str).str.contains("Country", case=False, na=False).any(), axis=1).idxmax()

    if not header_index.any():
        return None

    df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=header_index)

    df_sheet = df_sheet.rename(columns=lambda x: str(x).strip())

    resistance_cols = [col for col in df_sheet.columns if re.search(r"%R", col, re.IGNORECASE)]
    print(resistance_cols)

    for col in resistance_cols:
        df_sheet[col] = pd.to_numeric(df_sheet[col], errors="coerce")

    resistance_values = df_sheet[resistance_cols].values.flatten()
    return resistance_values[~np.isnan(resistance_values)]


def calculate_totals(df, columns):
    """Calculates the total values for specific columns."""
    return df[columns].sum()

def plot_totals(summary_df):
    """Plots the totals as a bar chart."""

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df['Year'], summary_df['Total Patients'], color='#53377A', edgecolor='black', align='edge', linewidth=0.5, width=1)

    plt.title('Tested Isolates by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of isolates')
    plt.xticks(summary_df['Year'])
    plt.tight_layout()
    plt.show()


def plot_resistance_distribution(resistance_percentages, num_tested):
    """Plots a histogram for resistance percentage distribution relative to the number of tested isolates."""
    plt.figure(figsize=(8, 5))

    weights = num_tested / np.sum(num_tested)

    sns.histplot(resistance_percentages, bins=30, weights=weights, kde=True, color="#53377A", edgecolor="black", alpha=0.7)

    plt.xlabel("Percentage of Resistance (%)")
    plt.ylabel("Proportion of Total Tested Isolates")
    plt.title("Distribution of Resistance Adjusted by Number of Tested Isolates")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
