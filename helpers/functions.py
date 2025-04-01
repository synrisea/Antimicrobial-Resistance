import pandas as pd
import os
import warnings
warnings.simplefilter("ignore", UserWarning)

def checkFiles(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    

    if not file_path.lower().endswith(('.xlsx', '.xls')):
        raise ValueError(f"Файл {file_path} не является Excel-файлом.")

    try:
        excel_file = pd.ExcelFile(file_path)
        if not excel_file.sheet_names:
            print(f"{file_path} - does not conatin any sheets")

        for sheet in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)

            if df.empty:
                print(f'{file_path} - {sheet} - пуст')
    except Exception as e:
        raise RuntimeError(f"Ошибка при открытии Excel-файла {file_path}: {str(e)}")