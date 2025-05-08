import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import optuna
import re 

print("--- Запуск анализа модели CatBoost с подбором гиперпараметров Optuna ---")

def sanitize_filename(name):
    """Очищает строку для использования в качестве части имени файла или каталога."""
    name = name.replace('%', 'perc') 
    name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    name = re.sub(r'_+', '_', name) 
    name = name.strip('_')
    return name if name else "unknown_target"


target_variable = 'log2_tet_mic' 

N_CV_SPLITS = 3
N_OPTUNA_TRIALS = 50 

safe_target_name = sanitize_filename(target_variable)
OPTUNA_RESULTSDIR = f'optuna_study_catboost_{safe_target_name}'
OPTUNA_RESULTS_DIR = os.path.join('reports', OPTUNA_RESULTSDIR)
MODELS_DIR = 'models' 

os.makedirs(OPTUNA_RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) 


# --- 1. Загрузка данных ---
data_path = os.path.join('data', 'processed', 'amr.csv') 
try:
    print(f"Загрузка данных из: {data_path}")
    df = pd.read_csv(data_path)
    print("Данные успешно загружены.")
    print(f"Размер датасета: {df.shape}")
except FileNotFoundError:
    print(f"Ошибка: Файл не найден по пути {data_path}. Проверьте структуру папок.")
    exit()

# --- 2. Определение признаков и целевой переменной ---
print(f"--- Обучение модели для целевой переменной: {target_variable} ---")

feature_columns = [
    'Microorganism', 'Country', 'Year', 'Continent', 'Beta.lactamase', 'Group',
    'NG_MAST', 'N', 'Number of laboratories', '3GCREC', '3GCRKP',
    'AMINOGLYCOSIDES', 'AMINOPENICILLINS', 'AZM', 'CARBAPENEMS', 'CEFTAZIDIME',
    'CFX', 'CIP', 'CRO', 'FLUOROQUINOLONES', 'GENTAMICIN', 'MACROLIDES',
    'METICILLIN', 'PEN', 'PENICILLINS', 'PIPERACILLIN+TAZOBACTAM', 'TET',
    'THIRD-GENERATION CEPHALOSPORINS', 'VANCOMYCIN'
]
categorical_features = ['Microorganism', 'Country', 'Continent', 'Beta.lactamase']

# Проверка наличия всех колонок, включая новую целевую
missing_cols = [col for col in feature_columns if col not in df.columns] 
if target_variable not in df.columns:
    missing_cols.append(target_variable)

if missing_cols:
    print(f"Ошибка: Следующие колонки не найдены в датасете: {missing_cols}")
    print(f"Доступные колонки: {df.columns.tolist()}")
    exit()

# --- 3. Предобработка ---
print("Предобработка данных...")
if 'Year' in df.columns:
    df.sort_values('Year', inplace=True)
else:
    print("Ошибка: Колонка 'Year' отсутствует.")
    exit()

for col in categorical_features:
    if col in df.columns and df[col].isnull().any(): 
        mode_val = df[col].mode()[0]
        df.loc[:, col] = df[col].fillna(mode_val)

numeric_features = [col for col in feature_columns if col not in categorical_features]
for col in numeric_features:
    if col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                print(f"Внимание: Медиана для '{col}' не может быть вычислена. Заполняем нулем.")
                df.loc[:, col] = df[col].fillna(0)
            else:
                df.loc[:, col] = df[col].fillna(median_val)

if target_variable in df.columns:
    df.loc[:, target_variable] = pd.to_numeric(df[target_variable], errors='coerce')
    df.dropna(subset=[target_variable], inplace=True)
    print(f"Целевая переменная '{target_variable}' обработана как числовая.")
else:
    print(f"Критическая ошибка: Целевая колонка {target_variable} не найдена перед разделением данных!")
    exit()

# --- 4. Разделение на данные для CV/обучения и финальный тест ---
print("Разделение данных...")
if df['Year'].nunique() > 1:
    split_year = df['Year'].max() - 1
    if split_year < df['Year'].min(): 
        split_year = df['Year'].max()
    
    df_for_cv_train = df[df['Year'] < split_year].copy()
    df_final_test = df[df['Year'] >= split_year].copy()

    if df_for_cv_train.empty and not df.empty:
        print("Предупреждение: df_for_cv_train пуст после разделения по годам. Возможно, слишком мало уникальных лет.")
        print(f"Уникальные года в df: {df['Year'].unique()}")
        print(f"split_year: {split_year}")

        print("Используем весь датасет для CV, финальный тест будет отсутствовать.")
        df_for_cv_train = df.copy()
        df_final_test = pd.DataFrame()

else:
    print("В данных только один год или года не различаются для разделения. Финальный тест по времени не создается.")
    df_for_cv_train = df.copy()
    df_final_test = pd.DataFrame()


X_for_cv_train = df_for_cv_train[feature_columns]
y_for_cv_train = df_for_cv_train[target_variable]

if not df_final_test.empty:
    X_final_test = df_final_test[feature_columns]
    y_final_test = df_final_test[target_variable]
    print(f"Размер данных для финального теста: X={X_final_test.shape}, y={y_final_test.shape}")
else:
    X_final_test, y_final_test = pd.DataFrame(), pd.Series(dtype='float64')
    print("Финальный тестовый набор отсутствует или пуст.")

print(f"Размер данных для CV и обучения финальной модели: X={X_for_cv_train.shape}, y={y_for_cv_train.shape}")

if X_for_cv_train.empty or y_for_cv_train.empty:
    print("Ошибка: Нет данных для обучения после разделения (X_for_cv_train или y_for_cv_train пусты). Выход.")
    exit()


# --- 5. Optuna: Определение функции objective ---
def objective(trial):
    trial_num = trial.number
    trial_dir = os.path.join(OPTUNA_RESULTS_DIR, f'trial_{trial_num:04d}')
    os.makedirs(trial_dir, exist_ok=True)

    params = {
        'iterations': trial.suggest_int('iterations', 500, 2500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0, step=0.05),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100, log=True),
        'loss_function': 'RMSE', 
        'eval_metric': 'RMSE', 
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 50, 200, step=25)
    }

    with open(os.path.join(trial_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    fold_rmses = []
    fold_actual_iterations = []
    fold_maes = []

    print(f"Optuna Trial {trial_num} ({target_variable}): Starting CV with {N_CV_SPLITS} splits.")

    for fold, (train_index, val_index) in enumerate(tscv.split(X_for_cv_train, y_for_cv_train)):
        X_train_fold, X_val_fold = X_for_cv_train.iloc[train_index], X_for_cv_train.iloc[val_index]
        y_train_fold, y_val_fold = y_for_cv_train.iloc[train_index], y_for_cv_train.iloc[val_index]

        if X_train_fold.empty or X_val_fold.empty:
            print(f"  Trial {trial_num}, Fold {fold + 1}: Skipped due to empty data.")
            return float('inf')

        train_pool_fold = Pool(data=X_train_fold, label=y_train_fold, cat_features=categorical_features)
        val_pool_fold = Pool(data=X_val_fold, label=y_val_fold, cat_features=categorical_features)

        model_fold = CatBoostRegressor(**params)
        model_fold.fit(train_pool_fold, eval_set=val_pool_fold)

        y_pred_fold = model_fold.predict(X_val_fold)
        rmse_fold = root_mean_squared_error(y_val_fold, y_pred_fold)
        mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)

        fold_rmses.append(rmse_fold)
        fold_maes.append(mae_fold)
        fold_actual_iterations.append(model_fold.get_best_iteration() if model_fold.get_best_iteration() is not None else params['iterations'])
        print(f"  Trial {trial_num}, Fold {fold + 1}: RMSE={rmse_fold:.4f}, Actual Iterations={fold_actual_iterations[-1]}")

    fold_results_df = pd.DataFrame({
        'fold': range(1, len(fold_rmses) + 1),
        'rmse': fold_rmses,
        'mae': fold_maes,
        'actual_iterations': fold_actual_iterations
    })
    fold_results_df.to_csv(os.path.join(trial_dir, 'fold_metrics.csv'), index=False)

    average_rmse = np.mean(fold_rmses) if fold_rmses else float('inf')
    std_rmse = np.std(fold_rmses) if fold_rmses else float('nan')
    average_mae = np.mean(fold_maes) if fold_maes else float('inf')
    avg_actual_iter = np.mean(fold_actual_iterations) if fold_actual_iterations else float('nan')

    with open(os.path.join(trial_dir, 'trial_summary.txt'), 'w') as f:
        f.write(f"Average RMSE: {average_rmse:.4f}\n")
        f.write(f"Std RMSE: {std_rmse:.4f}\n")
        f.write(f"Average MAE: {average_mae:.4f}\n")
        f.write(f"Average Actual Iterations: {avg_actual_iter:.0f}\n")

    print(f"Optuna Trial {trial_num} ({target_variable}): Average CV RMSE = {average_rmse:.4f}")
    return average_rmse


# --- 6. Запуск исследования Optuna ---
print(f"\n--- Запуск Optuna для подбора гиперпараметров ({N_OPTUNA_TRIALS} испытаний) для '{target_variable}' ---")

optuna_study_name = f'catboost_tuning_{safe_target_name}'
study = optuna.create_study(direction='minimize', study_name=optuna_study_name)
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, timeout=3600) # Таймаут 1 час

print(f"\n--- Подбор гиперпараметров Optuna для '{target_variable}' завершен ---")
print(f"Лучшее значение (минимальный средний RMSE по CV): {study.best_value:.4f}")
print("Лучшие гиперпараметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

best_hyperparams = study.best_params.copy()


# --- 7. Обучение финальной модели с лучшими гиперпараметрами ---
print(f"\n--- Обучение финальной модели с лучшими гиперпараметрами для '{target_variable}' ---")
if not X_for_cv_train.empty:
    final_train_pool = Pool(data=X_for_cv_train, label=y_for_cv_train, cat_features=categorical_features)
    
    final_model_params = best_hyperparams.copy()
    final_model_params['verbose'] = 100

    final_eval_pool = None
    if not X_final_test.empty:
        final_eval_pool = Pool(data=X_final_test, label=y_final_test, cat_features=categorical_features)
        print("Используем финальный тестовый набор для early stopping финальной модели.")
    else:
        print("Финальный тестовый набор отсутствует. Early stopping для финальной модели будет на основе трейна или отключен.")

    final_model = CatBoostRegressor(**final_model_params)

    print(f"Обучение финальной модели на данных X_for_cv_train для '{target_variable}'...")
    if final_eval_pool:
        final_model.fit(final_train_pool, eval_set=final_eval_pool)
    else:
        final_model.fit(final_train_pool)

    # --- 8. Оценка финальной модели на финальном тестовом наборе ---
    if not X_final_test.empty:
        print(f"\n--- Оценка финальной модели на финальном тестовом наборе для '{target_variable}' ---")
        y_pred_final = final_model.predict(X_final_test)

        mae_final = mean_absolute_error(y_final_test, y_pred_final)
        rmse_final = root_mean_squared_error(y_final_test, y_pred_final)
        r2_final = r2_score(y_final_test, y_pred_final)

        print(f"Финальный тест MAE:  {mae_final:.4f}")
        print(f"Финальный тест RMSE: {rmse_final:.4f}")
        print(f"Финальный тест R2:   {r2_final:.4f}")

        final_metrics_path = os.path.join(OPTUNA_RESULTS_DIR, 'final_model_metrics.json')
        with open(final_metrics_path, 'w') as f:
            json.dump({
                'target_variable': target_variable,
                'MAE': mae_final,
                'RMSE': rmse_final,
                'R2': r2_final,
                'best_hyperparameters': best_hyperparams
            }, f, indent=4)
        print(f"Метрики финальной модели и лучшие параметры сохранены в: {final_metrics_path}")

    else:
        print("\nФинальный тестовый набор отсутствует, оценка на нем не проводится.")

    # --- 9. Важность признаков для финальной модели ---
    print(f"\n--- Важность признаков финальной модели для '{target_variable}' ---")
    try:
        feature_importance_values = final_model.get_feature_importance(final_train_pool)
        feature_names = X_for_cv_train.columns
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
        importance_df = importance_df.sort_values(by='importance', ascending=False)

        print("Топ 10 признаков:")
        try:
            from tabulate import tabulate
            print(tabulate(importance_df.head(10), headers='keys', tablefmt='pipe', showindex=False))
        except ImportError:
            print(importance_df.head(10).to_string(index=False))
        
        feature_importance_plot_path = os.path.join(MODELS_DIR, f'feature_importance_optuna_final_{safe_target_name}.png')
        plt.figure(figsize=(12, max(6, len(importance_df['feature'][:15]) * 0.4)))
        plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
        plt.xlabel("Важность признаков")
        plt.ylabel("Признак")
        plt.title(f"Важность признаков для '{target_variable}' (Optuna)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(feature_importance_plot_path)
        print(f"График важности признаков сохранен в {feature_importance_plot_path}")
        plt.close() 

    except Exception as e:
        print(f"Ошибка при выводе/сохранении важности признаков: {e}")

    # --- 10. Сохранение финальной модели ---

    model_save_path = os.path.join(MODELS_DIR, f'catboost_optuna_final_{safe_target_name}.cbm')
    print(f"\nСохранение обученной финальной модели в {model_save_path}...")
    final_model.save_model(model_save_path)
    print("Финальная модель успешно сохранена.")

else:
    print(f"Нет данных для обучения финальной модели (X_for_cv_train пуст) для '{target_variable}'.")

print(f"\n--- Анализ модели CatBoost с Optuna для '{target_variable}' завершен ---")