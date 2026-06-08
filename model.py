import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import os
import optuna
import mlflow
import joblib
import mlflow.xgboost
#Настройка бд для mlflow
db_path = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(db_path)

#Создаем или используем уже готовый эксперимент
experiment_name = "SWS_Helix_Optimization"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except Exception:
    # Если эксперимент уже создан, просто берем его
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

print('---Загрузка данных---')
df = pd.read_csv(r'G:\Projects_Py\ab_tests_and_more\helix_sws_dataset.csv')
print("Реальные колонки в файле:", df.columns.tolist())
X = df[['shag_spirali_mm', 'vnutr_diameter_d_mm', 'r_provodnika', 'dielectric_eps']]
y = df['polosa_4astot_ghz'] 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
features = list(X.columns)

# Настраиваем MLflow эксперимент
mlflow.set_experiment("SWS_Helix_Optimization")


# --- 2. Определение целевой функции для Optuna ---
def objective(trial):
    # Задаем пространство поиска гиперпараметров
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
    }

    # Включаем автоматическое вложенное логирование для каждого триала в MLflow
    with mlflow.start_run(
        run_name=f"Optuna_Trial_{trial.number}", nested=True
    ) as run:
        # Обучаем модель с текущими параметрами
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        # Считаем метрику
        preds = model.predict(X_test)
        rmse = np.sqrt(root_mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        # Логируем параметры и метрики текущего триала в MLflow
        mlflow.log_params(params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2_Score", r2)
        # Optuna минимизирует возвращаемое значение rmse
        return rmse


# --- Запуск оптимизации Optuna ---
# Создаем родительский запуск в MLflow для всей сессии подбора
with mlflow.start_run(run_name="Optuna_Hyperparameter_Tuning") as parent_run:

    # Создаем исследование Optuna rmse - > min
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)  #30 итераций 

    print("\n[Успех] Подбор параметров завершен!")
    print(f"Лучшие параметры: {study.best_params}")
    print(f"Наименьший RMSE: {study.best_value:.4f}")

    # --- 4. Обучение финальной модели ---
    best_params = study.best_params
    best_params["random_state"] = 42

    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    final_preds = best_model.predict(X_test)
    final_rmse = np.sqrt(root_mean_squared_error(y_test, final_preds))
    final_r2 = r2_score(y_test, final_preds)

    # Логируем результаты best модели в главный запуск MLflow
    mlflow.log_dict(best_params, "best_parameters.json")
    mlflow.log_metric("Final_RMSE", final_rmse)
    mlflow.log_metric("Final_R2_Score", final_r2)

    # Сохраняем финальную модель в артефакты MLflow
    mlflow.xgboost.log_model(best_model.get_booster(), "best_spiral_model")

    # --- Экспорт для Docker API ---
    # Переводим в Booster для совместимости с нашим app.py
    best_model.get_booster().save_model("xgb_model.json")
    joblib.dump(features, "model_features.joblib")

    print(
        f"\nФинальная модель сохранена локально! Итоговый R2: {final_r2:.4f}, RMSE: {final_rmse:.2f}"
    )