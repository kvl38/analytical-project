import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import mlflow.catboost
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from training.data_loader import dfs_loader
from training.config import *
from training.preprocessing import data_converter
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def getting_best_model(df: pd.DataFrame) -> None:
    """
    Выполняет подбор и обучение лучшей модели машинного обучения с помощью GridSearchCV,
    логирует параметры и метрики в MLflow, регистрирует модель как Production.

    Параметры:
    -----------
    df : pd.DataFrame
        Предобработанный датафрейм, содержащий входные признаки и целевую переменную.

    Возвращает:
    -----------
    None
        Функция сохраняет лучшую модель в MLflow и ничего не возвращает.
    """
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Рассмотрим размеры сформированных выборок
    logger.info(f"Размеры сформированных обучающей и тренировочной выборок: {X_train.shape, X_test.shape}")
    logger.info("Данные успешно загружены.")

    # Категориальные и непрерывные признаки
    for column in df.select_dtypes(include='object').columns:
        df[column] = df[column].astype('category')

    ohe_columns = list(df.select_dtypes(include='category').columns)
    num_columns = INTERVAL_COLUMNS

    # Препроцессор
    data_preprocessor = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ohe_columns),
        ('num', 'passthrough', num_columns)
    ],
        remainder='passthrough'
    )

    # Веса классов
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # Пайплайн
    pipe = Pipeline([
        ('preprocessor', data_preprocessor),
        ('models', LogisticRegression(random_state=RANDOM_STATE))
    ])

    param_grid = [
        {
            'models': [LogisticRegression(max_iter=1000, penalty='l2', class_weight='balanced')],
            'models__C': [0.01, 0.1, 1],
            'preprocessor__num': [StandardScaler(), MinMaxScaler()]
        },

        {
            'models': [RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')],
            'models__n_estimators': [100, 200],
            'models__max_depth': [7, 10],
            'models__min_samples_split': [3, 7],
            'preprocessor__num': ['passthrough']
        },

        {
            'models': [CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, class_weights=class_weights_dict)],
            'models__iterations': [500, 1000],
            'models__depth': [3, 5, 7],
            'preprocessor__num': ['passthrough']
        }
    ]

    # Обучение и логирование
    logger.info("Настройка MLflow и запуск GridSearchCV...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Model_to_prod")
    run_name = "CatBoost_Classification_3"

    with mlflow.start_run(run_name=run_name):
        grid_search = GridSearchCV(pipe, param_grid, cv=CV, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        logger.info("GridSearch завершён.")


        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("roc_auc", grid_search.best_score_)
        best_model = grid_search.best_estimator_

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="BestModel"
        )

        # Дать системе время зарегистрировать модель
        time.sleep(2)

        client = MlflowClient()
        model_name = "BestModel"
        versions = client.search_model_versions(f"name = '{model_name}'")
        if not versions:
            raise ValueError("Model was not registered properly.")
        latest_version = max(int(v.version) for v in versions)
        client.set_registered_model_alias( name=model_name, alias="Production", version=latest_version)
        logger.info(f"Лучшая модель сохранена в MLflow: {grid_search.best_params_}")

if __name__ == "__main__":
    logger.info("Подготовка модели началась")
    # Загрузка и предобработка данных
    merged_df = dfs_loader(CONTRACT_LINK, PERSONAL_LINK, INTERNET_LINK, PHONE_LINK)
    merged_df = data_converter(merged_df)
    merged_df.drop(COLUMNS_TO_DELETE, axis=1, inplace=True)

    # Получение ml-модели
    getting_best_model(merged_df)
    logger.info("Модель готова к использованию")
