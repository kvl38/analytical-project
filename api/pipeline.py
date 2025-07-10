import pandas as pd
from datetime import datetime
from api.config import curr_day
from training.config import COLUMNS_TO_DELETE
import logging

logger = logging.getLogger(__name__)

def preprocess_input(input_dict: dict) -> pd.DataFrame:
    """
    Выполняет предобработку входных данных для модели:
    преобразует словарь в DataFrame, считает количество дней с начала подписки,
    удаляет ненужные столбцы и приводит строковые признаки к категориальным.

    Параметры:
    -----------
    input_dict : dict
        Словарь с входными признаками одного объекта.

    Возвращает:
    -----------
    pd.DataFrame
        Предобработанный DataFrame, готовый для подачи в модель.
    """
    logger.info("Запуск предобработки данных")

    df = pd.DataFrame([input_dict])

    current_day = pd.to_datetime(curr_day)
    df['num_of_days'] = df.apply(lambda row: (current_day - pd.to_datetime(row['begin_date'])).days, axis=1)

    df.drop(['begin_date'], axis=1, inplace=True)
    df.drop(COLUMNS_TO_DELETE[1:], axis=1, inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    logger.info(f"Данные предобработаны:\n{df.head()}")

    return df
