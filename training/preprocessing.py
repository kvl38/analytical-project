import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

def renaming_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Переименовывает столбцы DataFrame из camelCase/PascalCase в snake_case.

    Параметры:
    -----------
    df : pd.DataFrame
        Исходный DataFrame с названиями столбцов в camelCase или PascalCase.

    Возвращает:
    -----------
    pd.DataFrame
        Новый DataFrame с переименованными столбцами в формате snake_case.
    """

    def to_snake_case(col_name: str) -> str:
        """
        Преобразует отдельное имя столбца, заменяя переходы от строчных букв к заглавным
        на нижние подчёркивания и приводит строку к нижнему регистру.

        Параметры:
        -----------
        name : str
            Исходное имя столбца.

        Возвращает:
        -----------
        str
            Имя столбца, преобразованное в snake_case.
        """

        stage_1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', col_name)
        stage_2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', stage_1)
        return stage_2.lower()

    new_columns = {col: to_snake_case(col) for col in df.columns}
    return df.rename(columns=new_columns)

def data_converter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет предобработку исходного датафрейма для задачи машинного обучения:
    переименование столбцов, генерация целевой переменной, очистка и форматирование данных.

    Параметры:
    -----------
    df : pd.DataFrame
        Исходный датафрейм, объединённый из нескольких источников.

    Возвращает:
    -----------
    pd.DataFrame
        Подготовленный датафрейм без дубликатов и с готовыми признаками для моделирования.
    """
    logger.info("Запуск предобработки данных")

    # Переименуем столбцы
    df = renaming_columns(df)

    # Создаем целевой столбец 'subscriber_left', отражающий факт ухода клиента:
    # 1 — клиент ушел
    # 0 — клиент остался
    df['subscriber_left'] = df['end_date'].apply(lambda x: 1 if x != 'No' else 0)

    # Производим замену в столбце 'total_charges'
    df.loc[df['total_charges'] == ' ', 'total_charges'] = \
        df.loc[df['total_charges'] == ' ', 'monthly_charges']

    # Производим замену в столбце 'end_date'
    df['end_date'] = df['end_date'].replace('No', '2020-02-01')

    # Изменяем тип данных
    df['begin_date'] = pd.to_datetime(df['begin_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['total_charges'] = df['total_charges'].astype(float)

    # Рассчёт количества дней подписки абонентов
    df['num_of_days'] = df.apply(lambda row: (row['end_date'] - row['begin_date']).days, axis=1)

    # Удаляем столбцы
    df.drop(['begin_date', 'end_date'], axis=1, inplace=True)

    # Приведём значения столбца senior_citizen к единому формату категориальных признаков в рамках этого проекта
    df['senior_citizen'] = df['senior_citizen'].apply(lambda x: 'Yes' if x == 1 else 'No')

    # Удаляем столбец 'customer_id' из датафрейма
    df.drop('customer_id', axis=1, inplace=True)

    # Заменяем пропуски в столбцах 'internet_service' и 'multiple_lines' на 'not_in_use'
    df['internet_service'] = df['internet_service'].fillna('not_in_use')
    df['multiple_lines'] = df['multiple_lines'].fillna('not_in_use')

    # Заполняем пропуски значением 'No' в столбцах доп. услуг
    internet_columns = ['online_security', 'online_backup', 'device_protection',
                        'tech_support', 'streaming_tv', 'streaming_movies']
    df[internet_columns] = df[internet_columns].fillna('No')

    # Удаление дубликатов
    df = df.drop_duplicates()

    logger.info("Данные предобработаны")
    logger.info(f"Размер предобработанного датафрейма {df.shape}")
    return df


