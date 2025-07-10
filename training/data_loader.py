import pandas as pd
import logging

logger = logging.getLogger(__name__)

def data_loader(link: str) -> pd.DataFrame:
    """
    Загружает данные из CSV-файла по переданной ссылке.

    Параметры:
    -----------
    link : str
        URL-адрес CSV-файла.

    Возвращает:
    -----------
    pd.DataFrame
        Загруженные данные.
    """
    try:
        df = pd.read_csv(link)
        return df
    except Exception as e:
        logger.error(f"Произошла ошибка при загрузке данных: {e}")
        return pd.DataFrame()


def dfs_loader(contract_link: str, personal_link: str, internet_link:str, phone_link:str) -> pd.DataFrame:
    """
    Загружает и объединяет несколько датафреймов по ключу 'customerID'.

    Параметры:
    -----------
    contract_link : str
        Ссылка на CSV-файл с контрактными данными клиентов.
    personal_link : str
        Ссылка на CSV-файл с персональной информацией о клиентах.
    internet_link : str
        Ссылка на CSV-файл с информацией об интернет-услугах клиентов.
    phone_link : str
        Ссылка на CSV-файл с информацией о телефонии клиентов.

    Возвращает:
    -----------
    pd.DataFrame
        Объединённый датафрейм со всеми данными, соединёнными по 'customerID'.
    """
    logger.info("Запуск загрузки данных")
    # Загружаем исходные данные
    contract_df = data_loader(contract_link)
    personal_df = data_loader(personal_link)
    internet_df = data_loader(internet_link)
    phone_df = data_loader(phone_link)

    # Объединяем все данные по столбцу 'customerID'
    merged_df = contract_df.merge(personal_df, on='customerID', how='left') \
                           .merge(internet_df, on='customerID', how='left') \
                           .merge(phone_df, on='customerID', how='left')

    logger.info("Исходные данные загружены и объединены")
    logger.info(f"Размер объединенного датафрейма {merged_df.shape}")
    return merged_df