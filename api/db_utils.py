import psycopg2
import json
import logging
from api.db_config import POSTGRES

logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Создаёт и возвращает соединение с базой данных PostgreSQL.

    Возвращает:
    -----------
    conn : psycopg2.extensions.connection
        Активное соединение с базой данных PostgreSQL.
    """
    return psycopg2.connect(**POSTGRES)


def log_to_db(input_data: dict, prediction: str) -> None:
    """
    Записывает входные данные и результат предсказания в таблицу prediction_logs базы данных PostgreSQL.

    Параметры:
    -----------
    input_data : dict
        Словарь с признаками, переданными в модель.

    prediction : str
        Текстовое предсказание модели (например, 'Абонент уйдет' или 'Абонент останется').

    Возвращает:
    -----------
    None
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO prediction_logs (input_data, prediction)
            VALUES (%s, %s)
        """
        cursor.execute(insert_query, (json.dumps(input_data), prediction))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Предсказание успешно залогировано в БД.")
    except Exception as e:
        logger.error(f"Ошибка при логировании в PostgreSQL: {e}")
