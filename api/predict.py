import mlflow.pyfunc
from api.pipeline import preprocess_input
import logging
from training.config import MLFLOW_TRACKING_URI
from api.db_utils import log_to_db

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    logger.info("Загрузка модели из MLflow...")
    model = mlflow.pyfunc.load_model("models:/BestModel@Production")
    logger.info("Модель успешно загружена.")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {e}")
    model = None

def get_prediction(features: dict) -> dict:
    """
    Выполняет предсказание на основе входных признаков и возвращает читаемый результат.

    Параметры:
    -----------
    features : dict
        Словарь с входными признаками для модели.

    Возвращает:
    -----------
    dict
        Словарь с ключом 'prediction' и значением в виде строки:
        'Абонент уйдет' если предсказание 1, 'Абонент останется' если 0, или None, если модель не загружена.
    """
    if model is None:
        logger.error("Модель не загружена, предсказание невозможно.")
        return {"prediction": None}

    df = preprocess_input(features)

    try:
        pred = model.predict(df)
    except Exception as e:
        logger.exception("Ошибка при вызове model.predict(df)")
        raise

    label = "Абонент уйдет" if int(pred[0]) == 1 else "Абонент останется"
    logger.info(f"Предсказание модели: {label}")

    # логирование в базу
    try:
        log_to_db(features, label)
    except Exception as e:
        logger.error(f"Ошибка логирования в БД: {e}")

    return {"prediction": label}
