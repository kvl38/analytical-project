from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from api.models import InputData, PredictionResult
from api.predict import get_prediction
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Подключаем папку с шаблонами для фронтенда
templates = Jinja2Templates(directory="api/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Отображает главную HTML-страницу веб-приложения.

    Параметры:
    -----------
    request : Request
        Объект запроса FastAPI, необходимый для корректного рендеринга шаблона.

    Возвращает:
    -----------
    _TemplateResponse
        HTML-страница, сформированная с помощью Jinja2.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResult)
def predict(data: InputData) -> PredictionResult:
    """
    Обрабатывает POST-запрос для предсказания на основе входных данных.

    Параметры:
    -----------
    data : InputData
        Входные данные, соответствующие схеме InputData.

    Возвращает:
    -----------
    PredictionResult
        Результат предсказания в формате JSON.
    """
    input_dict = data.dict()
    logger.info(f"Получен запрос: {input_dict}")
    result = get_prediction(input_dict)
    logger.info(f"Предсказание: {result}")
    return result
