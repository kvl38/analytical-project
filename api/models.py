from pydantic import BaseModel

class InputData(BaseModel):
    begin_date: str
    type: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    gender: str
    senior_citizen: str
    partner: str
    dependents: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    multiple_lines: str

class PredictionResult(BaseModel):
    prediction: str