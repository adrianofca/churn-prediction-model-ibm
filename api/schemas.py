"""Schemas Pydantic para validação do request/response do endpoint /predict."""
from pydantic import BaseModel


class PredictRequest(BaseModel):

    gender: str

    SeniorCitizen: int

    Partner: str

    Dependents: str

    tenure: int

    PhoneService: str

    MultipleLines: str

    InternetService: str

    OnlineSecurity: str

    OnlineBackup: str

    DeviceProtection: str

    TechSupport: str

    StreamingTV: str

    StreamingMovies: str

    Contract: str

    PaperlessBilling: str

    PaymentMethod: str

    MonthlyCharges: float

    TotalCharges: float


class PredictResponse(BaseModel):

    probability: float

    prediction: int
