"""API REST do modelo de churn: expõe GET /health e POST /predict."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.logging_config import configure_logging
from api.middleware import LatencyMiddleware
from api.model_loader import predict
from api.schemas import PredictRequest, PredictResponse

logger = configure_logging()

app = FastAPI(
    title="Telco Churn API",
    version="1.0"
)

app.add_middleware(
    LatencyMiddleware
)


@app.on_event("startup")
def log_routes():
    logger.info(f"Rotas registradas: {[getattr(route, 'path', '') for route in app.routes]}")


@app.get("/health")
def health():

    logger.info("[HEALTH] API saudável")

    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=PredictResponse
)
def predict_churn(
    request: PredictRequest
):

    probability, prediction = predict(
        request.model_dump()
    )

    logger.info(
        f"[PREDICTION] prediction={prediction} probability={probability:.4f}"
    )

    return PredictResponse(
        probability=probability,
        prediction=prediction
    )


@app.exception_handler(Exception)
async def handle_error(
    request: Request,
    exc: Exception
):

    logger.exception(
        f"Erro interno: {str(exc)}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erro interno"
        }
    )
