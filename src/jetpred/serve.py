import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from jetpred import config, metrics

MAX_BATCH_SIZE = 1000


@dataclass
class LoadedModel:
    model: object
    feature_names: list[str]
    version: str


def load_from_registry() -> LoadedModel:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", config.MLFLOW_TRACKING_URI))
    name = os.getenv("MODEL_NAME", config.MODEL_NAME)
    alias = os.getenv("MODEL_ALIAS", config.MODEL_ALIAS)

    version = MlflowClient().get_model_version_by_alias(name, alias).version
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    feature_names = model.metadata.get_input_schema().input_names()
    return LoadedModel(model, feature_names, str(version))


def load_from_directory(model_dir) -> LoadedModel:
    model = mlflow.pyfunc.load_model(str(model_dir))
    feature_names = model.metadata.get_input_schema().input_names()
    version_file = Path(model_dir) / "VERSION"
    version = version_file.read_text().strip() if version_file.exists() else "unknown"
    return LoadedModel(model, feature_names, version)


def default_loader() -> LoadedModel:
    model_dir = os.getenv("MODEL_DIR")
    return load_from_directory(model_dir) if model_dir else load_from_registry()


class PredictRequest(BaseModel):
    jets: list[dict[str, float]] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class JetPrediction(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[JetPrediction]


def create_app(loader=default_loader) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.loaded = loader()
        yield

    app = FastAPI(title="Jet classifier", lifespan=lifespan)

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            route = request.scope.get("route")
            path = route.path if route else "unmatched"
            metrics.HTTP_REQUESTS.labels(request.method, path, status).inc()
            metrics.HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - start)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        loaded = getattr(request.app.state, "loaded", None)
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "ready", "model_version": loaded.version}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request):
        loaded = request.app.state.loaded
        expected = set(loaded.feature_names)

        for i, jet in enumerate(payload.jets):
            missing, unknown = expected - jet.keys(), jet.keys() - expected
            if missing or unknown:
                raise HTTPException(
                    status_code=422,
                    detail={"jet_index": i, "missing": sorted(missing), "unknown": sorted(unknown)},
                )

        # The model signature expects float32; JSON numbers arrive as float64
        frame = pd.DataFrame(payload.jets)[loaded.feature_names].astype("float32")
        start = time.perf_counter()
        output = loaded.model.predict(frame)
        metrics.INFERENCE_LATENCY.observe(time.perf_counter() - start)
        metrics.BATCH_SIZE.observe(len(frame))

        prob_columns = [c for c in output.columns if c.startswith("prob_")]
        predictions = [
            JetPrediction(
                prediction=row["prediction"],
                probabilities={c.removeprefix("prob_"): float(row[c]) for c in prob_columns},
            )
            for _, row in output.iterrows()
        ]
        for p in predictions:
            metrics.PREDICTIONS.labels(p.prediction, loaded.version).inc()
            metrics.PREDICTION_CONFIDENCE.observe(max(p.probabilities.values()))
            
        return PredictResponse(model_version=loaded.version, predictions=predictions)

    return app


app = create_app()
