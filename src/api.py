"""
Etapa 5 — Serviço demonstrável.

API FastAPI que recebe dados de um cliente e retorna a oferta recomendada
pelo modelo Thompson Sampling treinado no notebook 02_bandit_mlflow.ipynb.

Rodar localmente:
    uvicorn src.api:app --reload --port 8000

Testar:
    curl -X POST http://localhost:8000/recomendar -H "Content-Type: application/json" \
        -d '{"idade": 35, "profissao": "admin.", "estado_civil": "married"}'
"""
import os
import pickle
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.bandit import ThompsonSampling

MODEL_PATH = os.getenv("MODEL_PATH", "data/thompson_model.pkl")

app = FastAPI(
    title="Datathon 7MLET - API de Recomendação Adaptativa",
    description="Recebe dados de um cliente e retorna a oferta recomendada via Thompson Sampling.",
    version="1.0.0",
)

_model: Optional[ThompsonSampling] = None


def load_model() -> ThompsonSampling:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modelo não encontrado em {MODEL_PATH}. Rode o notebook 02_bandit_mlflow.ipynb primeiro."
            )
        with open(MODEL_PATH, "rb") as f:
            state = pickle.load(f)
        _model = ThompsonSampling.from_state(state)
    return _model


class Cliente(BaseModel):
    idade: int
    profissao: str
    estado_civil: str
    escolaridade: Optional[str] = None
    tem_emprestimo: Optional[bool] = False


class Recomendacao(BaseModel):
    oferta_recomendada: str
    probabilidades_estimadas: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recomendar", response_model=Recomendacao)
def recomendar(cliente: Cliente):
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Nota: neste MVP o contexto do cliente ainda não influencia diretamente
    # a escolha do braço (bandit não-contextual). Para uma versão contextual,
    # o vetor de features do cliente entraria como covariável no modelo Beta
    # (ex: regressão logística bayesiana por braço).
    braco = model.select_arm()

    probs = {
        a: model.alpha[a] / (model.alpha[a] + model.beta[a])
        for a in model.arms
    }

    return Recomendacao(oferta_recomendada=braco, probabilidades_estimadas=probs)


@app.post("/feedback")
def feedback(braco: str, converteu: bool):
    """Endpoint opcional para realimentar o bandit com o resultado observado."""
    model = load_model()
    if braco not in model.arms:
        raise HTTPException(status_code=400, detail=f"Braço inválido. Opções: {model.arms}")
    model.update(braco, int(converteu))
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model.state(), f)
    return {"status": "atualizado", "braco": braco, "reward": int(converteu)}
