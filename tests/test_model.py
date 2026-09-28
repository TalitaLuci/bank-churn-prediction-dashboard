"""
Testes de sanidade do modelo treinado (models/modelo_churn_xgboost.pkl).

Garantem que o artefato salvo pelo notebook 03 continua carregável e compatível
com o formato de entrada esperado pelo dashboard (src/app.py).

Rodar a partir da raiz do projeto com:
    pytest tests/
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_churn_xgboost.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"


@pytest.fixture(scope="module")
def modelo():
    assert MODEL_PATH.exists(), f"Modelo não encontrado em {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def metadata():
    assert METADATA_PATH.exists(), f"Metadata não encontrado em {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def cliente_exemplo():
    # Mesmas colunas montadas em src/app.py para o simulador.
    return pd.DataFrame([{
        "vintage": 2000,
        "age": 45,
        "dependents": 0,
        "current_balance": 3500.0,
        "previous_month_end_balance": 3500.0,
        "average_monthly_balance_prevQ": 3500.0,
        "average_monthly_balance_prevQ2": 3500.0,
        "current_month_credit": 0.0,
        "previous_month_credit": 0.0,
        "current_month_debit": 0.0,
        "previous_month_debit": 0.0,
        "current_month_balance": 3500.0,
        "previous_month_balance": 3500.0,
        "saldo_negativo": 0,
        "queda_saldo": 0.0,
        "sem_transacao_recente": 0,
        "gender": "Male",
        "occupation": "salaried",
        "customer_nw_category": 2,
    }])


def test_metadata_tem_limiar_e_metricas(metadata):
    assert "limiar_negocio" in metadata
    assert 0 < metadata["limiar_negocio"] < 1
    assert "metricas_teste" in metadata
    for chave in ["recall", "precision", "f1", "roc_auc"]:
        assert chave in metadata["metricas_teste"]


def test_modelo_prediz_probabilidade_valida(modelo, cliente_exemplo):
    proba = modelo.predict_proba(cliente_exemplo)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_modelo_e_deterministico(modelo, cliente_exemplo):
    # Mesma entrada deve sempre gerar a mesma probabilidade (sem aleatoriedade
    # residual no pipeline treinado).
    p1 = modelo.predict_proba(cliente_exemplo)[0, 1]
    p2 = modelo.predict_proba(cliente_exemplo)[0, 1]
    assert p1 == p2
