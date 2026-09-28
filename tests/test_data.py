"""
Testes de integridade do dataset tratado (data/processed/churn_clean.csv).

Rodar a partir da raiz do projeto com:
    pytest tests/
"""
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "churn_clean.csv"

COLUNAS_ESPERADAS = {
    "customer_id", "vintage", "age", "gender", "dependents", "occupation",
    "city", "customer_nw_category", "branch_code", "current_balance",
    "previous_month_end_balance", "average_monthly_balance_prevQ",
    "average_monthly_balance_prevQ2", "current_month_credit",
    "previous_month_credit", "current_month_debit", "previous_month_debit",
    "current_month_balance", "previous_month_balance", "last_transaction",
    "churn", "saldo_negativo",
}


@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), (
        f"Dataset tratado não encontrado em {DATA_PATH}. "
        "Rode notebooks/01_data_cleaning.ipynb antes de testar."
    )
    return pd.read_csv(DATA_PATH)


def test_colunas_esperadas_presentes(df):
    faltando = COLUNAS_ESPERADAS - set(df.columns)
    assert not faltando, f"Colunas ausentes no dataset tratado: {faltando}"


def test_sem_duplicatas_de_cliente(df):
    assert df["customer_id"].duplicated().sum() == 0


def test_colunas_criticas_sem_ausentes(df):
    # Colunas que o notebook 01 garante estarem 100% preenchidas após o tratamento.
    criticas = ["dependents", "gender", "occupation", "age", "current_balance", "churn"]
    for col in criticas:
        assert df[col].isnull().sum() == 0, f"Coluna '{col}' ainda tem valores ausentes"


def test_dependents_sem_outliers_grosseiros(df):
    # O tratamento no notebook 01 limita 'dependents' a <= 10.
    assert df["dependents"].max() <= 10


def test_churn_e_binario(df):
    assert set(df["churn"].unique()) <= {0, 1}


def test_taxa_de_churn_dentro_do_esperado(df):
    # A taxa de churn documentada no README é ~18,5%; uma margem ampla evita
    # falso-positivo por pequenas variações, mas detecta quebras reais no pipeline.
    taxa = df["churn"].mean()
    assert 0.10 < taxa < 0.30
