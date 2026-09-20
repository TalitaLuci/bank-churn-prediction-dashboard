"""
Dashboard de Previsão de Churn Bancário
=========================================
Duas abas:
  - Análise: achados da EDA de negócio (notebook 02)
  - Simulador: previsão em tempo real usando o modelo treinado (notebook 03)

Para rodar:
    streamlit run src/app.py
(rode a partir da pasta raiz do projeto, bank-churn-prediction-dashboard/)
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Caminhos: resolvidos a partir da localização deste arquivo, não da pasta
# de onde o comando é executado.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_churn_xgboost.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
DATA_PATH = BASE_DIR / "data" / "processed" / "churn_clean.csv"

st.set_page_config(page_title="Previsão de Churn Bancário", page_icon="🏦", layout="wide")
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Carregamento de dados e modelo, com cache.
# @st.cache_data / @st.cache_resource evitam recarregar o CSV e o modelo
# a cada clique do usuário.
# ---------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata


@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH, parse_dates=["last_transaction"])
    df["queda_saldo"] = df["average_monthly_balance_prevQ"] - df["current_balance"]
    df["sem_transacao_recente"] = df["last_transaction"].isnull().astype(int)
    return df


modelo, metadata = carregar_modelo()
df = carregar_dados()
LIMIAR_NEGOCIO = metadata["limiar_negocio"]

st.title("🏦 Previsão de Churn Bancário")
st.caption(
    "Modelo XGBoost treinado para identificar clientes com risco de cancelamento, "
    "com base em comportamento financeiro recente. Valores monetários estão na "
    "unidade original do dataset (não confirmada oficialmente como Rupias)."
)

tab_analise, tab_simulador = st.tabs(["📊 Análise", "🔮 Simulador"])

# ===========================================================================
# ABA 1 — ANÁLISE
# Reaproveita os achados do notebook 02_eda_negocio.ipynb
# ===========================================================================
with tab_analise:
    st.header("Panorama geral")

    taxa_churn = df["churn"].mean()
    saldo_total = df["current_balance"].sum()
    saldo_perdido = df.loc[df["churn"] == 1, "current_balance"].sum()
    pct_perdido = saldo_perdido / saldo_total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Taxa de churn", f"{taxa_churn*100:.1f}%")
    col2.metric("Saldo já perdido (churn)", f"{saldo_perdido:,.0f}")
    col3.metric("% do saldo total perdido", f"{pct_perdido:.1f}%")

    st.divider()

    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.subheader("Taxa de churn por ocupação")
        churn_ocupacao = (
            df.groupby("occupation", observed=True)["churn"].mean().sort_values(ascending=False) * 100
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_ocupacao.plot(kind="barh", color="#2E86AB", ax=ax)
        ax.set_xlabel("Taxa de churn (%)")
        ax.invert_yaxis()
        st.pyplot(fig)
        st.caption("Autônomos cancelam quase 2x mais que funcionários de empresa.")

    with col_dir:
        st.subheader("Queda de saldo: ativo vs cancelou")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="churn", y="queda_saldo", showfliers=False, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ativo", "Cancelou"])
        ax.set_ylabel("Queda de saldo (trimestre anterior − atual)")
        st.pyplot(fig)
        st.caption(
            "Quem cancela esvazia a conta antes de sair — o sinal mais forte encontrado na análise."
        )

    st.divider()

    st.subheader("Explore por ocupação")
    ocupacao_filtro = st.selectbox(
        "Filtrar clientes por ocupação", ["Todas"] + sorted(df["occupation"].unique().tolist())
    )
    df_filtrado = df if ocupacao_filtro == "Todas" else df[df["occupation"] == ocupacao_filtro]
    st.dataframe(
        df_filtrado[
            ["age", "occupation", "customer_nw_category", "current_balance", "queda_saldo", "churn"]
        ].head(200),
        width="stretch",
    )

# ===========================================================================
# ABA 2 — SIMULADOR
# Formulário -> monta um DataFrame de 1 linha com as MESMAS colunas usadas
# no treino -> pipeline.predict_proba() -> classificação de risco
# ===========================================================================
with tab_simulador:
    st.header("Simule um cliente")
    st.caption(
        "Preencha os campos principais. Os campos avançados já vêm com valores "
        "padrão razoáveis — ajuste só se quiser mais precisão."
    )

    with st.form("form_simulador"):
        st.subheader("Informações principais")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Idade", min_value=18, max_value=100, value=45)
            gender = st.selectbox("Gênero", ["Male", "Female", "Not Informed"])
            occupation = st.selectbox(
                "Ocupação", ["self_employed", "salaried", "retired", "student", "company"]
            )
        with c2:
            vintage = st.number_input(
                "Tempo de relacionamento (dias)", min_value=0, max_value=5000, value=2000
            )
            dependents = st.number_input("Dependentes", min_value=0, max_value=10, value=0)
            nw_categoria = st.selectbox(
                "Categoria de patrimônio", ["Alto", "Médio", "Baixo"], index=1
            )
        with c3:
            current_balance = st.number_input(
                "Saldo atual", min_value=-10000.0, value=3500.0, step=100.0
            )
            average_monthly_balance_prevQ = st.number_input(
                "Saldo médio no trimestre anterior", min_value=0.0, value=3500.0, step=100.0
            )
            sem_transacao = st.checkbox("Sem transação registrada no último ano")

        with st.expander("Campos avançados (opcional — melhoram a precisão)"):
            a1, a2 = st.columns(2)
            with a1:
                previous_month_end_balance = st.number_input(
                    "Saldo no fim do mês anterior", value=float(current_balance)
                )
                average_monthly_balance_prevQ2 = st.number_input(
                    "Saldo médio 2 trimestres atrás", value=float(average_monthly_balance_prevQ)
                )
                current_month_credit = st.number_input("Crédito no mês atual", value=0.0)
                previous_month_credit = st.number_input("Crédito no mês anterior", value=0.0)
            with a2:
                current_month_debit = st.number_input("Débito no mês atual", value=0.0)
                previous_month_debit = st.number_input("Débito no mês anterior", value=0.0)
                current_month_balance = st.number_input(
                    "Saldo do mês atual", value=float(current_balance)
                )
                previous_month_balance = st.number_input(
                    "Saldo do mês anterior", value=float(current_balance)
                )

        enviado = st.form_submit_button("Calcular risco de churn", type="primary")

    if enviado:
        mapa_nw = {"Alto": 1, "Médio": 2, "Baixo": 3}

        # Monta a linha exatamente com as colunas que o pipeline espera
        # (mesma ordem/nome usados no treino em 03_modelagem.ipynb)
        entrada = pd.DataFrame([{
            "vintage": vintage,
            "age": age,
            "dependents": dependents,
            "current_balance": current_balance,
            "previous_month_end_balance": previous_month_end_balance,
            "average_monthly_balance_prevQ": average_monthly_balance_prevQ,
            "average_monthly_balance_prevQ2": average_monthly_balance_prevQ2,
            "current_month_credit": current_month_credit,
            "previous_month_credit": previous_month_credit,
            "current_month_debit": current_month_debit,
            "previous_month_debit": previous_month_debit,
            "current_month_balance": current_month_balance,
            "previous_month_balance": previous_month_balance,
            "saldo_negativo": int(current_balance < 0),
            "queda_saldo": average_monthly_balance_prevQ - current_balance,
            "sem_transacao_recente": int(sem_transacao),
            "gender": gender,
            "occupation": occupation,
            "customer_nw_category": mapa_nw[nw_categoria],
        }])

        probabilidade = modelo.predict_proba(entrada)[0, 1]
        risco_alto = probabilidade >= LIMIAR_NEGOCIO

        st.divider()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Probabilidade de churn", f"{probabilidade*100:.1f}%")
        with col_b:
            if risco_alto:
                st.error(
                    f"⚠️ **Risco alto** — acima do limiar de negócio ({LIMIAR_NEGOCIO*100:.1f}%). "
                    "Priorizar contato de retenção."
                )
            else:
                st.success(
                    f"✅ **Risco baixo** — abaixo do limiar de negócio ({LIMIAR_NEGOCIO*100:.1f}%)."
                )

        st.caption(
            f"Limiar calibrado no notebook de modelagem para capturar ~75% dos "
            f"clientes que realmente cancelam (recall de negócio), aceitando mais "
            f"falsos positivos em troca de menos falsos negativos."
        )
