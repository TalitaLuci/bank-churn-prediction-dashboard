# Bank Churn Prediction Dashboard

Modelo preditivo e dashboard interativo para identificar clientes de um banco com risco de cancelamento (churn) e estimar o impacto em receita.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

🔗 **[Acesse o dashboard ao vivo](https://bank-churn-prediction-dashboard-2azfey4fpwgbqutgvc8hx8.streamlit.app)**

![Tela inicial do dashboard de previsão de churn bancário](./assets/tela-inicial.png)

---

## 📌 Sumário

- [Contexto de negócio](#-contexto-de-negócio)
- [Visualização dos resultados](#-visualização-dos-resultados)
- [Coleta de dados](#-coleta-de-dados)
- [Modelagem](#-modelagem)
- [Conclusões](#-conclusões)
- [Dashboard](#-dashboard)
- [Como executar](#-como-executar)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Tecnologias](#-tecnologias)
- [Autor](#-autor)

---

## 🎯 Contexto de negócio

O banco enfrenta uma taxa de churn de **18,5%** — bem acima do que é considerado saudável no setor bancário. Isso já custou **13,1% de todo o saldo administrado na base** em contas que cancelaram. O objetivo deste projeto é identificar, antes do cancelamento acontecer, quais clientes estão em risco, para que a equipe de retenção possa agir a tempo.

## Objetivos

* Identificar clientes com alta probabilidade de cancelamento antes que aconteça
* Entender **por que** clientes cancelam, não só prever quem vai cancelar
* Priorizar quais clientes a equipe de retenção deve contatar primeiro
* Entregar os resultados em um dashboard interativo, sem depender de um analista para gerar cada relatório

## Perguntas de Negócio Respondidas

|Pergunta|Resposta|
|-|-|
|Qual a taxa de churn?|18,5%|
|Quanto já foi perdido?|13,1% do saldo total da base|
|Existe perfil demográfico de quem cancela?|Não — idade, tempo de relacionamento e dependentes são quase iguais entre grupos|
|Ocupação importa?|Sim — autônomos cancelam quase 2x mais que funcionários de empresa (19,8% vs 10,0%)|
|Qual o sinal mais forte?|Queda de saldo no trimestre — quem cancela esvazia a conta antes de sair|
|Onde priorizar a retenção?|Cruzar clientes de alto saldo (top 10%) com sinal de queda de saldo|

---

## Estrutura do Repositório

```
bank-churn-prediction-dashboard/
├── assets/
│   ├── tela-inicial.png
│   ├── eda.png
│   └── simulador-churn.png
├── data/
│   ├── raw/                    # Dataset original (nunca editado)
│   └── processed/              # Dataset tratado, pronto para análise/modelagem
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Tratamento de dados: ausentes, outliers, tipos
│   ├── 02_eda_negocio.ipynb        # Análise exploratória orientada a negócio
│   └── 03_modelagem.ipynb          # Comparação de modelos, tuning, impacto de negócio
├── models/
│   ├── modelo_churn_xgboost.pkl    # Modelo final treinado
│   └── metadata.json               # Limiar de decisão e métricas do modelo
├── src/
│   └── app.py                      # Dashboard Streamlit (Análise + Simulador)
├── requirements.txt
└── README.md
```

## Dados

**Fonte:** [Bank Customer Churn Data](https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data) (Kaggle, por Penta Krishna Kishore) — 28.382 clientes, 21 colunas.

**Principais decisões de tratamento** (detalhadas em `01_data_cleaning.ipynb`):

* Valores ausentes em `dependents`, `city`, `gender` e `occupation` tratados individualmente, com decisão justificada para cada coluna (mediana, categoria própria, ou moda, conforme o caso)
* Outliers em `dependents` (valores como 52) tratados como erro de cadastro
* ~800 clientes com menos de 18 anos **mantidos intencionalmente** — provável conta de menor, comum no mercado bancário indiano
* Saldo negativo transformado em uma feature própria (`saldo_negativo`) em vez de removido, por ser um sinal potencialmente relevante

## Principais Insights (EDA de Negócio)

Detalhados em `02_eda_negocio.ipynb`:

![Aba Análise do dashboard, mostrando os gráficos de taxa de churn por ocupação e de queda de saldo entre clientes ativos e cancelados](./assets/eda.png)

* **O sinal mais forte não é quem o cliente é, é o que ele está fazendo com o dinheiro.** Clientes que cancelam tinham saldo médio *maior* no trimestre anterior, mas *esvaziaram* a conta antes de sair (queda média de 3.322, contra uma leve alta de 613 entre quem ficou) — essa feature (`queda_saldo`) se mostrou o preditor mais poderoso do projeto.
* **Ocupação é um sinal de negócio válido:** autônomos cancelam quase 2x mais que funcionários de empresa.
* **Uma hipótese testada e descartada:** esperava-se que a ausência de transação recente indicasse maior risco — o oposto se confirmou. Reportado com transparência, não escondido.
* **Perfil demográfico (idade, tempo de relacionamento, dependentes) não diferencia quem cancela** — reforça que comportamento financeiro > características fixas do cliente.

## Modelagem

Detalhado em `03_modelagem.ipynb`. Três modelos comparados com parâmetros padrão:

|Modelo|Recall|Precision|F1-Score|ROC-AUC|
|-|-|-|-|-|
|Logistic Regression|0,647|0,340|0,446|0,740|
|Random Forest|0,432|0,722|0,540|0,849|
|XGBoost|0,568|0,621|0,594|0,815|

**XGBoost foi escolhido** pelo melhor equilíbrio entre recall e precisão, e depois otimizado com `RandomizedSearchCV` (5-fold cross-validation), elevando o F1-Score para **0,608** e o ROC-AUC para **0,836**.

**Ajuste de limiar orientado a negócio:** o limiar de decisão foi calibrado para capturar ~75% dos clientes que realmente cancelam (recall priorizado sobre acurácia, já que deixar passar um cliente que vai cancelar custa mais caro que uma ligação de retenção desnecessária).

**Impacto de negócio:** nesse limiar, o modelo sinaliza 1.728 clientes (30% da base de teste) e captura ~45% de todo o saldo em risco de cancelamento — uma redução de 15% no número de clientes que a equipe de retenção precisa contatar, comparado à versão sem tuning, para o mesmo recall.

**Limitação conhecida:** o teto de ~0,84 de ROC-AUC provavelmente reflete o limite real dos dados disponíveis — não há informação sobre motivo de cancelamento, satisfação do cliente, ou histórico além de um trimestre.

## Dashboard

Construído em Streamlit, com duas abas:

* **📊 Análise** — os principais achados da EDA de negócio, com gráficos interativos e filtro por ocupação
* **🔮 Simulador** — formulário que recebe dados de um cliente (real ou hipotético) e retorna a probabilidade de churn em tempo real, usando o modelo treinado

![Aba Simulador do dashboard, mostrando o formulário com os campos principais (idade, ocupação, saldo atual, saldo médio do trimestre anterior) e o resultado da previsão com a probabilidade de churn e a classificação de risco](./assets/simulador-churn.png)

### Como rodar

```bash
git clone https://github.com/TalitaLuci/bank-churn-prediction-dashboard.git
cd bank-churn-prediction-dashboard
pip install -r requirements.txt
streamlit run src/app.py
```

O dashboard abre automaticamente em `http://localhost:8501`.



## Observações

* **Moeda não confirmada:** valores tratados sem símbolo monetário até confirmação da fonte oficial dos dados
* **Sem dado de motivo de cancelamento:** o modelo prevê *que* o cliente vai cancelar, não *por quê* além do que os dados financeiros revelam
* **Próximo passo natural:** incorporar dado de satisfação/motivo de contato do cliente, se disponível, para complementar o sinal puramente financeiro


---

## 🛠 Tecnologias

- **Linguagem:** Python 3.11+
- **Análise:** pandas, numpy
- **Visualização:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn, XGBoost
- **Dashboard:** Streamlit
- **Serialização:** joblib

---


## 👩‍💻 Autora

**Talita Luci**

[![GitHub](https://img.shields.io/badge/GitHub-TalitaLuci-black)](https://github.com/TalitaLuci)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-TalitaLuci-blue)](https://www.linkedin.com/in/talita-luci)

