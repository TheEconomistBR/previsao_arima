import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(page_title="Painel de Previsão Agro", layout="wide")

# ========== ESTILO ==========
st.markdown("""
<style>
h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
section[data-testid="stSidebar"] {
    background-color: #f8f9fa;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ========== CABEÇALHO ==========
st.title("📈 Painel de Previsão de Preços Agropecuários")
st.caption("Comparação de modelos ARIMA e Prophet com avaliação de desempenho (RMSE)")

# ========== SIDEBAR ==========
st.sidebar.header("Parâmetros da Previsão")
horizontes = {"6 meses": 6, "12 meses": 12, "24 meses": 24, "48 meses": 48}
horizonte_nome = st.sidebar.radio("Horizonte:", list(horizontes.keys()))
n_meses = horizontes[horizonte_nome]

# Parâmetros ARIMA
st.sidebar.subheader("Parâmetros ARIMA")
p = st.sidebar.slider("AR (p)", 0, 5, 2)
d = st.sidebar.slider("Diferença (d)", 0, 2, 1)
q = st.sidebar.slider("MA (q)", 0, 5, 2)

# ========== FUNÇÕES ==========
@st.cache_data
def carregar_base():
    df = pd.read_csv("dados/base_unificada_cepea.csv", encoding="latin1")
    df.columns = [col.lower().strip() for col in df.columns]
    return df

def preparar_serie(df):
    meses = {
        'janeiro': 'January', 'fevereiro': 'February', 'março': 'March',
        'abril': 'April', 'maio': 'May', 'junho': 'June',
        'julho': 'July', 'agosto': 'August', 'setembro': 'September',
        'outubro': 'October', 'novembro': 'November', 'dezembro': 'December'
    }
    df['preco_deflacionado'] = df['preco_deflacionado'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
    df['mes_en'] = df['mes'].str.lower().str.strip().map(meses)
    df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes_en'], format='%Y-%B', errors='coerce')
    return df.dropna(subset=['data']).sort_values('data').set_index('data')

def prever_arima(serie, steps, p, d, q):
    modelo = ARIMA(serie, order=(p,d,q)).fit()
    previsao = modelo.get_forecast(steps=steps)
    return (
        pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS'),
        previsao.predicted_mean,
        previsao.conf_int()
    )

def prever_prophet(serie, steps):
    dfp = serie.reset_index().rename(columns={"data": "ds", "preco_deflacionado": "y"})
    modelo = Prophet()
    modelo.fit(dfp)
    futuro = modelo.make_future_dataframe(periods=steps, freq="MS")
    previsao = modelo.predict(futuro)
    previsao = previsao.set_index('ds')
    return (
        previsao.index[-steps:],
        previsao['yhat'][-steps:],
        previsao[['yhat_lower', 'yhat_upper']][-steps:]
    )

def calcular_rmse(serie, p, d, q):
    if len(serie) < 36:
        return None, None

    treino = serie[:-12]
    real = serie[-12:]

    try:
        arima_model = ARIMA(treino, order=(p, d, q)).fit()
        arima_pred = arima_model.forecast(12)
        arima_rmse = mean_squared_error(real, arima_pred, squared=False)
    except:
        arima_rmse = None

    try:
        dfp = treino.reset_index().rename(columns={"data": "ds", "preco_deflacionado": "y"})
        model = Prophet()
        model.fit(dfp)
        futuro = model.make_future_dataframe(periods=12, freq="MS")
        previsao = model.predict(futuro).set_index('ds')
        prophet_pred = previsao['yhat'][-12:]
        prophet_rmse = mean_squared_error(real, prophet_pred, squared=False)
    except:
        prophet_rmse = None

    return arima_rmse, prophet_rmse

def grafico_modelo(serie, previsao, media, intervalo, nome, cor, estilo, unidade):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie, name="Histórico", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=previsao, y=media, name=nome, line=dict(color=cor, dash=estilo)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([previsao, previsao[::-1]]),
        y=np.concatenate([intervalo.iloc[:, 0], intervalo.iloc[:, 1][::-1]]) if isinstance(intervalo, pd.DataFrame)
        else np.concatenate([intervalo['yhat_lower'], intervalo['yhat_upper'][::-1]]),
        fill='toself', fillcolor='rgba(200,200,200,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False
    ))
    fig.update_layout(
        xaxis_title="Ano",
        yaxis_title=f"Preço deflacionado (R$/{unidade})",
        height=450,
        template="plotly_white"
    )
    return fig

# ========== APLICAÇÃO ==========
df_base = carregar_base()
produtos = sorted(df_base['produto'].unique())
produto = st.selectbox("Selecione um produto:", produtos)

df_prod = preparar_serie(df_base[df_base['produto'] == produto])
serie = df_prod['preco_deflacionado']
unidade = "litro" if "leite" in produto.lower() else "saca"

# Previsões
arima_out = prophet_out = None
try:
    arima_out = prever_arima(serie, n_meses, p, d, q)
except Exception as e:
    st.warning(f"Erro no ARIMA: {e}")
try:
    prophet_out = prever_prophet(df_prod, n_meses)
except Exception as e:
    st.warning(f"Erro no Prophet: {e}")

# Cálculo de RMSE
arima_rmse, prophet_rmse = calcular_rmse(serie, p, d, q)

# ========== TABS COM GRÁFICOS ==========
tab1, tab2 = st.tabs(["📊 ARIMA", "📈 Prophet"])

with tab1:
    if arima_out:
        st.plotly_chart(grafico_modelo(df_prod, *arima_out, "ARIMA", "orange", "dot", unidade), use_container_width=True)

with tab2:
    if prophet_out:
        st.plotly_chart(grafico_modelo(df_prod, *prophet_out, "Prophet", "blue", "dash", unidade), use_container_width=True)

# ========== MÉTRICAS ==========
st.markdown("### 🧮 Desempenho (últimos 12 meses)")
col1, col2 = st.columns(2)
with col1:
    st.metric("ARIMA RMSE", f"{arima_rmse:.2f}" if arima_rmse else "Erro")
with col2:
    st.metric("Prophet RMSE", f"{prophet_rmse:.2f}" if prophet_rmse else "Erro")

# ========== RODAPÉ ==========
st.markdown("---")
st.markdown(f"""
📊 Desenvolvido por **Lucas França e Paola Conti**  
📅 Última atualização: Maio/2025  
🔍 Modelos aplicados: ARIMA({p},{d},{q}) e Prophet  
📩 Contato: contato@ufsm.com.br
""")
