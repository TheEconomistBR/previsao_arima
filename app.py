import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(page_title="Previsão ARIMA", layout="wide")

# ========== ESTILO ==========
st.markdown("""
<style>
h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Previsão de Preços com ARIMA")
st.caption("Personalize os parâmetros do modelo ARIMA e visualize a previsão.")

# ========== EXPLICAÇÃO METODOLÓGICA ==========
with st.expander("📘 Sobre a Metodologia", expanded=False):
    st.markdown("""
**Modelo ARIMA (AutoRegressive Integrated Moving Average)**

- Requer que a série seja estacionária ou transformada em estacionária.
- Seus componentes:
  - **p (autoregressivo)**: número de períodos passados considerados.
  - **d (diferenciação)**: quantas vezes subtrair a série de si mesma para remover tendência.
  - **q (média móvel)**: número de erros passados utilizados para correção.

**Exemplo clássico**: ARIMA(2,1,2) — bastante usado para séries econômicas mensais com tendência leve.

As previsões neste painel utilizam dados deflacionados e são feitas para horizontes de 6 a 48 meses.
""")

# ========== SIDEBAR ==========
st.sidebar.header("Parâmetros")
horizontes = {"6 meses": 6, "12 meses": 12, "24 meses": 24, "48 meses": 48}
horizonte_nome = st.sidebar.radio("Horizonte da Previsão", list(horizontes.keys()))
n_meses = horizontes[horizonte_nome]

st.sidebar.subheader("Parâmetros ARIMA")
p = st.sidebar.slider("AR (p)", 0, 5, 2)
d = st.sidebar.slider("Diferenciação (d)", 0, 2, 1)
q = st.sidebar.slider("MA (q)", 0, 5, 2)

# ========== FUNÇÕES ==========
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

def calcular_rmse(serie, p, d, q):
    if len(serie) < 36:
        return None
    treino = serie[:-12]
    real = serie[-12:]
    try:
        modelo = ARIMA(treino, order=(p,d,q)).fit()
        pred = modelo.forecast(12)
        return mean_squared_error(real, pred, squared=False)
    except:
        return None

def gerar_grafico(serie, datas, media, intervalo, unidade):
    customdata = np.stack([media, intervalo.iloc[:, 0], intervalo.iloc[:, 1]], axis=-1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=serie.index, y=serie,
        name="Histórico", mode="lines",
        line=dict(color="black")
    ))

    fig.add_trace(go.Scatter(
        x=datas, y=media,
        name="Previsão ARIMA",
        mode="lines+markers",
        line=dict(color="orange"),
        marker=dict(size=6),
        customdata=customdata,
        hovertemplate=(
            "<b>Data:</b> %{x|%b/%Y}<br>" +
            f"<b>Previsão:</b> R$ %{{customdata[0]:.2f}}/{unidade}<br>" +
            "<b>IC Inferior:</b> R$ %{customdata[1]:.2f}<br>" +
            "<b>IC Superior:</b> R$ %{customdata[2]:.2f}<extra></extra>"
        )
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([datas, datas[::-1]]),
        y=np.concatenate([intervalo.iloc[:, 0], intervalo.iloc[:, 1][::-1]]),
        fill='toself', fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'), showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=f"Previsão para {n_meses} meses com ARIMA({p},{d},{q})",
        xaxis_title="Data",
        yaxis_title=f"Preço deflacionado (R$/{unidade})",
        template="plotly_white",
        height=480
    )

    return fig

# ========== EXECUÇÃO ==========
df_base = carregar_base()
produtos = sorted(df_base['produto'].unique())
produto = st.selectbox("Produto:", produtos)

df_prod = preparar_serie(df_base[df_base['produto'] == produto])
serie = df_prod['preco_deflacionado']
unidade = "litro" if "leite" in produto.lower() else "saca"

try:
    datas_prev, media_prev, intervalo_prev = prever_arima(serie, n_meses, p, d, q)
    rmse = calcular_rmse(serie, p, d, q)
    fig = gerar_grafico(serie, datas_prev, media_prev, intervalo_prev, unidade)
    st.plotly_chart(fig, use_container_width=True)

    # RMSE
    if rmse:
        st.metric("RMSE (últimos 12 meses)", f"{rmse:.2f}")
    else:
        st.warning("Não foi possível calcular o RMSE (dados insuficientes).")

except Exception as e:
    st.error(f"Erro ao calcular previsão: {e}")

# ========== RODAPÉ ==========
st.markdown("---")
st.markdown(f"""
📊 Desenvolvido por **Lucas França e Paola Conti**  
📅 Atualizado em Maio/2025  
🔍 Modelo ARIMA({p},{d},{q}) aplicado  
📩 Contato: contato@ufsm.com.br
""")
