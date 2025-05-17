import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import os

st.set_page_config(page_title="Painel de Previsão Agro", layout="wide")

# ======= ESTILO VISUAL =======
st.markdown("""
<style>
h1 { color: #2e7d32; text-align: center; font-size: 36px; }
section[data-testid="stSidebar"] { background-color: #f0fdf4; padding-top: 2rem; }
.logo-container img { margin-top: 10px; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ======= LOGO E TÍTULO =======
col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image("static/images/logo.png", width=150)
    st.markdown('</div>', unsafe_allow_html=True)

with col_title:
    st.markdown("## **Previsão de Preços Agro**")
    st.markdown("### Análise e previsão de preços agropecuários no RS com modelo ARIMA.")

st.markdown("---")

# ======= SIDEBAR =======
st.sidebar.markdown("## 📈 Parâmetros da Previsão")

opcoes_horizonte = {
    "6 meses": 6,
    "12 meses": 12,
    "24 meses": 24,
    "48 meses": 48
}

escolha = st.sidebar.radio("Selecione o horizonte de previsão:", list(opcoes_horizonte.keys()))
horizonte = opcoes_horizonte[escolha]

st.sidebar.caption("Resultados baseados no modelo ARIMA(2,1,2). Estimativas aproximadas.")

# ======= FUNÇÃO DE PREVISÃO =======
def prever_serie(df_produto, nome, unidade):
    df = df_produto.copy()
    df['preco_deflacionado'] = df['preco_deflacionado'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)

    meses_pt_en = {
        'janeiro': 'January', 'fevereiro': 'February', 'março': 'March',
        'abril': 'April', 'maio': 'May', 'junho': 'June',
        'julho': 'July', 'agosto': 'August', 'setembro': 'September',
        'outubro': 'October', 'novembro': 'November', 'dezembro': 'December'
    }

    df['mes_en'] = df['mes'].str.strip().str.lower().map(meses_pt_en)
    df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes_en'], format='%Y-%B', errors='coerce')
    df = df.dropna(subset=['data']).sort_values('data')
    df.set_index('data', inplace=True)

    serie = df['preco_deflacionado']

    if len(serie.dropna()) < 24 or serie.nunique() < 5:
        st.warning(f"A série de {nome} tem dados insuficientes ou pouca variação.")
        return go.Figure()

    try:
        modelo = ARIMA(serie, order=(2,1,2)).fit()
    except Exception as e:
        st.error(f"Erro ao ajustar ARIMA para {nome}: {str(e)}")
        return go.Figure()

    previsao = modelo.get_forecast(steps=horizonte)
    media = previsao.predicted_mean
    intervalo = previsao.conf_int()
    datas = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=horizonte, freq='MS')
    customdata = np.stack((media, intervalo.iloc[:, 0], intervalo.iloc[:, 1]), axis=-1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=serie.index, y=serie,
        mode='lines',
        name='Histórico',
        line=dict(color='black', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=datas,
        y=media,
        mode='lines',
        name='Previsão',
        line=dict(color='orange', dash='dot'),
        customdata=customdata,
        hovertemplate=
            '<b>Data:</b> %{x|%b/%Y}<br>' +
            f'<b>Previsão:</b> R$ %{{customdata[0]:.2f}}/{unidade}<br>' +
            f'<b>IC Inferior:</b> R$ %{{customdata[1]:.2f}}<br>' +
            f'<b>IC Superior:</b> R$ %{{customdata[2]:.2f}}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([datas, datas[::-1]]),
        y=np.concatenate([intervalo.iloc[:, 0], intervalo.iloc[:, 1][::-1]]),
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        title=f'{nome} - Previsão para {horizonte} meses',
        xaxis_title='Ano',
        yaxis_title=f'Preço deflacionado (R$/{unidade})',
        template='plotly_white',
        height=500
    )
    return fig

# ======= CARREGAR CSV COM TODOS OS PRODUTOS =======
try:
    df = pd.read_csv("dados/base_unificada_cepea.csv", encoding='latin1', sep=';')
except Exception as e:
    st.error(f"Erro ao carregar o CSV: {e}")
    st.stop()

df.columns = [col.strip().lower() for col in df.columns]
produtos_disponiveis = df['produto'].unique()

# ======= GERAR UM DASHBOARD POR PRODUTO =======
for produto in produtos_disponiveis:
    st.markdown("---")
    col_img, col_txt = st.columns([1, 6])
    with col_img:
        img_path = f"static/images/{produto.lower()}.png"
        if os.path.exists(img_path):
            st.image(img_path, width=120)
    with col_txt:
        st.subheader(f"📊 {produto}")

    df_prod = df[df['produto'] == produto]
    unidade = "litro" if "leite" in produto.lower() else "saca"
    fig = prever_serie(df_prod, produto, unidade)
    st.plotly_chart(fig, use_container_width=True)

# ======= ASSINATURA FINAL =======
st.markdown("---")
st.markdown("""
📊 Desenvolvido por **Lucas França e Paola Conti**  
📅 Última atualização: Maio/2025  
🔎 Modelo: ARIMA(2,1,2)  
📬 Contato: contato@ufsm.com.br
""")

