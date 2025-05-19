import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import os

# ========== CONFIGURAÇÃO DA PÁGINA ==========
st.set_page_config(page_title="Painel de Previsão Agro", layout="wide")

# ========== ESTILIZAÇÃO ==========
st.markdown("""
<style>
/* Cabeçalhos e fontes */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    padding-top: 1.5rem;
}
/* Estilo de gráfico */
.stPlotlyChart {
    border-radius: 6px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}
/* Cartão por produto */
.card-container {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 0 8px rgba(0, 0, 0, 0.04);
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ========== LOGO E TÍTULO ==========
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("static/images/logo.png", width=120)
with col_title:
    st.markdown("## **Painel de Previsão de Preços Agropecuários**")
    st.caption("Análises temporais com ARIMA para o estado do Rio Grande do Sul.")

st.markdown("---")

# ========== SIDEBAR ==========
st.sidebar.header("🔧 Parâmetros")
horizontes = {"6 meses": 6, "12 meses": 12, "24 meses": 24, "48 meses": 48}
horizonte = st.sidebar.radio("Horizonte de Previsão:", list(horizontes.keys()))
n_meses = horizontes[horizonte]
st.sidebar.caption("Modelo ARIMA(2,1,2) aplicado. Estimativas sujeitas a revisão.")

# ========== FUNÇÃO DE PREVISÃO ==========
def gerar_previsao(df, nome, unidade):
    df = df.copy()
    df['preco_deflacionado'] = df['preco_deflacionado'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)

    meses = {
        'janeiro': 'January', 'fevereiro': 'February', 'março': 'March',
        'abril': 'April', 'maio': 'May', 'junho': 'June',
        'julho': 'July', 'agosto': 'August', 'setembro': 'September',
        'outubro': 'October', 'novembro': 'November', 'dezembro': 'December'
    }

    df['mes_en'] = df['mes'].str.lower().str.strip().map(meses)
    df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes_en'], format='%Y-%B', errors='coerce')
    df = df.dropna(subset=['data']).sort_values('data').set_index('data')

    serie = df['preco_deflacionado']

    if len(serie.dropna()) < 24 or serie.nunique() < 5:
        st.warning(f"⚠️ Série '{nome}' possui dados insuficientes ou pouca variação.")
        return go.Figure()

    try:
        modelo = ARIMA(serie, order=(2, 1, 2)).fit()
        previsao = modelo.get_forecast(steps=n_meses)
    except Exception as e:
        st.error(f"Erro no ARIMA para {nome}: {e}")
        return go.Figure()

    media = previsao.predicted_mean
    intervalo = previsao.conf_int()
    datas_prev = pd.date_range(start=serie.index[-1] + pd.offsets.MonthBegin(1), periods=n_meses, freq='MS')
    customdata = np.stack([media, intervalo.iloc[:, 0], intervalo.iloc[:, 1]], axis=-1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=serie.index, y=serie, mode='lines', name='Histórico',
        line=dict(color='black', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=datas_prev, y=media, mode='lines', name='Previsão',
        line=dict(color='orange', dash='dot'),
        customdata=customdata,
        hovertemplate=(
            '<b>Data:</b> %{x|%b/%Y}<br>' +
            f'<b>Previsão:</b> R$ %{{customdata[0]:.2f}}/{unidade}<br>' +
            f'<b>IC Inferior:</b> R$ %{{customdata[1]:.2f}}<br>' +
            f'<b>IC Superior:</b> R$ %{{customdata[2]:.2f}}<extra></extra>'
        )
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([datas_prev, datas_prev[::-1]]),
        y=np.concatenate([intervalo.iloc[:, 0], intervalo.iloc[:, 1][::-1]]),
        fill='toself', fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo='skip'
    ))

    fig.update_layout(
        title=f"{nome} - Previsão para {n_meses} meses",
        xaxis_title="Ano",
        yaxis_title=f"Preço deflacionado (R$/{unidade})",
        template="plotly_white",
        height=480,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig

# ========== CARREGAMENTO DOS DADOS ==========
try:
    df_base = pd.read_csv("dados/base_unificada_cepea.csv", encoding="latin1")
    df_base.columns = [col.lower().strip() for col in df_base.columns]
    if 'produto' not in df_base.columns:
        st.error(f"Coluna 'produto' ausente. Verifique as colunas disponíveis: {df_base.columns.tolist()}")
        st.stop()
except Exception as e:
    st.error(f"Erro ao carregar base de dados: {e}")
    st.stop()

# ========== DASHBOARD POR PRODUTO ==========
for produto in df_base['produto'].unique():
    st.markdown('<div class="card-container">', unsafe_allow_html=True)

    col_img, col_info = st.columns([1, 6])
    with col_img:
        img_path = f"static/images/{produto.lower()}.png"
        if os.path.exists(img_path):
            st.image(img_path, width=100)
    with col_info:
        st.subheader(f"📊 {produto.capitalize()}")

    df_prod = df_base[df_base['produto'] == produto]
    unidade = "litro" if "leite" in produto.lower() else "saca"
    grafico = gerar_previsao(df_prod, produto, unidade)
    st.plotly_chart(grafico, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ========== RODAPÉ ==========
st.markdown("---")
st.markdown("""
📊 Desenvolvido por **Lucas França e Paola Conti**  
📅 Última atualização: Maio/2025  
🔍 Modelo aplicado: ARIMA(2,1,2)  
📩 Contato: contato@ufsm.com.br
""")
