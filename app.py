# --- streamlit_sarima_app.py ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Previsão Agro-ARIMA", layout="wide")

# ======= ESTILO =======
st.markdown("""
<style>
h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
footer {visibility: hidden;}
.sidebar-logo {
    display: flex;
    justify-content: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ======= CABEÇALHO =======
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("static/images/logo.png", width=150)
with col2:
    st.title("📈 Previsão de Preços Agrícolas no Rio Grande do Sul")
    st.caption("Modelo ARIMA/SARIMA ajustado para séries temporais deflacionadas com RMSE dos últimos 12 meses.")

with st.expander("👩‍🎓 Sobre este Projeto Acadêmico", expanded=False):
    st.markdown("""
Projeto independente dos mestrandos Lucas França e Paola Conti (PPGAGR/UFSM-PM), com aplicação de modelos de séries temporais a preços agropecuários deflacionados. 

Combina:
- Modelagem ARIMA/SARIMA
- Visualização interativa
- Métricas como RMSE

✨ **Com orientação de:** Prof. Gabriel Nunes, Nilson Costa e Vinícius Carvalho.
📅 **Ano-base:** 2025
""")

# ======= METODOLOGIA =======
with st.expander("📘 Sobre a Metodologia", expanded=False):
    st.markdown("""
### Fontes e Deflação
- Dados CEPEA/ESALQ
- Preços deflacionados via IGP-DI/FGV

### O que é ARIMA/SARIMA?
- **ARIMA(p,d,q)** modela relações não sazonais
- **SARIMA(p,d,q)(P,D,Q)m** incorpora sazonalidade (como padrões anuais em dados mensais)

### Sazonalidade
Este painel permite incluir sazonalidade com **ciclo anual (m=12)**, ajustando o modelo para oscilações sazonais de preço (ex: colheita ou entre-safra).

### AutoARIMA
Escolhe automaticamente os melhores parâmetros (p,d,q)(P,D,Q)m, otimizando o modelo via AIC/BIC.

### RMSE
Erro quadrático médio da previsão dos últimos 12 meses: quanto menor, melhor.
""")

# ======= SIDEBAR =======
st.sidebar.image("static/images/logo_ufsm.png", width=120)
st.sidebar.header("Parâmetros")

horizontes = {"6 meses": 6, "12 meses": 12, "24 meses": 24, "48 meses": 48}
horizonte_nome = st.sidebar.radio("Horizonte de Previsão", list(horizontes.keys()))
n_meses = horizontes[horizonte_nome]

usar_autoarima = st.sidebar.checkbox("Usar AutoARIMA", value=True)
usar_sazonalidade = st.sidebar.checkbox("Incluir sazonalidade (SARIMA)", value=True)
periodo_sazonal = st.sidebar.number_input("Período sazonal (meses)", min_value=1, value=12)

if not usar_autoarima:
    st.sidebar.subheader("Parâmetros manuais")
    p = st.sidebar.slider("AR (p)", 0, 5, 2)
    d = st.sidebar.slider("Diferenciação (d)", 0, 2, 1)
    q = st.sidebar.slider("MA (q)", 0, 5, 2)
    P = st.sidebar.slider("SAR (P)", 0, 2, 1)
    D = st.sidebar.slider("SD (D)", 0, 2, 1)
    Q = st.sidebar.slider("SMA (Q)", 0, 2, 1)
    s = periodo_sazonal
else:
    p = d = q = P = D = Q = s = None

# ======= FUNÇÕES =======
def carregar_base():
    df = pd.read_csv("dados/base_unificada_cepea.csv", encoding="latin1")
    df.columns = [col.lower().strip() for col in df.columns]
    return df

def preparar_serie(df):
    meses = { 'janeiro': 'January', 'fevereiro': 'February', 'março': 'March', 'abril': 'April', 'maio': 'May', 'junho': 'June',
              'julho': 'July', 'agosto': 'August', 'setembro': 'September', 'outubro': 'October', 'novembro': 'November', 'dezembro': 'December' }
    df['preco_deflacionado'] = df['preco_deflacionado'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
    df['mes_en'] = df['mes'].str.lower().str.strip().map(meses)
    df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes_en'], format='%Y-%B', errors='coerce')
    return df.dropna(subset=['data']).sort_values('data').set_index('data')

def encontrar_melhor_arima(serie, sazonal, m):
    modelo = auto_arima(serie, seasonal=sazonal, m=m, stepwise=True, suppress_warnings=True, error_action='ignore')
    return modelo.order, modelo.seasonal_order

def prever_arima(serie, steps, p, d, q, P=0, D=0, Q=0, s=0, sazonal=False):
    if sazonal:
        modelo = ARIMA(serie, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit()
    else:
        modelo = ARIMA(serie, order=(p, d, q)).fit()
    previsao = modelo.get_forecast(steps=steps)
    return (
        pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS'),
        previsao.predicted_mean,
        previsao.conf_int()
    )

def calcular_rmse(serie, p, d, q, P, D, Q, s, sazonal):
    if len(serie) < 36:
        return None
    treino = serie[:-12]
    real = serie[-12:]
    try:
        if sazonal:
            modelo = ARIMA(treino, order=(p,d,q), seasonal_order=(P,D,Q,s)).fit()
        else:
            modelo = ARIMA(treino, order=(p,d,q)).fit()
        pred = modelo.forecast(12)
        return mean_squared_error(real, pred, squared=False)
    except:
        return None

def gerar_grafico(serie, datas, media, intervalo, unidade, titulo_modelo):
    customdata = np.stack([media, intervalo.iloc[:, 0], intervalo.iloc[:, 1]], axis=-1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie, name="Histórico", mode="lines", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=datas, y=media, name="Previsão", mode="lines+markers", line=dict(color="orange"),
                             customdata=customdata, hovertemplate="""<b>Data:</b> %{x|%b/%Y}<br><b>Previsão:</b> R$ %{customdata[0]:.2f}/""" + unidade + "<br><b>IC Inferior:</b> R$ %{customdata[1]:.2f}<br><b>IC Superior:</b> R$ %{customdata[2]:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.concatenate([datas, datas[::-1]]),
                             y=np.concatenate([intervalo.iloc[:, 0], intervalo.iloc[:, 1][::-1]]),
                             fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig.update_layout(title=titulo_modelo, xaxis_title="Data", yaxis_title=f"Preço deflacionado (R$/{unidade})", template="plotly_white", height=480)
    return fig

# ======= EXECUÇÃO =======
df_base = carregar_base()
produtos = sorted(df_base['produto'].unique())
produto = st.selectbox("Produto:", produtos)
df_prod = preparar_serie(df_base[df_base['produto'] == produto])
serie = df_prod['preco_deflacionado']
unidade = "litro" if "leite" in produto.lower() else "saca"

if usar_autoarima:
    try:
        (p, d, q), (P, D, Q, s) = encontrar_melhor_arima(serie, usar_sazonalidade, periodo_sazonal)
        st.sidebar.success(f"Melhor modelo: ARIMA({p},{d},{q})" + (f" × ({P},{D},{Q}){s}" if usar_sazonalidade else ""))
    except Exception as e:
        st.sidebar.error(f"Erro AutoARIMA: {e}")
        p, d, q, P, D, Q, s = 2, 1, 2, 0, 0, 0, 0

try:
    datas_prev, media_prev, intervalo_prev = prever_arima(serie, n_meses, p, d, q, P, D, Q, s, usar_sazonalidade)
    rmse = calcular_rmse(serie, p, d, q, P, D, Q, s, usar_sazonalidade)
    titulo_modelo = f"Previsão com ARIMA({p},{d},{q})" + (f" × ({P},{D},{Q}){s}" if usar_sazonalidade else "")
    fig = gerar_grafico(serie, datas_prev, media_prev, intervalo_prev, unidade, titulo_modelo)
    st.plotly_chart(fig, use_container_width=True)

    if rmse:
        st.metric("RMSE (12 meses)", f"{rmse:.2f}")
    else:
        st.warning("Dados insuficientes para calcular o RMSE.")
except Exception as e:
    st.error(f"Erro ao gerar previsão: {e}")





# ======= RODAPÉ =======
st.markdown("---")

# Define a descrição do modelo
modelo_usado = "AutoARIMA (seleção automática)" if usar_autoarima else f"ARIMA escolhido manualmente ({p}, {d}, {q})"

st.markdown(f"""
🔧 **Projeto desenvolvido por:** Lucas França e Paola Conti  
🎓 Programa de Pós-Graduação em Agronegócios – UFSM (PPGAGR/UFSM-PM)  
📅 Atualizado em Maio/2025  
📈 Modelo utilizado: {modelo_usado}  
📬 Contato: [lucas.tanaro@acad.ufsm.br](mailto:lucas.tanaro@acad.ufsm.br)
""")


