import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(page_title="Previsão ARIMA", layout="wide")

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

# ======= CABEÇALHO COM LOGO =======
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("static/images/logo.png", width=80)
with col2:
    st.title("📈 Previsão de Preços com ARIMA")
    st.caption("Modelo ARIMA ajustado para séries temporais deflacionadas com RMSE dos últimos 12 meses.")
with col3:
    st.image("static/images/logo2.png", width=80)

# ======= EXPLICAÇÃO METODOLÓGICA =======
with st.expander("📘 Sobre a Metodologia", expanded=False):
    st.markdown("""
### 🔍 O que é ARIMA?

**ARIMA** significa:  
- **AR (p)**: Parte autoregressiva  
- **I (d)**: Parte de integração (diferença)  
- **MA (q)**: Parte de média móvel  

O modelo ARIMA é útil para **previsão de séries temporais** e combina três componentes:

---

### 🧩 Parâmetros do modelo ARIMA(p, d, q):

- **p (Autoregressivo - AR):**
  - Número de observações passadas consideradas como preditores.
  - **Exemplo:** p = 1 → o valor atual depende diretamente do mês anterior.  
    p = 3 → depende dos últimos três meses.
  - ⏳ Quanto maior o p, mais o modelo "olha para trás".

- **d (Diferenciação - I):**
  - Quantas vezes a série é diferenciada para torná-la estacionária.
  - **Exemplo:** d = 1 → subtrai o valor do mês atual pelo mês anterior.
  - 📉 Se a série já parece estável, use d = 0. Se há tendência, use d = 1.

- **q (Média Móvel - MA):**
  - Número de erros passados usados para corrigir a previsão atual.
  - **Exemplo:** q = 1 → o erro do último mês é usado para ajustar a previsão atual.
  - 🔧 Alta sensibilidade a flutuações recentes.

---

### 📘 AutoARIMA

Se você ativar o **AutoARIMA**, o sistema escolhe automaticamente os melhores valores de p, d e q com base em testes estatísticos (AIC, BIC). Isso ajuda a evitar ajustes manuais e otimiza o desempenho da previsão.

---

### 📏 RMSE

O RMSE (Root Mean Squared Error) calcula o erro médio da previsão nos últimos 12 meses. Valores menores indicam previsões mais próximas dos valores reais.
""")

# ======= SIDEBAR COM LOGO =======
st.sidebar.markdown('<div class="sidebar-logo"><img src="static/images/logo_ufsm.png" width="120"></div>', unsafe_allow_html=True)
st.sidebar.header("Parâmetros")

horizontes = {"6 meses": 6, "12 meses": 12, "24 meses": 24, "48 meses": 48}
horizonte_nome = st.sidebar.radio("Horizonte de Previsão", list(horizontes.keys()))
n_meses = horizontes[horizonte_nome]

usar_autoarima = st.sidebar.checkbox("Usar AutoARIMA (sugerir melhor modelo)", value=True)

if not usar_autoarima:
    st.sidebar.subheader("Parâmetros ARIMA manuais")
    p = st.sidebar.slider("AR (p)", 0, 5, 2)
    d = st.sidebar.slider("Diferenciação (d)", 0, 2, 1)
    q = st.sidebar.slider("MA (q)", 0, 5, 2)
else:
    p = d = q = None

# ======= FUNÇÕES =======
def carregar_base():
    df = pd.read_csv("dados/base_unificada_cepea.csv", encoding="latin1")
    df.columns = [col.lower().strip() for col in df.columns]
    return df

def preparar_serie(df):
    meses = { 'janeiro': 'January', 'fevereiro': 'February', 'março': 'March', 'abril': 'April', 'maio': 'May', 'junho': 'June', 'julho': 'July', 'agosto': 'August', 'setembro': 'September', 'outubro': 'October', 'novembro': 'November', 'dezembro': 'December' }
    df['preco_deflacionado'] = df['preco_deflacionado'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
    df['mes_en'] = df['mes'].str.lower().str.strip().map(meses)
    df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes_en'], format='%Y-%B', errors='coerce')
    return df.dropna(subset=['data']).sort_values('data').set_index('data')

def encontrar_melhor_arima(serie):
    modelo = auto_arima(serie, seasonal=False, stepwise=True, suppress_warnings=True)
    return modelo.order

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
    fig.add_trace(go.Scatter(x=serie.index, y=serie, name="Histórico", mode="lines", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=datas, y=media, name="Previsão ARIMA", mode="lines+markers", line=dict(color="orange"), marker=dict(size=6), customdata=customdata, hovertemplate="""<b>Data:</b> %{x|%b/%Y}<br><b>Previsão:</b> R$ %{customdata[0]:.2f}/""" + unidade + "<br><b>IC Inferior:</b> R$ %{customdata[1]:.2f}<br><b>IC Superior:</b> R$ %{customdata[2]:.2f}<extra></extra>"""))
    fig.add_trace(go.Scatter(x=np.concatenate([datas, datas[::-1]]), y=np.concatenate([intervalo.iloc[:, 0], intervalo.iloc[:, 1][::-1]]), fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo='skip'))
    fig.update_layout(title=f"Previsão para {n_meses} meses com ARIMA({p},{d},{q})", xaxis_title="Data", yaxis_title=f"Preço deflacionado (R$/{unidade})", template="plotly_white", height=480)
    return fig

def gerar_pdf(produto, modelo_arima, datas, media, intervalo, unidade):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    logo_path = "static/images/logo.png"
    if os.path.exists(logo_path):
        story.append(Image(logo_path, width=2*inch, height=1*inch))
    story.append(Paragraph("EconoMetrika Inteligência em Negócios", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Previsão de Preço - {produto.title()}</b>", styles["Heading2"]))
    story.append(Paragraph(f"Modelo ARIMA aplicado: ({modelo_arima[0]},{modelo_arima[1]},{modelo_arima[2]})", styles["Normal"]))
    story.append(Spacer(1, 12))
    dados = [["Data", "Previsão (R$)", "IC Inferior", "IC Superior"]]
    for i in range(len(datas)):
        dados.append([
            datas[i].strftime("%b/%Y"),
            f"{media[i]:.2f}",
            f"{intervalo.iloc[i, 0]:.2f}",
            f"{intervalo.iloc[i, 1]:.2f}"
        ])
    story.append(Table(dados))
    story.append(Spacer(1, 24))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(datas, media, label="Previsão", color="orange")
    ax.fill_between(datas, intervalo.iloc[:, 0], intervalo.iloc[:, 1], color="orange", alpha=0.3)
    ax.set_title(f"Previsão ARIMA - {produto}")
    ax.set_xlabel("Data")
    ax.set_ylabel(f"R$/{unidade}")
    ax.legend()
    plt.tight_layout()
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png")
    plt.close(fig)
    img_buffer.seek(0)
    story.append(Image(img_buffer, width=400, height=200))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ======= EXECUÇÃO PRINCIPAL =======
df_base = carregar_base()
produtos = sorted(df_base['produto'].unique())
produto = st.selectbox("Produto:", produtos)
df_prod = preparar_serie(df_base[df_base['produto'] == produto])
serie = df_prod['preco_deflacionado']
unidade = "litro" if "leite" in produto.lower() else "saca"

if usar_autoarima:
    try:
        p, d, q = encontrar_melhor_arima(serie)
        st.sidebar.success(f"Melhor modelo sugerido: ARIMA({p},{d},{q})")
    except Exception as e:
        st.sidebar.error(f"Erro ao rodar AutoARIMA: {e}")
        p, d, q = 2, 1, 2

try:
    datas_prev, media_prev, intervalo_prev = prever_arima(serie, n_meses, p, d, q)
    rmse = calcular_rmse(serie, p, d, q)
    fig = gerar_grafico(serie, datas_prev, media_prev, intervalo_prev, unidade)
    st.plotly_chart(fig, use_container_width=True)
    if rmse:
        st.metric("RMSE (últimos 12 meses)", f"{rmse:.2f}")
        st.caption("O RMSE (Root Mean Squared Error) mede o erro médio entre os valores reais e previstos nos últimos 12 meses da série histórica. Quanto menor o RMSE, melhor o desempenho do modelo na previsão recente.")
    else:
        st.warning("Não foi possível calcular o RMSE (dados insuficientes).")

    # PDF
    pdf_buffer = gerar_pdf(produto, (p, d, q), datas_prev, media_prev, intervalo_prev, unidade)
    st.download_button(
        label="📄 Gerar PDF da previsão",
        data=pdf_buffer,
        file_name=f"previsao_arima_{produto.lower().replace(' ', '_')}.pdf",
        mime="application/pdf"
    )

except Exception as e:
    st.error(f"Erro ao calcular previsão: {repr(e)}")


# ======= RODAPÉ =======
st.markdown("---")
st.markdown(f"""
📊 Desenvolvido por **Lucas França e Paola Conti**  
📅 Atualizado em Maio/2025  
🔍 Modelo ARIMA({p},{d},{q}) aplicado  
📩 Contato: contato@ufsm.com.br
""")
