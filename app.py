import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import glob
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Ignorar avisos para melhorar a experiência do usuário
warnings.filterwarnings('ignore')

# ======= CONFIGURAÇÃO DA PÁGINA =======
st.set_page_config(
    page_title="Painel de Previsão Agro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======= ESTILO VISUAL =======
st.markdown("""
<style>
    /* Cores do agronegócio: verde, marrom, dourado */
    :root {
        --main-green: #2e7d32;
        --light-green: #f0fdf4;
        --dark-green: #1b5e20;
        --wheat-color: #F5DEB3;
        --soil-brown: #8B4513;
    }
    
    h1, h2, h3 {
        color: var(--main-green);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1 {
        text-align: center;
        font-size: 36px;
        border-bottom: 2px solid var(--wheat-color);
        padding-bottom: 10px;
    }
    
    h2 {
        font-size: 28px;
        border-left: 4px solid var(--main-green);
        padding-left: 10px;
    }
    
    h3 {
        font-size: 22px;
    }
    
    .stButton button {
        background-color: var(--main-green);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background-color: var(--dark-green);
    }
    
    section[data-testid="stSidebar"] {
        background-color: var(--light-green);
        padding-top: 2rem;
        border-right: 1px solid var(--main-green);
    }
    
    .logo-container img {
        margin-top: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background-color: var(--light-green);
        border-left: 4px solid var(--main-green);
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .product-icon {
        font-size: 24px;
        margin-right: 10px;
    }
    
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ======= FUNÇÕES AUXILIARES =======

def carregar_dados(produto):
    """
    Carrega os dados de um produto específico da pasta 'dados'.
    
    Args:
        produto (str): Nome do produto a ser carregado
        
    Returns:
        DataFrame: DataFrame com os dados do produto ou None em caso de erro
    """
    try:
        # Caminho para o arquivo CSV do produto
        caminho_arquivo = f"dados/{produto}.csv"
        
        # Verificar se o arquivo existe
        if not os.path.exists(caminho_arquivo):
            st.error(f"Arquivo de dados para {produto} não encontrado em {caminho_arquivo}")
            return None
        
        # Carregar os dados do CSV
        df = pd.read_csv(caminho_arquivo, encoding='utf-8', sep=',')
        
        # Verificar se as colunas necessárias existem
        colunas_necessarias = ['ano', 'mes', 'preco_deflacionado', 'produto']
        for coluna in colunas_necessarias:
            if coluna not in df.columns:
                st.error(f"Coluna '{coluna}' não encontrada no arquivo de {produto}")
                return None
        
        # Limpar e preparar os dados
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Converter preços para formato numérico
        df['preco_deflacionado'] = df['preco_deflacionado'].astype(str).str.replace(' ', '')
        df['preco_deflacionado'] = df['preco_deflacionado'].str.replace(',', '.').astype(float)
        
        # Mapear nomes dos meses para inglês para criar datas
        meses_pt_en = {
            'janeiro': 'January', 'fevereiro': 'February', 'março': 'March', 'marco': 'March',
            'abril': 'April', 'maio': 'May', 'junho': 'June',
            'julho': 'July', 'agosto': 'August', 'setembro': 'September',
            'outubro': 'October', 'novembro': 'November', 'dezembro': 'December'
        }
        
        # Criar coluna de mês em inglês e data
        df['mes_en'] = df['mes'].str.strip().str.lower().map(meses_pt_en)
        df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes_en'], format='%Y-%B', errors='coerce')
        
        # Remover linhas com datas inválidas e ordenar
        df = df.dropna(subset=['data']).sort_values('data')
        
        # Definir data como índice
        df.set_index('data', inplace=True)
        
        return df
    
    except Exception as e:
        st.error(f"Erro ao carregar dados de {produto}: {str(e)}")
        return None

def avaliar_modelos(serie, horizonte):
    """
    Avalia diferentes modelos de previsão para selecionar o melhor.
    
    Args:
        serie (Series): Série temporal a ser modelada
        horizonte (int): Horizonte de previsão desejado
        
    Returns:
        tuple: Modelo treinado, nome do modelo, métricas de avaliação
    """
    # Verificar se há dados suficientes
    if len(serie) < 24:
        return None, "Dados insuficientes", {}
    
    # Dividir dados em treino e teste (últimos 6 meses para teste)
    n_teste = min(6, len(serie) // 4)
    treino = serie[:-n_teste]
    teste = serie[-n_teste:]
    
    resultados = {}
    modelos = {}
    
    # Testar modelo ARIMA
    try:
        modelo_arima = ARIMA(treino, order=(2,1,2)).fit()
        previsao_arima = modelo_arima.forecast(steps=n_teste)
        
        mae_arima = mean_absolute_error(teste, previsao_arima)
        mse_arima = mean_squared_error(teste, previsao_arima)
        rmse_arima = np.sqrt(mse_arima)
        
        resultados['ARIMA'] = {
            'MAE': mae_arima,
            'RMSE': rmse_arima
        }
        modelos['ARIMA'] = ARIMA(serie, order=(2,1,2)).fit()
    except Exception as e:
        resultados['ARIMA'] = {'erro': str(e)}
    
    # Testar modelo SARIMA
    try:
        modelo_sarima = SARIMAX(treino, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        previsao_sarima = modelo_sarima.forecast(steps=n_teste)
        
        mae_sarima = mean_absolute_error(teste, previsao_sarima)
        mse_sarima = mean_squared_error(teste, previsao_sarima)
        rmse_sarima = np.sqrt(mse_sarima)
        
        resultados['SARIMA'] = {
            'MAE': mae_sarima,
            'RMSE': rmse_sarima
        }
        modelos['SARIMA'] = SARIMAX(serie, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except Exception as e:
        resultados['SARIMA'] = {'erro': str(e)}
    
    # Testar modelo Prophet
    try:
        # Preparar dados para Prophet
        df_prophet = pd.DataFrame({'ds': treino.index, 'y': treino.values})
        
        modelo_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        modelo_prophet.fit(df_prophet)
        
        # Fazer previsões
        futuro = modelo_prophet.make_future_dataframe(periods=n_teste, freq='MS')
        previsao = modelo_prophet.predict(futuro)
        
        # Extrair previsões para o período de teste
        previsao_prophet = previsao.tail(n_teste)['yhat'].values
        
        mae_prophet = mean_absolute_error(teste, previsao_prophet)
        mse_prophet = mean_squared_error(teste, previsao_prophet)
        rmse_prophet = np.sqrt(mse_prophet)
        
        resultados['Prophet'] = {
            'MAE': mae_prophet,
            'RMSE': rmse_prophet
        }
        
        # Treinar modelo final com todos os dados
        df_prophet_completo = pd.DataFrame({'ds': serie.index, 'y': serie.values})
        modelo_prophet_final = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        modelo_prophet_final.fit(df_prophet_completo)
        modelos['Prophet'] = modelo_prophet_final
    except Exception as e:
        resultados['Prophet'] = {'erro': str(e)}
    
    # Selecionar o melhor modelo com base no RMSE
    melhor_modelo = None
    menor_rmse = float('inf')
    nome_melhor = "ARIMA"  # Padrão caso nenhum modelo funcione
    
    for nome, metricas in resultados.items():
        if 'RMSE' in metricas and metricas['RMSE'] < menor_rmse:
            menor_rmse = metricas['RMSE']
            nome_melhor = nome
            melhor_modelo = modelos.get(nome)
    
    return melhor_modelo, nome_melhor, resultados

def fazer_previsao(serie, modelo, nome_modelo, horizonte):
    """
    Realiza previsão usando o modelo selecionado.
    
    Args:
        serie (Series): Série temporal completa
        modelo: Modelo treinado
        nome_modelo (str): Nome do modelo ('ARIMA', 'SARIMA', 'Prophet')
        horizonte (int): Número de períodos a prever
        
    Returns:
        tuple: Datas futuras, valores previstos, intervalos de confiança
    """
    if modelo is None:
        return None, None, None
    
    # Datas futuras para previsão
    ultima_data = serie.index[-1]
    datas_futuras = pd.date_range(ultima_data + pd.offsets.MonthBegin(1), periods=horizonte, freq='MS')
    
    # Fazer previsão com base no tipo de modelo
    if nome_modelo == 'ARIMA' or nome_modelo == 'SARIMA':
        previsao = modelo.get_forecast(steps=horizonte)
        media = previsao.predicted_mean
        intervalo = previsao.conf_int()
        
        return datas_futuras, media, intervalo
    
    elif nome_modelo == 'Prophet':
        # Criar dataframe futuro para Prophet
        futuro = pd.DataFrame({'ds': datas_futuras})
        
        # Fazer previsão
        previsao = modelo.predict(futuro)
        
        # Extrair valores e intervalos
        media = previsao['yhat'].values
        intervalo_inf = previsao['yhat_lower'].values
        intervalo_sup = previsao['yhat_upper'].values
        
        # Criar DataFrame de intervalo de confiança no formato esperado
        intervalo = pd.DataFrame({
            'lower': intervalo_inf,
            'upper': intervalo_sup
        }, index=datas_futuras)
        
        return datas_futuras, pd.Series(media, index=datas_futuras), intervalo
    
    return None, None, None

def criar_grafico_previsao(serie, datas_futuras, media, intervalo, nome_produto, unidade, nome_modelo):
    """
    Cria um gráfico interativo com a série histórica e previsão.
    
    Args:
        serie (Series): Série temporal histórica
        datas_futuras (DatetimeIndex): Datas para previsão
        media (Series): Valores médios previstos
        intervalo (DataFrame): Intervalos de confiança
        nome_produto (str): Nome do produto
        unidade (str): Unidade de medida
        nome_modelo (str): Nome do modelo utilizado
        
    Returns:
        Figure: Objeto de figura do Plotly
    """
    # Criar figura
    fig = go.Figure()
    
    # Adicionar série histórica
    fig.add_trace(go.Scatter(
        x=serie.index, 
        y=serie,
        mode='lines',
        name='Histórico',
        line=dict(color='#2e7d32', width=2)
    ))
    
    # Adicionar linha de previsão
    if media is not None and datas_futuras is not None:
        # Preparar dados para hover
        customdata = np.stack((
            media, 
            intervalo.iloc[:, 0], 
            intervalo.iloc[:, 1]
        ), axis=-1)
        
        fig.add_trace(go.Scatter(
            x=datas_futuras,
            y=media,
            mode='lines',
            name='Previsão',
            line=dict(color='#FFA500', width=2, dash='dot'),
            customdata=customdata,
            hovertemplate=
                '<b>Data:</b> %{x|%b/%Y}<br>' +
                f'<b>Previsão:</b> R$ %{{customdata[0]:.2f}}/{unidade}<br>' +
                f'<b>IC Inferior:</b> R$ %{{customdata[1]:.2f}}<br>' +
                f'<b>IC Superior:</b> R$ %{{customdata[2]:.2f}}<extra></extra>'
        ))
        
        # Adicionar intervalo de confiança
        fig.add_trace(go.Scatter(
            x=np.concatenate([datas_futuras, datas_futuras[::-1]]),
            y=np.concatenate([intervalo.iloc[:, 0], intervalo.iloc[:, 1][::-1]]),
            fill='toself',
            fillcolor='rgba(46, 125, 50, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Configurar layout
    fig.update_layout(
        title=f'{nome_produto.capitalize()} - Previsão para {len(datas_futuras)} meses (Modelo: {nome_modelo})',
        xaxis_title='Ano',
        yaxis_title=f'Preço deflacionado (R$/{unidade})',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def obter_icone_produto(produto):
    """
    Retorna o ícone correspondente ao produto.
    
    Args:
        produto (str): Nome do produto
        
    Returns:
        str: Emoji correspondente ao produto
    """
    icones = {
        'leite': '🥛',
        'milho': '🌽',
        'soja': '🫘',
        'arroz': '🍚',
        'trigo': '🌾'
    }
    return icones.get(produto.lower(), '📊')

def obter_unidade_produto(produto):
    """
    Retorna a unidade de medida para cada produto.
    
    Args:
        produto (str): Nome do produto
        
    Returns:
        str: Unidade de medida
    """
    unidades = {
        'leite': 'litro',
        'milho': 'saca',
        'soja': 'saca',
        'arroz': 'saca',
        'trigo': 'saca'
    }
    return unidades.get(produto.lower(), 'unidade')

def listar_produtos_disponiveis():
    """
    Lista todos os produtos disponíveis na pasta de dados.
    
    Returns:
        list: Lista de nomes de produtos disponíveis
    """
    arquivos = glob.glob('dados/*.csv')
    produtos = [os.path.splitext(os.path.basename(arquivo))[0] for arquivo in arquivos]
    return sorted(produtos)

# ======= INTERFACE PRINCIPAL =======

# Logo e título
col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image("static/logo.png", width=150)
    st.markdown('</div>', unsafe_allow_html=True)

with col_title:
    st.markdown("# **Previsão de Preços Agro**")
    st.markdown("### Sistema de análise e previsão de preços agropecuários com modelos avançados de séries temporais")

st.markdown("---")

# ======= SIDEBAR =======
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="static/logo_ufsm.png" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("## 📈 Parâmetros da Previsão")

# Opções de horizonte
opcoes_horizonte = {
    "6 meses": 6,
    "12 meses": 12,
    "24 meses": 24,
    "48 meses": 48
}

# Seleção de horizonte
escolha = st.sidebar.radio("Selecione o horizonte de previsão:", list(opcoes_horizonte.keys()))
horizonte = opcoes_horizonte[escolha]

# Lista de produtos disponíveis
produtos_disponiveis = listar_produtos_disponiveis()

if not produtos_disponiveis:
    st.error("Nenhum produto encontrado na pasta 'dados'. Verifique se os arquivos CSV estão presentes.")
else:
    # Seleção de produtos para visualização
    produtos_selecionados = st.sidebar.multiselect(
        "Selecione os produtos para visualizar:",
        produtos_disponiveis,
        default=produtos_disponiveis[:min(2, len(produtos_disponiveis))]
    )
    
    # Informações sobre os modelos
    st.sidebar.markdown("## 🧠 Modelos de Previsão")
    st.sidebar.markdown("""
    O sistema testa automaticamente diferentes modelos e seleciona o mais preciso para cada produto:
    
    - **ARIMA**: Modelo autorregressivo integrado de médias móveis
    - **SARIMA**: ARIMA com componente sazonal
    - **Prophet**: Modelo de decomposição de séries temporais do Facebook
    """)
    
    st.sidebar.caption("A seleção é baseada no erro quadrático médio (RMSE) em dados de teste.")
    
    # Área principal
    if not produtos_selecionados:
        st.warning("Por favor, selecione pelo menos um produto para visualizar.")
    else:
        # Criar abas para cada produto selecionado
        tabs = st.tabs([f"{obter_icone_produto(produto)} {produto.capitalize()}" for produto in produtos_selecionados])
        
        # Para cada produto, criar uma aba com análise e previsão
        for i, produto in enumerate(produtos_selecionados):
            with tabs[i]:
                # Carregar dados
                df = carregar_dados(produto)
                
                if df is None or len(df) < 12:
                    st.error(f"Dados insuficientes para {produto}. É necessário pelo menos 12 meses de histórico.")
                    continue
                
                # Extrair série temporal
                serie = df['preco_deflacionado']
                
                # Obter unidade do produto
                unidade = obter_unidade_produto(produto)
                
                # Avaliar modelos e selecionar o melhor
                with st.spinner(f"Analisando dados e selecionando melhor modelo para {produto}..."):
                    modelo, nome_modelo, metricas = avaliar_modelos(serie, horizonte)
                
                # Fazer previsão
                datas_futuras, media, intervalo = fazer_previsao(serie, modelo, nome_modelo, horizonte)
                
                # Mostrar métricas do modelo
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"<div class='metric-card'><h3>📊 Modelo Selecionado</h3><p>{nome_modelo}</p></div>", unsafe_allow_html=True)
                
                with col2:
                    if nome_modelo in metricas and 'MAE' in metricas[nome_modelo]:
                        mae = metricas[nome_modelo]['MAE']
                        st.markdown(f"<div class='metric-card'><h3>📉 Erro Médio Absoluto</h3><p>R$ {mae:.2f}</p></div>", unsafe_allow_html=True)
                
                with col3:
                    if nome_modelo in metricas and 'RMSE' in metricas[nome_modelo]:
                        rmse = metricas[nome_modelo]['RMSE']
                        st.markdown(f"<div class='metric-card'><h3>🎯 Erro Quadrático Médio</h3><p>R$ {rmse:.2f}</p></div>", unsafe_allow_html=True)
                
                # Criar e mostrar gráfico
                fig = criar_grafico_previsao(serie, datas_futuras, media, intervalo, produto, unidade, nome_modelo)
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar tabela de previsão
                if media is not None and datas_futuras is not None:
                    st.markdown("### 📋 Tabela de Previsão")
                    
                    # Criar DataFrame com previsões
                    df_previsao = pd.DataFrame({
                        'Data': datas_futuras,
                        'Previsão (R$)': media.values,
                        'Intervalo Inferior (R$)': intervalo.iloc[:, 0].values,
                        'Intervalo Superior (R$)': intervalo.iloc[:, 1].values
                    })
                    
                    # Formatar datas
                    df_previsao['Data'] = df_previsao['Data'].dt.strftime('%b/%Y')
                    
                    # Mostrar tabela
                    st.dataframe(df_previsao, use_container_width=True)
                    
                    # Opção para download
                    csv = df_previsao.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Baixar previsão de {produto}",
                        data=csv,
                        file_name=f'previsao_{produto}_{horizonte}_meses.csv',
                        mime='text/csv',
                    )
                
                # Mostrar estatísticas descritivas
                st.markdown("### 📊 Estatísticas Históricas")
                
                # Estatísticas anuais
                df_anual = df.copy()
                df_anual['Ano'] = df_anual.index.year
                stats_anuais = df_anual.groupby('Ano')['preco_deflacionado'].agg(['mean', 'min', 'max'])
                stats_anuais.columns = ['Média', 'Mínimo', 'Máximo']
                
                st.markdown(f"#### Preços Anuais de {produto.capitalize()} (R$/{unidade})")
                st.dataframe(stats_anuais.style.format("{:.2f}"), use_container_width=True)

# ======= ASSINATURA FINAL =======
st.markdown("---")
st.markdown("""
📊 Este painel foi desenvolvido para análise e previsão de preços agropecuários.

📅 Última atualização: Maio/2025

🔎 Modelos utilizados: ARIMA, SARIMA e Prophet com seleção automática do mais preciso para cada produto.

💬 Para mais informações, entre em contato.
""")
