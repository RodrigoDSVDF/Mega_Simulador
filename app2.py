# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import altair as alt
import itertools
import warnings
import time
from collections import Counter
from typing import List, Tuple, Any, Dict
from fpdf import FPDF
# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================

warnings.filterwarnings("ignore")
st.set_page_config(
    layout="wide", 
    page_title="Análise Mega-Sena AI", 
    page_icon="🎲",
    initial_sidebar_state="collapsed"
)

# Constantes Globais
COLUNAS_BOLAS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
ALL_NUMBERS = list(range(1, 61))

# =============================================================================
# 0. DESIGN SYSTEM & CSS
# =============================================================================

def inject_custom_css():
    """Injeta CSS para remover a sidebar, criar botões estilizados e estilizar login."""
    st.markdown(
        f"""
        <style>
        /* 1. REMOVER SIDEBAR E ELEMENTOS PADRÃO */
        section[data-testid="stSidebar"] {{ display: none !important; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        
        /* 2. ESTILO GERAL (FUNDO E TEXTO) */
        .stApp {{
            background-color: #0E1117;
            color: #E0E0E0;
        }}
        
        /* LOGIN CONTAINER (PREMIUM) */
        .premium-gate {{
            background: #1F2937;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            border: 2px solid #00C896; /* Borda Verde Neon */
            margin-top: 20px;
            text-align: center;
        }}
        
        /* INPUT FIELDS (LOGIN) */
        .stTextInput > div > div > input {{
            background-color: #111827;
            color: #f8fafc;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 10px 15px;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: #00C896;
            box-shadow: 0 0 10px rgba(0, 200, 150, 0.2);
        }}
        
        h1, h2, h3 {{
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 600;
        }}
        
        h1 {{ color: #00C896; border-bottom: 2px solid #00C896; padding-bottom: 10px; }}
        h2 {{ color: #00C896; margin-top: 30px; border-left: 4px solid #00C896; padding-left: 10px; }}
        h3 {{ color: #E0E0E0; font-size: 1.2rem; margin-top: 20px; }}
        p, label {{ color: #E0E0E0; }}
        
        /* 3. BOTÕES DE NAVEGAÇÃO */
        div.stButton > button {{
            background-color: #1F2937 !important;
            color: #9CA3AF !important;
            border: 1px solid #374151 !important;
            border-radius: 12px !important;
            padding: 10px 10px !important;
            transition: all 0.3s ease !important;
            font-weight: 500 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
            width: 100% !important;
            font-size: 0.9rem !important;
        }}
        
        div.stButton > button:hover {{
            background-color: #374151 !important;
            border-color: #00C896 !important;
            color: #00C896 !important;
            transform: translateY(-2px) !important;
        }}
        
        div.stButton > button[kind="primary"] {{
            background: linear-gradient(145deg, #1F2937, #111827) !important;
            border: 2px solid #00C896 !important;
            color: #00C896 !important;
            box-shadow: 0 0 15px rgba(0, 200, 150, 0.5) !important;
            font-weight: 700 !important;
        }}
        
        /* 5. CARDS DE MÉTRICAS */
        [data-testid="stMetric"] {{
            background-color: #1F2937;
            border-radius: 12px;
            padding: 15px;
            border: 1px solid #374151;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: #9CA3AF !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: #00C896 !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }}
        
        /* 6. TABELAS */
        .dataframe {{
            border: 1px solid #374151;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .dataframe th {{
            background-color: #1F2937 !important;
            color: #00C896 !important;
            font-weight: 600 !important;
        }}
        
        .dataframe td {{
            background-color: #111827 !important;
            color: #E0E0E0 !important;
        }}
        
        /* 7. CHECKBOX & SLIDER */
        .stCheckbox > label {{ color: #E0E0E0; }}
        .stSlider > div > div {{ background-color: #00C896; }}
        
        /* 8. EXPANDER */
        .streamlit-expanderHeader {{
            background-color: #1F2937;
            border: 1px solid #374151;
            border-radius: 8px;
            color: #E0E0E0;
        }}
        
        /* 9. CONTAINER DE JOGOS */
        .jogo-card {{
            background: linear-gradient(145deg, #1F2937, #111827);
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }}
        
        .jogo-card:hover {{
            border-color: #00C896;
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 200, 150, 0.1);
        }}
        
        .numero-bola {{
            display: inline-block;
            width: 40px;
            height: 40px;
            line-height: 40px;
            border-radius: 50%;
            background: linear-gradient(145deg, #00C896, #008B6A);
            color: white;
            text-align: center;
            font-weight: bold;
            margin: 0 5px;
            box-shadow: 0 4px 6px -1px rgba(0, 200, 150, 0.3);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# 1. CARREGAR DADOS
# =============================================================================

def carregar_dados_caixa():
    """
    Carrega dados históricos da Mega-Sena.
    Retorna um DataFrame ou None em caso de erro.
    """
    try:
        # Tenta carregar do CSV local primeiro
        csv_path = "Dados_mega_sena.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding='utf-8', sep=';')
        else:
            # Fallback para dados online da Caixa
            url = "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Converte para DataFrame
            linhas = []
            for item in data:
                linha = {
                    'Concurso': item.get('numero'),
                    'Data': item.get('dataApuracao'),
                    'B1': item.get('dezenasSorteadasOrdemSorteio')[0],
                    'B2': item.get('dezenasSorteadasOrdemSorteio')[1],
                    'B3': item.get('dezenasSorteadasOrdemSorteio')[2],
                    'B4': item.get('dezenasSorteadasOrdemSorteio')[3],
                    'B5': item.get('dezenasSorteadasOrdemSorteio')[4],
                    'B6': item.get('dezenasSorteadasOrdemSorteio')[5],
                }
                linhas.append(linha)
            df = pd.DataFrame(linhas)
        
        # Ordena por concurso
        if 'Concurso' in df.columns:
            df = df.sort_values('Concurso', ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def validar_dados(df):
    """Valida se o DataFrame contém dados essenciais."""
    if df is None or df.empty:
        return False
    colunas_necessarias = ['Concurso', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6']
    return all(col in df.columns for col in colunas_necessarias)

# =============================================================================
# 2. COMPONENTES DE INTERFACE
# =============================================================================

def draw_navigation():
    """Renderiza a barra de navegação horizontal."""
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1.5, 1, 1, 1, 1, 1, 1, 1.5])
    
    pages = [
        ("Visão Geral", "📊"),
        ("Frequência", "📈"),
        ("Pares/Impares", "🔢"),
        ("Combinações", "🔄"),
        ("Quentes/Frios", "🔥❄️"),
        ("∑ Somas", "🧮"),
        ("Previsões AI", "🤖")
    ]
    
    with col2:
        if st.button(f"{pages[0][1]} {pages[0][0]}", use_container_width=True):
            st.session_state['current_page'] = pages[0][0]
    with col3:
        if st.button(f"{pages[1][1]} {pages[1][0]}", use_container_width=True):
            st.session_state['current_page'] = pages[1][0]
    with col4:
        if st.button(f"{pages[2][1]} {pages[2][0]}", use_container_width=True):
            st.session_state['current_page'] = pages[2][0]
    with col5:
        if st.button(f"{pages[3][1]} {pages[3][0]}", use_container_width=True):
            st.session_state['current_page'] = pages[3][0]
    with col6:
        if st.button(f"{pages[4][1]} {pages[4][0]}", use_container_width=True):
            st.session_state['current_page'] = pages[4][0]
    with col7:
        if st.button(f"{pages[5][1]} {pages[5][0]}", use_container_width=True):
            st.session_state['current_page'] = pages[5][0]
    with col8:
        if st.button(f"{pages[6][1]} {pages[6][0]}", use_container_width=True):
            st.session_state['current_page'] = pages[6][0]
    
    # Inicializa a página atual se não existir
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Visão Geral"
    
    st.markdown("---")

# =============================================================================
# 3. ANÁLISES ESTATÍSTICAS
# =============================================================================

def analise_frequencia(df, top_n=10):
    """Retorna as N bolas mais frequentes e menos frequentes."""
    todas_bolas = pd.concat([df[f'B{i}'] for i in range(1, 7)])
    freq = todas_bolas.value_counts().sort_values(ascending=False)
    
    mais_frequentes = freq.head(top_n).index.tolist()
    menos_frequentes = freq.tail(top_n).index.tolist()
    
    return mais_frequentes, menos_frequentes, freq

def analise_pares_impares(df):
    """Calcula a distribuição de pares e ímpares por sorteio."""
    df_calc = df.copy()
    for i in range(1, 7):
        df_calc[f'B{i}_par'] = df_calc[f'B{i}'] % 2 == 0
    
    pares_por_sorteio = df_calc[[f'B{i}_par' for i in range(1, 7)]].sum(axis=1)
    distribuicao = pares_por_sorteio.value_counts().sort_index()
    
    return distribuicao

def analise_combinacoes(df, max_comb=2):
    """Analisa combinações frequentes de números."""
    todas_combinacoes = []
    for _, row in df.iterrows():
        numeros = sorted([row[f'B{i}'] for i in range(1, 7)])
        comb = list(itertools.combinations(numeros, max_comb))
        todas_combinacoes.extend(comb)
    
    freq_combinacoes = Counter(todas_combinacoes)
    mais_comuns = freq_combinacoes.most_common(20)
    
    return mais_comuns

def analise_quentes_frios(df, window=20):
    """Identifica números quentes (recentes) e frios (ausentes)."""
    if len(df) < window:
        window = len(df)
    
    df_recente = df.head(window)
    df_antigo = df.tail(len(df) - window) if len(df) > window else df
    
    todas_recentes = pd.concat([df_recente[f'B{i}'] for i in range(1, 7)])
    todas_antigas = pd.concat([df_antigo[f'B{i}'] for i in range(1, 7)])
    
    quentes = todas_recentes.value_counts().head(10).index.tolist()
    frios = todas_antigas.value_counts().tail(10).index.tolist()
    
    return quentes, frios

def analise_somas(df):
    """Analisa a distribuição das somas dos números por sorteio."""
    df['Soma'] = sum(df[f'B{i}'] for i in range(1, 7))
    return df['Soma'].describe()

# =============================================================================
# 4. MODELO PREDITIVO
# =============================================================================

def preparar_dados_timeseries(df, lookback=10):
    """Prepara dados para modelagem de séries temporais."""
    X, y = [], []
    
    # Concatena todas as bolas de cada sorteio
    sorteios = []
    for _, row in df.iterrows():
        sorteios.append(sorted([int(row[f'B{i}']) for i in range(1, 7)]))
    
    # Cria sequências
    for i in range(lookback, len(sorteios)):
        X.append(np.array(sorteios[i-lookback:i]).flatten())
        y.append(sorteios[i])
    
    return np.array(X), np.array(y)

def treinar_modelo(df, lookback=10):
    """Treina um modelo de regressão logística para prever cada número."""
    try:
        X, y = preparar_dados_timeseries(df, lookback)
        
        if len(X) == 0:
            return None, None
        
        # Modelo para cada posição (0-5)
        modelos = []
        scalers = []
        
        for pos in range(6):
            # Extrai labels para esta posição
            y_pos = np.array([sorteio[pos] for sorteio in y])
            
            # Converte para problema binário por número (1-60)
            y_bin = np.zeros((len(y_pos), 60))
            for i, num in enumerate(y_pos):
                y_bin[i, num-1] = 1
            
            # Treina classificador
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Usa regressão logística com calibração
            base_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            model = CalibratedClassifierCV(base_model, cv=TimeSeriesSplit(n_splits=3))
            model.fit(X_scaled, y_bin)
            
            modelos.append(model)
            scalers.append(scaler)
        
        return modelos, scalers
    except Exception as e:
        st.error(f"Erro no treinamento: {str(e)}")
        return None, None

def prever_probab(modelos, scalers, df, lookback=10, top_n=15):
    """Faz previsões usando o modelo treinado."""
    if modelos is None or scalers is None:
        return None
    
    try:
        # Prepara os últimos dados
        X, _ = preparar_dados_timeseries(df, lookback)
        if len(X) == 0:
            return None
        
        ultimo_x = X[-1].reshape(1, -1)
        
        # Previsões para cada posição
        preds_todas_pos = []
        
        for pos in range(6):
            scaler = scalers[pos]
            model = modelos[pos]
            
            X_scaled = scaler.transform(ultimo_x)
            probas = model.predict_proba(X_scaled)
            
            # Combina probabilidades de todas as classes
            prob_por_numero = []
            for class_idx in range(60):
                # Média das probabilidades entre os classificadores calibrados
                prob = np.mean([proba[0][class_idx] for proba in probas])
                prob_por_numero.append((class_idx + 1, prob))
            
            preds_todas_pos.append(prob_por_numero)
        
        # Combina probabilidades de todas as posições
        prob_agregada = np.zeros(60)
        for pos_preds in preds_todas_pos:
            for num, prob in pos_preds:
                prob_agregada[num-1] += prob
        
        # Normaliza
        prob_agregada = prob_agregada / prob_agregada.sum()
        
        # Top N números
        top_indices = np.argsort(prob_agregada)[-top_n:][::-1]
        top_preds = [(idx+1, prob_agregada[idx]) for idx in top_indices]
        
        return top_preds
    except Exception as e:
        st.error(f"Erro na previsão: {str(e)}")
        return None

# =============================================================================
# 5. GERAÇÃO DE JOGOS
# =============================================================================

def gerar_combinacoes(preds, num_jogos=10, max_repeticao=2):
    """Gera combinações baseadas nas probabilidades previstas."""
    numeros, probs = zip(*preds)
    probs = np.array(probs)
    probs = probs / probs.sum()  # Normaliza
    
    jogos = []
    while len(jogos) < num_jogos:
        # Amostra números baseados nas probabilidades
        jogo = np.random.choice(
            numeros,
            size=6,
            replace=False,
            p=probs
        )
        jogo = sorted(jogo)
        
        # Verifica critérios básicos
        soma = sum(jogo)
        pares = sum(1 for x in jogo if x % 2 == 0)
        
        # Critérios de validação
        if (100 <= soma <= 200 and 
            2 <= pares <= 4 and 
            jogo not in jogos):
            jogos.append(jogo)
    
    return jogos

def gerar_pdf_bytes(jogos):
    """Gera bytes de PDF com os jogos."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Palpites Mega-Sena - Gerados por IA", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "", 12)
    for i, jogo in enumerate(jogos, 1):
        pdf.cell(0, 10, f"Jogo {i}: {' - '.join(map(str, jogo))}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 10, "Lembre-se: trata-se apenas de uma análise estatística. Jogue com responsabilidade.")
    
    return pdf.output(dest='S').encode('latin1')

# =============================================================================
# 6. PÁGINAS DE VISUALIZAÇÃO
# =============================================================================

def page_visao_geral(df):
    """Página principal com visão geral."""
    st.header("📊 Visão Geral dos Dados")
    
    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Concursos", len(df))
    with col2:
        st.metric("Concurso Mais Recente", df['Concurso'].iloc[0])
    with col3:
        data_mais_recente = pd.to_datetime(df['Data'].iloc[0], dayfirst=True, errors='coerce')
        if pd.notna(data_mais_recente):
            st.metric("Data Mais Recente", data_mais_recente.strftime("%d/%m/%Y"))
    with col4:
        st.metric("Período Abrangido", f"{df['Concurso'].min()} - {df['Concurso'].max()}")
    
    # Tabela com últimos resultados
    st.subheader("Últimos 10 Resultados")
    df_display = df.head(10).copy()
    df_display = df_display[['Concurso', 'Data', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6']]
    st.dataframe(df_display, use_container_width=True)
    
    # Estatísticas descritivas
    st.subheader("📈 Estatísticas por Posição")
    stats_data = []
    for i in range(1, 7):
        bola_col = f'B{i}'
        stats_data.append({
            'Posição': f'Bola {i}',
            'Mínimo': int(df[bola_col].min()),
            'Máximo': int(df[bola_col].max()),
            'Média': f"{df[bola_col].mean():.1f}",
            'Mediana': int(df[bola_col].median()),
            'Moda': int(df[bola_col].mode().iloc[0])
        })
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)

def page_frequencia(df):
    """Página de análise de frequência."""
    st.header("📈 Frequência dos Números")
    
    mais_frequentes, menos_frequentes, freq = analise_frequencia(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Top 10 Mais Frequentes")
        df_mais = pd.DataFrame({
            'Número': mais_frequentes,
            'Frequência': [freq[num] for num in mais_frequentes]
        })
        st.dataframe(df_mais, use_container_width=True)
        
        # Gráfico de barras
        chart_data = pd.DataFrame({
            'Número': [str(x) for x in mais_frequentes],
            'Frequência': [freq[num] for num in mais_frequentes]
        })
        bar_chart = alt.Chart(chart_data).mark_bar(color='#00C896').encode(
            x=alt.X('Número:O', sort='-y'),
            y='Frequência:Q',
            tooltip=['Número', 'Frequência']
        ).properties(height=300)
        st.altair_chart(bar_chart, use_container_width=True)
    
    with col2:
        st.subheader("❄️ Top 10 Menos Frequentes")
        df_menos = pd.DataFrame({
            'Número': menos_frequentes,
            'Frequência': [freq[num] for num in menos_frequentes]
        })
        st.dataframe(df_menos, use_container_width=True)
        
        # Gráfico de barras
        chart_data = pd.DataFrame({
            'Número': [str(x) for x in menos_frequentes],
            'Frequência': [freq[num] for num in menos_frequentes]
        })
        bar_chart = alt.Chart(chart_data).mark_bar(color='#9CA3AF').encode(
            x=alt.X('Número:O', sort='y'),
            y='Frequência:Q',
            tooltip=['Número', 'Frequência']
        ).properties(height=300)
        st.altair_chart(bar_chart, use_container_width=True)
    
    # Frequência completa
    st.subheader("📊 Frequência Completa (1-60)")
    freq_completa = pd.DataFrame({
        'Número': list(range(1, 61)),
        'Frequência': [freq.get(num, 0) for num in range(1, 61)]
    })
    
    # Heatmap visual
    heatmap_chart = alt.Chart(freq_completa).mark_rect().encode(
        x=alt.X('Número:O', title='Número'),
        color=alt.Color('Frequência:Q', scale=alt.Scale(scheme='viridis'))
    ).properties(height=100)
    st.altair_chart(heatmap_chart, use_container_width=True)

def page_pares_impares(df):
    """Página de análise de pares e ímpares."""
    st.header("🔢 Distribuição Pares/Ímpares")
    
    distribuicao = analise_pares_impares(df)
    
    # Estatísticas
    total_sorteios = len(df)
    st.write(f"**Total de sorteios analisados:** {total_sorteios}")
    
    # Tabela de distribuição
    dist_df = pd.DataFrame({
        'Pares no Sorteio': distribuicao.index,
        'Quantidade de Sorteios': distribuicao.values,
        'Percentual': (distribuicao.values / total_sorteios * 100).round(2)
    })
    st.dataframe(dist_df, use_container_width=True)
    
    # Gráfico de barras
    chart_data = pd.DataFrame({
        'Pares': [str(x) for x in distribuicao.index],
        'Sorteios': distribuicao.values
    })
    
    bars = alt.Chart(chart_data).mark_bar(color='#00C896').encode(
        x=alt.X('Pares:O', title='Quantidade de Pares no Sorteio'),
        y=alt.Y('Sorteios:Q', title='Número de Sorteios'),
        tooltip=['Pares', 'Sorteios']
    ).properties(height=400)
    
    st.altair_chart(bars, use_container_width=True)
    
    # Insights
    st.subheader("💡 Insights")
    moda_pares = distribuicao.idxmax()
    percent_moda = (distribuicao.max() / total_sorteios * 100).round(2)
    
    st.info(f"""
    A configuração mais comum é ter **{moda_pares} números pares** no sorteio, 
    ocorrendo em **{percent_moda}%** dos concursos analisados.
    
    Em geral, a maioria dos sorteios tem entre 2 e 4 números pares.
    """)

def page_combinacoes(df):
    """Página de análise de combinações."""
    st.header("🔄 Combinações Frequentes")
    
    st.write("Analisando combinações de 2 números que mais aparecem juntos:")
    
    comb_mais_comuns = analise_combinacoes(df, max_comb=2)
    
    # Tabela de combinações
    comb_data = []
    for comb, freq in comb_mais_comuns:
        comb_data.append({
            'Número 1': comb[0],
            'Número 2': comb[1],
            'Frequência Conjunta': freq
        })
    
    comb_df = pd.DataFrame(comb_data)
    st.dataframe(comb_df, use_container_width=True)
    
    # Gráfico de rede (simplificado)
    st.subheader("🔗 Mapa de Conexões")
    
    # Preparar dados para o gráfico
    connections = []
    for comb, freq in comb_mais_comuns[:15]:  # Limita para visualização
        connections.append({
            'source': comb[0],
            'target': comb[1],
            'value': freq
        })
    
    # Exibir como tabela expandida para melhor visualização
    expander = st.expander("Ver todas as combinações (até 50)")
    with expander:
        comb_data_all = []
        for comb, freq in comb_mais_comuns[:50]:
            comb_data_all.append({
                'Combinação': f"{comb[0]} - {comb[1]}",
                'Frequência': freq
            })
        comb_df_all = pd.DataFrame(comb_data_all)
        st.dataframe(comb_df_all, use_container_width=True)

def page_quentes(df):
    """Página de análise de números quentes e frios."""
    st.header("🔥❄️ Números Quentes e Frios")
    
    window = st.slider(
        "Período para análise (últimos N sorteios):",
        min_value=10,
        max_value=100,
        value=30,
        help="Define quantos sorteios recentes considerar como 'quentes'"
    )
    
    quentes, frios = analise_quentes_frios(df, window)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🔥 Quentes (últimos {window} sorteios)")
        df_quentes = pd.DataFrame({'Número': quentes})
        st.dataframe(df_quentes, use_container_width=True)
        
        # Display como bolas
        st.write("Visualização:")
        html_quentes = "<div style='margin: 10px 0;'>"
        for num in quentes:
            html_quentes += f"<span class='numero-bola'>{num}</span>"
        html_quentes += "</div>"
        st.markdown(html_quentes, unsafe_allow_html=True)
    
    with col2:
        st.subheader(f"❄️ Frios (excluindo últimos {window} sorteios)")
        df_frios = pd.DataFrame({'Número': frios})
        st.dataframe(df_frios, use_container_width=True)
        
        # Display como bolas
        st.write("Visualização:")
        html_frios = "<div style='margin: 10px 0;'>"
        for num in frios:
            html_frios += f"<span class='numero-bola' style='background: linear-gradient(145deg, #9CA3AF, #6B7280);'>{num}</span>"
        html_frios += "</div>"
        st.markdown(html_frios, unsafe_allow_html=True)
    
    # Análise temporal
    st.subheader("📈 Evolução Temporal")
    
    # Selecionar números para acompanhar
    numeros_selecionados = st.multiselect(
        "Selecione números para acompanhar:",
        options=list(range(1, 61)),
        default=quentes[:3] + frios[:2]
    )
    
    if numeros_selecionados:
        # Calcular frequência acumulada
        df_sorted = df.sort_values('Concurso')
        freq_acumulada = {num: [] for num in numeros_selecionados}
        concursos = []
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            concursos.append(row['Concurso'])
            sorteio_nums = [row[f'B{i}'] for i in range(1, 7)]
            for num in numeros_selecionados:
                freq = sum(1 for x in sorteio_nums if x == num)
                if i == 1:
                    freq_acumulada[num].append(freq)
                else:
                    freq_acumulada[num].append(freq_acumulada[num][-1] + freq)
        
        # Preparar dados para o gráfico
        chart_data = []
        for num in numeros_selecionados:
            for concurso, freq in zip(concursos, freq_acumulada[num]):
                chart_data.append({
                    'Concurso': concurso,
                    'Número': f'Número {num}',
                    'Frequência Acumulada': freq
                })
        
        chart_df = pd.DataFrame(chart_data)
        
        line_chart = alt.Chart(chart_df).mark_line().encode(
            x='Concurso:O',
            y='Frequência Acumulada:Q',
            color='Número:N',
            tooltip=['Concurso', 'Número', 'Frequência Acumulada']
        ).properties(height=400)
        
        st.altair_chart(line_chart, use_container_width=True)

def page_somas(df):
    """Página de análise de somas."""
    st.header("🧮 Análise das Somas")
    
    # Calcular somas
    df['Soma'] = sum(df[f'B{i}'] for i in range(1, 7))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Estatísticas Descritivas")
        stats = df['Soma'].describe()
        stats_df = pd.DataFrame({
            'Estatística': ['Mínimo', 'Máximo', 'Média', 'Mediana', 'Desvio Padrão'],
            'Valor': [
                int(stats['min']),
                int(stats['max']),
                f"{stats['mean']:.1f}",
                int(stats['50%']),
                f"{stats['std']:.1f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.subheader("Distribuição mais Comum")
        freq_somas = df['Soma'].value_counts().head(10)
        freq_df = pd.DataFrame({
            'Soma': freq_somas.index,
            'Frequência': freq_somas.values
        })
        st.dataframe(freq_df, use_container_width=True)
    
    # Histograma
    st.subheader("📊 Distribuição de Frequência")
    
    hist_chart = alt.Chart(df).mark_bar(color='#00C896').encode(
        alt.X('Soma:Q', bin=alt.Bin(maxbins=30), title='Soma dos Números'),
        alt.Y('count()', title='Frequência'),
        tooltip=['count()']
    ).properties(height=400)
    
    st.altair_chart(hist_chart, use_container_width=True)
    
    # Somas por período
    st.subheader("📈 Evolução Temporal das Somas")
    
    df_sorted = df.sort_values('Concurso')
    line_chart = alt.Chart(df_sorted).mark_line(color='#00C896').encode(
        x=alt.X('Concurso:O', title='Concurso'),
        y=alt.Y('Soma:Q', title='Soma'),
        tooltip=['Concurso', 'Soma']
    ).properties(height=400)
    
    st.altair_chart(line_chart, use_container_width=True)
    
    # Insights
    st.subheader("💡 Insights")
    
    media_soma = df['Soma'].mean()
    mediana_soma = df['Soma'].median()
    
    st.info(f"""
    **Características das Somas:**
    - **Média:** {media_soma:.1f} pontos
    - **Mediana:** {mediana_soma:.0f} pontos
    - **Faixa típica:** A maioria das somas está entre {int(media_soma - 20)} e {int(media_soma + 20)}
    - **Distribuição:** Normalmente seguem uma distribuição aproximadamente normal
    
    **Recomendação:** Ao escolher números, tente somas próximas da média histórica ({media_soma:.0f} ± 15 pontos).
    """)

def page_ai(df):
    """Página de previsões com IA."""
    st.header("🤖 Previsões com Inteligência Artificial")
    
    # Inicializar estado da sessão
    if 'jogos_gerados' not in st.session_state:
        st.session_state['jogos_gerados'] = False
    if 'email_enviado' not in st.session_state:
        st.session_state['email_enviado'] = False
    
    st.write("""
    Esta seção utiliza aprendizado de máquina para analisar padrões históricos 
    e sugerir combinações com base em probabilidades calculadas.
    
    **Como funciona:**
    1. O modelo analisa sequências temporais dos sorteios
    2. Calcula probabilidades para cada número (1-60)
    3. Gera combinações otimizadas estatisticamente
    """)
    
    # Aceite de termos
    aceite = st.checkbox(
        "✅ Li e aceito os termos: Entendo que são apenas sugestões estatísticas e não garantia de acertos.",
        value=False
    )
    
    if aceite:
        # Verificar se já gerou jogos ou se é a primeira vez
        if not st.session_state['jogos_gerados']:
            # PRIMEIRA VEZ: Gerar diretamente sem pedir email
            st.success("🎉 **Primeira geração liberada!** Você pode gerar jogos uma vez sem necessidade de email.")
            
            # Configurações do modelo
            with st.expander("⚙️ Configurações Avançadas", expanded=False):
                lookback = st.slider("Período de análise (lookback):", 5, 20, 10)
                num_jogos = st.slider("Quantidade de jogos:", 5, 20, 10)
                top_n = st.slider("Top N números:", 15, 40, 25)
            
            # Botão para gerar
            if st.button("🚀 GERAR JOGOS COM IA", type="primary", use_container_width=True):
                with st.spinner("Analisando padrões históricos e calculando probabilidades..."):
                    time.sleep(1)
                    
                    # Treinar modelo
                    modelos, scalers = treinar_modelo(df, lookback)
                    
                    if modelos is not None:
                        # Fazer previsões
                        preds = prever_probab(modelos, scalers, df, lookback, top_n)
                        
                        if preds is not None:
                            # Gerar combinações
                            combs = gerar_combinacoes(preds, num_jogos)
                            
                            # Marcar que já gerou jogos
                            st.session_state['jogos_gerados'] = True
                            
                            # Exibir resultados
                            st.success(f"✅ {num_jogos} jogos gerados com sucesso!")
                            
                            # Mostrar probabilidades
                            st.subheader("🎯 Probabilidades dos Números")
                            df_probs = pd.DataFrame(preds, columns=['Número', 'Probabilidade'])
                            df_probs['Probabilidade'] = (df_probs['Probabilidade'] * 100).round(2)
                            st.dataframe(df_probs, use_container_width=True)
                            
                            # Mostrar jogos gerados
                            st.subheader("🎰 Jogos Sugeridos")
                            for i, jogo in enumerate(combs, 1):
                                html_jogo = f"""
                                <div class='jogo-card'>
                                    <h4>Jogo {i}:</h4>
                                    <div style='margin: 15px 0;'>
                                """
                                for num in jogo:
                                    html_jogo += f"<span class='numero-bola'>{num}</span>"
                                html_jogo += "</div></div>"
                                st.markdown(html_jogo, unsafe_allow_html=True)
                            
                            # Métricas dos jogos
                            st.subheader("📊 Análise das Combinações")
                            show_all = st.checkbox("Mostrar análise detalhada de todos os números")
                            
                            if show_all:
                                todas_preds = []
                                for pos in range(6):
                                    if modelos[pos] is not None:
                                        # Previsões detalhadas para esta posição
                                        pass
                            
                            # Estatísticas resumidas
                            st.write("**Resumo estatístico dos jogos gerados:**")
                            for i, jogo in enumerate(combs, 1):
                                c = jogo
                                soma = sum(c)
                                pares = sum(1 for x in c if x % 2 == 0)
                                impares = 6 - pares
                                baixos = sum(1 for x in c if x <= 30)
                                altos = 6 - baixos
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Soma", soma)
                                m2.metric("Par/Ímpar", f"{pares}/{impares}")
                                m3.metric("Baixo/Alto", f"{baixos}/{altos}")
                                m4.metric("Média", f"{soma/6:.1f}")
                            
                            st.markdown("---")
                            st.subheader("💾 Salvar Jogos")
                            
                            # Gera PDF
                            pdf_bytes = gerar_pdf_bytes(combs)
                            st.download_button(
                                label="📄 BAIXAR JOGOS EM PDF",
                                data=pdf_bytes,
                                file_name='palpites_megasena_.pdf',
                                mime='application/pdf',
                                type="primary",
                                use_container_width=True
                            )
                    else:
                        st.error("Não foi possível treinar o modelo. Verifique os dados.")
        else:
            # JÁ GEROU JOGOS UMA VEZ: Mostrar formulário de email
            st.warning("⚠️ **Você já usou sua geração gratuita.**")
            
            if not st.session_state['email_enviado']:
                st.markdown("""
                <div class='premium-gate'>
                    <h3>🔓 Desbloqueie Novas Gerações</h3>
                    <p>Para continuar gerando jogos, insira seu email abaixo.</p>
                </div>
                """, unsafe_allow_html=True)
                
                email = st.text_input("📧 Seu melhor e-mail:")
                
                if st.button("🔓 DESBLOQUEAR", type="primary", use_container_width=True):
                    if email and "@" in email and "." in email:
                        # Aqui você pode adicionar lógica para enviar o email
                        # Por enquanto, apenas marcamos como enviado
                        st.session_state['email_enviado'] = True
                        st.success("✅ E-mail registrado! Agora você pode gerar novos jogos.")
                        st.rerun()
                    else:
                        st.error("Por favor, insira um e-mail válido.")
            else:
                # EMAIL JÁ ENVIADO: Liberar para novas gerações
                st.success("✅ **Email verificado!** Você pode gerar novos jogos.")
                
                # Configurações do modelo
                with st.expander("⚙️ Configurações Avançadas", expanded=False):
                    lookback = st.slider("Período de análise (lookback):", 5, 20, 10)
                    num_jogos = st.slider("Quantidade de jogos:", 5, 20, 10)
                    top_n = st.slider("Top N números:", 15, 40, 25)
                
                # Botão para gerar
                if st.button("🚀 GERAR NOVOS JOGOS", type="primary", use_container_width=True):
                    with st.spinner("Analisando padrões históricos e calculando probabilidades..."):
                        time.sleep(1)
                        
                        # Treinar modelo
                        modelos, scalers = treinar_modelo(df, lookback)
                        
                        if modelos is not None:
                            # Fazer previsões
                            preds = prever_probab(modelos, scalers, df, lookback, top_n)
                            
                            if preds is not None:
                                # Gerar combinações
                                combs = gerar_combinacoes(preds, num_jogos)
                                
                                # Exibir resultados
                                st.success(f"✅ {num_jogos} novos jogos gerados com sucesso!")
                                
                                # Mostrar probabilidades
                                st.subheader("🎯 Probabilidades dos Números")
                                df_probs = pd.DataFrame(preds, columns=['Número', 'Probabilidade'])
                                df_probs['Probabilidade'] = (df_probs['Probabilidade'] * 100).round(2)
                                st.dataframe(df_probs, use_container_width=True)
                                
                                # Mostrar jogos gerados
                                st.subheader("🎰 Jogos Sugeridos")
                                for i, jogo in enumerate(combs, 1):
                                    html_jogo = f"""
                                    <div class='jogo-card'>
                                        <h4>Jogo {i}:</h4>
                                        <div style='margin: 15px 0;'>
                                    """
                                    for num in jogo:
                                        html_jogo += f"<span class='numero-bola'>{num}</span>"
                                    html_jogo += "</div></div>"
                                    st.markdown(html_jogo, unsafe_allow_html=True)
                                
                                # Métricas dos jogos
                                st.subheader("📊 Análise das Combinações")
                                show_all = st.checkbox("Mostrar análise detalhada de todos os números")
                                
                                if show_all:
                                    todas_preds = []
                                    for pos in range(6):
                                        if modelos[pos] is not None:
                                            # Previsões detalhadas para esta posição
                                            pass
                                
                                # Estatísticas resumidas
                                st.write("**Resumo estatístico dos jogos gerados:**")
                                for i, jogo in enumerate(combs, 1):
                                    c = jogo
                                    soma = sum(c)
                                    pares = sum(1 for x in c if x % 2 == 0)
                                    impares = 6 - pares
                                    baixos = sum(1 for x in c if x <= 30)
                                    altos = 6 - baixos
                                    m1, m2, m3, m4 = st.columns(4)
                                    m1.metric("Soma", soma)
                                    m2.metric("Par/Ímpar", f"{pares}/{impares}")
                                    m3.metric("Baixo/Alto", f"{baixos}/{altos}")
                                    m4.metric("Média", f"{soma/6:.1f}")
                                
                                st.markdown("---")
                                st.subheader("💾 Salvar Jogos")
                                
                                # Gera PDF
                                pdf_bytes = gerar_pdf_bytes(combs)
                                st.download_button(
                                    label="📄 BAIXAR JOGOS EM PDF",
                                    data=pdf_bytes,
                                    file_name='palpites_megasena_.pdf',
                                    mime='application/pdf',
                                    type="primary",
                                    use_container_width=True
                                )
                        else:
                            st.error("Não foi possível treinar o modelo. Verifique os dados.")
    else:
        st.info("Marque o aceite para habilitar o modelo preditivo.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    inject_custom_css()
    st.title("🎲 Análise Mega-Sena")
    
    # 1. Carregar
    df = carregar_dados_caixa()
    if not validar_dados(df):
        st.error("Erro crítico: Banco de dados indisponível.")
        return
    
    # 2. Navegação
    draw_navigation()
    
    # 3. Roteamento
    page = st.session_state['current_page']
    
    if page == "Visão Geral":
        page_visao_geral(df)
    elif page == "Frequência":
        page_frequencia(df)
    elif page == "Pares/Impares":
        page_pares_impares(df)
    elif page == "Combinações":
        page_combinacoes(df)
    elif page == "Quentes/Frios":
        page_quentes(df)
    elif page == "∑ Somas":
        page_somas(df)
    elif page == "Previsões AI":
        page_ai(df)

if __name__ == "__main__":
    main()
