#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import altair as alt
import itertools
import warnings
from collections import Counter
from typing import List, Tuple, Any, Dict, Optional
from fpdf import FPDF
from datetime import datetime, timedelta
import io

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Desabilitar aviso de InsecureRequest
warnings.filterwarnings('ignore', category=requests.packages.urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================
st.set_page_config(
    layout="wide",
    page_title="Análise Mega-Sena com Simulador de Jogos", 
    page_icon="🎲",
    initial_sidebar_state="collapsed"
)

# Constantes Globais
COLUNAS_BOLAS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
ALL_NUMBERS = list(range(1, 61))
PRIMOS_1_A_60 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

# =============================================================================
# 0. DESIGN SYSTEM & CSS (CORRIGIDO)
# =============================================================================

def inject_custom_css():
    """Injeta CSS customizado com tema escuro moderno e correção de alinhamento."""
    st.markdown(
        """
        <style>
            /* REMOVER ELEMENTOS PADRÃO */
            section[data-testid="stSidebar"] { display: none !important; }
            #MainMenu { visibility: hidden; }
            footer { visibility: hidden; }
            .stDeployButton { display: none; }
            
            /* ESTILO GERAL */
            .stApp {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #f8fafc;
            }
            
            /* TIPOGRAFIA */
            h1, h2, h3, h4 {
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-weight: 600;
                margin-bottom: 1rem;
            }
            
            h1 { 
                color: #00e6b8; 
                border-bottom: 3px solid #00e6b8; 
                padding-bottom: 15px; 
                text-align: center; 
                font-size: 2.5rem; 
            }
            
            h2 { 
                color: #00e6b8; 
                border-left: 4px solid #00e6b8; 
                padding-left: 15px; 
                margin-top: 2rem; 
            }
            
            /* --- CORREÇÃO DO BUG DE ALINHAMENTO AQUI --- */
            h3 { 
                color: #cbd5e1; 
                font-size: 1.3rem; 
                /* Garante altura mínima para 2 linhas, evitando desalinhamento */
                min-height: 3.5rem; 
                display: flex;
                align-items: end; /* Alinha texto embaixo */
                margin-bottom: 10px !important;
            }
            
            p, label { color: #e2e8f0; line-height: 1.6; }

            /* BOTÕES DE NAVEGAÇÃO */
            div.stButton > button {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
                color: #94a3b8 !important;
                border: 2px solid #475569 !important;
                border-radius: 12px !important;
                padding: 12px 8px !important;
                transition: all 0.3s ease !important;
                font-weight: 600 !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
                width: 100% !important;
                font-size: 0.95rem !important;
                margin: 4px 0;
                min-height: 80px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                line-height: 1.3;
            }
            
            div.stButton > button:hover {
                background: linear-gradient(135deg, #334155 0%, #475569 100%) !important;
                border-color: #00e6b8 !important;
                color: #00e6b8 !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(0, 230, 184, 0.3) !important;
            }

            /* BOTÃO SELECIONADO */
            div.stButton > button[kind="primary"],
            div.stButton > button[data-testid="stBaseButton-primary"] {
                background: linear-gradient(135deg, #00e6b8 0%, #00b894 100%) !important;
                border: 2px solid #00e6b8 !important;
                color: #0f172a !important;
                box-shadow: 0 0 25px rgba(0, 230, 184, 0.5) !important;
                transform: translateY(-2px) !important;
                font-weight: 700 !important;
            }

            /* CARDS DE MÉTRICAS */
            [data-testid="stMetric"] {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid #475569;
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
                transition: transform 0.2s ease;
            }
            
            [data-testid="stMetric"]:hover {
                transform: translateY(-3px);
            }
            
            [data-testid="stMetric"] label { 
                color: #94a3b8 !important; 
                font-weight: 600 !important;
            }
            
            [data-testid="stMetric"] div[data-testid="stMetricValue"] { 
                color: #00e6b8 !important; 
                font-weight: 800 !important;
                font-size: 1.4rem !important;
            }

            /* BOLAS DA LOTERIA */
            .lotto-number {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(145deg, #00e6b8, #00b894);
                color: #0f172a;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                font-size: 15px;
                font-weight: 800;
                margin: 3px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                border: 2px solid #ffffff40;
                transition: transform 0.2s ease;
            }
            
            .lotto-number:hover {
                transform: scale(1.1);
            }

            /* TABELAS */
            .custom-table-header {
                display: flex;
                background: linear-gradient(135deg, #00e6b8 0%, #00b894 100%);
                color: #0f172a;
                padding: 15px 12px;
                font-weight: 700;
                border-radius: 12px 12px 0 0;
            }
            
            .custom-table-row {
                display: flex;
                padding: 12px;
                border-bottom: 1px solid #475569;
                color: #e2e8f0;
                align-items: center;
                transition: background-color 0.2s;
            }
            
            .custom-table-row:hover {
                background-color: #334155;
            }

            /* LOADING SPINNER CUSTOM */
            .stSpinner > div {
                border-color: #00e6b8 !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# 1. FUNÇÕES DE DADOS E PDF
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="📡 Sincronizando dados da Caixa...")
def carregar_dados_caixa() -> Optional[pd.DataFrame]:
    """Carrega dados da Mega-Sena com fallback robusto."""
    folder = 'dados_mega_sena'
    os.makedirs(folder, exist_ok=True)
    
    urls = [
        "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena",
        "https://asloterias.com.br/download.php?modalidade=Mega-Sena",
    ]
    
    caminho_arquivo = os.path.join(folder, 'mega_sena.xlsx')
    
    # Tentar múltiplas fontes
    for url in urls:
        try:
            response = requests.get(url, timeout=30, verify=False)
            if response.status_code == 200:
                with open(caminho_arquivo, 'wb') as f:
                    f.write(response.content)
                break
        except Exception:
            continue
    
    # Fallback para dados locais se disponível
    if not os.path.exists(caminho_arquivo):
        st.warning("📡 Usando dados de exemplo. Conectando à internet para dados atualizados...")
        return criar_dados_exemplo()
    
    try:
        return processar_arquivo_excel(caminho_arquivo)
    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {e}")
        return criar_dados_exemplo()

def criar_dados_exemplo() -> pd.DataFrame:
    """Cria dados de exemplo para demonstração."""
    end_date = datetime.now() - timedelta(days=7)
    start_date = end_date - timedelta(days=2000*7)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='W-WED')
    np.random.seed(42)
    
    dados = []
    for i, date in enumerate(dates[::-1], 1):
        numeros = sorted(np.random.choice(ALL_NUMBERS, size=6, replace=False))
        dados.append({
            'Concurso': i,
            'Data': date,
            'B1': numeros[0], 'B2': numeros[1], 'B3': numeros[2],
            'B4': numeros[3], 'B5': numeros[4], 'B6': numeros[5]
        })
    
    return pd.DataFrame(dados)

def processar_arquivo_excel(caminho: str) -> Optional[pd.DataFrame]:
    """Processa o arquivo Excel baixado da Caixa."""
    try:
        df_raw = pd.read_excel(caminho, header=None, nrows=10)
        linha_cabecalho = None
        
        for i in range(len(df_raw)):
            linha = df_raw.iloc[i].astype(str).str.lower().values
            if any('concurso' in str(cell) for cell in linha) and any('bola' in str(cell) for cell in linha):
                linha_cabecalho = i
                break
        
        if linha_cabecalho is not None:
            df = pd.read_excel(caminho, header=linha_cabecalho)
        else:
            df = pd.read_excel(caminho, header=0)
        
        df_clean = mapear_colunas_dataframe(df)
        
        if not validar_dataframe(df_clean):
            return None
            
        return df_clean
        
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
        return None

def mapear_colunas_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia colunas do dataframe para formato padronizado."""
    df_clean = pd.DataFrame()
    df_cols_lower = {str(col).lower().strip(): col for col in df.columns}
    
    mapping = {
        'Concurso': ['concurso', 'número', 'numero', 'n°', 'ndoconcurso'],
        'Data': ['data', 'data sorteio', 'data do sorteio', 'datasorteio'],
        'B1': ['bola 1', 'bola 01', 'bola1', '1ª bola', '1abola'],
        'B2': ['bola 2', 'bola 02', 'bola2', '2ª bola', '2abola'],
        'B3': ['bola 3', 'bola 03', 'bola3', '3ª bola', '3abola'],
        'B4': ['bola 4', 'bola 04', 'bola4', '4ª bola', '4abola'],
        'B5': ['bola 5', 'bola 05', 'bola5', '5ª bola', '5abola'],
        'B6': ['bola 6', 'bola 06', 'bola6', '6ª bola', '6abola']
    }
    
    for target_col, patterns in mapping.items():
        for pattern in patterns:
            if pattern in df_cols_lower:
                df_clean[target_col] = df[df_cols_lower[pattern]]
                break
    
    if len(df_clean.columns) < 4 and len(df.columns) >= 8:
        df_clean['Concurso'] = df.iloc[:, 0]
        df_clean['Data'] = df.iloc[:, 1]
        for i, col in enumerate(COLUNAS_BOLAS, 2):
            if i < len(df.columns):
                df_clean[col] = df.iloc[:, i]
    
    return df_clean

def validar_dataframe(df: pd.DataFrame) -> bool:
    """Valida se o dataframe tem estrutura correta e limpa datas futuras."""
    if df.empty or len(df) < 10:
        return False
    
    colunas_necessarias = ['Concurso', 'Data', 'B1', 'B2']
    if not all(col in df.columns for col in colunas_necessarias):
        return False
    
    try:
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce', dayfirst=True)
        df['Concurso'] = pd.to_numeric(df['Concurso'], errors='coerce')
        for col in COLUNAS_BOLAS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        data_atual = datetime.now()
        df = df[df['Data'] <= data_atual]
        df.dropna(subset=['Concurso', 'Data', 'B1', 'B2'], inplace=True)
        
        df = df[df['Concurso'] > 0]
        for col in COLUNAS_BOLAS:
            if col in df.columns:
                df = df[(df[col] >= 1) & (df[col] <= 60)]
        
        df.sort_values('Data', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return len(df) > 10
        
    except Exception:
        return False

def gerar_pdf_bytes(palpites: List[List[int]], titulo: str = "MEGA-SENA - PALPITES GERADOS") -> bytes:
    """Gera PDF profissional com os palpites."""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, titulo, 0, 1, 'C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, f"Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Courier", 'B', 14)
    
    for i, palpite in enumerate(palpites, 1):
        numeros_formatados = "  ".join([f"{n:02d}" for n in palpite])
        texto = f"JOGO {i:02d}:  {numeros_formatados}"
        pdf.cell(0, 12, texto, 1, 1, 'C')
        pdf.ln(3)
    
    pdf.ln(15)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, "*** Gerado por Sistema de Análise Preditiva - Boa Sorte! ***", 0, 1, 'C')
    pdf.cell(0, 8, "*** Use com responsabilidade - Apostas devem ser conscientes ***", 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# =============================================================================
# 2. FUNÇÕES ESTATÍSTICAS
# =============================================================================

@st.cache_data
def is_primo(n: int) -> bool:
    return n in PRIMOS_1_A_60

@st.cache_data
def get_primos_compostos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df.columns]
    
    df_analise = df[bolas_cols].apply(
        lambda row: pd.Series({
            'Qtd_Primos': sum(is_primo(n) for n in row),
            'Qtd_Compostos': 6 - sum(is_primo(n) for n in row)
        }), axis=1
    )
    
    dist_primos = df_analise['Qtd_Primos'].value_counts(normalize=True).sort_index() * 100
    dist_primos = dist_primos.reset_index()
    dist_primos.columns = ['Qtd_Primos', 'Percentual']
    dist_primos['Label'] = dist_primos['Qtd_Primos'].apply(lambda x: f"{x} Prim{'os' if x != 1 else 'o'} / {6-x} Compost{'os' if 6-x != 1 else 'o'}")
    
    todos_numeros = df[bolas_cols].values.flatten()
    freq_total = Counter(todos_numeros)
    
    primos_sorteados = sum(freq_total[n] for n in PRIMOS_1_A_60 if n in freq_total)
    compostos_sorteados = sum(freq_total[n] for n in ALL_NUMBERS if not is_primo(n) and n != 1)
    
    df_resumo = pd.DataFrame({
        'Categoria': ['Primos', 'Compostos', 'Número 1'],
        'Total_Sorteado': [primos_sorteados, compostos_sorteados, freq_total.get(1, 0)],
        'Percentual': [
            primos_sorteados / len(todos_numeros) * 100,
            compostos_sorteados / len(todos_numeros) * 100,
            freq_total.get(1, 0) / len(todos_numeros) * 100
        ]
    })
    
    df_teorico = pd.DataFrame({
        'Categoria': ['Primos', 'Compostos', 'Número 1'],
        'Qtd_Universo': [len(PRIMOS_1_A_60), 60 - len(PRIMOS_1_A_60) - 1, 1],
        'Probabilidade_Teorica': [
            len(PRIMOS_1_A_60) / 60 * 6,
            (60 - len(PRIMOS_1_A_60) - 1) / 60 * 6,
            1 / 60 * 6
        ]
    })
    
    return dist_primos, df_resumo, df_teorico

@st.cache_data
def get_frequencia(df: pd.DataFrame) -> List[Tuple[int, int]]:
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df.columns]
    todos_numeros = df[bolas_cols].values.flatten()
    frequencia = Counter(todos_numeros)
    
    for num in ALL_NUMBERS:
        frequencia.setdefault(num, 0)
    
    return sorted(frequencia.items(), key=lambda x: (x[1], -x[0]), reverse=True)

@st.cache_data
def get_pares_impares(df: pd.DataFrame) -> pd.Series:
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df.columns]
    qtd_impares = (df[bolas_cols] % 2 == 1).sum(axis=1)
    return qtd_impares.value_counts(normalize=True).sort_index() * 100

@st.cache_data
def get_frequencia_faixas(df: pd.DataFrame) -> pd.Series:
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df.columns]
    todos_numeros = df[bolas_cols].values.flatten()
    
    faixas = pd.cut(todos_numeros, bins=[0, 10, 20, 30, 40, 50, 60], 
                    labels=['01-10', '11-20', '21-30', '31-40', '41-50', '51-60'])
    return faixas.value_counts().sort_index()

@st.cache_data
def get_atrasados(df: pd.DataFrame) -> List[Tuple[int, int]]:
    if df.empty:
        return []
    
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df.columns]
    df_melted = df.melt(id_vars=['Concurso'], value_vars=bolas_cols, value_name='Numero')
    
    ultima_aparicao = df_melted.groupby('Numero')['Concurso'].max()
    ultimo_concurso = df['Concurso'].max()
    
    atrasos = {num: ultimo_concurso - ultima_aparicao.get(num, 0) for num in ALL_NUMBERS}
    return sorted(atrasos.items(), key=lambda x: x[1], reverse=True)

@st.cache_data
def get_quentes_frios(df: pd.DataFrame, window: int = 50) -> Tuple[List, List]:
    if len(df) < window:
        window = len(df)
    
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df.columns]
    ultimos_sorteios = df.tail(window)
    numeros_recentes = ultimos_sorteios[bolas_cols].values.flatten()
    
    freq_recentes = Counter(numeros_recentes)
    for num in ALL_NUMBERS:
        freq_recentes.setdefault(num, 0)
    
    freq_ordenada = freq_recentes.most_common()
    return freq_ordenada[:20], freq_ordenada[-20:][::-1]

@st.cache_data
def get_combinacoes(df: pd.DataFrame) -> Tuple[List, List]:
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df.columns]
    
    duplas = Counter()
    triplas = Counter()
    
    for _, linha in df.iterrows():
        numeros = sorted([n for n in linha[bolas_cols].values if pd.notna(n)])
        
        for dupla in itertools.combinations(numeros, 2):
            duplas[tuple(sorted(dupla))] += 1
            
        for tripla in itertools.combinations(numeros, 3):
            triplas[tuple(sorted(tripla))] += 1
    
    return duplas.most_common(30), triplas.most_common(30)

@st.cache_data
def get_vizinhos(df: pd.DataFrame, numero: int) -> List[Tuple[int, int]]:
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df.columns]
    vizinhos = Counter()
    
    for _, linha in df.iterrows():
        numeros = [n for n in linha[bolas_cols].values if pd.notna(n)]
        if numero in numeros:
            for outro_num in numeros:
                if outro_num != numero:
                    vizinhos[outro_num] += 1
                    
    return vizinhos.most_common(10)

# =============================================================================
# 3. FUNÇÕES DE MACHINE LEARNING
# =============================================================================

def compute_basic_freqs_fast(df_ml: pd.DataFrame, window: Optional[int] = None) -> Dict[int, int]:
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df_ml.columns]
    dados = df_ml.tail(window)[bolas_cols].values.flatten() if window else df_ml[bolas_cols].values.flatten()
    freq = pd.Series(dados).value_counts()
    return {n: freq.get(n, 0) for n in ALL_NUMBERS}

def exponential_moving_freq_fast(df_ml: pd.DataFrame, span: int = 20) -> Dict[int, float]:
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df_ml.columns]
    
    data_list = []
    for _, row in df_ml.iterrows():
        row_dict = {i: 0 for i in ALL_NUMBERS}
        for n in row[bolas_cols].values:
            if pd.notna(n):
                row_dict[int(n)] = 1
        data_list.append(row_dict)
    
    if not data_list:
        return {n: 0.0 for n in ALL_NUMBERS}
        
    df_ohe = pd.DataFrame(data_list)
    ema = df_ohe.ewm(span=span, adjust=False).mean().iloc[-1]
    return {n: float(ema.get(n, 0.0)) for n in ALL_NUMBERS}

def last_appearance_distance_fast(df_ml: pd.DataFrame, max_dist: int = 1000) -> Dict[int, int]:
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df_ml.columns]
    
    melted = []
    for col in bolas_cols:
        for idx, val in df_ml[col].items():
            if pd.notna(val):
                melted.append({'index': idx, 'numero': int(val)})
    
    if not melted:
        return {n: max_dist for n in ALL_NUMBERS}
    
    df_m = pd.DataFrame(melted)
    last_appearance = df_m.groupby('numero')['index'].max()
    current_idx = df_ml.index.max() + 1
    
    return {n: int(current_idx - last_appearance.get(n, -1)) for n in ALL_NUMBERS}

def build_features_table_fast(df_ml: pd.DataFrame) -> pd.DataFrame:
    if len(df_ml) == 0:
        return pd.DataFrame()
    
    freq_all = compute_basic_freqs_fast(df_ml)
    freq_50 = compute_basic_freqs_fast(df_ml, 50)
    freq_10 = compute_basic_freqs_fast(df_ml, 10)
    
    ema_20 = exponential_moving_freq_fast(df_ml, 20)
    ema_50 = exponential_moving_freq_fast(df_ml, 50)
    
    last_dist = last_appearance_distance_fast(df_ml, len(df_ml) + 100)
    
    data = []
    for num in ALL_NUMBERS:
        data.append({
            'numero': num,
            'freq_all': freq_all.get(num, 0),
            'freq_50': freq_50.get(num, 0),
            'freq_10': freq_10.get(num, 0),
            'ema_20': ema_20.get(num, 0.0),
            'ema_50': ema_50.get(num, 0.0),
            'last_dist': last_dist.get(num, len(df_ml)),
            'is_even': num % 2,
            'is_leq30': 1 if num <= 30 else 0,
            'decena': (num - 1) // 10,
            'is_mult_5': 1 if num % 5 == 0 else 0,
            'is_primo': 1 if is_primo(num) else 0
        })
    
    features = pd.DataFrame(data).set_index('numero')
    
    for col in ['freq_all', 'freq_50', 'freq_10']:
        max_val = max(1, features[col].max())
        features[f'{col}_norm'] = features[col] / max_val
    
    features['last_dist_norm'] = features['last_dist'] / max(1, features['last_dist'].max())
    
    return features

def create_training_dataset_fast(df_ml: pd.DataFrame, sample_fraction: float = 0.3) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df_sorted = df_ml.reset_index(drop=True)
    n = len(df_sorted)
    
    start_idx = max(50, int(0.15 * n))
    time_points = list(range(start_idx, n - 1))
    
    if len(time_points) > 100:
        step = max(1, len(time_points) // 100)
        time_points = time_points[::step]
    
    examples, targets = [], []
    bolas_cols = [col for col in COLUNAS_BOLAS if col in df_sorted.columns]
    
    progress_bar = st.progress(0, text="🔄 Processando dados históricos...")
    
    for i, t in enumerate(time_points):
        progress = (i + 1) / len(time_points)
        progress_bar.progress(progress, text=f"📊 Analisando período {t}/{n} ({progress:.1%})")
        
        df_until = df_sorted.iloc[:t + 1]
        features = build_features_table_fast(df_until)
        
        proximo_sorteio = set(df_sorted.loc[t + 1, bolas_cols].tolist())
        
        numeros_amostra = list(ALL_NUMBERS)
        if sample_fraction < 1.0:
            n_amostra = max(15, int(60 * sample_fraction))
            numeros_sorteados = list(proximo_sorteio)
            numeros_nao_sorteados = [x for x in ALL_NUMBERS if x not in proximo_sorteio]
            
            n_restante = max(0, n_amostra - len(numeros_sorteados))
            amostra_nao_sorteados = np.random.choice(numeros_nao_sorteados, n_restante, replace=False).tolist()
            
            numeros_amostra = numeros_sorteados + amostra_nao_sorteados
        
        for num in numeros_amostra:
            examples.append(features.loc[num].values)
            targets.append(1 if num in proximo_sorteio else 0)
    
    progress_bar.empty()
    
    return np.array(examples) if examples else np.empty((0, 0)), np.array(targets, dtype=int), df_sorted

@st.cache_resource(ttl=3600, show_spinner="🧠 Treinando modelo de IA...")
def treinar_modelo_avancado(df: pd.DataFrame, use_sampling: bool = True) -> Tuple[Any, Any, pd.DataFrame]:
    if len(df) < 80:
        raise ValueError("📊 Dados insuficientes para treino (mínimo 80 sorteios)")
    
    sample_frac = 0.4 if use_sampling and len(df) > 500 else 0.8
    
    with st.spinner("🎯 Preparando modelo de machine learning..."):
        X, y, df_processed = create_training_dataset_fast(df, sample_frac)
    
    if len(X) == 0:
        raise ValueError("❌ Erro ao gerar features para treino")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    base_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        C=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    n_splits = max(2, min(5, len(X_scaled) // 1000))
    cv = TimeSeriesSplit(n_splits=n_splits) if len(X_scaled) >= 500 else 3
    
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        cv=cv,
        method='sigmoid',
        n_jobs=-1
    )
    
    calibrated_model.fit(X_scaled, y)
    
    return calibrated_model, scaler, df_processed

def gerar_previsoes_avancadas(df_processed: pd.DataFrame, model: Any, scaler: Any) -> List[Tuple[int, float]]:
    features = build_features_table_fast(df_processed)
    X_scaled = scaler.transform(features.values)
    
    try:
        probabilities = model.predict_proba(X_scaled)[:, 1]
    except NotFittedError:
        st.warning("⚠️ Modelo não treinado corretamente. Usando frequência básica para previsões.")
        freqs = compute_basic_freqs_fast(df_processed)
        total_freq = sum(freqs.values())
        probabilities = np.array([freqs.get(num, 1) / total_freq for num in ALL_NUMBERS])
    except Exception as e:
        st.error(f"Erro ao gerar previsões: {e}")
        return [(num, 1/60) for num in ALL_NUMBERS]
    
    exponents = np.exp(probabilities - np.max(probabilities))
    relative_probs = exponents / exponents.sum()
    
    return sorted([(int(num), float(relative_probs[i])) for i, num in enumerate(features.index)], 
                  key=lambda x: x[1], reverse=True)

def safe_weighted_choice(population: List[int], weights: List[float], k: int) -> List[int]:
    try:
        weights_array = np.maximum(np.array(weights, dtype=float), 0)
        if weights_array.sum() == 0:
            weights_array = np.ones_like(weights_array)
        
        probabilities = weights_array / weights_array.sum()
        indices = np.random.choice(len(population), size=k, replace=False, p=probabilities)
        return [population[i] for i in indices]
    except Exception:
        return list(np.random.choice(population, size=k, replace=False))

@st.cache_data
def gerar_combinacoes_avancadas(predictions: List[Tuple[int, float]], n_combinacoes: int = 8, diversificar: bool = True) -> List[List[int]]:
    numeros, pesos = [p[0] for p in predictions], [p[1] for p in predictions]
    candidatos, pesos_candidatos = numeros[:30], pesos[:30]
    
    combinacoes = set()
    tentativas = 0
    
    while len(combinacoes) < n_combinacoes and tentativas < 500:
        tentativas += 1
        
        combinacao = safe_weighted_choice(candidatos, pesos_candidatos, 6)
        combinacao_tuple = tuple(sorted(combinacao))
        
        if diversificar:
            pares = sum(1 for x in combinacao if x % 2 == 0)
            primos = sum(1 for x in combinacao if is_primo(x))
            soma_total = sum(combinacao)
            
            if pares < 2 or pares > 4:
                continue
            if primos < 1 or primos > 4:
                continue
            if soma_total < 100 or soma_total > 250:
                continue
        
        combinacoes.add(combinacao_tuple)
    
    return [list(comb) for comb in list(combinacoes)[:n_combinacoes]]

# =============================================================================
# 4. INTERFACE E NAVEGAÇÃO
# =============================================================================

def draw_navigation():
    pages = {
        "📊 Visão Geral": "visao_geral",
        "📈 Frequência": "frequencia", 
        "⚖️ Pares/Ímpares": "pares_impares",
        "🔢 Primos/Compostos": "primos_compostos",
        "🤝 Combinações": "combinacoes",
        "🔥 Quentes/Frios": "quentes_frios",
        "➕ ∑ Somas": "somas",
        "🤖 Previsões AI": "previsoes_ai"
    }
    
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "visao_geral"
    
    cols = st.columns(len(pages))
    
    for col, (label, page_key) in zip(cols, pages.items()):
        is_active = st.session_state['current_page'] == page_key
        btn_type = "primary" if is_active else "secondary"
        
        if col.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
            st.session_state['current_page'] = page_key
            st.rerun()
    
    st.markdown("---")

# =============================================================================
# 5. PÁGINAS DE CONTEÚDO
# =============================================================================

def page_visao_geral(df: pd.DataFrame):
    st.header("📊 Visão Geral dos Sorteios")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Sorteios", f"{len(df):,}")
    
    with col2:
        st.metric("Último Concurso", f"{df['Concurso'].iloc[-1]}")
    
    with col3:
        st.metric("Data Mais Recente", df['Data'].iloc[-1].strftime('%d/%m/%Y'))
    
    with col4:
        primeiro_ano = df['Data'].min().year
        ultimo_ano = df['Data'].max().year
        st.metric("Período Abrangido", f"{primeiro_ano}-{ultimo_ano}")
    
    st.divider()
    
    st.subheader("🎯 Últimos 20 Resultados")
    
    df_recent = df[df['Data'] <= datetime.now()].tail(20).sort_values('Data', ascending=False)
    
    header_cols = st.columns([1, 1, 3, 1])
    with header_cols[0]:
        st.markdown(f'<div class="custom-table-header" style="width: 100%; border-radius: 12px 0 0 0; background: linear-gradient(135deg, #00e6b8 0%, #00b894 100%);">Concurso</div>', unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown(f'<div class="custom-table-header" style="width: 100%; background: linear-gradient(135deg, #00e6b8 0%, #00b894 100%);">Data</div>', unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown(f'<div class="custom-table-header" style="width: 100%; background: linear-gradient(135deg, #00e6b8 0%, #00b894 100%); text-align: center;">Números Sorteados</div>', unsafe_allow_html=True)
    with header_cols[3]:
        st.markdown(f'<div class="custom-table-header" style="width: 100%; border-radius: 0 12px 0 0; background: linear-gradient(135deg, #00e6b8 0%, #00b894 100%); text-align: right;">Soma</div>', unsafe_allow_html=True)
        
    for i, row in df_recent.iterrows():
        numeros = [int(row[col]) for col in COLUNAS_BOLAS if pd.notna(row[col])]
        soma = sum(numeros)
        
        bolas_html = "".join([f'<span class="lotto-number">{n:02d}</span>' for n in numeros])
        data_fmt = row['Data'].strftime('%d/%m/%Y')
        
        row_cols = st.columns([1, 1, 3, 1])
        
        with row_cols[0]:
            st.markdown(f"<div class='custom-table-row' style='border-bottom: none; border-radius: 0;'><strong>{row['Concurso']}</strong></div>", unsafe_allow_html=True)
        with row_cols[1]:
            st.markdown(f"<div class='custom-table-row' style='border-bottom: none; border-radius: 0;'>{data_fmt}</div>", unsafe_allow_html=True)
        with row_cols[2]:
            st.markdown(f"<div class='custom-table-row' style='border-bottom: none; justify-content: center; border-radius: 0;'>{bolas_html}</div>", unsafe_allow_html=True)
        with row_cols[3]:
            st.markdown(f"<div class='custom-table-row' style='border-bottom: none; justify-content: flex-end; border-radius: 0;'><strong>{soma}</strong></div>", unsafe_allow_html=True)
            
        st.markdown("---")

    st.divider()
    
    st.subheader("📈 Evolução Temporal dos Sorteios")
    
    df_anual = df.copy()
    df_anual['Ano'] = df_anual['Data'].dt.year
    sorteios_por_ano = df_anual.groupby('Ano').size().reset_index(name='Quantidade')
    
    if not sorteios_por_ano.empty:
        chart = alt.Chart(sorteios_por_ano).mark_bar(color='#00e6b8', cornerRadius=5).encode(
            x=alt.X('Ano:O', title='Ano', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Quantidade:Q', title='Quantidade de Sorteios'),
            tooltip=['Ano', 'Quantidade']
        ).properties(
            height=400,
            title='Sorteios por Ano'
        )
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Não há dados suficientes para gerar o gráfico de evolução temporal.")

def page_frequencia(df: pd.DataFrame):
    st.header("📈 Frequência dos Números")
    st.markdown("Distribuição completa da frequência de todos os números de 1 a 60.")
    
    freq_data = get_frequencia(df)
    df_freq = pd.DataFrame(freq_data, columns=['Número', 'Frequência'])
    
    if df_freq.empty:
        st.warning("Não há dados de frequência disponíveis.")
        return
    
    try:
        df_freq['Número_str'] = df_freq['Número'].astype(str)
        
        chart = alt.Chart(df_freq).mark_bar(color='#00e6b8').encode(
            x=alt.X('Número_str:N', 
                    title='Número (1 a 60)', 
                    sort=[str(n) for n in ALL_NUMBERS], 
                    axis=alt.Axis(labelAngle=0)), 
            y=alt.Y('Frequência:Q', title='Frequência'),
            tooltip=[alt.Tooltip('Número:Q', title='Número'), 'Frequência']
        ).properties(
            height=500,
            title='Frequência de Sorteio por Número'
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Erro crítico ao gerar gráfico de frequência: {e}")
        st.info("Exibindo a tabela de dados como alternativa.")
        st.dataframe(df_freq, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 15 Mais Frequentes")
        st.dataframe(
            df_freq.head(15),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Número": st.column_config.NumberColumn(format="%d"),
                "Frequência": st.column_config.NumberColumn(format="%d")
            }
        )
    
    with col2:
        st.subheader("🧊 Top 15 Menos Frequentes")
        st.dataframe(
            df_freq.tail(15).sort_values('Frequência'),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Número": st.column_config.NumberColumn(format="%d"),
                "Frequência": st.column_config.NumberColumn(format="%d")
            }
        )

def page_pares_impares(df: pd.DataFrame):
    st.header("⚖️ Análise de Pares e Ímpares")
    
    dist_pares_impares = get_pares_impares(df).reset_index()
    dist_pares_impares.columns = ['Qtd_Impares', 'Percentual']
    dist_pares_impares['Label'] = dist_pares_impares['Qtd_Impares'].apply(
        lambda x: f"{x} Ímpare{'s' if x != 1 else ''} / {6-x} Pare{'s' if 6-x != 1 else ''}"
    )
    
    dist_faixas = get_frequencia_faixas(df).reset_index()
    dist_faixas.columns = ['Faixa', 'Frequência']
    dist_faixas['Percentual'] = (dist_faixas['Frequência'] / dist_faixas['Frequência'].sum() * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição Par/Ímpar")
        
        pie_chart = alt.Chart(dist_pares_impares).mark_arc(innerRadius=80).encode(
            theta=alt.Theta('Percentual:Q', stack=True),
            color=alt.Color('Label:N', 
                          scale=alt.Scale(scheme='set2'),
                          legend=alt.Legend(title="Combinação", orient='bottom')),
            tooltip=['Label', alt.Tooltip('Percentual:Q', format='.1f')]
        ).properties(
            height=400,
            title='Proporção das Combinações Par/Ímpar'
        )
        
        st.altair_chart(pie_chart, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Distribuição por Faixas")
        
        bar_chart = alt.Chart(dist_faixas).mark_bar(color='#00e6b8').encode(
            x=alt.X('Faixa:N', title='Faixa de Dezenas', sort=None),
            y=alt.Y('Frequência:Q', title='Frequência'),
            tooltip=['Faixa', 'Frequência', 'Percentual']
        ).properties(
            height=400,
            title='Frequência por Faixa de Dezenas'
        )
        
        st.altair_chart(bar_chart, use_container_width=True)
    
    st.divider()
    
    st.subheader("📋 Estatísticas Detalhadas")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        total_numeros = len(df) * 6
        total_impares = (df[COLUNAS_BOLAS] % 2 == 1).sum().sum()
        percentual_impares = (total_impares / total_numeros * 100)
        st.metric("Percentual de Ímpares", f"{percentual_impares:.1f}%")
    
    with col_stat2:
        combinacao_mais_comum = dist_pares_impares.loc[dist_pares_impares['Percentual'].idxmax()]
        st.metric("Combinação Mais Comum", combinacao_mais_comum['Label'])
    
    with col_stat3:
        faixa_mais_comum = dist_faixas.loc[dist_faixas['Frequência'].idxmax(), 'Faixa']
        st.metric("Faixa Mais Frequente", faixa_mais_comum)

def page_primos_compostos(df: pd.DataFrame):
    st.header("🔢 Análise de Primos e Compostos")
    st.markdown("Distribuição estatística de números primos e compostos nos sorteios históricos.")
    
    dist_primos, resumo, teorico = get_primos_compostos(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    media_primos = (dist_primos['Qtd_Primos'] * dist_primos['Percentual']).sum() / 100
    primos_universo = len(PRIMOS_1_A_60)
    percentual_primos_universo = (primos_universo / 60) * 100
    
    with col1:
        st.metric("Média de Primos por Sorteio", f"{media_primos:.2f}")
    
    with col2:
        st.metric("Primos no Universo (1-60)", primos_universo)
    
    with col3:
        st.metric("% Primos no Universo", f"{percentual_primos_universo:.1f}%")
    
    with col4:
        combinacao_mais_comum_primos = dist_primos.loc[dist_primos['Percentual'].idxmax(), 'Label']
        st.metric("Combinação Mais Comum", combinacao_mais_comum_primos.split(' / ')[0])
    
    st.divider()
    
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
        st.subheader("📊 Distribuição por Sorteio")
        
        pie_primos = alt.Chart(dist_primos).mark_arc(innerRadius=60).encode(
            theta=alt.Theta('Percentual:Q', stack=True),
            color=alt.Color('Label:N', 
                          scale=alt.Scale(scheme='set1'),
                          legend=alt.Legend(title="Primos/Compostos", orient='bottom')),
            tooltip=['Label', alt.Tooltip('Percentual:Q', format='.1f')]
        ).properties(
            height=400,
            title='Distribuição de Primos por Sorteio'
        )
        
        st.altair_chart(pie_primos, use_container_width=True)
    
    with col_graf2:
        st.subheader("📈 Frequência Agregada")
        
        bar_resumo = alt.Chart(resumo).mark_bar(color='#00e6b8').encode(
            x=alt.X('Categoria:N', title='Categoria'),
            y=alt.Y('Total_Sorteado:Q', title='Total de Sorteios'),
            tooltip=['Categoria', 'Total_Sorteado', 'Percentual']
        ).properties(
            height=400,
            title='Frequência Total por Categoria'
        )
        
        st.altair_chart(bar_resumo, use_container_width=True)
    
    st.divider()
    
    st.subheader("🎯 Análise Individual dos Primos")
    
    freq_data = get_frequencia(df)
    df_freq = pd.DataFrame(freq_data, columns=['Número', 'Frequência'])
    df_primos = df_freq[df_freq['Número'].isin(PRIMOS_1_A_60)]
    
    col_primos1, col_primos2 = st.columns(2)
    
    with col_primos1:
        st.markdown("##### 🏆 Primos Mais Sorteados")
        st.dataframe(
            df_primos.head(10),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Número": st.column_config.NumberColumn(format="%d"),
                "Frequência": st.column_config.NumberColumn(format="%d")
            }
        )
    
    with col_primos2:
        st.markdown("##### 🧊 Primos Menos Sorteados")
        st.dataframe(
            df_primos.tail(10).sort_values('Frequência'),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Número": st.column_config.NumberColumn(format="%d"),
                "Frequência": st.column_config.NumberColumn(format="%d")
            }
        )

def page_combinacoes(df: pd.DataFrame):
    st.header("🤝 Análise de Combinações")
    
    tipo_analise = st.radio(
        "Selecione o tipo de análise:",
        ["Duplas Mais Frequentes", "Triplas Mais Frequentes", "Números Vizinhos"],
        horizontal=True
    )
    
    st.divider()
    
    if tipo_analise == "Números Vizinhos":
        col_sel, col_viz = st.columns([1, 2])
        
        with col_sel:
            numero_analise = st.selectbox(
                "Selecione o número para análise:",
                options=ALL_NUMBERS,
                format_func=lambda x: f"{x:02d}",
                index=9
            )
        
        vizinhos = get_vizinhos(df, numero_analise)
        df_vizinhos = pd.DataFrame(vizinhos, columns=['Número', 'Frequência Conjunta'])
        
        with col_viz:
            st.subheader(f"🔗 Números que Mais Saem com o {numero_analise:02d}")
            
            chart_vizinhos = alt.Chart(df_vizinhos).mark_bar(color='#00e6b8').encode(
                x=alt.X('Frequência Conjunta:Q', title='Frequência Conjunta'),
                y=alt.Y('Número:O', sort='-x', title='Número Vizinho'),
                tooltip=['Número', 'Frequência Conjunta']
            ).properties(
                height=400,
                title=f'Frequência Conjunta com o Número {numero_analise:02d}'
            )
            
            st.altair_chart(chart_vizinhos, use_container_width=True)
        
        st.dataframe(
            df_vizinhos,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Número": st.column_config.NumberColumn(format="%d"),
                "Frequência Conjunta": st.column_config.NumberColumn(format="%d")
            }
        )
    
    else:
        duplas, triplas = get_combinacoes(df)
        
        if tipo_analise == "Duplas Mais Frequentes":
            st.subheader("🏆 Top 20 Duplas Mais Frequentes")
            df_duplas = pd.DataFrame(duplas[:20], columns=['Dupla', 'Frequência'])
            df_duplas['Dupla_Formatada'] = df_duplas['Dupla'].apply(lambda x: f"{x[0]:02d} e {x[1]:02d}")
            
            chart_duplas = alt.Chart(df_duplas).mark_bar(color='#00e6b8').encode(
                x=alt.X('Frequência:Q', title='Frequência'),
                y=alt.Y('Dupla_Formatada:O', sort='-x', title='Dupla'),
                tooltip=['Dupla_Formatada', 'Frequência']
            ).properties(
                height=500,
                title='Duplas Mais Frequentes'
            )
            
            st.altair_chart(chart_duplas, use_container_width=True)
            
            st.dataframe(
                df_duplas[['Dupla_Formatada', 'Frequência']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Dupla_Formatada": "Dupla",
                    "Frequência": st.column_config.NumberColumn(format="%d")
                }
            )
        
        else:
            st.subheader("🏆 Top 20 Triplas Mais Frequentes")
            df_triplas = pd.DataFrame(triplas[:20], columns=['Tripla', 'Frequência'])
            df_triplas['Tripla_Formatada'] = df_triplas['Tripla'].apply(lambda x: f"{x[0]:02d}, {x[1]:02d} e {x[2]:02d}")
            
            chart_triplas = alt.Chart(df_triplas).mark_bar(color='#00b894').encode(
                x=alt.X('Frequência:Q', title='Frequência'),
                y=alt.Y('Tripla_Formatada:O', sort='-x', title='Tripla'),
                tooltip=['Tripla_Formatada', 'Frequência']
            ).properties(
                height=500,
                title='Triplas Mais Frequentes'
            )
            
            st.altair_chart(chart_triplas, use_container_width=True)
            
            st.dataframe(
                df_triplas[['Tripla_Formatada', 'Frequência']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Tripla_Formatada": "Tripla",
                    "Frequência": st.column_config.NumberColumn(format="%d")
                }
            )

# =============================================================================
# PAGINA QUENTES E FRIOS (CORRIGIDA)
# =============================================================================
def page_quentes_frios(df: pd.DataFrame):
    """Página de análise de números quentes e frios."""
    st.header("🔥❄️ Números Quentes, Frios e Atrasados")
    
    janela = st.slider("Janela de Análise (número de sorteios recentes):", 10, 200, 50, 10)
    
    atrasados = get_atrasados(df)
    quentes, frios = get_quentes_frios(df, janela)
    
    # Configuração de colunas com gap para melhor visualização
    col1, col2, col3 = st.columns(3, gap="medium")
    
    # --- FIX CRÍTICO: Altura fixa para as tabelas ---
    TABLE_HEIGHT = 560 

    with col1:
        st.subheader(f"🔥 Quentes ({janela} concursos)")
        df_quentes = pd.DataFrame(quentes[:15], columns=['Número', 'Frequência'])
        st.dataframe(
            df_quentes,
            use_container_width=True,
            hide_index=True,
            height=TABLE_HEIGHT,  # Altura fixa
            column_config={
                "Número": st.column_config.NumberColumn(format="%d"),
                "Frequência": st.column_config.NumberColumn(format="%d")
            }
        )
    
    with col2:
        st.subheader(f"❄️ Frios ({janela} concursos)")
        df_frios = pd.DataFrame(frios[:15], columns=['Número', 'Frequência'])
        st.dataframe(
            df_frios,
            use_container_width=True,
            hide_index=True,
            height=TABLE_HEIGHT, # Altura fixa
            column_config={
                "Número": st.column_config.NumberColumn(format="%d"),
                "Frequência": st.column_config.NumberColumn(format="%d")
            }
        )
    
    with col3:
        st.subheader("⏰ Mais Atrasados (Geral)")
        df_atrasados = pd.DataFrame(atrasados[:15], columns=['Número', 'Atraso'])
        st.dataframe(
            df_atrasados,
            use_container_width=True,
            hide_index=True,
            height=TABLE_HEIGHT, # Altura fixa
            column_config={
                "Número": st.column_config.NumberColumn(format="%d"),
                "Atraso": st.column_config.NumberColumn(format="%d")
            }
        )
    
    st.divider()
    
    st.subheader("📊 Gráfico de Atrasos (Top 20)")
    
    df_atrasados_chart = pd.DataFrame(atrasados[:20], columns=['Número', 'Atraso'])
    
    chart_atrasados = alt.Chart(df_atrasados_chart).mark_bar(color='#ef4444').encode(
        x=alt.X('Atraso:Q', title='Sorteios em Atraso'),
        y=alt.Y('Número:O', sort='-x', title='Número'),
        tooltip=['Número', 'Atraso']
    ).properties(
        height=500,
        title='Números Mais Atrasados'
    )
    
    st.altair_chart(chart_atrasados, use_container_width=True)

def page_somas(df: pd.DataFrame):
    st.header("➕ Análise das Somas das Dezenas")
    st.markdown("Distribuição estatística das somas dos 6 números sorteados em cada concurso.")
    
    df_soma = df.copy()
    df_soma['Soma'] = df_soma[COLUNAS_BOLAS].sum(axis=1)
    
    media_soma = df_soma['Soma'].mean()
    mediana_soma = df_soma['Soma'].median()
    moda_soma = df_soma['Soma'].mode().iloc[0] if not df_soma['Soma'].mode().empty else 0
    std_soma = df_soma['Soma'].std()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Média das Somas", f"{media_soma:.1f}")
    
    with col2:
        st.metric("Mediana", f"{mediana_soma:.1f}")
    
    with col3:
        st.metric("Moda (Mais Comum)", f"{moda_soma:.0f}")
    
    with col4:
        st.metric("Desvio Padrão", f"{std_soma:.1f}")
    
    st.divider()
    
    st.subheader("📈 Distribuição das Somas (Curva de Sino)")
    
    histograma = alt.Chart(df_soma).mark_bar(color='#00e6b8', opacity=0.8).encode(
        x=alt.X('Soma:Q', bin=alt.Bin(maxbins=40), title='Valor da Soma'),
        y=alt.Y('count()', title='Frequência'),
        tooltip=[alt.Tooltip('count()', title='Frequência'), alt.Tooltip('Soma:Q', bin=True, title='Faixa')]
    ).properties(
        height=500,
        title='Distribuição Normal das Somas dos Sorteios'
    )
    
    media_line = alt.Chart(pd.DataFrame({'media': [media_soma]})).mark_rule(
        color='#ef4444',
        strokeWidth=2,
        strokeDash=[5, 5]
    ).encode(
        x='media:Q',
        tooltip=[alt.Tooltip('media:Q', title='Média')]
    )
    
    st.altair_chart(histograma + media_line, use_container_width=True)
    
    st.info("""
    💡 **Insight Estatístico:** A distribuição das somas segue uma curva normal (distribuição gaussiana) centrada entre **180 e 220**. 
    Somas muito baixas (<100) ou muito altas (>300) são estatisticamente raras na Mega-Sena.
    """)
    
    st.divider()
    
    st.subheader("📋 Somas dos Últimos 20 Sorteios")
    
    df_recentes = df_soma.tail(20).sort_values('Concurso', ascending=False)
    df_display = df_recentes[['Concurso', 'Data', 'Soma'] + COLUNAS_BOLAS].copy()
    df_display['Data'] = df_display['Data'].dt.strftime('%d/%m/%Y')
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Concurso": st.column_config.NumberColumn(format="%d"),
            "Soma": st.column_config.NumberColumn(format="%d")
        }
    )

def page_previsoes_ai(df: pd.DataFrame):
    st.header("🤖 Previsões com Inteligência Artificial")
    st.markdown("""
    **Sistema Preditivo Baseado em Machine Learning**
    
    Utiliza Regressão Logística com calibração temporal para prever probabilidades 
    dos números serem sorteados no próximo concurso.
    """)
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        n_combinacoes = st.slider("Número de Combinações:", 1, 15, 6)
        top_k = st.slider("Ranking Top K:", 10, 60, 15)
    
    with col_config2:
        diversificar = st.checkbox("Diversificar Combinações", True, 
                                 help="Balancear pares/ímpares, primos/compostos e somas")
        mostrar_todas_probs = st.checkbox("Mostrar Todas as Probabilidades", False)
    
    with col_config3:
        usar_amostragem = st.checkbox("Treino Rápido (Amostragem)", True,
                                    help="Reduz tempo de treino com amostragem inteligente")
        st.metric("Base de Treino", f"{len(df)} sorteios")
    
    st.divider()
    
    with st.expander("⚠️ Termos de Responsabilidade", expanded=True):
        st.markdown("""
        **Importante:**
        - Este é um sistema de análise estatística e predição **sem garantia de acertos**
        - Loteria é um jogo de azar com probabilidades fixas
        - Use as previsões como ferramenta de estudo estatístico
        - Aposte com responsabilidade e dentro de suas possibilidades
        """)
        
        aceite = st.checkbox("✅ Compreendo e aceito os termos acima")
    
    if aceite:
        if st.button("🚀 TREINAR MODELO E GERAR PREVISÕES", 
                    type="primary", 
                    use_container_width=True,
                    disabled=len(df) < 80):
            
            if len(df) < 80:
                st.error("❌ Dados insuficientes para treino. São necessários pelo menos 80 sorteios.")
                return
            
            try:
                with st.spinner("🧠 Treinando modelo de machine learning..."):
                    modelo, scaler, df_processado = treinar_modelo_avancado(df, usar_amostragem)
                    previsoes = gerar_previsoes_avancadas(df_processado, modelo, scaler)
                
                st.success("✅ Modelo treinado e calibrado com sucesso!")
                
                st.subheader(f"🎯 Top {top_k} Probabilidades Preditivas")
                
                df_top = pd.DataFrame(previsoes[:top_k], columns=['Número', 'Probabilidade'])
                df_top['Probabilidade_Percentual'] = df_top['Probabilidade'] * 100
                
                chart_probs = alt.Chart(df_top).mark_bar(color='#00e6b8').encode(
                    x=alt.X('Probabilidade_Percentual:Q', title='Probabilidade (%)'),
                    y=alt.Y('Número:O', sort='-x', title='Número'),
                    tooltip=['Número', alt.Tooltip('Probabilidade_Percentual:Q', format='.3f')]
                ).properties(
                    height=500,
                    title='Probabilidades Preditivas (Top 15)'
                )
                
                st.altair_chart(chart_probs, use_container_width=True)
                
                df_display = df_top.copy()
                df_display['Probabilidade'] = df_display['Probabilidade_Percentual'].map(lambda x: f"{x:.3f}%")
                st.dataframe(
                    df_display[['Número', 'Probabilidade']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Número": st.column_config.NumberColumn(format="%d"),
                        "Probabilidade": "Probabilidade (%)"
                    }
                )
                
                st.divider()
                
                st.subheader("💡 Combinações Sugeridas (Otimizadas)")
                
                combinacoes = gerar_combinacoes_avancadas(previsoes, n_combinacoes, diversificar)
                
                if not combinacoes:
                    st.warning("⚠️ Não foi possível gerar combinações diversificadas. Tente reduzir as restrições.")
                else:
                    for i, comb in enumerate(combinacoes, 1):
                        st.markdown(f"#### 🎯 Palpite {i}")
                        
                        bolas_html = "".join([f'<span class="lotto-number">{n:02d}</span>' for n in comb])
                        st.markdown(f"<div style='text-align: center; margin: 20px 0;'>{bolas_html}</div>", 
                                  unsafe_allow_html=True)
                        
                        soma = sum(comb)
                        pares = sum(1 for x in comb if x % 2 == 0)
                        impares = 6 - pares
                        primos = sum(1 for x in comb if is_primo(x))
                        baixos = sum(1 for x in comb if x <= 30)
                        altos = 6 - baixos
                        
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        with col_stat1:
                            st.metric("Soma", soma)
                        
                        with col_stat2:
                            st.metric("Par/Ímpar", f"{pares}/{impares}")
                        
                        with col_stat3:
                            st.metric("Primos", primos)
                        
                        with col_stat4:
                            st.metric("Baixo/Alto", f"{baixos}/{altos}")
                        
                        st.markdown("---")
                    
                    st.subheader("💾 Exportar Palpites")
                    
                    pdf_bytes = gerar_pdf_bytes(combinacoes, "MEGA-SENA - PALPITES IA")
                    
                    st.download_button(
                        label="📄 BAIXAR PALPITES EM PDF",
                        data=pdf_bytes,
                        file_name=f"palpites_megasena_ia_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                
                if mostrar_todas_probs:
                    st.divider()
                    st.subheader("📊 Probabilidades de Todos os Números")
                    
                    df_completo = pd.DataFrame(previsoes, columns=['Número', 'Probabilidade'])
                    df_completo['Probabilidade (%)'] = df_completo['Probabilidade'] * 100
                    
                    st.dataframe(
                        df_completo,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Número": st.column_config.NumberColumn(format="%d"),
                            "Probabilidade (%)": st.column_config.NumberColumn(format="%.4f")
                        }
                    )
            
            except Exception as e:
                st.error(f"❌ Erro no processamento: {str(e)}")
                st.info("💡 Tente ajustar as configurações ou usar a opção de treino rápido.")
    
    else:
        st.info("📝 Marque a aceitação dos termos para habilitar o modelo preditivo.")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Função principal da aplicação."""
    
    inject_custom_css()
    
    st.title("🎲 Analisador Mega-Sena AI")
    st.markdown("""
    <div style='text-align: center; color: #94a3b8; margin-bottom: 30px;'>
    Sistema completo de análise estatística e preditiva para a Mega-Sena
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("📡 Conectando à base de dados..."):
        df = carregar_dados_caixa()
    
    if df is None or df.empty:
        st.error("""
        ❌ **Erro Crítico:** Não foi possível carregar os dados da Mega-Sena.
        
        Possíveis causas:
        - Problema de conexão com a internet
        - Servidor da Caixa Econômica indisponível
        - Formato dos dados alterado
        
        Tente novamente em alguns minutos.
        """)
        return
    
    if not validar_dataframe(df):
        st.error("❌ Os dados carregados estão em formato inválido.")
        return
    
    draw_navigation()
    
    pagina = st.session_state['current_page']
    
    if pagina == "visao_geral":
        page_visao_geral(df)
    elif pagina == "frequencia":
        page_frequencia(df)
    elif pagina == "pares_impares":
        page_pares_impares(df)
    elif pagina == "primos_compostos":
        page_primos_compostos(df)
    elif pagina == "combinacoes":
        page_combinacoes(df)
    elif pagina == "quentes_frios":
        page_quentes_frios(df)
    elif pagina == "somas":
        page_somas(df)
    elif pagina == "previsoes_ai":
        page_previsoes_ai(df)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.9rem;'>
    🎯 Desenvolvido para fins educacionais e análise estatística • 
    Use com responsabilidade • 
    Versão 2.1
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
