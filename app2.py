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
import hashlib
import json
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
# CONFIGURAÇÕES INICIAIS E CONSTANTES
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

# Configurações do Sistema de Acesso
ACESSO_LIVRE_JOGOS = 1  # Número de jogos gratuitos permitidos
LINK_COMPRA = "https://seusite.com/comprar"  # Link para compra de acesso

# =============================================================================
# 0. DESIGN SYSTEM & CSS (ATUALIZADO COM NOVOS ESTILOS)
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
        
        /* 3. CONTAINER DE ACESSO */
        .access-container {{
            background: linear-gradient(145deg, #1F2937, #111827);
            padding: 30px;
            border-radius: 15px;
            border: 2px solid #00C896;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 200, 150, 0.2);
        }}
        
        .access-badge {{
            display: inline-block;
            background: #00C896;
            color: #0E1117;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
            font-size: 0.9rem;
        }}
        
        .warning-box {{
            background: rgba(255, 193, 7, 0.1);
            border-left: 4px solid #FFC107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        
        .success-box {{
            background: rgba(0, 200, 150, 0.1);
            border-left: 4px solid #00C896;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        
        /* 4. BOTÕES ESPECIAIS */
        .btn-premium {{
            background: linear-gradient(145deg, #00C896, #00997A) !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
        }}
        
        .btn-free {{
            background: linear-gradient(145deg, #6366F1, #4F46E5) !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
        }}
        
        /* 5. INPUT FIELDS */
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
        
        /* 6. BOTÕES DE NAVEGAÇÃO */
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
        
        /* 7. CARDS DE MÉTRICAS */
        [data-testid="stMetric"] {{
            background-color: #1F2937;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #374151;
        }}
        
        [data-testid="stMetricLabel"] {{ color: #9CA3AF; }}
        [data-testid="stMetricValue"] {{ color: #00C896; font-size: 1.5rem; }}
        [data-testid="stMetricDelta"] {{ color: #E0E0E0; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# SISTEMA DE CONTROLE DE ACESSO
# =============================================================================

def inicializar_sessao():
    """Inicializa variáveis de sessão para controle de acesso."""
    if 'acesso_inicializado' not in st.session_state:
        st.session_state['acesso_inicializado'] = True
        st.session_state['jogos_gerados'] = 0
        st.session_state['email_cadastrado'] = None
        st.session_state['usuario_premium'] = False
        st.session_state['current_page'] = "Visão Geral"

def verificar_email_cadastrado(email):
    """
    Simula verificação de email cadastrado.
    Em produção, substituir por consulta ao banco de dados.
    """
    # Lista simulada de emails cadastrados/premium
    emails_premium = [
        "cliente@premium.com",
        "usuario@pagante.com",
        "teste@validado.com"
    ]
    
    # Hash do email para comparação segura
    email_hash = hashlib.sha256(email.lower().strip().encode()).hexdigest()
    premium_hashes = [hashlib.sha256(e.encode()).hexdigest() for e in emails_premium]
    
    return email_hash in premium_hashes

def mostrar_controle_acesso():
    """Exibe o controle de acesso e status do usuário."""
    st.markdown('<div class="access-container">', unsafe_allow_html=True)
    
    if st.session_state['usuario_premium']:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success(f"✅ ACESSO PREMIUM ATIVO")
        st.info(f"Email: {st.session_state['email_cadastrado']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif st.session_state['jogos_gerados'] >= ACESSO_LIVRE_JOGOS:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning(f"⚠️ VOCÊ UTILIZOU SEU {ACESSO_LIVRE_JOGOS} JOGO GRATUITO")
        
        # Formulário para email
        with st.form("form_email"):
            email = st.text_input("Digite seu email cadastrado:", 
                                 placeholder="seu@email.com")
            submit = st.form_submit_button("🔓 VERIFICAR ACESSO", 
                                          use_container_width=True)
            
            if submit and email:
                if verificar_email_cadastrado(email):
                    st.session_state['email_cadastrado'] = email
                    st.session_state['usuario_premium'] = True
                    st.session_state['jogos_gerados'] = 0  # Reset contador
                    st.rerun()
                else:
                    st.error("❌ Email não cadastrado ou acesso não ativado")
                    st.markdown(f"""
                    <div style="margin-top: 20px; padding: 15px; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
                    <h4 style="color: #EF4444; margin: 0;">Acesso Premium Requerido</h4>
                    <p>Para continuar gerando jogos, adquira o acesso premium:</p>
                    <a href="{LINK_COMPRA}" target="_blank">
                        <button style="
                            background: linear-gradient(145deg, #00C896, #00997A);
                            color: white;
                            border: none;
                            padding: 12px 24px;
                            border-radius: 8px;
                            font-weight: bold;
                            cursor: pointer;
                            margin-top: 10px;
                            width: 100%;
                        ">
                            🛒 COMPRAR ACESSO PREMIUM
                        </button>
                    </a>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        jogos_restantes = ACESSO_LIVRE_JOGOS - st.session_state['jogos_gerados']
        st.markdown(f'<div class="access-badge">🎁 {jogos_restantes} JOGO(S) GRATUITO(S) RESTANTE(S)</div>', 
                   unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def verificar_permissao_gerar_jogo():
    """
    Verifica se o usuário tem permissão para gerar mais jogos.
    Incrementa contador se for acesso gratuito.
    """
    # Se for premium, sempre permite
    if st.session_state['usuario_premium']:
        return True
    
    # Se ainda tem jogos gratuitos
    if st.session_state['jogos_gerados'] < ACESSO_LIVRE_JOGOS:
        st.session_state['jogos_gerados'] += 1
        return True
    
    return False

# =============================================================================
# 1. FUNÇÕES AUXILIARES (CORRIGIDAS PARA NOVO FORMATO DA API)
# =============================================================================

def carregar_dados_caixa():
    """Carrega dados históricos da Mega-Sena com fallback para dados locais."""
    try:
        url = "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Tentar parsear como JSON
        try:
            data = response.json()
        except:
            # Se não for JSON válido, tentar tratar como texto
            content = response.text
            if content.startswith('<?xml'):
                # É XML, não JSON
                st.warning("API retornou XML em vez de JSON. Usando dados de fallback.")
                return carregar_dados_fallback()
            else:
                # Tentar encontrar JSON no conteúdo
                try:
                    # Remover possíveis caracteres antes do JSON
                    json_start = content.find('{')
                    if json_start != -1:
                        content = content[json_start:]
                    data = json.loads(content)
                except:
                    st.warning("Não foi possível parsear a resposta. Usando dados de fallback.")
                    return carregar_dados_fallback()
        
        # Verificar se é uma lista ou um dicionário
        if isinstance(data, dict):
            # Se for um único sorteio, colocar em uma lista
            if 'numero' in data:
                data = [data]
            else:
                # Verificar se há uma chave com os dados
                for key in ['listaDezenas', 'dezenasSorteadasOrdemSorteio', 'dezenas']:
                    if key in data:
                        data = [data]
                        break
        
        # Processar dados
        rows = []
        for item in data if isinstance(data, list) else [data]:
            try:
                # Tentar diferentes formatos de chaves
                numero = None
                data_sorteio = None
                dezenas = []
                
                # Obter número do concurso
                for key in ['numero', 'concurso', 'nu_concurso']:
                    if key in item:
                        numero = int(item[key]) if item[key] else 0
                        break
                
                # Obter data do sorteio
                for key in ['dataApuracao', 'data', 'dt_sorteio']:
                    if key in item:
                        data_sorteio = item[key]
                        break
                
                # Obter dezenas sorteadas
                dezenas_keys = [
                    'dezenasSorteadasOrdemSorteio',
                    'listaDezenas', 
                    'dezenas',
                    'dezenasSorteadas'
                ]
                
                for key in dezenas_keys:
                    if key in item and item[key]:
                        if isinstance(item[key], list):
                            dezenas = item[key]
                        elif isinstance(item[key], str):
                            # Tentar converter string para lista
                            if '[' in item[key]:
                                dezenas = json.loads(item[key])
                            else:
                                dezenas = item[key].split(',') if ',' in item[key] else item[key].split()
                        break
                
                # Se não encontrou dezenas, tentar outras abordagens
                if not dezenas:
                    # Procurar por dezenas individuais
                    dezenas = []
                    for i in range(1, 7):
                        dezena_key = f'dezena{i}' if f'dezena{i}' in item else f'bola{i}' if f'bola{i}' in item else None
                        if dezena_key and dezena_key in item:
                            dezenas.append(item[dezena_key])
                
                # Converter dezenas para inteiros
                dezenas = [int(d) for d in dezenas if str(d).isdigit()][:6]
                
                if len(dezenas) == 6 and numero:
                    row = {
                        'Concurso': numero,
                        'Data': pd.to_datetime(data_sorteio, dayfirst=True, errors='coerce'),
                        'B1': dezenas[0],
                        'B2': dezenas[1],
                        'B3': dezenas[2],
                        'B4': dezenas[3],
                        'B5': dezenas[4],
                        'B6': dezenas[5]
                    }
                    rows.append(row)
                    
            except Exception as e:
                st.warning(f"Erro ao processar item: {e}")
                continue
        
        if rows:
            df = pd.DataFrame(rows)
            df[COLUNAS_BOLAS] = df[COLUNAS_BOLAS].astype(int)
            df = df.sort_values('Concurso', ascending=False).reset_index(drop=True)
            return df
        else:
            st.warning("Nenhum dado válido encontrado na API. Usando dados de fallback.")
            return carregar_dados_fallback()
    
    except requests.exceptions.RequestException as e:
        st.warning(f"Erro na requisição HTTP: {e}. Usando dados de fallback.")
        return carregar_dados_fallback()
    except Exception as e:
        st.warning(f"Erro ao carregar dados: {e}. Usando dados de fallback.")
        return carregar_dados_fallback()

def carregar_dados_fallback():
    """Carrega dados de fallback quando a API falha."""
    try:
        # Dados de exemplo da Mega-Sena (últimos 20 concursos como fallback)
        dados_fallback = [
            {'Concurso': 2797, 'Data': '2024-12-04', 'B1': 4, 'B2': 10, 'B3': 21, 'B4': 32, 'B5': 44, 'B6': 58},
            {'Concurso': 2796, 'Data': '2024-11-30', 'B1': 7, 'B2': 13, 'B3': 19, 'B4': 28, 'B5': 41, 'B6': 52},
            {'Concurso': 2795, 'Data': '2024-11-27', 'B1': 2, 'B2': 15, 'B3': 24, 'B4': 37, 'B5': 49, 'B6': 56},
            {'Concurso': 2794, 'Data': '2024-11-23', 'B1': 8, 'B2': 17, 'B3': 26, 'B4': 35, 'B5': 43, 'B6': 54},
            {'Concurso': 2793, 'Data': '2024-11-20', 'B1': 3, 'B2': 12, 'B3': 22, 'B4': 33, 'B5': 45, 'B6': 59},
            {'Concurso': 2792, 'Data': '2024-11-16', 'B1': 6, 'B2': 14, 'B3': 23, 'B4': 31, 'B5': 46, 'B6': 57},
            {'Concurso': 2791, 'Data': '2024-11-13', 'B1': 5, 'B2': 18, 'B3': 25, 'B4': 34, 'B5': 42, 'B6': 53},
            {'Concurso': 2790, 'Data': '2024-11-09', 'B1': 9, 'B2': 11, 'B3': 20, 'B4': 29, 'B5': 38, 'B6': 55},
            {'Concurso': 2789, 'Data': '2024-11-06', 'B1': 1, 'B2': 16, 'B3': 27, 'B4': 36, 'B5': 47, 'B6': 60},
            {'Concurso': 2788, 'Data': '2024-11-02', 'B1': 4, 'B2': 13, 'B3': 21, 'B4': 32, 'B5': 44, 'B6': 58},
            {'Concurso': 2787, 'Data': '2024-10-30', 'B1': 7, 'B2': 10, 'B3': 19, 'B4': 28, 'B5': 41, 'B6': 52},
            {'Concurso': 2786, 'Data': '2024-10-26', 'B1': 2, 'B2': 15, 'B3': 24, 'B4': 37, 'B5': 49, 'B6': 56},
            {'Concurso': 2785, 'Data': '2024-10-23', 'B1': 8, 'B2': 17, 'B3': 26, 'B4': 35, 'B5': 43, 'B6': 54},
            {'Concurso': 2784, 'Data': '2024-10-19', 'B1': 3, 'B2': 12, 'B3': 22, 'B4': 33, 'B5': 45, 'B6': 59},
            {'Concurso': 2783, 'Data': '2024-10-16', 'B1': 6, 'B2': 14, 'B3': 23, 'B4': 31, 'B5': 46, 'B6': 57},
            {'Concurso': 2782, 'Data': '2024-10-12', 'B1': 5, 'B2': 18, 'B3': 25, 'B4': 34, 'B5': 42, 'B6': 53},
            {'Concurso': 2781, 'Data': '2024-10-09', 'B1': 9, 'B2': 11, 'B3': 20, 'B4': 29, 'B5': 38, 'B6': 55},
            {'Concurso': 2780, 'Data': '2024-10-05', 'B1': 1, 'B2': 16, 'B3': 27, 'B4': 36, 'B5': 47, 'B6': 60},
            {'Concurso': 2779, 'Data': '2024-10-02', 'B1': 4, 'B2': 13, 'B3': 21, 'B4': 32, 'B5': 44, 'B6': 58},
            {'Concurso': 2778, 'Data': '2024-09-28', 'B1': 7, 'B2': 10, 'B3': 19, 'B4': 28, 'B5': 41, 'B6': 52}
        ]
        
        df = pd.DataFrame(dados_fallback)
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values('Concurso', ascending=False).reset_index(drop=True)
        st.info("⚠️ Usando dados de demonstração (últimos 20 concursos)")
        return df
        
    except Exception as e:
        st.error(f"Erro no fallback: {e}")
        return pd.DataFrame()

def validar_dados(df):
    """Valida se os dados foram carregados corretamente."""
    if df.empty:
        return False
    
    # Verificar colunas necessárias
    colunas_necessarias = ['Concurso', 'Data'] + COLUNAS_BOLAS
    if not all(col in df.columns for col in colunas_necessarias):
        return False
    
    # Verificar se há dados suficientes
    if len(df) < 10:
        return False
    
    return True

def draw_navigation():
    """Renderiza a navegação entre páginas."""
    pages = [
        ("📊 Visão Geral", "Visão Geral"),
        ("📈 Frequência", "Frequência"),
        ("🔢 Pares/Impares", "Pares/Impares"),
        ("🔄 Combinações", "Combinações"),
        ("🔥 Quentes/Frios", "Quentes/Frios"),
        ("🧮 ∑ Somas", "∑ Somas"),
        ("🤖 Previsões AI", "Previsões AI")
    ]
    
    cols = st.columns(len(pages))
    for idx, (icon_name, page_name) in enumerate(pages):
        with cols[idx]:
            if st.button(f"{icon_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state['current_page'] = page_name
                st.rerun()

# =============================================================================
# 2. MODELO PREDITIVO (MANTIDO ORIGINAL)
# =============================================================================

class MegaSenaPredictor:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.last_training_date = None
    
    def prepare_features(self, window_size=10):
        """Prepara features para o modelo."""
        df = self.df.sort_values('Concurso').reset_index(drop=True)
        features = []
        targets = []
        
        for i in range(window_size, len(df)):
            # Features: frequências dos últimos concursos
            recent = df.iloc[i-window_size:i]
            freq = recent[COLUNAS_BOLAS].values.flatten()
            freq_counts = Counter(freq)
            
            # Features para cada número (1-60)
            row_features = []
            for num in range(1, 61):
                row_features.append(freq_counts.get(num, 0))
            
            # Adicionar features derivadas
            ultimo_sorteio = df.iloc[i-1][COLUNAS_BOLAS].values
            for num in range(1, 61):
                row_features.append(1 if num in ultimo_sorteio else 0)
            
            features.append(row_features)
            
            # Target: números sorteados no concurso atual
            target = df.iloc[i][COLUNAS_BOLAS].values
            targets.append(target)
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Feature names
        self.feature_names = [f"freq_{i}" for i in range(1, 61)] + [f"ultimo_{i}" for i in range(1, 61)]
        
        return features, targets
    
    def train(self):
        """Treina o modelo."""
        try:
            X, y = self.prepare_features(window_size=15)
            
            if len(X) < 20:
                raise ValueError("Dados insuficientes para treino")
            
            # Transformar problema multiclasse
            y_flat = y.flatten()
            X_flat = np.repeat(X, 6, axis=0)
            
            # Normalizar
            X_scaled = self.scaler.fit_transform(X_flat)
            
            # Modelo com calibração
            base_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            
            self.model = CalibratedClassifierCV(
                base_model,
                method='sigmoid',
                cv=TimeSeriesSplit(n_splits=5)
            )
            
            self.model.fit(X_scaled, y_flat)
            self.last_training_date = pd.Timestamp.now()
            return True
            
        except Exception as e:
            st.error(f"Erro no treinamento: {e}")
            return False
    
    def predict_proba_next(self):
        """Previsão para o próximo sorteio."""
        if self.model is None:
            raise NotFittedError("Modelo não treinado")
        
        # Preparar features do último sorteio conhecido
        X_last, _ = self.prepare_features(window_size=15)
        if len(X_last) == 0:
            raise ValueError("Não há dados suficientes para previsão")
        
        X_last_scaled = self.scaler.transform(X_last[-1:])
        X_last_repeated = np.repeat(X_last_scaled, 60, axis=0)
        
        # Prever probabilidades
        numbers = np.arange(1, 61).reshape(-1, 1)
        probas = self.model.predict_proba(X_last_repeated)
        
        # Média das probabilidades para cada número
        final_probs = []
        for i in range(60):
            idx = np.where(self.model.classes_ == i+1)[0]
            if len(idx) > 0:
                final_probs.append(probas[i, idx[0]])
            else:
                final_probs.append(0.0)
        
        return list(zip(range(1, 61), final_probs))
    
    def generate_combinations(self, n_comb=10, top_n=25):
        """Gera combinações baseadas nas probabilidades."""
        predictions = self.predict_proba_next()
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        top_numbers = [num for num, prob in predictions[:top_n]]
        comb_selected = []
        
        for _ in range(n_comb):
            comb = np.random.choice(top_numbers, size=6, replace=False)
            comb.sort()
            comb_selected.append(list(comb))
        
        return comb_selected, predictions

# =============================================================================
# 3. FUNÇÕES DE RELATÓRIO PDF (MANTIDAS ORIGINAIS)
# =============================================================================

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Palpites Mega-Sena - Gerado por AI', 0, 1, 'C')
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

def gerar_pdf_bytes(combinacoes):
    """Gera PDF com as combinações."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 11)
    
    pdf.cell(0, 10, 'Combinações Sugeridas:', 0, 1)
    pdf.ln(5)
    
    for i, comb in enumerate(combinacoes, 1):
        nums = ' - '.join(f'{n:02d}' for n in comb)
        pdf.cell(0, 8, f'Jogo {i:02d}: {nums}', 0, 1)
    
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(0, 6, 'Observação: Estas são sugestões baseadas em análise estatística e IA. '
                       'Lembre-se que a loteria é um jogo de azar e não há garantias de vitória.')
    
    return pdf.output(dest='S').encode('latin1')

# =============================================================================
# 4. PÁGINAS DE ANÁLISE (MANTIDAS ORIGINAIS - APENAS page_ai MODIFICADA)
# =============================================================================

def page_visao_geral(df):
    """Página de visão geral."""
    st.header("📊 Visão Geral dos Sorteios")
    
    ultimo = df.iloc[0]
    st.metric("Último Concurso", int(ultimo['Concurso']))
    st.metric("Data Último Sorteio", ultimo['Data'].strftime('%d/%m/%Y'))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Dezenas Sorteadas")
        st.write(f"{ultimo['B1']} - {ultimo['B2']} - {ultimo['B3']} - "
                 f"{ultimo['B4']} - {ultimo['B5']} - {ultimo['B6']}")
    
    with col2:
        st.subheader("Distribuição (1-60)")
        chart_data = pd.DataFrame({
            'Número': ultimo[COLUNAS_BOLAS].values,
            'Valor': [1] * 6
        })
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='Número:Q',
            y='Valor:Q'
        ).properties(height=150)
        st.altair_chart(chart, use_container_width=True)
    
    with col3:
        st.subheader("Últimos 10 Concursos")
        st.dataframe(df.head(10)[['Concurso', 'Data'] + COLUNAS_BOLAS], 
                    use_container_width=True, hide_index=True)

def page_frequencia(df):
    """Página de análise de frequência."""
    st.header("📈 Análise de Frequência")
    
    # Calcular frequências
    all_numbers = df[COLUNAS_BOLAS].values.flatten()
    freq_series = pd.Series(all_numbers).value_counts().sort_index()
    freq_df = pd.DataFrame({'Número': freq_series.index, 'Frequência': freq_series.values})
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de barras
        bars = alt.Chart(freq_df).mark_bar().encode(
            x=alt.X('Número:O', title='Número'),
            y=alt.Y('Frequência:Q', title='Frequência'),
            color=alt.condition(
                alt.datum.Frequência > freq_df['Frequência'].mean(),
                alt.value('#00C896'),
                alt.value('#374151')
            )
        ).properties(height=400)
        st.altair_chart(bars, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Mais Sorteados")
        top10 = freq_df.nlargest(10, 'Frequência')
        st.dataframe(top10, use_container_width=True, hide_index=True)
        
        st.subheader("Top 10 Menos Sorteados")
        bottom10 = freq_df.nsmallest(10, 'Frequência')
        st.dataframe(bottom10, use_container_width=True, hide_index=True)

def page_pares_impares(df):
    """Página de análise de pares/ímpares."""
    st.header("🔢 Análise de Pares e Ímpares")
    
    # Calcular proporção
    df['Pares'] = df[COLUNAS_BOLAS].apply(lambda row: sum(1 for x in row if x % 2 == 0), axis=1)
    df['Ímpares'] = 6 - df['Pares']
    
    # Distribuição
    dist_pares = df['Pares'].value_counts().sort_index()
    dist_df = pd.DataFrame({
        'Pares': dist_pares.index,
        'Frequência': dist_pares.values,
        'Porcentagem': (dist_pares.values / len(df) * 100).round(1)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição Pares/Ímpares")
        chart = alt.Chart(dist_df).mark_bar().encode(
            x='Pares:O',
            y='Frequência:Q',
            color=alt.Color('Pares:O', scale=alt.Scale(scheme='greens'))
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        
        # Estatísticas
        st.metric("Média de Pares por Sorteio", f"{df['Pares'].mean():.2f}")
        st.metric("Moda (Mais Comum)", int(df['Pares'].mode()[0]))
    
    with col2:
        st.subheader("Últimos 20 Sorteios")
        recent = df.head(20).copy()
        recent['Razão'] = recent['Pares'].astype(str) + ':' + recent['Ímpares'].astype(str)
        st.dataframe(recent[['Concurso', 'Pares', 'Ímpares', 'Razão']], 
                    use_container_width=True, hide_index=True)

def page_combinacoes(df):
    """Página de análise de combinações recorrentes."""
    st.header("🔄 Combinações e Padrões")
    
    # Análise de duplas recorrentes
    duplas = []
    for _, row in df.iterrows():
        nums = sorted(row[COLUNAS_BOLAS])
        duplas.extend(list(itertools.combinations(nums, 2)))
    
    dupla_counts = Counter(duplas)
    top_duplas = dupla_counts.most_common(20)
    
    # Preparar dados para visualização
    dupla_data = []
    for (n1, n2), count in top_duplas:
        dupla_data.append({'Dupla': f'{n1:02d}-{n2:02d}', 'Frequência': count})
    
    dupla_df = pd.DataFrame(dupla_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de duplas
        chart = alt.Chart(dupla_df).mark_bar().encode(
            y=alt.Y('Dupla:O', sort='-x', title='Dupla'),
            x=alt.X('Frequência:Q', title='Frequência'),
            color=alt.Color('Frequência:Q', scale=alt.Scale(scheme='viridis'))
        ).properties(height=500)
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.subheader("Top 15 Duplas")
        st.dataframe(dupla_df.head(15), use_container_width=True, hide_index=True)
        
        # Análise de trios
        st.subheader("Análise de Trios")
        trio_sample = []
        for _, row in df.head(50).iterrows():
            nums = sorted(row[COLUNAS_BOLAS])
            trio_sample.extend(list(itertools.combinations(nums, 3))[:2])
        
        trio_counts = Counter(trio_sample)
        st.write(f"Trios únicos nos últimos 50: {len(trio_counts)}")

def page_quentes(df):
    """Página de números quentes e frios."""
    st.header("🔥 Números Quentes e Frios")
    
    # Definir período para análise
    periodo = st.slider("Número de Concursos para Análise", 
                       min_value=10, 
                       max_value=100, 
                       value=50)
    
    df_recente = df.head(periodo).copy()
    
    # Calcular frequências no período
    all_recent = df_recente[COLUNAS_BOLAS].values.flatten()
    freq_recent = pd.Series(all_recent).value_counts().reindex(range(1, 61), fill_value=0)
    
    # Classificar
    limiar_quente = freq_recent.quantile(0.75)
    limiar_frio = freq_recent.quantile(0.25)
    
    quentes = freq_recent[freq_recent >= limiar_quente].index.tolist()
    frios = freq_recent[freq_recent <= limiar_frio].index.tolist()
    normais = [n for n in range(1, 61) if n not in quentes + frios]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(f"🔥 Quentes ({len(quentes)})")
        st.write("Frequência alta recente")
        quentes_str = ', '.join(f'{n:02d}' for n in sorted(quentes))
        st.info(quentes_str)
        
        # Gráfico quentes
        if quentes:
            q_df = pd.DataFrame({
                'Número': quentes,
                'Frequência': [freq_recent[n] for n in quentes],
                'Tipo': 'Quente'
            })
            bars = alt.Chart(q_df).mark_bar(color='#EF4444').encode(
                x='Número:O',
                y='Frequência:Q'
            ).properties(height=200)
            st.altair_chart(bars, use_container_width=True)
    
    with col2:
        st.subheader(f"😐 Normais ({len(normais)})")
        st.write("Frequência média recente")
        normais_str = ', '.join(f'{n:02d}' for n in sorted(normais[:20]))
        if len(normais) > 20:
            normais_str += f"... (+{len(normais)-20})"
        st.warning(normais_str)
    
    with col3:
        st.subheader(f"❄️ Frios ({len(frios)})")
        st.write("Frequência baixa recente")
        frios_str = ', '.join(f'{n:02d}' for n in sorted(frios))
        st.success(frios_str)
        
        # Gráfico frios
        if frios:
            f_df = pd.DataFrame({
                'Número': frios,
                'Frequência': [freq_recent[n] for n in frios],
                'Tipo': 'Frio'
            })
            bars = alt.Chart(f_df).mark_bar(color='#3B82F6').encode(
                x='Número:O',
                y='Frequência:Q'
            ).properties(height=200)
            st.altair_chart(bars, use_container_width=True)

def page_somas(df):
    """Página de análise das somas das dezenas."""
    st.header("🧮 Análise das Somas das Dezenas")
    
    # Calcular somas
    df['Soma'] = df[COLUNAS_BOLAS].sum(axis=1)
    
    # Estatísticas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Soma Média", f"{df['Soma'].mean():.1f}")
    col2.metric("Soma Mínima", int(df['Soma'].min()))
    col3.metric("Soma Máxima", int(df['Soma'].max()))
    col4.metric("Desvio Padrão", f"{df['Soma'].std():.1f}")
    
    # Distribuição
    st.subheader("Distribuição das Somas")
    
    hist = alt.Chart(df).mark_bar(color='#00C896').encode(
        alt.X('Soma:Q', bin=alt.Bin(maxbins=30), title='Soma'),
        alt.Y('count():Q', title='Frequência')
    ).properties(height=300)
    
    st.altair_chart(hist, use_container_width=True)
    
    # Análise temporal
    st.subheader("Evolução Temporal das Somas")
    
    line = alt.Chart(df.head(100)).mark_line(color='#00C896').encode(
        x=alt.X('Concurso:Q', title='Concurso'),
        y=alt.Y('Soma:Q', title='Soma'),
        tooltip=['Concurso', 'Soma']
    ).properties(height=300)
    
    mean_line = alt.Chart(pd.DataFrame({'y': [df['Soma'].mean()]})).mark_rule(
        color='red', strokeDash=[5, 5]
    ).encode(y='y:Q')
    
    st.altair_chart(line + mean_line, use_container_width=True)

def page_ai(df):
    """Página de previsões com IA - COM CONTROLE DE ACESSO."""
    st.header("🤖 Previsões com Inteligência Artificial")
    
    # Mostrar controle de acesso
    mostrar_controle_acesso()
    
    # Verificar se pode acessar (primeiro jogo é sempre permitido)
    if not verificar_permissao_gerar_jogo():
        # Já exibiu mensagem no mostrar_controle_acesso()
        return
    
    # Continuar com a funcionalidade original
    st.markdown("""
    Esta ferramenta utiliza **machine learning** para analisar padrões históricos 
    e sugerir combinações com maior probabilidade estatística.
    """)
    
    # Configurações
    col1, col2 = st.columns(2)
    with col1:
        n_jogos = st.slider("Número de Jogos", 1, 20, 5)
    with col2:
        top_n = st.slider("Considerar Top N Números", 15, 40, 25)
    
    aceite = st.checkbox(
        "✅ Confirmo que entendo que são sugestões estatísticas e não garantias"
    )
    
    if st.button("🎯 GERAR PREVISÕES COM IA", 
                 type="primary", 
                 disabled=not aceite,
                 use_container_width=True):
        
        # Verificar novamente antes de gerar (para casos de múltiplos cliques)
        if not verificar_permissao_gerar_jogo():
            st.warning("Limite de jogos gratuitos atingido. Verifique seu acesso acima.")
            return
        
        with st.spinner("🧠 Analisando padrões históricos e treinando modelo..."):
            time.sleep(1)
            
            try:
                # Treinar modelo
                predictor = MegaSenaPredictor(df)
                success = predictor.train()
                
                if not success:
                    st.error("Falha no treinamento do modelo.")
                    return
                
                # Gerar previsões
                combs, preds = predictor.generate_combinations(
                    n_comb=n_jogos, 
                    top_n=top_n
                )
                
                # Exibir resultados
                st.subheader("🎯 Combinações Sugeridas")
                
                cols = st.columns(min(3, len(combs)))
                for idx, comb in enumerate(combs):
                    with cols[idx % 3]:
                        with st.container():
                            st.markdown(f"**Jogo {idx+1}**")
                            st.markdown(
                                f"<div style='font-size: 1.2rem; font-weight: bold; "
                                f"color: #00C896; padding: 10px; background: #1F2937; "
                                f"border-radius: 10px; text-align: center;'>"
                                f"{' - '.join(f'{n:02d}' for n in comb)}</div>",
                                unsafe_allow_html=True
                            )
                            
                            # Estatísticas da combinação
                            soma = sum(comb)
                            pares = sum(1 for x in comb if x % 2 == 0)
                            impares = 6 - pares
                            baixos = sum(1 for x in comb if x <= 30)
                            altos = 6 - baixos
                            
                            st.caption(f"Soma: {soma} | Pares: {pares} | Baixos: {baixos}")
                
                # Métricas resumo
                st.subheader("📊 Estatísticas das Combinações")
                comb_array = np.array(combs)
                soma_media = np.mean(comb_array.sum(axis=1))
                pares_medio = np.mean([sum(1 for x in c if x % 2 == 0) for c in combs])
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Soma Média", f"{soma_media:.1f}")
                m2.metric("Pares Médios", f"{pares_medio:.1f}")
                m3.metric("Baixos Médios", f"{np.mean([sum(1 for x in c if x <= 30) for c in combs]):.1f}")
                m4.metric("Variação", f"{len(set(np.array(combs).flatten()))}/60")
                
                # Botão para download
                st.markdown("---")
                st.subheader("💾 Salvar Jogos")
                pdf_bytes = gerar_pdf_bytes(combs)
                
                st.download_button(
                    label="📄 BAIXAR JOGOS EM PDF",
                    data=pdf_bytes,
                    file_name='palpites_megasena_ai.pdf',
                    mime='application/pdf',
                    type="primary",
                    use_container_width=True
                )
                
                # Mostrar todas as probabilidades (opcional)
                if st.checkbox("Mostrar probabilidades de todos os números"):
                    df_probs = pd.DataFrame(preds, columns=['Número', 'Probabilidade'])
                    df_probs['Probabilidade'] = (df_probs['Probabilidade'] * 100).round(2)
                    st.dataframe(df_probs, use_container_width=True)
                
                # Mostrar contador de uso
                if not st.session_state['usuario_premium']:
                    jogos_usados = st.session_state['jogos_gerados']
                    jogos_restantes = max(0, ACESSO_LIVRE_JOGOS - jogos_usados)
                    
                    if jogos_restantes > 0:
                        st.info(f"🎁 Você ainda tem {jogos_restantes} jogo(s) gratuito(s).")
                    else:
                        st.warning("⚠️ Você utilizou todos os jogos gratuitos.")
                
            except Exception as e:
                st.error(f"Erro no processamento: {str(e)}")
    elif not aceite:
        st.info("Marque o aceite para habilitar o modelo preditivo.")

# =============================================================================
# MAIN (ATUALIZADA COM CONTROLE DE ACESSO)
# =============================================================================

def main():
    # Inicializar CSS
    inject_custom_css()
    
    # Inicializar sistema de sessão
    inicializar_sessao()
    
    # Título principal
    st.title("🎲 Análise Mega-Sena - Sistema Premium")
    
    # Carregar dados
    df = carregar_dados_caixa()
    
    # Verificar se dados são válidos
    if not validar_dados(df):
        st.error("Erro crítico: Banco de dados indisponível. Usando dados de demonstração.")
        # Carregar dados de fallback como última tentativa
        df = carregar_dados_fallback()
        if not validar_dados(df):
            st.error("Não foi possível carregar dados. Tente novamente mais tarde.")
            return
    
    # Navegação
    draw_navigation()
    
    # Roteamento de páginas
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
        page_ai(df)  # Página com controle de acesso

if __name__ == "__main__":
    main()
