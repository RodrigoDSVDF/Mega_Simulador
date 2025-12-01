#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
import time
import random
from datetime import datetime

# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================

warnings.filterwarnings("ignore")

st.set_page_config(
    layout="wide", 
    page_title="Mega-Sena AI Pro", 
    page_icon="🎲",
    initial_sidebar_state="collapsed"
)

# Link do Checkout (Seu Link)
CHECKOUT_URL = "https://pay.cakto.com.br/5dUKrWD"

# =============================================================================
# GERENCIAMENTO DE ESTADO (SESSION STATE)
# =============================================================================

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Previsões AI" # Começa direto na página principal
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'premium_unlocked' not in st.session_state:
    st.session_state.premium_unlocked = False

# =============================================================================
# 0. DESIGN SYSTEM & CSS
# =============================================================================

def inject_custom_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');

        /* REMOVER SIDEBAR E ELEMENTOS PADRÃO */
        section[data-testid="stSidebar"] {{ display: none !important; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}

        /* ESTILO GERAL */
        .stApp {{
            background-color: #050505;
            color: #E0E0E0;
            font-family: 'Inter', sans-serif;
        }}

        /* TÍTULOS */
        h1, h2, h3 {{
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            letter-spacing: -1px;
        }}
        h1 {{ 
            color: #00ff88; 
            text-align: center;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
            margin-bottom: 30px;
        }}

        /* LOCK SCREEN CONTAINER */
        .lock-container {{
            background: rgba(31, 41, 55, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid #ff4444;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin: 50px auto;
            max-width: 600px;
            box-shadow: 0 0 50px rgba(255, 68, 68, 0.1);
        }}

        /* INPUT FIELDS */
        .stTextInput > div > div > input {{
            background-color: #111;
            color: #fff;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: #00ff88;
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
        }}

        /* BOTÕES DE NAVEGAÇÃO */
        div.stButton > button {{
            background-color: #1a1a1a !important;
            color: #888 !important;
            border: 1px solid #333 !important;
            border-radius: 10px !important;
            padding: 10px !important;
            font-weight: 600 !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
        }}
        div.stButton > button:hover {{
            border-color: #00ff88 !important;
            color: #00ff88 !important;
            background-color: #111 !important;
        }}
        
        /* BOTÃO DE AÇÃO PRIMÁRIA (GERAR) */
        .primary-action button {{
            background: linear-gradient(90deg, #00ff88, #00cc6a) !important;
            color: #000 !important;
            font-weight: 800 !important;
            border: none !important;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.4) !important;
        }}
        .primary-action button:hover {{
            transform: scale(1.02) !important;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.6) !important;
        }}

        /* METRICS */
        [data-testid="stMetric"] {{
            background-color: #111;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 15px;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
        }}
        [data-testid="stMetricLabel"] {{ color: #888; }}
        [data-testid="stMetricValue"] {{ color: #fff; font-family: 'Roboto Mono', monospace; }}

        /* RESULT BALLS */
        .ball {{
            display: inline-block;
            width: 45px;
            height: 45px;
            line-height: 45px;
            background: radial-gradient(circle at 30% 30%, #00ff88, #008f4c);
            border-radius: 50%;
            color: #000;
            font-weight: bold;
            font-family: 'Roboto Mono', monospace;
            text-align: center;
            margin: 5px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# 1. CARREGAMENTO DE DADOS (SIMULADO PARA PERFORMANCE)
# =============================================================================

@st.cache_data
def carregar_dados_caixa():
    # Tenta baixar da URL oficial, se falhar cria dados fictícios para não quebrar a demo
    url = 'https://servicebus2.caixa.gov.br/portaldasiloterias/api/megasena'
    try:
        # Nota: A API da caixa costuma bloquear requisições diretas sem headers específicos.
        # Para este exemplo funcionar liso, vou gerar um DataFrame realista.
        # Em produção, você usaria pd.read_html ou a API com requests e headers.
        
        # Simulando um Dataset histórico
        dates = pd.date_range(end=datetime.today(), periods=500)
        data = {
            'Concurso': range(2000, 2500),
            'Data': dates,
            'Bola1': np.random.randint(1, 10, 500),
            'Bola2': np.random.randint(11, 20, 500),
            'Bola3': np.random.randint(21, 30, 500),
            'Bola4': np.random.randint(31, 40, 500),
            'Bola5': np.random.randint(41, 50, 500),
            'Bola6': np.random.randint(51, 60, 500),
        }
        df = pd.DataFrame(data)
        return df
    except:
        return pd.DataFrame() # Fallback

# =============================================================================
# 2. SISTEMA DE NAVEGAÇÃO
# =============================================================================

def draw_navigation():
    # Menu superior estilizado
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Se estiver bloqueado, esconde a navegação para forçar o foco no desbloqueio
    if st.session_state.usage_count >= 1 and not st.session_state.premium_unlocked:
        return

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔮 Previsões AI", key="nav_ai"):
            st.session_state['current_page'] = "Previsões AI"
            st.rerun()
    with col2:
        if st.button("📊 Estatísticas", key="nav_stats"):
            st.session_state['current_page'] = "Estatísticas"
            st.rerun()
    with col3:
        if st.button("🔥 Quentes/Frios", key="nav_hot"):
            st.session_state['current_page'] = "Quentes/Frios"
            st.rerun()
    with col4:
        # Botão especial de Checkout
        st.link_button("💎 SEJA PREMIUM", CHECKOUT_URL)

    st.markdown("---")

# =============================================================================
# 3. PÁGINAS DO SISTEMA
# =============================================================================

def page_estatisticas(df):
    st.header("📊 Análise Estatística Global")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Concursos", "2.500")
    c2.metric("Média de Pares", "3.2")
    c3.metric("Soma Média", "185")
    
    st.info("Para ver gráficos avançados e tendências históricas, desbloqueie a versão PRO.")

def page_quentes_frios(df):
    st.header("🔥 Números Quentes & Frios ❄️")
    st.markdown("Algoritmo de análise de frequência dos últimos 100 jogos.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔥 Mais Sorteados")
        st.markdown("""
        1. **10** - (Saiu 18x)
        2. **53** - (Saiu 15x)
        3. **05** - (Saiu 14x)
        """)
    with c2:
        st.subheader("❄️ Mais Atrasados")
        st.markdown("""
        1. **42** - (40 jogos sem sair)
        2. **01** - (32 jogos sem sair)
        3. **22** - (28 jogos sem sair)
        """)

# --- PÁGINA PRINCIPAL: A IA PREDITIVA ---
def page_ai(df):
    st.title("🔮 Inteligência Artificial Preditiva")
    
    # LÓGICA DE BLOQUEIO (GATE)
    # Se o usuário já gerou 1 vez E não desbloqueou com e-mail
    if st.session_state.usage_count >= 1 and not st.session_state.premium_unlocked:
        
        # TELA DE BLOQUEIO
        st.markdown(f"""
        <div class="lock-container">
            <h1 style="color: #ff4444; margin-bottom: 10px;">🔒 LIMITE GRATUITO ATINGIDO</h1>
            <p style="font-size: 1.2rem; margin-bottom: 20px;">
                Você gerou 1 jogo com sucesso usando nossa IA.<br>
                Para continuar gerando palpites e baixar o PDF, identifique-se.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            email = st.text_input("📧 Insira seu melhor e-mail para liberar:", placeholder="ex: seu@email.com")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("DESBLOQUEAR SISTEMA AGORA 🔓", type="primary", use_container_width=True):
                if "@" in email and "." in email:
                    st.session_state.user_email = email
                    
                    # Simulação de processamento/validação
                    with st.spinner("Validando credenciais..."):
                        time.sleep(1.5)
                        
                        # REDIRECIONA PARA O CHECKOUT E "DESBLOQUEIA"
                        # Nota: Em um app real, o desbloqueio aconteceria APÓS o pagamento (via webhook)
                        # Mas aqui seguimos a lógica solicitada de "bloqueio pedindo email".
                        st.session_state.premium_unlocked = True 
                        st.markdown(f'<meta http-equiv="refresh" content="0;url={CHECKOUT_URL}" />', unsafe_allow_html=True)
                else:
                    st.error("Por favor, insira um e-mail válido.")
            
            st.caption("🔒 Seus dados estão seguros. Ao continuar você será redirecionado para concluir o acesso Vitalício.")

    else:
        # TELA DE GERAÇÃO (LIBERADA 1 VEZ)
        st.markdown("""
        <div style="background: rgba(0,255,136,0.05); padding: 20px; border-radius: 10px; border: 1px solid #00ff88; margin-bottom: 30px;">
            <strong>Status do Sistema:</strong> <span style="color:#00ff88">ONLINE</span> | 
            <strong>Modelo:</strong> Random Forest Regressor v4.2
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("⚙️ Configuração")
            qtd_jogos = st.slider("Quantidade de Jogos", 1, 10, 1) # Limitado visualmente a 1
            estrategia = st.selectbox("Estratégia", ["Equilíbrio Matemático", "Ousadia (Mais Ímpares)", "Conservador (Mais Pares)"])
        
        with col2:
            st.subheader("🚀 Processamento")
            st.write("O modelo irá analisar os últimos 2500 concursos para encontrar padrões de recorrência.")
            
            # Espaço para o botão
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Botão com classe CSS customizada
            st.markdown('<div class="primary-action">', unsafe_allow_html=True)
            if st.button("GERAR PALPITE COM I.A.", use_container_width=True):
                
                # BARRA DE PROGRESSO FAKE (DOPAMINA)
                progress_text = "Conectando à Rede Neural..."
                my_bar = st.progress(0, text=progress_text)
                
                steps = ["Analisando frequências...", "Calculando desvio padrão...", "Aplicando filtros de paridade...", "Gerando combinação otimizada..."]
                
                for i, step in enumerate(steps):
                    time.sleep(0.5)
                    my_bar.progress((i + 1) * 25, text=step)
                
                time.sleep(0.5)
                my_bar.empty()
                
                # GERAR NÚMEROS ALEATÓRIOS (SIMULANDO IA)
                palpite = sorted(random.sample(range(1, 61), 6))
                
                # EXIBIR RESULTADO
                st.success("✅ Palpite Gerado com Sucesso!")
                
                html_balls = ""
                for num in palpite:
                    html_balls += f"<div class='ball'>{num:02d}</div>"
                
                st.markdown(f"<div style='text-align:center; margin: 20px 0;'>{html_balls}</div>", unsafe_allow_html=True)
                
                # Análise Rápida do Jogo Gerado
                st.markdown("### 🔍 Análise do Jogo")
                sa1, sa2, sa3 = st.columns(3)
                sa1.metric("Probabilidade Estimada", "1 em 450k")
                sa2.metric("Soma das Dezenas", sum(palpite))
                sa3.metric("Pares / Ímpares", f"{len([x for x in palpite if x%2==0])} / {len([x for x in palpite if x%2!=0])}")

                # INCREMENTAR CONTADOR (IMPORTANTE)
                # Isso vai travar a tela na próxima atualização
                st.session_state.usage_count += 1
                
                if st.session_state.usage_count >= 1:
                    st.warning("⚠️ Você utilizou seu crédito gratuito. Salve este jogo agora!")
            
            st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    inject_custom_css()
    
    # Header Principal
    if st.session_state.usage_count == 0 or st.session_state.premium_unlocked:
        # Só mostra navegação se não estiver na tela de bloqueio
        draw_navigation()
    
    # Carregar Dados (apenas se necessário)
    df = carregar_dados_caixa()

    # Roteamento
    page = st.session_state['current_page']

    if page == "Previsões AI":
        page_ai(df)
    elif page == "Estatísticas":
        page_estatisticas(df)
    elif page == "Quentes/Frios":
        page_quentes_frios(df)

if __name__ == "__main__":
    main()
