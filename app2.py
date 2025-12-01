# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import warnings
import time
from fpdf import FPDF
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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
# 1. DESIGN SYSTEM & CSS
# =============================================================================
def inject_custom_css():
    st.markdown(
        f"""
        <style>
        /* 1. REMOVER SIDEBAR E ELEMENTOS PADRÃO */
        section[data-testid="stSidebar"] {{ display: none !important; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}

        /* 2. ESTILO GERAL */
        .stApp {{
            background-color: #0E1117;
            color: #E0E0E0;
        }}

        /* 3. GATE PREMIUM (BLOQUEIO) */
        .premium-gate {{
            background: linear-gradient(145deg, #1F2937, #111827);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border: 2px solid #00C896;
            margin-top: 20px;
            text-align: center;
        }}
        .premium-gate h3 {{ color: #00C896 !important; }}
        .premium-gate p {{ color: #ccc; margin-bottom: 20px; }}

        /* 4. RESULTADO CARD */
        .game-card {{
            background-color: #1F2937;
            border: 1px solid #374151;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .game-numbers {{
            font-size: 28px;
            font-weight: bold;
            color: #00C896;
            letter-spacing: 3px;
        }}

        /* 5. BOTÕES */
        div.stButton > button {{
            background-color: #1F2937 !important;
            color: #9CA3AF !important;
            border: 1px solid #374151 !important;
            border-radius: 12px !important;
            width: 100% !important;
        }}
        div.stButton > button:hover {{
            border-color: #00C896 !important;
            color: #00C896 !important;
        }}
        div.stButton > button[kind="primary"] {{
            border-color: #00C896 !important;
            color: #00C896 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# 2. FUNÇÕES AUXILIARES E DE DADOS (SIMULADAS PARA EXECUÇÃO)
# =============================================================================

@st.cache_data
def carregar_dados_caixa():
    # Simulação de dados para o exemplo funcionar
    data = {
        'Concurso': range(1, 101),
        'Data': pd.date_range(start='2023-01-01', periods=100),
    }
    for col in COLUNAS_BOLAS:
        data[col] = np.random.randint(1, 61, 100)
    return pd.DataFrame(data)

def validar_dados(df):
    return not df.empty

def gerar_pdf_bytes(games):
    # Gera um PDF simples com os jogos
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Palpites Mega-Sena AI", ln=1, align="C")
    pdf.ln(10)
    for i, game in enumerate(games):
        game_str = " - ".join([str(n).zfill(2) for n in game])
        pdf.cell(200, 10, txt=f"Jogo {i+1}: {game_str}", ln=1, align="C")
    return pdf.output(dest='S').encode('latin-1')

# =============================================================================
# 3. LÓGICA ML (SIMPLIFICADA DO SEU CÓDIGO)
# =============================================================================

# (Mantenha suas funções de ML originais aqui. 
# Para o exemplo rodar, criei esta função de simulação)
def gerar_jogo_simulado_ai():
    import random
    numbers = sorted(random.sample(range(1, 61), 6))
    return numbers

# =============================================================================
# 4. PÁGINAS DO SISTEMA
# =============================================================================

def page_visao_geral(df):
    st.header("📊 Visão Geral")
    st.info("Estatísticas gerais dos sorteios carregados.")
    st.dataframe(df.tail(10), use_container_width=True)

def page_frequencia(df): st.info("Análise de Frequência (Em desenvolvimento)")
def page_pares_impares(df): st.info("Análise Pares/Ímpares (Em desenvolvimento)")
def page_combinacoes(df): st.info("Análise de Combinações (Em desenvolvimento)")
def page_quentes(df): st.info("Números Quentes e Frios (Em desenvolvimento)")
def page_somas(df): st.info("Análise de Somas (Em desenvolvimento)")

# =============================================================================
# 5. PÁGINA GERADOR COM BLOQUEIO (CRÍTICO)
# =============================================================================

def page_ai(df):
    st.title("🤖 Gerador de Jogos com IA")
    
    # --- CONTROLE DE ESTADO (SESSION STATE) ---
    if 'games_generated_count' not in st.session_state:
        st.session_state['games_generated_count'] = 0
        
    if 'generated_games_history' not in st.session_state:
        st.session_state['generated_games_history'] = []

    if 'is_premium' not in st.session_state:
        st.session_state['is_premium'] = False  # Mude para True via callback de pagamento real

    col_config, col_result = st.columns([1, 2])

    with col_config:
        st.subheader("Configuração")
        st.write("O modelo utiliza regressão logística calibrada para identificar padrões.")
        
        # LÓGICA DE BLOQUEIO
        # Permite gerar se for premium OU se contagem < 1
        pode_gerar = st.session_state['is_premium'] or (st.session_state['games_generated_count'] < 1)

        if pode_gerar:
            st.success(f"Jogos gratuitos restantes: {1 - st.session_state['games_generated_count']}")
            if st.button("🔮 GERAR JOGO OTIMIZADO", type="primary"):
                with st.spinner("Processando dados históricos..."):
                    time.sleep(1.5) # Simula processamento
                    novo_jogo = gerar_jogo_simulado_ai()
                    
                    # Salva no histórico e incrementa contador
                    st.session_state['generated_games_history'].insert(0, novo_jogo)
                    st.session_state['games_generated_count'] += 1
                    st.rerun()
        else:
            # --- TELA DE BLOQUEIO (GATE) ---
            st.markdown("""
            <div class="premium-gate">
                <h3>🔒 Limite Atingido</h3>
                <p>Você utilizou sua geração gratuita de teste.</p>
                <p style="font-size:0.9rem">Desbloqueie a versão PRO para:</p>
                <ul style="text-align:left; color:#bbb; margin-bottom:20px">
                    <li>Gerações Ilimitadas</li>
                    <li>Download em PDF</li>
                    <li>Acesso aos algoritmos avançados</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Botão de Pagamento
            st.link_button(
                label="🔓 DESBLOQUEAR AGORA (R$ 19,90)", 
                url="https://seulinkdepagamento.com.br", # <--- COLOQUE SEU LINK AQUI
                type="primary",
                use_container_width=True
            )

    with col_result:
        st.subheader("Jogos Gerados")
        
        history = st.session_state['generated_games_history']
        
        if history:
            # Exibe o jogo mais recente com destaque
            latest = history[0]
            str_nums = " - ".join([f"{n:02d}" for n in latest])
            
            st.markdown(f"""
            <div class="game-card">
                <div style="font-size:14px; color:#aaa; margin-bottom:5px">PALPITE GERADO PELA IA</div>
                <div class="game-numbers">{str_nums}</div>
            </div>
            """, unsafe_allow_html=True)

            # Estatísticas do jogo
            soma = sum(latest)
            pares = sum(1 for x in latest if x % 2 == 0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Soma", soma)
            c2.metric("Pares", pares)
            c3.metric("Ímpares", 6 - pares)
            
            st.divider()
            
            # Botão de PDF (Disponível apenas para o jogo gerado ou bloqueado também?)
            # Aqui deixei disponível para baixar o jogo gratuito
            pdf_data = gerar_pdf_bytes([latest])
            st.download_button(
                label="📄 Baixar Jogo em PDF",
                data=pdf_data,
                file_name="jogo_mega_ai.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        else:
            st.info("Clique em Gerar para iniciar a análise.")

# =============================================================================
# 6. NAVEGAÇÃO E MAIN
# =============================================================================

def draw_navigation():
    pages = ["Visão Geral", "Frequência", "Pares/Impares", "Combinações", "Quentes/Frios", "∑ Somas", "Previsões AI"]
    
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Previsões AI" # Inicia na página da AI para teste
        
    cols = st.columns(len(pages))
    for i, page_name in enumerate(pages):
        # Destaca o botão da página atual
        is_active = st.session_state['current_page'] == page_name
        if cols[i].button(
            page_name.replace(" ", "\n"), 
            key=f"nav_{i}", 
            type="primary" if is_active else "secondary"
        ):
            st.session_state['current_page'] = page_name
            st.rerun()

def main():
    inject_custom_css()
    
    st.markdown("# 🎲 Mega-Sena Analyzer Pro")
    
    # Carregar dados
    df = carregar_dados_caixa()
    if not validar_dados(df):
        st.error("Erro ao carregar banco de dados.")
        return

    # Navegação
    draw_navigation()
    st.markdown("---")

    # Roteamento
    pg = st.session_state['current_page']
    
    if pg == "Visão Geral": page_visao_geral(df)
    elif pg == "Frequência": page_frequencia(df)
    elif pg == "Pares/Impares": page_pares_impares(df)
    elif pg == "Combinações": page_combinacoes(df)
    elif pg == "Quentes/Frios": page_quentes(df)
    elif pg == "∑ Somas": page_somas(df)
    elif pg == "Previsões AI": page_ai(df)

if __name__ == "__main__":
    main()

