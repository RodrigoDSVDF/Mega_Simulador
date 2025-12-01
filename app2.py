#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
import warnings
from datetime import datetime
from fpdf import FPDF

# Scikit-learn (Simulado na lógica, importado para estrutura)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

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

# --- MOCK DATA & CONFIGURAÇÕES DE NEGÓCIO ---
# Substitua o link abaixo pelo seu link real do Stripe/Hotmart/Eduzz
LINK_COMPRA = "https://seulinkdepagamento.com.br/checkout-vip"
EMAILS_PREMIUM_DB = ["vip@usuario.com", "admin@teste.com", "cliente@premium.com"]

# Constantes
COLUNAS_BOLAS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']

# =============================================================================
# 1. FUNÇÕES AUXILIARES (BACKEND SIMULADO)
# =============================================================================

@st.cache_data
def carregar_dados_caixa():
    """Gera um DataFrame simulado para o código funcionar sem arquivo externo."""
    np.random.seed(42)
    n_sorteios = 200
    data = {
        'Concurso': np.arange(2500, 2500 + n_sorteios),
        'Data': [datetime.today()] * n_sorteios,
    }
    # Gera sorteios aleatórios (simulando a Mega)
    bolas = []
    for _ in range(n_sorteios):
        sorteio = sorted(np.random.choice(range(1, 61), 6, replace=False))
        bolas.append(sorteio)
    
    df_bolas = pd.DataFrame(bolas, columns=COLUNAS_BOLAS)
    return pd.concat([pd.DataFrame(data), df_bolas], axis=1)

def gerar_pdf_bytes(jogos):
    """Gera um PDF simples com os jogos para download."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Palpites Mega-Sena AI", ln=1, align="C")
    pdf.ln(10)
    
    for i, jogo in enumerate(jogos, 1):
        txt_jogo = f"Jogo {i}: " + " - ".join(map(str, jogo))
        pdf.cell(0, 10, txt=txt_jogo, ln=1)
        
    return pdf.output(dest='S').encode('latin-1') # Retorna bytes

# Lógica Simulada de IA
def treinar_modelo_avancado(df, usar_amostragem):
    time.sleep(1.5) # Simula tempo de treino
    return "modelo_mock", "scaler_mock", df

def gerar_previsoes_avancadas(df, mod, scl):
    # Retorna uma lista de tuplas (Número, Probabilidade)
    numeros = list(range(1, 61))
    probs = np.random.uniform(0.01, 0.99, 60)
    probs = probs / probs.sum() # Normaliza
    dados = list(zip(numeros, probs))
    return sorted(dados, key=lambda x: x[1], reverse=True)

def gerar_combinacoes_avancadas(preds, n_comb, diversificar):
    # Gera combinações baseadas nos tops, garantindo aleatoriedade controlada
    top_nums = [x[0] for x in preds[:20]]
    jogos = []
    for _ in range(n_comb):
        # Pega 6 números aleatórios dos top 20 para variar
        jogo = sorted(np.random.choice(top_nums, 6, replace=False).tolist())
        jogos.append(jogo)
    return jogos

# =============================================================================
# 2. GESTÃO DE ESTADO (SESSION STATE)
# =============================================================================

def inicializar_session_state():
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'Visão Geral'
    
    # Controle de tentativas Free
    if 'geracoes_realizadas' not in st.session_state:
        st.session_state['geracoes_realizadas'] = 0
    
    # Status Premium
    if 'is_premium' not in st.session_state:
        st.session_state['is_premium'] = False
        
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = None

def verificar_login(email):
    """Valida email na base mockada."""
    email_clean = email.lower().strip()
    if email_clean in EMAILS_PREMIUM_DB:
        st.session_state['is_premium'] = True
        st.session_state['user_email'] = email_clean
        return True
    return False

# =============================================================================
# 3. DESIGN SYSTEM & CSS
# =============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
    /* REMOVER SIDEBAR E ELEMENTOS PADRÃO */
    section[data-testid="stSidebar"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ESTILO GERAL */
    .stApp { background-color: #0E1117; color: #E0E0E0; }

    /* CONTAINER DE NAVEGAÇÃO */
    .nav-container {
        display: flex;
        justify_content: center;
        gap: 15px;
        padding: 10px;
        background-color: #1F2937;
        border-radius: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    
    /* BADGES */
    .premium-badge { background-color: #00C896; color: #000; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    .free-badge { background-color: #E65100; color: #fff; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    
    /* BLOQUEIO (LOCK SCREEN) */
    .lock-screen {
        background-color: #1F2937;
        padding: 40px;
        border-radius: 20px;
        border: 2px solid #374151;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* BOTÃO DE COMPRA ESTILIZADO */
    .btn-buy {
        background-color: #00C896; color: white; padding: 15px 32px;
        text-align: center; text-decoration: none; display: inline-block;
        font-size: 16px; margin: 10px 0; cursor: pointer; border: none;
        border-radius: 8px; width: 100%; font-weight: bold;
        transition: 0.3s;
    }
    .btn-buy:hover { background-color: #00a87e; transform: scale(1.02); }
    
    /* NUMEROS DA LOTERIA */
    .lotto-number {
        display: inline-block; width: 35px; height: 35px;
        background-color: #00C896; color: #000; border-radius: 50%;
        text-align: center; line-height: 35px; font-weight: bold; margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

def draw_navigation():
    """Desenha o menu superior personalizado."""
    pages = ["Visão Geral", "Frequência", "Pares/Impares", "Quentes/Frios", "∑ Somas", "Previsões AI"]
    
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    cols = st.columns(len(pages))
    for i, page_name in enumerate(pages):
        if cols[i].button(page_name, key=f"nav_{i}", use_container_width=True):
            st.session_state['current_page'] = page_name
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 4. PÁGINAS DE ANÁLISE (Resumidas para focar na IA)
# =============================================================================

def page_visao_geral(df):
    st.header("📊 Visão Geral")
    st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

def page_frequencia(df):
    st.header("📈 Frequência dos Números")
    # Logica simples de contagem
    flat_list = df[COLUNAS_BOLAS].values.flatten()
    counts = pd.Series(flat_list).value_counts().reset_index()
    counts.columns = ['Número', 'Qtd']
    st.altair_chart(alt.Chart(counts.head(20)).mark_bar().encode(
        x=alt.X('Número:O', sort='-y'), y='Qtd'
    ), use_container_width=True)

def page_pares_impares(df):
    st.header("⚖️ Pares e Ímpares")
    st.info("Página de exemplo: Análise de paridade.")

def page_quentes(df):
    st.header("🔥❄️ Quentes e Frios")
    st.info("Página de exemplo: Números mais sorteados recentemente.")

def page_somas(df):
    st.header("∑ Análise das Somas")
    st.info("Página de exemplo: Distribuição Normal das somas.")

# =============================================================================
# 5. PÁGINA "PREVISÕES AI" (LÓGICA DE BLOQUEIO IMPLEMENTADA)
# =============================================================================

def page_ai(df):
    st.header("🤖 Inteligência Artificial Preditiva")

    # ----------------------------------------------------
    # HEADER COM STATUS
    # ----------------------------------------------------
    col_status, col_blank = st.columns([2, 3])
    with col_status:
        if st.session_state['is_premium']:
            st.markdown(f'<div class="premium-badge">👑 VIP: {st.session_state["user_email"]}</div>', unsafe_allow_html=True)
        else:
            usados = st.session_state['geracoes_realizadas']
            st.markdown(f'<div class="free-badge">🆓 MODO GRATUITO: {usados}/1 Jogo Gerado</div>', unsafe_allow_html=True)
    
    st.divider()

    # ----------------------------------------------------
    # VERIFICAÇÃO DE PERMISSÃO (GATEKEEPER)
    # ----------------------------------------------------
    pode_jogar = st.session_state['is_premium'] or (st.session_state['geracoes_realizadas'] < 1)

    # SEÇÃO 1: CONFIGURAÇÃO (Visível mas desabilitada se bloqueado)
    st.markdown("#### Configuração do Modelo")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_comb = st.slider("Qtd. Jogos:", 1, 10, 3, disabled=not pode_jogar)
    with c2:
        div = st.checkbox("Diversificar", True, disabled=not pode_jogar)
    with c3:
        amostra = st.checkbox("Modo Rápido", True, disabled=not pode_jogar)

    st.markdown("---")

    # SEÇÃO 2: LÓGICA DE EXECUÇÃO VS BLOQUEIO
    if pode_jogar:
        # --- USUÁRIO LIBERADO ---
        aceite = st.checkbox("✅ Entendo que loteria é um jogo de azar e não há garantias.")
        
        if aceite:
            if st.button("🚀 TREINAR IA E GERAR PALPITE", type="primary", use_container_width=True):
                
                # 1. Incrementa contador se for Free
                if not st.session_state['is_premium']:
                    st.session_state['geracoes_realizadas'] += 1
                
                # 2. Processamento
                with st.spinner("Calibrando Redes Neurais e analisando padrões..."):
                    mod, scl, df_s = treinar_modelo_avancado(df, amostra)
                    preds = gerar_previsoes_avancadas(df_s, mod, scl)
                    jogos = gerar_combinacoes_avancadas(preds, n_comb, div)
                    
                    # 3. Exibição dos Resultados
                    st.success("Cálculos Finalizados!")
                    
                    st.subheader("💡 Seus Palpites Otimizados")
                    for i, jogo in enumerate(jogos, 1):
                        html_balls = "".join([f'<span class="lotto-number">{n}</span>' for n in jogo])
                        st.markdown(f"**Jogo {i}:** {html_balls}", unsafe_allow_html=True)
                        st.caption(f"Soma: {sum(jogo)} | Pares: {len([x for x in jogo if x%2==0])}")

                    # 4. Botão de Download PDF
                    pdf_data = gerar_pdf_bytes(jogos)
                    st.download_button("📄 BAIXAR PDF", data=pdf_data, file_name="palpites_ai.pdf", mime="application/pdf", use_container_width=True)

                    # 5. Se for Free, força refresh após alguns segundos para bloquear
                    if not st.session_state['is_premium']:
                        st.warning("⚠️ Você utilizou seu jogo gratuito. O sistema será bloqueado em 5 segundos.")
                        time.sleep(5)
                        st.rerun()

        else:
            st.info("Por favor, marque o aceite acima para desbloquear o botão.")

    else:
        # --- USUÁRIO BLOQUEADO (TELA DE VENDA) ---
        st.markdown("""
        <div class="lock-screen">
            <h1>🔒 Limite Gratuito Atingido</h1>
            <p style="font-size: 1.2em;">Você já gerou sua previsão gratuita de hoje.</p>
            <p>Para acesso ilimitado, análises profundas e exportação PDF, torne-se Premium.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_login, col_venda = st.columns([1, 1])
        
        # Coluna Login
        with col_login:
            st.markdown("### 🔑 Já sou Cliente")
            with st.form("frm_login"):
                email_input = st.text_input("Seu e-mail de compra:")
                btn_log = st.form_submit_button("Desbloquear")
                if btn_log:
                    if verificar_login(email_input):
                        st.success("Login efetuado! Recarregando...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("E-mail não encontrado.")

        # Coluna Venda
        with col_venda:
            st.markdown("### 💎 Quero Acesso Total")
            st.markdown("""
            - ✅ Gerações Ilimitadas
            - ✅ Filtros Avançados
            - ✅ Download em PDF
            - ✅ Suporte Prioritário
            """)
            st.markdown(f'<a href="{LINK_COMPRA}" target="_blank"><button class="btn-buy">🛒 COMPRAR AGORA</button></a>', unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    inicializar_session_state()
    inject_custom_css()
    
    st.title("🎲 Mega-Sena Analytics Pro")
    
    # Carregar Dados
    df = carregar_dados_caixa()
    
    # Navegação
    draw_navigation()
    
    # Roteamento
    page = st.session_state['current_page']
    
    if page == "Visão Geral":
        page_visao_geral(df)
    elif page == "Frequência":
        page_frequencia(df)
    elif page == "Pares/Impares":
        page_pares_impares(df)
    elif page == "Quentes/Frios":
        page_quentes(df)
    elif page == "∑ Somas":
        page_somas(df)
    elif page == "Previsões AI":
        page_ai(df)

if __name__ == "__main__":
    main()
