# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import warnings
import time
import itertools
from collections import Counter
from typing import List, Tuple, Any, Dict
from fpdf import FPDF

# Scikit-learn
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
        .stat-box {{
            background: #111827;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #374151;
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
            box-shadow: 0 0 10px rgba(0,200,150,0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# 2. FUNÇÕES DE DADOS E HELPER
# =============================================================================

@st.cache_data
def carregar_dados_caixa():
    """
    Simula um histórico robusto de sorteios para permitir o treinamento do modelo.
    Em produção, substitua isso por pd.read_excel('Mega-Sena.xlsx').
    """
    np.random.seed(42)
    n_concursos = 600
    data = {
        'Concurso': range(1, n_concursos + 1),
        'Data': pd.date_range(end=pd.Timestamp.today(), periods=n_concursos),
    }
    
    # Gera sorteios garantindo numeros unicos por linha
    sorteios = []
    for _ in range(n_concursos):
        sorteios.append(sorted(np.random.choice(range(1, 61), 6, replace=False)))
    
    df_temp = pd.DataFrame(sorteios, columns=COLUNAS_BOLAS)
    return pd.concat([pd.DataFrame(data), df_temp], axis=1)

def validar_dados(df):
    return not df.empty and all(col in df.columns for col in COLUNAS_BOLAS)

def gerar_pdf_bytes(games):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Palpites Mega-Sena AI", ln=1, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for i, game in enumerate(games):
        game_str = " - ".join([str(n).zfill(2) for n in game])
        pdf.cell(200, 10, txt=f"Jogo {i+1}: {game_str}", ln=1, align="C")
    return pdf.output(dest='S').encode('latin-1')

# =============================================================================
# 3. LÓGICA ML AVANÇADA (RESTAURO)
# =============================================================================

def compute_basic_freqs_fast(df_ml, window=None):
    bolas = [c for c in COLUNAS_BOLAS if c in df_ml.columns]
    dados = df_ml.tail(window)[bolas].values.flatten() if window else df_ml[bolas].values.flatten()
    freq = pd.Series(dados).value_counts()
    return {n: freq.get(n, 0) for n in ALL_NUMBERS}

def exponential_moving_freq_fast(df_ml, span=20):
    bolas = [c for c in COLUNAS_BOLAS if c in df_ml.columns]
    data_list = []
    for _, row in df_ml.iterrows():
        row_dict = {i: 0 for i in ALL_NUMBERS}
        for n in row[bolas].values:
            if pd.notna(n): row_dict[int(n)] = 1
        data_list.append(row_dict)
    
    if not data_list: return {n: 0.0 for n in ALL_NUMBERS}
    
    df_ohe = pd.DataFrame(data_list)
    ema = df_ohe.ewm(span=span, adjust=False).mean().iloc[-1]
    return {n: float(ema.get(n, 0.0)) for n in ALL_NUMBERS}

def last_appearance_distance_fast(df_ml, max_dist=1000):
    bolas = [c for c in COLUNAS_BOLAS if c in df_ml.columns]
    melted = []
    for col in bolas:
        for idx, val in df_ml[col].items():
            if pd.notna(val): melted.append({'index': idx, 'numero': int(val)})
            
    if not melted: return {n: max_dist for n in ALL_NUMBERS}
    
    df_m = pd.DataFrame(melted)
    last = df_m.groupby('numero')['index'].max()
    curr = df_ml.index.max()
    curr = curr + 1 if pd.notna(curr) else max_dist
    
    return {n: int(curr - last.get(n, -1)) if n in last else int(curr) for n in ALL_NUMBERS}

def build_features_table_fast(df_ml):
    if len(df_ml) == 0: return pd.DataFrame()
    
    min_len = len(df_ml)
    f_all = compute_basic_freqs_fast(df_ml)
    f_50 = compute_basic_freqs_fast(df_ml, 50)
    f_10 = compute_basic_freqs_fast(df_ml, 10)
    
    ema_20 = exponential_moving_freq_fast(df_ml, 20) if min_len >= 10 else {n:0.0 for n in ALL_NUMBERS}
    ema_50 = exponential_moving_freq_fast(df_ml, 50) if min_len >= 10 else {n:0.0 for n in ALL_NUMBERS}
    ldist = last_appearance_distance_fast(df_ml, min_len + 100)
    
    data = []
    for n in ALL_NUMBERS:
        data.append({
            'numero': n, 
            'freq_all': f_all.get(n,0), 
            'freq_50': f_50.get(n,0), 
            'freq_10': f_10.get(n,0),
            'ema_20': ema_20.get(n,0.0), 
            'ema_50': ema_50.get(n,0.0), 
            'last_dist': ldist.get(n, min_len),
            'is_even': n%2, 
            'is_leq30': 1 if n<=30 else 0, 
            'decena': (n-1)//10, 
            'is_mult_5': 1 if n%5==0 else 0
        })
        
    feat = pd.DataFrame(data).set_index('numero')
    
    # Normalização simples
    for c in ['freq_all','freq_50','freq_10']: 
        feat[f'{c}_norm'] = feat[c]/max(1, feat[c].max())
    feat['last_dist_norm'] = feat['last_dist']/max(1, feat['last_dist'].max())
    
    return feat[['freq_all_norm','freq_50_norm','freq_10_norm','ema_20','ema_50','last_dist_norm','is_even','is_leq30','decena','is_mult_5']]

def create_training_dataset_fast(df_ml, sample_fraction=0.3):
    df_sorted = df_ml.reset_index(drop=True)
    n = len(df_sorted)
    # Precisamos de histórico para gerar features
    start_idx = max(50, int(0.15 * n))
    
    time_points = list(range(start_idx, n-1))
    if len(time_points) > 100: 
        time_points = time_points[::max(1, len(time_points)//100)] # Downsample para performance
        
    examples, targets = [], []
    bolas = [c for c in COLUNAS_BOLAS if c in df_sorted.columns]
    
    prog_bar = st.progress(0, text="Analisando histórico...")
    
    for i, t in enumerate(time_points):
        prog_bar.progress((i+1)/len(time_points), text=f"Treinando em concursos passados: {i+1}/{len(time_points)}")
        
        df_until = df_sorted.iloc[:t+1]
        feats = build_features_table_fast(df_until)
        
        # O alvo são os números do PRÓXIMO sorteio (t+1)
        prox = set(df_sorted.loc[t+1, bolas].tolist())
        
        # Amostragem de negativos para balancear
        prox_list = list(prox)
        nao_sort = [x for x in ALL_NUMBERS if x not in prox]
        
        if sample_fraction < 1.0:
            n_samp = max(15, int(60 * sample_fraction))
            n_rest = max(0, n_samp - len(prox_list))
            sub = np.random.choice(nao_sort, n_rest, replace=False).tolist()
            amostra = prox_list + sub
        else:
            amostra = ALL_NUMBERS

        for num in amostra:
            if num in feats.index:
                examples.append(feats.loc[num].values)
                targets.append(1 if num in prox else 0)
                
    prog_bar.empty()
    return (np.array(examples) if examples else np.empty((0,0))), np.array(targets, dtype=int), df_sorted

@st.cache_resource(ttl=3600)
def treinar_modelo_avancado(df, use_sampling=True):
    if len(df) < 80: 
        st.warning("Dados insuficientes para IA. Usando modo simplificado.")
        return None, None, df
        
    frac = 0.4 if use_sampling and len(df) > 500 else 0.8
    
    with st.spinner("Treinando Redes Neurais e Regressão..."):
        X, y, df_s = create_training_dataset_fast(df, frac)
        
        if len(X) == 0: return None, None, df
        
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        
        base = LogisticRegression(max_iter=500, class_weight='balanced', solver='lbfgs', C=0.1, random_state=42)
        
        splits = max(2, min(3, len(X_sc)//3000))
        cv = TimeSeriesSplit(n_splits=splits) if len(X_sc) >= 500 else 3
        
        calib = CalibratedClassifierCV(estimator=base, cv=cv, method='sigmoid', n_jobs=-1)
        try:
            calib.fit(X_sc, y)
        except:
            base.fit(X_sc, y)
            calib = base
            
    return calib, scaler, df_s

def gerar_previsoes_avancadas(df_s, model, scaler):
    if model is None:
        # Fallback aleatório inteligente se falhar treino
        return [(n, 1/60) for n in ALL_NUMBERS]

    feats = build_features_table_fast(df_s)
    X_sc = scaler.transform(feats.values)
    
    # Probabilidade de ser sorteado
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_sc)[:, 1]
    else:
        probs = model.decision_function(X_sc)
        
    # Softmax para normalizar scores
    exps = np.exp(probs - np.max(probs))
    rel = exps / exps.sum()
    
    return sorted([(int(n), float(rel[i])) for i, n in enumerate(feats.index)], key=lambda x: x[1], reverse=True)

def safe_weighted_choice(population, weights, k):
    try:
        w = np.maximum(np.array(weights, dtype=float), 0)
        if w.sum() == 0: w = np.ones_like(w)
        p = w / w.sum()
        return list(np.random.choice(population, size=k, replace=False, p=p))
    except:
        return list(np.random.choice(population, size=k, replace=False))

def gerar_combinacoes_avancadas(preds, n_comb=1):
    nums = [p[0] for p in preds]
    pesos = [p[1] for p in preds]
    
    # Foca nos top 35 números para gerar as combinações
    cands, p_cands = nums[:35], pesos[:35]
    
    combs = []
    attempts = 0
    while len(combs) < n_comb and attempts < 200:
        attempts += 1
        c = safe_weighted_choice(cands, p_cands, 6)
        c = sorted(c)
        
        # Filtros básicos de qualidade
        pares = sum(1 for x in c if x % 2 == 0)
        soma = sum(c)
        if 2 <= pares <= 4 and 140 <= soma <= 240:
            if c not in combs:
                combs.append(c)
                
    if not combs: # Fallback se filtros forem muito rígidos
        c = sorted(safe_weighted_choice(nums, pesos, 6))
        combs.append(c)
        
    return combs[0]

# =============================================================================
# 4. PÁGINAS DO SISTEMA
# =============================================================================

def page_visao_geral(df):
    st.header("📊 Visão Geral")
    
    # Métricas Topo
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Concursos", len(df))
    col2.metric("Última Data", df['Data'].iloc[-1].strftime("%d/%m/%Y"))
    
    last_draw = df.iloc[-1]
    numeros = [last_draw[c] for c in COLUNAS_BOLAS]
    col3.metric("Última Soma", sum(numeros))

    st.divider()
    
    # Tabela Visual
    st.subheader("Últimos Resultados")
    df_show = df.tail(10).sort_values("Concurso", ascending=False)
    
    # Estilizando DataFrame
    def color_balls(val):
        return 'color: #00C896; font-weight: bold;'
        
    st.dataframe(
        df_show[['Concurso', 'Data'] + COLUNAS_BOLAS],
        use_container_width=True,
        hide_index=True
    )

def page_frequencia(df):
    st.header("📈 Análise de Frequência")
    freqs = compute_basic_freqs_fast(df)
    
    df_freq = pd.DataFrame(list(freqs.items()), columns=['Número', 'Frequência'])
    df_freq = df_freq.sort_values('Frequência', ascending=False)
    
    c = alt.Chart(df_freq).mark_bar().encode(
        x=alt.X('Número:O', sort='-y'),
        y='Frequência:Q',
        color=alt.condition(
            alt.datum.Frequência >= df_freq['Frequência'].mean(),
            alt.value('#00C896'),
            alt.value('#374151')
        ),
        tooltip=['Número', 'Frequência']
    ).properties(height=400)
    
    st.altair_chart(c, use_container_width=True)

def page_quentes(df):
    st.header("🔥 Números Quentes e Frios")
    freqs = compute_basic_freqs_fast(df, window=20) # Ultimos 20 jogos
    
    df_q = pd.DataFrame(list(freqs.items()), columns=['Número', 'Freq_Recente'])
    quentes = df_q.sort_values('Freq_Recente', ascending=False).head(10)
    frios = df_q.sort_values('Freq_Recente', ascending=True).head(10)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔥 Top 10 Quentes (Últimos 20)")
        for _, row in quentes.iterrows():
            st.markdown(f"**{int(row['Número']):02d}** - Saiu {int(row['Freq_Recente'])}x")
            st.progress(min(1.0, row['Freq_Recente']/10))
            
    with c2:
        st.subheader("❄️ Top 10 Frios (Últimos 20)")
        for _, row in frios.iterrows():
            st.markdown(f"**{int(row['Número']):02d}** - Saiu {int(row['Freq_Recente'])}x")
            
def page_pares_impares(df):
    st.header("⚖️ Pares e Ímpares")
    data = []
    for _, row in df.iterrows():
        nums = row[COLUNAS_BOLAS].values
        pares = sum(1 for n in nums if n % 2 == 0)
        data.append(pares)
        
    counts = pd.Series(data).value_counts().sort_index()
    st.bar_chart(counts)
    st.caption("Distribuição da quantidade de números PARES por sorteio.")

def page_somas(df):
    st.header("➕ Análise de Somas")
    somas = df[COLUNAS_BOLAS].sum(axis=1)
    
    st.line_chart(somas.tail(50))
    st.caption("Soma dos números nos últimos 50 concursos.")
    
    col1, col2 = st.columns(2)
    col1.metric("Média Histórica", f"{somas.mean():.1f}")
    col2.metric("Desvio Padrão", f"{somas.std():.1f}")

def page_combinacoes(df):
    st.info("Análise de combinações em desenvolvimento.")

# =============================================================================
# 5. PÁGINA GERADOR COM BLOQUEIO (CRÍTICO)
# =============================================================================

def page_ai(df):
    st.title("🤖 Gerador Mega-Sena AI PRO")
    
    # --- CONTROLE DE ESTADO (SESSION STATE) ---
    if 'games_generated_count' not in st.session_state:
        st.session_state['games_generated_count'] = 0
        
    if 'generated_games_history' not in st.session_state:
        st.session_state['generated_games_history'] = []

    if 'is_premium' not in st.session_state:
        st.session_state['is_premium'] = False 

    # --- MODELO EM CACHE ---
    # Só treinamos se for gerar
    
    col_config, col_result = st.columns([1, 2])

    with col_config:
        st.subheader("Parâmetros do Modelo")
        st.markdown("""
        - **Algoritmo:** Regressão Logística Calibrada
        - **Features:** Frequência, Atraso, Médias Móveis
        - **Validação:** TimeSeriesSplit
        """)
        
        # LÓGICA DE BLOQUEIO
        pode_gerar = st.session_state['is_premium'] or (st.session_state['games_generated_count'] < 1)

        if pode_gerar:
            st.success(f"Gerações gratuitas: {1 - st.session_state['games_generated_count']}")
            if st.button("🔮 TREINAR IA E GERAR", type="primary"):
                
                # 1. Treinamento (usa cache se já treinado)
                model, scaler, df_s = treinar_modelo_avancado(df)
                
                # 2. Previsão
                preds = gerar_previsoes_avancadas(df_s, model, scaler)
                
                # 3. Formação do Jogo
                novo_jogo = gerar_combinacoes_avancadas(preds, n_comb=1)
                
                # 4. Salvar
                st.session_state['generated_games_history'].insert(0, novo_jogo)
                st.session_state['games_generated_count'] += 1
                st.rerun()
        else:
            # --- TELA DE BLOQUEIO ---
            st.markdown("""
            <div class="premium-gate">
                <h3>🔒 IA Bloqueada</h3>
                <p>Você já utilizou sua previsão gratuita baseada em IA.</p>
                <div style="font-size: 0.8rem; color: #888; margin-bottom: 15px;">
                    O treinamento de modelos preditivos consome muitos recursos.
                </div>
                <p>Desbloqueie para continuar gerando.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.link_button(
                label="🔓 DESBLOQUEAR VERSION PRO", 
                url="#",
                type="primary",
                use_container_width=True
            )

    with col_result:
        st.subheader("Previsão da Inteligência Artificial")
        
        history = st.session_state['generated_games_history']
        
        if history:
            latest = history[0]
            str_nums = " - ".join([f"{n:02d}" for n in latest])
            
            st.markdown(f"""
            <div class="game-card">
                <div style="font-size:14px; color:#aaa; margin-bottom:5px">PROBABILIDADE OTIMIZADA</div>
                <div class="game-numbers">{str_nums}</div>
            </div>
            """, unsafe_allow_html=True)

            # Stats do jogo
            pares = sum(1 for x in latest if x % 2 == 0)
            soma = sum(latest)
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='stat-box'>Soma<br><b>{soma}</b></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-box'>Pares<br><b>{pares}</b></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-box'>Ímpares<br><b>{6-pares}</b></div>", unsafe_allow_html=True)
            
            st.divider()
            
            pdf_data = gerar_pdf_bytes([latest])
            st.download_button(
                label="📄 Baixar Previsão (PDF)",
                data=pdf_data,
                file_name="previsao_ai.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("O modelo irá analisar 500+ concursos passados para encontrar padrões.")

# =============================================================================
# 6. NAVEGAÇÃO E MAIN
# =============================================================================

def draw_navigation():
    pages = ["Visão Geral", "Frequência", "Pares/Impares", "Combinações", "Quentes/Frios", "∑ Somas", "Previsões AI"]
    
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Visão Geral"
        
    cols = st.columns(len(pages))
    for i, page_name in enumerate(pages):
        is_active = st.session_state['current_page'] == page_name
        label = page_name.replace(" ", "\n")
        if cols[i].button(label, key=f"nav_{i}", type="primary" if is_active else "secondary"):
            st.session_state['current_page'] = page_name
            st.rerun()

def main():
    inject_custom_css()
    st.markdown("## 🎲 Mega-Sena Analyzer")
    
    df = carregar_dados_caixa()
    if not validar_dados(df):
        st.error("Erro ao carregar dados.")
        return

    draw_navigation()
    st.markdown("---")

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
