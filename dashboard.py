# ====================================================================
# FICHIER : dashboard.py
# Dashboard Streamlit Premium - Classification des Iris
# Version finale avec prédictions locales
# ====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Iris Classification Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================================================================
# CHARGEMENT DU MODÈLE LOCAL
# ====================================================================

@st.cache_resource
def load_model_files():
    """Charger le modèle et le scaler"""
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('model_info.pkl', 'rb') as file:
            model_info = pickle.load(file)
        return model, scaler, model_info
    except Exception as e:
        return None, None, None

MODEL, SCALER, MODEL_INFO = load_model_files()

# ====================================================================
# STYLE CSS PERSONNALISÉ
# ====================================================================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1a2947;
        --bg-tertiary: #243557;
        --accent-primary: #06b6d4;
        --accent-secondary: #0891b2;
        --accent-light: #22d3ee;
        --text-primary: #ffffff;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --border-color: #334155;
        --success: #10b981;
        --danger: #ef4444;
        --shadow-sm: 0 2px 8px rgba(6, 182, 212, 0.1);
        --shadow-md: 0 8px 24px rgba(6, 182, 212, 0.15);
        --shadow-lg: 0 16px 40px rgba(6, 182, 212, 0.2);
    }
    
    html, body {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 20px;
        padding: clamp(1.5rem, 5vw, 3.5rem);
        margin-bottom: clamp(1.5rem, 4vw, 3rem);
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .main-header > * {
        position: relative;
        z-index: 1;
    }
    
    .main-title {
        font-size: clamp(1.8rem, 8vw, 3.5rem);
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-light) 0%, var(--accent-primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: var(--text-secondary);
        font-size: clamp(0.85rem, 2vw, 1.1rem);
        font-weight: 400;
        max-width: 600px;
    }
    
    /* Cards */
    .glass-card {
        background: rgba(26, 41, 71, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(6, 182, 212, 0.1);
        border-radius: 16px;
        padding: clamp(1.25rem, 3vw, 2rem);
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
        border-color: rgba(6, 182, 212, 0.4);
        background: rgba(26, 41, 71, 0.8);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(36, 53, 87, 0.4);
        border: 1px solid rgba(6, 182, 212, 0.15);
        border-radius: 14px;
        padding: clamp(1rem, 2.5vw, 1.8rem);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        border-color: rgba(6, 182, 212, 0.4);
        transform: translateY(-6px) scale(1.02);
        box-shadow: var(--shadow-lg);
        background: rgba(36, 53, 87, 0.7);
    }
    
    .metric-label {
        color: var(--text-muted);
        font-size: clamp(0.65rem, 1.5vw, 0.8rem);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.75rem;
    }
    
    .metric-value {
        background: linear-gradient(135deg, var(--accent-light) 0%, var(--accent-primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-description {
        color: var(--text-muted);
        font-size: clamp(0.7rem, 1.5vw, 0.85rem);
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        color: var(--text-primary);
        font-size: clamp(1.3rem, 4vw, 2rem);
        font-weight: 700;
        margin: clamp(1.5rem, 3vw, 2.5rem) 0 clamp(0.75rem, 2vw, 1.5rem) 0;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, var(--accent-primary), transparent) 1;
        position: relative;
    }
    
    /* Stat Box */
    .stat-box {
        background: rgba(36, 53, 87, 0.4);
        border: 1px solid rgba(6, 182, 212, 0.15);
        border-radius: 12px;
        padding: clamp(0.9rem, 2vw, 1.4rem);
        transition: all 0.3s ease;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stat-box:hover {
        border-color: rgba(6, 182, 212, 0.4);
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
        background: rgba(36, 53, 87, 0.6);
    }
    
    .stat-title {
        color: var(--text-muted);
        font-size: clamp(0.6rem, 1.3vw, 0.75rem);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        background: linear-gradient(135deg, var(--accent-light) 0%, var(--accent-primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: clamp(1.2rem, 3vw, 2rem);
        font-weight: 800;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        color: #0f172a;
        border: none;
        border-radius: 10px;
        padding: clamp(0.6rem, 1.5vw, 0.9rem) clamp(1.2rem, 3vw, 1.8rem);
        font-weight: 700;
        font-size: clamp(0.85rem, 2vw, 1rem);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-md);
        width: 100%;
        letter-spacing: 0.01em;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--accent-light) 0%, var(--accent-primary) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Inputs */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(36, 53, 87, 0.4) !important;
        border: 1px solid rgba(6, 182, 212, 0.2) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        padding: clamp(0.6rem, 1.5vw, 0.9rem) !important;
        transition: all 0.3s ease !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: clamp(0.85rem, 2vw, 1rem) !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.25) !important;
        background: rgba(36, 53, 87, 0.6) !important;
    }
    
    label {
        color: var(--text-secondary) !important;
        font-weight: 700 !important;
        font-size: clamp(0.8rem, 1.8vw, 0.95rem) !important;
        letter-spacing: 0.01em !important;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: clamp(0.4rem, 1vw, 0.6rem) clamp(0.8rem, 2vw, 1.2rem);
        border-radius: 20px;
        font-size: clamp(0.7rem, 1.5vw, 0.85rem);
        font-weight: 700;
        background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
        color: white;
        box-shadow: var(--shadow-md);
        letter-spacing: 0.01em;
    }
    
    /* Navigation Buttons Container */
    .nav-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-muted);
        padding: clamp(1.5rem, 3vw, 2.5rem);
        margin-top: clamp(2rem, 5vw, 4rem);
        border-top: 1px solid rgba(6, 182, 212, 0.1);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem;
        }
        
        .section-header {
            font-size: 1.4rem;
            margin: 1.5rem 0 0.8rem 0;
        }
        
        .metric-card, .stat-box {
            min-height: 120px;
        }
        
        .glass-card {
            padding: 1.25rem;
        }
        
        .stColumns {
            gap: 0.75rem !important;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.8rem;
        }
        
        .main-subtitle {
            font-size: 0.9rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .stat-value {
            font-size: 1.2rem;
        }
        
        .stButton > button {
            padding: 0.6rem 1rem;
            font-size: 0.85rem;
        }
        
        .stColumns {
            gap: 0.5rem !important;
        }
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ====================================================================
# FONCTIONS UTILITAIRES
# ====================================================================

@st.cache_data
def load_iris_data():
    """Charger le dataset Iris complet"""
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species'].map(species_names)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    return df

def create_metric_card(label, value, description=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-description">{description}</div>' if description else ''}
    </div>
    """

def create_stat_box(title, value):
    return f"""
    <div class="stat-box">
        <div class="stat-title">{title}</div>
        <div class="stat-value">{value}</div>
    </div>
    """

# ====================================================================
# HEADER PRINCIPAL
# ====================================================================

st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Iris Classification Platform</h1>
        <p class="main-subtitle">Système de classification avancé des espèces d'iris utilisant le Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# ====================================================================
# NAVIGATION
# ====================================================================

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Navigation buttons - horizontal inline layout
st.markdown("""
    <style>
    .nav-buttons {
        display: flex;
        gap: 0.75rem;
        justify-content: flex-start;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    
    .nav-buttons button {
        flex: 1;
        min-width: 140px;
        padding: 0.7rem 1.2rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        white-space: nowrap;
    }
    
    @media (max-width: 768px) {
        .nav-buttons {
            gap: 0.5rem;
        }
        
        .nav-buttons button {
            flex: 0 1 calc(50% - 0.25rem) !important;
            min-width: 100px !important;
            font-size: 0.8rem !important;
            padding: 0.6rem 0.9rem !important;
        }
    }
    
    @media (max-width: 480px) {
        .nav-buttons {
            gap: 0.4rem;
        }
        
        .nav-buttons button {
            flex: 0 1 calc(50% - 0.2rem) !important;
            min-width: 85px !important;
            font-size: 0.75rem !important;
            padding: 0.5rem 0.7rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Create horizontal layout for navigation buttons
nav_cols = st.columns([1, 1, 1, 1, 1], gap="small")

nav_buttons = [
    ("Dashboard", "nav_dashboard"),
    ("Prédiction Simple", "nav_predict"),
    ("Prédictions Multiples", "nav_batch"),
    ("Visualisations", "nav_viz"),
    ("À propos", "nav_about")
]

for col, (label, key) in zip(nav_cols, nav_buttons):
    with col:
        if st.button(label, use_container_width=True, key=key):
            st.session_state.current_page = label

st.markdown("<br>", unsafe_allow_html=True)

# ====================================================================
# PAGE 1: DASHBOARD
# ====================================================================
if st.session_state.current_page == "Dashboard":
    df = load_iris_data()
    
    # Section 1: Vue d'ensemble du modèle
    st.markdown('<div class="section-header">Vue d\'ensemble du modèle</div>', unsafe_allow_html=True)
    
    if MODEL_INFO:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_metric_card("Modèle", MODEL_INFO['model_name'], "Algorithme utilisé"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Exactitude", f"{MODEL_INFO['accuracy']*100:.2f}%", "Performance sur test"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Features", len(MODEL_INFO['features']), "Variables prédictives"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Classes", len(MODEL_INFO['species']), "Espèces identifiables"), unsafe_allow_html=True)
    
    # Section 2: Statistiques du dataset
    st.markdown('<div class="section-header">Analyse statistique du dataset</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Statistiques descriptives")
        
        stats = df.describe().T[['mean', 'std', 'min', 'max']].round(2)
        stats.columns = ['Moyenne', 'Écart-type', 'Minimum', 'Maximum']
        
        # Créer un tableau stylisé
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Variable'] + list(stats.columns),
                fill_color='#1a1a1a',
                align='left',
                font=dict(color='white', size=12, family='Inter')
            ),
            cells=dict(
                values=[stats.index] + [stats[col] for col in stats.columns],
                fill_color='#242424',
                align='left',
                font=dict(color='#b3b3b3', size=11, family='Inter')
            ))
        ])
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Répartition des espèces")
        
        species_counts = df['species'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=species_counts.index,
            values=species_counts.values,
            hole=0.6,
            marker=dict(colors=['#06b6d4', '#8b5cf6', '#10b981']),
            textfont=dict(color='#0f172a', size=14, family='Poppins', weight='bold')
        )])
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                font=dict(color='#cbd5e1', family='Poppins', size=11),
                bgcolor='rgba(26, 41, 71, 0.8)',
                bordercolor='rgba(6, 182, 212, 0.2)',
                borderwidth=1
            ),
            height=250,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 3: Corrélations
    st.markdown('<div class="section-header">Matrice de corrélation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        corr_matrix = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            y=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            colorscale=[[0, '#1a1a1a'], [0.5, '#ff6b9d'], [1, '#ff8fb3']],
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12, "color": "white"},
            colorbar=dict(
                tickfont=dict(color='white'),
                title=dict(text="Corrélation", font=dict(color='white'))
            )
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter'),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Corrélations fortes")
        
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    strong_corr.append({
                        'Var 1': corr_matrix.columns[i].replace('_', ' ').title(),
                        'Var 2': corr_matrix.columns[j].replace('_', ' ').title(),
                        'Corrélation': f"{corr_matrix.iloc[i, j]:.3f}"
                    })
        
        if strong_corr:
            for corr in strong_corr:
                st.markdown(f"""
                <div style="background: rgba(36, 53, 87, 0.4); padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; border-left: 3px solid #06b6d4; border: 1px solid rgba(6, 182, 212, 0.2);">
                    <div style="color: #cbd5e1; font-size: 0.85rem; font-family: Poppins;">{corr['Var 1']} ↔ {corr['Var 2']}</div>
                    <div style="color: #22d3ee; font-size: 1.25rem; font-weight: 700; margin-top: 0.25rem; font-family: Poppins;">{corr['Corrélation']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 4: Distribution des variables
    st.markdown('<div class="section-header">Distribution des caractéristiques</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    titles = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = {'setosa': '#06b6d4', 'versicolor': '#8b5cf6', 'virginica': '#10b981'}
    
    for idx, (var, title) in enumerate(zip(variables, titles)):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        for species in df['species'].unique():
            species_data = df[df['species'] == species][var]
            
            fig.add_trace(
                go.Violin(
                    y=species_data,
                    name=species,
                    marker_color=colors[species],
                    showlegend=(idx == 0),
                    box_visible=True,
                    meanline_visible=True
                ),
                row=row, col=col
            )


    
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', family='Poppins'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 41, 71, 0.8)',
            bordercolor='rgba(6, 182, 212, 0.2)',
            borderwidth=1,
            font=dict(family='Poppins', size=11)
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(6, 182, 212, 0.1)', zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 5: Métriques rapides
    st.markdown('<div class="section-header">Métriques du dataset</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    metrics = [
        ("Total Échantillons", f"{len(df)}"),
        ("Variables", f"{len(df.columns)-1}"),
        ("Espèces", f"{df['species'].nunique()}"),
        ("Valeurs Manquantes", f"{df.isnull().sum().sum()}"),
        ("Moyenne Sepal", f"{df['sepal_length'].mean():.2f}"),
        ("Moyenne Petal", f"{df['petal_length'].mean():.2f}")
    ]
    
    for col, (label, value) in zip([col1, col2, col3, col4, col5, col6], metrics):
        with col:
            st.markdown(create_stat_box(label, value), unsafe_allow_html=True)

            
# ====================================================================
# PAGE 2: PRÉDICTION SIMPLE
# ====================================================================

elif st.session_state.current_page == "Prédiction Simple":
    st.markdown('<div class="section-header">Prédiction d\'espèce</div>', unsafe_allow_html=True)
    
    if MODEL is None or SCALER is None:
        st.error("❌ Modèle non disponible")
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("Entrez les mesures de la fleur en centimètres")
        
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input("Longueur du Sépale (cm)", 0.0, 10.0, 5.1, 0.1)
            petal_length = st.number_input("Longueur du Pétale (cm)", 0.0, 10.0, 1.4, 0.1)
        with col2:
            sepal_width = st.number_input("Largeur du Sépale (cm)", 0.0, 10.0, 3.5, 0.1)
            petal_width = st.number_input("Largeur du Pétale (cm)", 0.0, 10.0, 0.2, 0.1)
        
        if st.button("Lancer la prédiction", use_container_width=True):
            try:
                features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                features_scaled = SCALER.transform(features)
                prediction = MODEL.predict(features_scaled)[0]
                
                probabilities = []
                if hasattr(MODEL, 'predict_proba'):
                    proba = MODEL.predict_proba(features_scaled)[0]
                    probabilities = [{'species': species, 'probability': f'{prob*100:.2f}'} for species, prob in zip(MODEL_INFO['species'], proba)]
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 3rem;">
                    <div style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; font-family: Poppins; letter-spacing: 0.1em; font-weight: 700;">Espèce prédite</div>
                    <div style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #22d3ee 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 1rem 0; font-family: Poppins;">{prediction.upper()}</div>
                    <div class="status-badge">Confiance élevée</div>
                </div>
                """, unsafe_allow_html=True)
                
                if probabilities:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### Probabilités par espèce")
                    st.dataframe(pd.DataFrame(probabilities), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
        else:
            st.markdown('</div>', unsafe_allow_html=True)

# ====================================================================
# PAGE 3: PRÉDICTIONS MULTIPLES
# ====================================================================

elif st.session_state.current_page == "Prédictions Multiples":
    st.markdown('<div class="section-header">Prédictions par lot</div>', unsafe_allow_html=True)
    
    if MODEL is None or SCALER is None:
        st.error("❌ Modèle non disponible")
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        with st.expander("📋 Format du fichier CSV requis"):
            example_df = pd.DataFrame({'sepal_length': [5.1, 6.2, 5.9], 'sepal_width': [3.5, 2.8, 3.0], 'petal_length': [1.4, 4.8, 5.1], 'petal_width': [0.2, 1.8, 1.8]})
            st.dataframe(example_df, use_container_width=True)
            st.download_button(label="📥 Télécharger l'exemple", data=example_df.to_csv(index=False), file_name="exemple_iris.csv", mime="text/csv", use_container_width=True)
        
        uploaded_file = st.file_uploader("Sélectionner un fichier CSV", type=['csv', 'txt'])
        
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode('utf-8')
                uploaded_file.seek(0)
                separator = '\t' if '\t' in content.split('\n')[0] else (';' if ';' in content.split('\n')[0] else ',')
                df = pd.read_csv(uploaded_file, sep=separator)
                
                column_mapping = {
                    'SepalLength': 'sepal_length', 'SepalWidth': 'sepal_width', 'PetalLength': 'petal_length', 'PetalWidth': 'petal_width',
                    'Sepal.Length': 'sepal_length', 'Sepal.Width': 'sepal_width', 'Petal.Length': 'petal_length', 'Petal.Width': 'petal_width'
                }
                df = df.rename(columns=column_mapping)
                if 'Species' in df.columns:
                    df = df.drop('Species', axis=1)
                
                required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Colonnes manquantes")
                else:
                    df = df[required_cols]
                    st.success(f"✅ Fichier chargé : {len(df)} échantillons")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    if st.button("🚀 Lancer les prédictions", use_container_width=True):
                        try:
                            df_clean = df.copy()
                            for col in required_cols:
                                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                            df_clean = df_clean.dropna()
                            
                            if len(df_clean) == 0:
                                st.error("❌ Aucune donnée valide")
                            else:
                                features_scaled = SCALER.transform(df_clean.values)
                                predictions = MODEL.predict(features_scaled)
                                results_df = df_clean.copy()
                                results_df['prediction'] = predictions
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                pred_counts = pd.Series(predictions).value_counts()
                                with col1:
                                    st.markdown(create_metric_card("Total", f"{len(results_df)}", "Prédictions"), unsafe_allow_html=True)
                                with col2:
                                    st.markdown(create_metric_card("Setosa", f"{pred_counts.get('setosa', 0)}", "Échantillons"), unsafe_allow_html=True)
                                with col3:
                                    st.markdown(create_metric_card("Versicolor", f"{pred_counts.get('versicolor', 0)}", "Échantillons"), unsafe_allow_html=True)
                                with col4:
                                    st.markdown(create_metric_card("Virginica", f"{pred_counts.get('virginica', 0)}", "Échantillons"), unsafe_allow_html=True)
                                
                                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                                st.markdown("#### Répartition des prédictions")
                                fig = go.Figure(data=[go.Bar(x=pred_counts.index, y=pred_counts.values, marker=dict(color=['#06b6d4', '#8b5cf6', '#10b981']), text=pred_counts.values, textposition='auto', textfont=dict(color='#0f172a', size=14, family='Poppins'))])
                                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1', family='Poppins'), height=350, xaxis=dict(showgrid=False, title="Espèce"), yaxis=dict(showgrid=True, gridcolor='rgba(6, 182, 212, 0.1)', title="Nombre"), margin=dict(l=40, r=40, t=40, b=40))
                                st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                                st.markdown("#### Résultats détaillés")
                                st.dataframe(results_df, use_container_width=True)
                                csv = results_df.to_csv(index=False)
                                st.download_button(label="📥 Télécharger les résultats", data=csv, file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"❌ Erreur: {str(e)}")
            except Exception as e:
                st.error(f"❌ Erreur de lecture")
        
        st.markdown('</div>', unsafe_allow_html=True)
# ====================================================================
# PAGE 4: VISUALISATIONS
# ====================================================================
elif st.session_state.current_page == "Visualisations":
    
    st.markdown('<div class="section-header">Exploration visuelle des données</div>', unsafe_allow_html=True)
    
    df = load_iris_data()
    
    viz_type = st.selectbox(
        "Type de visualisation",
        ["Scatter Plot 2D", "Scatter Plot 3D", "Box Plot", "Distribution", "Pairplot"]
    )
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    colors_map = {'setosa': '#06b6d4', 'versicolor': '#8b5cf6', 'virginica': '#10b981'}
    
    if viz_type == "Scatter Plot 2D":
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Axe X", df.columns[:-1], index=2)
        with col2:
            y_var = st.selectbox("Axe Y", df.columns[:-1], index=3)
        
        fig = px.scatter(
            df, x=x_var, y=y_var, color='species',
            color_discrete_map=colors_map,
            title=f"{x_var.replace('_', ' ').title()} vs {y_var.replace('_', ' ').title()}"
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Poppins'),
            height=500,
            xaxis=dict(showgrid=True, gridcolor='rgba(6, 182, 212, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(6, 182, 212, 0.1)'),
            legend=dict(bgcolor='rgba(26, 41, 71, 0.8)', bordercolor='rgba(6, 182, 212, 0.2)', borderwidth=1, font=dict(family='Poppins', size=11))
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot 3D":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("Axe X", df.columns[:-1], index=0)
        with col2:
            y_var = st.selectbox("Axe Y", df.columns[:-1], index=1)
        with col3:
            z_var = st.selectbox("Axe Z", df.columns[:-1], index=2)
        
        fig = px.scatter_3d(
            df, x=x_var, y=y_var, z=z_var, color='species',
            color_discrete_map=colors_map
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Poppins'),
            height=600,
            scene=dict(
                xaxis=dict(backgroundcolor='#0f172a', gridcolor='rgba(6, 182, 212, 0.1)'),
                yaxis=dict(backgroundcolor='#0f172a', gridcolor='rgba(6, 182, 212, 0.1)'),
                zaxis=dict(backgroundcolor='#0f172a', gridcolor='rgba(6, 182, 212, 0.1)')
            ),
            legend=dict(bgcolor='rgba(26, 41, 71, 0.8)', bordercolor='rgba(6, 182, 212, 0.2)', borderwidth=1, font=dict(family='Poppins', size=11))
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        var = st.selectbox("Variable à visualiser", df.columns[:-1])
        
        fig = go.Figure()
        for species in df['species'].unique():
            fig.add_trace(go.Box(
                y=df[df['species'] == species][var],
                name=species,
                marker_color=colors_map[species]
            ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Poppins'),
            height=500,
            title=f"Distribution de {var.replace('_', ' ').title()}",
            yaxis=dict(showgrid=True, gridcolor='rgba(6, 182, 212, 0.1)'),
            xaxis=dict(showgrid=False),
            legend=dict(bgcolor='rgba(26, 41, 71, 0.8)', bordercolor='rgba(6, 182, 212, 0.2)', borderwidth=1, font=dict(family='Poppins', size=11))
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Distribution":
        var = st.selectbox("Variable à visualiser", df.columns[:-1])
        
        fig = go.Figure()
        for species in df['species'].unique():
            fig.add_trace(go.Histogram(
                x=df[df['species'] == species][var],
                name=species,
                marker_color=colors_map[species],
                opacity=0.7
            ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Poppins'),
            height=500,
            title=f"Distribution de {var.replace('_', ' ').title()}",
            barmode='overlay',
            xaxis=dict(showgrid=True, gridcolor='rgba(6, 182, 212, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(6, 182, 212, 0.1)'),
            legend=dict(bgcolor='rgba(26, 41, 71, 0.8)', bordercolor='rgba(6, 182, 212, 0.2)', borderwidth=1, font=dict(family='Poppins', size=11))
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pairplot":
        st.info("Matrice de nuages de points de toutes les variables")
        
        vars_list = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        fig = make_subplots(
            rows=4, cols=4,
            subplot_titles=[f"{v1.split('_')[0][0].upper()}{v1.split('_')[0][1]} vs {v2.split('_')[0][0].upper()}{v2.split('_')[0][1]}" 
                          for v1 in vars_list for v2 in vars_list],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        for i, var1 in enumerate(vars_list):
            for j, var2 in enumerate(vars_list):
                for species in df['species'].unique():
                    species_df = df[df['species'] == species]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=species_df[var2],
                            y=species_df[var1],
                            mode='markers',
                            name=species,
                            marker=dict(color=colors_map[species], size=4),
                            showlegend=(i==0 and j==0)
                        ),
                        row=i+1, col=j+1
                    )
        
        fig.update_layout(
            height=1000,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Poppins', size=8),
            showlegend=True,
            legend=dict(bgcolor='rgba(26, 41, 71, 0.8)', bordercolor='rgba(6, 182, 212, 0.2)', borderwidth=1, font=dict(family='Poppins', size=10))
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(6, 182, 212, 0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(6, 182, 212, 0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ====================================================================
# PAGE 5: À PROPOS
# ====================================================================

elif st.session_state.current_page == "À propos":
    
    st.markdown('<div class="section-header">À propos du projet</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        #### Projet Académique
        
        Ce système de classification fait partie du **TP N°1: Classification des fleurs iris** dans le cadre du module 
        **Introduction à l'Intelligence Artificielle et Machine Learning (INFO4111)**.
        
        #### Institution
        
        **Université de Yaoundé 1**  
        École Normale Supérieure  
        Département d'Informatique et des Technologies Éducatives
        
        #### Objectifs pédagogiques
        
        - Maîtrise de Python pour la data science
        - Utilisation des bibliothèques ML (scikit-learn, pandas, numpy)
        - Exploration et visualisation de données
        - Préparation et normalisation des données
        - Entraînement et évaluation de modèles
        - Déploiement d'applications ML avec Flask et Streamlit
        
        #### Dataset Iris
        
        Le dataset Iris, collecté par Edgar Anderson et popularisé par Ronald Fisher en 1936, est l'un des ensembles 
        de données les plus célèbres en apprentissage automatique. Il contient 150 échantillons de fleurs iris répartis 
        en 3 espèces, avec 4 caractéristiques morphologiques mesurées pour chaque échantillon.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Technologies")
        
        technologies = [
            ("Python", "Langage principal"),
            ("scikit-learn", "Machine Learning"),
            ("Pandas", "Manipulation de données"),
            ("NumPy", "Calcul numérique"),
            ("Plotly", "Visualisations"),
            ("Flask", "API REST"),
            ("Streamlit", "Interface web")
        ]
        
        for tech, desc in technologies:
            st.markdown(f"""
            <div style="background: rgba(36, 53, 87, 0.4); padding: 0.75rem; border-radius: 10px; margin-bottom: 0.5rem; border-left: 3px solid #06b6d4; border: 1px solid rgba(6, 182, 212, 0.2);">
                <div style="color: #22d3ee; font-weight: 700; font-family: Poppins;">{tech}</div>
                <div style="color: #cbd5e1; font-size: 0.85rem; font-family: Poppins; margin-top: 0.3rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Informations du modèle
    if MODEL_INFO:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Informations du modèle déployé")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(create_stat_box("Algorithme", MODEL_INFO['model_name']), unsafe_allow_html=True)
        with col2:
            st.markdown(create_stat_box("Précision", f"{MODEL_INFO['accuracy']*100:.2f}%"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_stat_box("Date", MODEL_INFO['training_date'].split()[0]), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Credits
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    #### Crédits
    
    **Enseignant**: Stéphane C.K. TEKOUAB (PhD & Ing.)  
    **Année académique**: 2025-2026
    
    www.tekouabou.com
    """)
    st.markdown('</div>', unsafe_allow_html=True)