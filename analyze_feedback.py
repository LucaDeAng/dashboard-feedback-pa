import pandas as pd
import json
import random
from tqdm import tqdm
import os
import streamlit as st
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import numpy as np
from fpdf import FPDF
import folium
from streamlit_folium import folium_static
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from transformers import pipeline

# Configurazione della pagina
st.set_page_config(
    page_title="Dashboard Feedback Servizi Digitali | Comune di XXX",
    page_icon="🏛️",
    layout="wide"
)

# Configurazione API HuggingFace
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/MilaNLProc/feel-it-italian-sentiment"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Stili CSS personalizzati per il design system PA
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        padding: 0 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        color: #004D40;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #004D40;
        color: white;
        border-radius: 4px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .alert-box {
        background-color: #FFE0E0;
        color: #D32F2F;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
        border-left: 4px solid #D32F2F;
    }
    .header-pa {
        background-color: #004D40;
        color: white;
        padding: 24px;
        margin-bottom: 32px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Header PA
st.markdown("""
    <div class="header-pa">
        <h1 style='margin:0'>🏛️ Dashboard Feedback Servizi Digitali</h1>
        <p style='margin:8px 0 0 0'>Comune di XXX - Ufficio Transizione al Digitale</p>
    </div>
""", unsafe_allow_html=True)

def calcola_leggibilita(testo):
    """Calcola l'indice di leggibilità Gulpease"""
    # Implementazione semplificata
    parole = len(testo.split())
    frasi = len(re.split(r'[.!?]+', testo))
    lettere = len(re.sub(r'[^a-zA-Zàèéìòù]', '', testo))
    
    if frasi == 0:
        return 0
    
    return 89 - (300 * frasi + 10 * lettere) / parole

def estrai_keywords(testo):
    """Estrae le parole chiave dal testo"""
    # Rimuovi stopwords e punteggiatura
    testo = re.sub(r'[^\w\s]', '', testo.lower())
    parole = testo.split()
    
    # Rimuovi parole comuni
    stopwords = ['il', 'la', 'le', 'gli', 'i', 'e', 'è', 'sono', 'per', 'con', 'su', 'da', 'in', 'a']
    parole = [p for p in parole if p not in stopwords and len(p) > 3]
    
    return parole

@st.cache_resource
def load_sentiment_pipeline():
    """Carica il modello di analisi del sentiment."""
    return pipeline("sentiment-analysis", model="MilaNLProc/feel-it-italian-sentiment")

def analizza_sentiment_huggingface(testo: str) -> Optional[Dict[str, Any]]:
    """Analizza il sentiment del testo usando il modello locale."""
    try:
        classifier = load_sentiment_pipeline()
        risultato = classifier(testo)
        return risultato
    except Exception as e:
        st.warning(f"⚠️ Errore nella pipeline di sentiment: {str(e)}. Uso logica base per sentiment.")
        return None

def mappa_sentiment_huggingface(label: str, score: float) -> tuple[str, int, str]:
    """
    Mappa le etichette HuggingFace nel formato dell'app.
    Restituisce (sentiment, score, emoji)
    """
    label = label.upper()[:3]  # Normalizza tipo: 'positive' → 'POS'
    
    if label == "POS":
        return ("positivo", 3 if score > 0.8 else 2, "😊" if score > 0.8 else "🙂")
    elif label == "NEG":
        return ("negativo", -3 if score > 0.8 else -2, "😟" if score > 0.8 else "😕")
    else:  # NEU
        return ("neutro", 0, "😐")

def simula_analisi_feedback(testo: str, timestamp=None, area_geografica=None, ente=None) -> Dict[str, Any]:
    """Analizza il feedback usando l'API HuggingFace per il sentiment"""
    time.sleep(0.2)  # Rate limiting
    
    # Definizione dei topic con parole chiave associate
    topics = {
        "Accessibilità Servizi Digitali": ["spid", "accesso", "login", "autenticazione", "portale", "online", "digitale"],
        "Tempi di Risposta PA": ["attesa", "tempo", "lento", "velocità", "ritardo", "risposta", "attesa"],
        "Qualità Informazioni": ["informazione", "chiaro", "comprensibile", "documentazione", "istruzioni", "guida"],
        "Supporto Utente": ["assistenza", "aiuto", "supporto", "personale", "operatore", "chiamata", "telefono"],
        "Usabilità Piattaforme": ["interfaccia", "usabile", "facile", "difficile", "complicato", "intuitivo"],
        "Efficienza Procedurale": ["procedura", "pratica", "documento", "certificato", "modulo", "burocrazia"]
    }
    
    # Analisi sentiment con HuggingFace
    sentiment_result = analizza_sentiment_huggingface(testo)
    
    if sentiment_result and isinstance(sentiment_result, list) and len(sentiment_result) > 0:
        # Estrai label e score dalla risposta API
        result = sentiment_result[0]
        label = result.get("label", "NEU")
        score = result.get("score", 0)
        
        # Mappa il risultato con il score
        sentiment, sentiment_score, sentiment_emoji = mappa_sentiment_huggingface(label, score)
        
        # Determina priority basata sul sentiment e score
        if sentiment == "negativo":
            priority = "alta" if score > 0.8 else "media"
        elif sentiment == "positivo":
            priority = "bassa"
        else:
            priority = "media"
    else:
        # Fallback alla logica base
        parole_positive = ["eccellente", "ottimo", "perfetto", "fantastico", "soddisfatto", "efficiente", "rapido"]
        parole_negative = ["deluso", "pessimo", "danneggiato", "problema", "male", "lento", "inefficiente"]
        
        testo_lower = testo.lower()
        score = 0
        for word in parole_positive:
            if word in testo_lower:
                score += 1
        for word in parole_negative:
            if word in testo_lower:
                score -= 1
        
        score = max(min(score, 5), -5)
        
        if score >= 2:
            sentiment = "positivo"
            priority = "bassa"
            sentiment_emoji = "😊"
        elif score <= -2:
            sentiment = "negativo"
            priority = "alta"
            sentiment_emoji = "😟"
        else:
            sentiment = "neutro"
            priority = "media"
            sentiment_emoji = "😐"
        
        sentiment_score = score
    
    # Analisi topic basata su parole chiave
    testo_lower = testo.lower()
    topic_scores = {}
    
    for topic, keywords in topics.items():
        score = 0
        for keyword in keywords:
            if keyword in testo_lower:
                score += 1
        topic_scores[topic] = score
    
    # Seleziona il topic con il punteggio più alto
    if topic_scores:
        topic = max(topic_scores.items(), key=lambda x: x[1])[0]
    else:
        topic = random.choice(list(topics.keys()))
    
    # Genera risposta suggerita in base al sentiment e topic
    if sentiment == "positivo":
        suggested_reply = f"Grazie per il tuo riscontro positivo su {topic}!"
    elif sentiment == "negativo":
        # Risposte più specifiche per feedback negativi
        if topic == "Accessibilità Servizi Digitali":
            suggested_reply = f"Ci dispiace per i problemi di accesso. Verificheremo immediatamente il servizio {topic}."
        elif topic == "Tempi di Risposta PA":
            suggested_reply = f"Ci scusiamo per i tempi di attesa. Stiamo lavorando per ottimizzare i processi di {topic}."
        elif topic == "Qualità Informazioni":
            suggested_reply = f"Apprezziamo il tuo feedback sulla chiarezza delle informazioni. Miglioreremo la documentazione relativa a {topic}."
        elif topic == "Supporto Utente":
            suggested_reply = f"Ci dispiace per le difficoltà nel supporto. Rafforzeremo il servizio di {topic}."
        elif topic == "Usabilità Piattaforme":
            suggested_reply = f"Grazie per il feedback sull'usabilità. Analizzeremo l'interfaccia di {topic} per renderla più intuitiva."
        else:
            suggested_reply = f"Ci dispiace per i disagi riscontrati riguardo a {topic}. Il tuo feedback è prezioso per migliorare il servizio."
    else:
        suggested_reply = f"Grazie per il tuo feedback sul tema {topic}. Continueremo a lavorare per migliorare."
    
    # Calcola leggibilità
    leggibilita = calcola_leggibilita(testo)
    
    # Estrai keywords
    keywords = estrai_keywords(testo)
    
    # Gestione timestamp
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str):
        try:
            timestamp = pd.to_datetime(timestamp)
        except:
            timestamp = datetime.now()
    
    return {
        "sentiment": sentiment,
        "topic": topic,
        "priority": priority,
        "suggested_action": "Revisione immediata necessaria" if sentiment == "negativo" else "Monitoraggio standard",
        "suggested_reply": suggested_reply,
        "sentiment_score": sentiment_score,
        "sentiment_label": "Positivo" if sentiment == "positivo" else "Negativo" if sentiment == "negativo" else "Neutro",
        "sentiment_emoji": sentiment_emoji,
        "leggibilita": leggibilita,
        "keywords": keywords,
        "timestamp": timestamp,
        "area_geografica": area_geografica or "Non specificata",
        "ente": ente or "Non specificato",
        "gestito": False,
        "operatore": None,
        "data_gestione": None
    }

def crea_struttura_progetto():
    """Crea la struttura delle cartelle e un file CSV di esempio"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    esempio_csv_path = "data/feedback.csv"
    if not os.path.exists(esempio_csv_path):
        # Dataset di esempio più ampio e variegato
        feedback_esempio = pd.DataFrame({
            "feedback": [
                "Il servizio SPID funziona perfettamente, molto soddisfatto della velocità!",
                "Impossibile completare la procedura online, sistema troppo lento e poco chiaro.",
                "La documentazione fornita è completa ma potrebbe essere più sintetica.",
                "Ottima assistenza telefonica, hanno risolto il mio problema in pochi minuti.",
                "Il sito continua a dare errore durante il caricamento dei documenti.",
                "Servizio eccellente, personale molto competente e disponibile.",
                "Tempi di attesa troppo lunghi per il rilascio dei documenti.",
                "Interfaccia del portale moderna e facile da usare.",
                "Non riesco a trovare le informazioni che cerco, tutto molto confuso.",
                "Grande miglioramento rispetto al sistema precedente, complimenti!"
            ],
            "timestamp": [
                "2024-03-01 10:00:00",
                "2024-03-02 11:30:00",
                "2024-03-03 09:15:00",
                "2024-03-04 14:20:00",
                "2024-03-05 16:45:00",
                "2024-03-06 10:30:00",
                "2024-03-07 12:00:00",
                "2024-03-08 15:15:00",
                "2024-03-09 11:45:00",
                "2024-03-10 13:30:00"
            ],
            "area_geografica": [
                "Milano",
                "Roma",
                "Napoli",
                "Torino",
                "Bologna",
                "Firenze",
                "Venezia",
                "Palermo",
                "Genova",
                "Bari"
            ]
        })
        feedback_esempio.to_csv(esempio_csv_path, index=False)
        return feedback_esempio
    df = pd.read_csv(esempio_csv_path)
    
    # Rimuovi colonne duplicate se presenti
    if any(df.columns.duplicated()):
        st.warning("⚠️ Rilevate colonne duplicate nel dataset")
        st.write("Colonne duplicate:", df.columns[df.columns.duplicated()].tolist())
        df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def generate_text_report(df):
    """Genera un report testuale dai risultati"""
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    report = [
        "REPORT ANALISI FEEDBACK SERVIZI PA",
        f"Data generazione: {now}\n",
        "STATISTICHE PRINCIPALI:",
        f"- Totale feedback analizzati: {len(df)}",
        f"- Sentiment positivo: {len(df[df['sentiment'] == 'positivo'])}",
        f"- Sentiment negativo: {len(df[df['sentiment'] == 'negativo'])}",
        f"- Sentiment neutro: {len(df[df['sentiment'] == 'neutro'])}\n",
        "DETTAGLIO FEEDBACK CRITICI (Priorità Alta):",
    ]
    
    critical = df[df['priority'] == 'alta']
    for idx, row in critical.iterrows():
        report.append(f"\nFeedback {idx+1}:")
        report.append(f"Testo: {row['feedback']}")
        report.append(f"Topic: {row['topic']}")
        report.append(f"Azione suggerita: {row['suggested_action']}")
    
    return "\n".join(report)

def create_gauge_chart(score, title):
    """Crea un grafico a tachimetro con plotly"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        number={'font': {'size': 36, 'color': 'black'}},
        gauge={
            'axis': {'range': [-5, 5], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "#007C55"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-5, -2], 'color': "#ffcdd2"},  # Rosso chiaro
                {'range': [-2, 2], 'color': "#f5f5f5"},   # Grigio chiaro
                {'range': [2, 5], 'color': "#c8e6c9"}     # Verde chiaro
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin={'l': 30, 'r': 30, 't': 30, 'b': 30},
        paper_bgcolor='white',
        font={'size': 16, 'color': 'black'}
    )
    return fig

def generate_wordcloud(texts):
    """Genera un wordcloud dai testi"""
    try:
        # Converti tutti i testi in stringhe e uniscili
        all_words = ' '.join([str(text).lower() for text in texts if pd.notna(text)])
        
        # Se non ci sono parole, ritorna None
        if not all_words.strip():
            return None
        
        # Rimuovi stopwords e caratteri speciali
        stopwords = ['il', 'la', 'le', 'gli', 'i', 'e', 'è', 'sono', 'per', 'con', 'su', 'da', 'in', 'a', 'che', 'di']
        words = ' '.join([word for word in all_words.split() if word.lower() not in stopwords and len(word) > 3])
        
        if not words.strip():
            return None
            
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=50,
            min_font_size=10,
            prefer_horizontal=0.7
        ).generate(words)
        
        # Crea figura con dimensioni specifiche e margini ridotti
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
    except Exception as e:
        print(f"Errore nella generazione del wordcloud: {str(e)}")
        return None

def log_interazione(feedback_id, operatore, azione):
    """Logga le interazioni in un file CSV"""
    log_data = {
        'timestamp': datetime.now(),
        'feedback_id': feedback_id,
        'operatore': operatore,
        'azione': azione
    }
    
    df_log = pd.DataFrame([log_data])
    df_log.to_csv('log_interazioni.csv', mode='a', header=not os.path.exists('log_interazioni.csv'), index=False)

def log_email(feedback_id, email, topic):
    """Logga l'invio di una email nel file CSV"""
    os.makedirs("output", exist_ok=True)
    log_path = "output/log_email.csv"
    
    log_data = {
        'timestamp': datetime.now(),
        'feedback_id': feedback_id,
        'email': email,
        'topic': topic,
        'stato': "Preparato per invio"
    }
    
    df_log = pd.DataFrame([log_data])
    df_log.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
    return True

def genera_corpo_email(feedback, sentiment, topic, risposta):
    """Genera il corpo dell'email da inviare"""
    return f"""
Gentile collega,

è stato ricevuto un nuovo feedback che richiede la tua attenzione.

DETTAGLI FEEDBACK:
----------------
Testo: {feedback}
Sentiment: {sentiment}
Topic: {topic}

RISPOSTA SUGGERITA:
----------------
{risposta}

Per favore, analizza il feedback e procedi con la gestione appropriata.

Cordiali saluti,
Sistema di Gestione Feedback
"""

def valida_email(email):
    """Valida il formato dell'email e il dominio"""
    domini_permessi = ['comune.xxx.it', 'ente.it']
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False
    dominio = email.split('@')[1].lower()
    return dominio in domini_permessi

def cluster_feedback_by_topic(df):
    """Raggruppa i feedback per topic e calcola le statistiche"""
    topic_stats = df.groupby('topic').agg({
        'sentiment_score': ['mean', 'count'],
        'sentiment': lambda x: x.value_counts().to_dict()
    }).round(2)
    
    topic_stats.columns = ['sentiment_medio', 'totale_feedback', 'distribuzione_sentiment']
    return topic_stats

def analyze_services(df):
    """Analizza i feedback per tipo di servizio"""
    servizi = {
        'digitali': ['spid', 'app', 'portale', 'online', 'digitale', 'sito'],
        'sportello': ['sportello', 'ufficio', 'persona', 'operatore'],
        'telefonici': ['telefono', 'chiamata', 'numero verde'],
        'documenti': ['documento', 'certificato', 'modulo', 'pratica']
    }
    
    for servizio, keywords in servizi.items():
        df[f'servizio_{servizio}'] = df['feedback'].str.lower().apply(
            lambda x: any(keyword in str(x).lower() for keyword in keywords)
        )
    
    service_stats = pd.DataFrame()
    for servizio in servizi.keys():
        mask = df[f'servizio_{servizio}']
        stats = {
            'totale': mask.sum(),
            'sentiment_medio': df[mask]['sentiment_score'].mean(),
            'positivi': df[mask & (df['sentiment'] == 'positivo')].shape[0],
            'negativi': df[mask & (df['sentiment'] == 'negativo')].shape[0]
        }
        service_stats = pd.concat([service_stats, pd.DataFrame([stats], index=[servizio])])
    
    return service_stats

def extract_negative_keywords(texts):
    """Estrae e analizza le keyword più frequenti nei feedback negativi"""
    # Stopwords personalizzate per l'analisi
    custom_stopwords = ['il', 'la', 'le', 'gli', 'i', 'e', 'è', 'sono', 'per', 'con', 'su', 'da', 'in', 'a', 
                       'che', 'di', 'non', 'ma', 'ho', 'ha', 'hanno', 'questo', 'questa', 'questi', 'queste']
    
    # Tokenizzazione e pulizia
    words = []
    for text in texts:
        # Rimuovi punteggiatura e converti in minuscolo
        clean_text = re.sub(r'[^\w\s]', '', str(text).lower())
        # Tokenizza
        tokens = clean_text.split()
        # Filtra stopwords e parole corte
        words.extend([word for word in tokens if word not in custom_stopwords and len(word) > 3])
    
    # Analisi frequenza
    word_freq = Counter(words)
    
    # Calcola il peso delle parole basato sulla frequenza e sulla presenza in feedback negativi
    word_weights = {}
    for word, freq in word_freq.items():
        # Conta in quanti feedback negativi appare la parola
        negative_occurrences = sum(1 for text in texts if word in str(text).lower())
        # Calcola un peso basato su frequenza e occorrenze negative
        weight = freq * (negative_occurrences / len(texts))
        word_weights[word] = weight
    
    return word_weights

# Creazione dei tab
tab_carica, tab_dashboard, tab_gestione = st.tabs([
    "📥 Carica dataset",
    "📊 Dashboard servizi digitali",
    "📬 Lista feedback da gestire"
])

# Tab 1: Carica dataset
with tab_carica:
    st.header("📥 Caricamento Dataset")
    
    upload_choice = st.radio(
        "Seleziona la fonte dei dati:",
        ["📋 Usa dataset dimostrativo", "📤 Carica file CSV personalizzato"]
    )
    
    # Inizializza df come None
    df = None
    
    if upload_choice == "📤 Carica file CSV personalizzato":
        uploaded_file = st.file_uploader("Carica il tuo CSV", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'feedback' not in df.columns:
                    st.error("❌ Il file CSV deve contenere una colonna 'feedback'")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Errore nel caricamento del file: {str(e)}")
                st.stop()
    else:
        try:
            df = crea_struttura_progetto()
        except Exception as e:
            st.error(f"❌ Errore nel caricamento del dataset dimostrativo: {str(e)}")
            st.stop()
    
    # Mostra l'anteprima solo se df è stato caricato correttamente
    if df is not None:
        st.write("### Anteprima dataset")
        # Aggiungo paginazione e mostro tutte le righe
        righe_per_pagina = st.selectbox("Righe per pagina:", [10, 25, 50, 100])
        num_pagine = len(df) // righe_per_pagina + (1 if len(df) % righe_per_pagina > 0 else 0)
        pagina = st.number_input("Pagina", min_value=1, max_value=num_pagine, value=1) - 1
        inizio = pagina * righe_per_pagina
        fine = min((pagina + 1) * righe_per_pagina, len(df))
        
        st.write(f"Visualizzazione righe {inizio+1}-{fine} di {len(df)}")
        st.dataframe(df.iloc[inizio:fine], use_container_width=True)
        
        if st.button("🚀 Avvia analisi", type="primary"):
            with st.spinner("Analisi in corso..."):
                risultati = []
                progress = st.progress(0)
                
                for idx, row in df.iterrows():
                    risultato = simula_analisi_feedback(
                        row['feedback'],
                        timestamp=row.get('timestamp'),
                        area_geografica=row.get('area_geografica')
                    )
                    risultati.append(risultato)
                    progress.progress((idx + 1) / len(df))
                
                df_risultati = pd.DataFrame(risultati)
                df_finale = pd.concat([df, df_risultati], axis=1)
                df_finale = df_finale.loc[:, ~df_finale.columns.duplicated()]
                
                # Salva in cache per gli altri tab
                st.session_state['risultati'] = df_finale
                st.session_state['analisi_completata'] = True
                
                # Salva risultati
                os.makedirs("output", exist_ok=True)
                df_finale.to_csv("output/feedback_analizzati.csv", index=False)
                
                st.success("✅ Analisi completata con successo!")
    else:
        st.info("⚠️ Seleziona una fonte dati per procedere con l'analisi")

# Tab 2: Dashboard
with tab_dashboard:
    if 'risultati' not in st.session_state:
        st.info("⚠️ Esegui prima l'analisi nel tab 'Carica dataset'")
        st.stop()
    
    df_finale = st.session_state['risultati']
    
    # Layout a colonne per metriche principali
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gauge chart sentiment medio
        avg_sentiment = df_finale['sentiment_score'].mean()
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_sentiment,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score Medio", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [-5, 5]},
                'bar': {'color': "#004D40"},
                'steps': [
                    {'range': [-5, -2], 'color': "#FFE0E0"},
                    {'range': [-2, 2], 'color': "#E0E0E0"},
                    {'range': [2, 5], 'color': "#E0F2F1"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # KPI principali
        total = len(df_finale)
        neg_perc = (df_finale['sentiment'] == 'negativo').mean() * 100
        neu_perc = (df_finale['sentiment'] == 'neutro').mean() * 100
        pos_perc = (df_finale['sentiment'] == 'positivo').mean() * 100
        
        st.markdown("""
            <div class="metric-card">
                <h3>📊 Distribuzione Sentiment</h3>
                <p>😊 Positivi: {:.1f}%</p>
                <p>😐 Neutri: {:.1f}%</p>
                <p>😟 Negativi: {:.1f}%</p>
            </div>
        """.format(pos_perc, neu_perc, neg_perc), unsafe_allow_html=True)
        
        if neg_perc > 30:
            st.markdown("""
                <div class="alert-box">
                    ⚠️ <strong>Attenzione!</strong><br>
                    La percentuale di feedback negativi supera il 30%
                </div>
            """, unsafe_allow_html=True)
    
    # Statistiche dettagliate
    st.subheader("📈 Statistiche dettagliate")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🏷️ Topic più frequenti")
        topic_counts = df_finale['topic'].value_counts()
        fig = px.bar(
            x=topic_counts.index,
            y=topic_counts.values,
            title="Distribuzione Topic"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📱 Canali più utilizzati")
        if 'canale' in df_finale.columns:
            channel_counts = df_finale['canale'].value_counts()
            fig = px.pie(
                values=channel_counts.values,
                names=channel_counts.index,
                title="Distribuzione Canali"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("#### 🗺️ Aree geografiche critiche")
        if 'area_geografica' in df_finale.columns:
            area_sentiment = df_finale.groupby('area_geografica')['sentiment_score'].mean()
            fig = px.bar(
                x=area_sentiment.index,
                y=area_sentiment.values,
                title="Sentiment medio per area"
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Gestione feedback
with tab_gestione:
    if 'risultati' not in st.session_state:
        st.info("⚠️ Esegui prima l'analisi nel tab 'Carica dataset'")
        st.stop()
    
    df_finale = st.session_state['risultati']
    
    # Filtri
    st.subheader("🔍 Filtri")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_filter = st.multiselect(
            "Filtra per sentiment",
            options=['positivo', 'neutro', 'negativo'],
            default=['negativo']
        )
    
    with col2:
        topic_filter = st.multiselect(
            "Filtra per topic",
            options=df_finale['topic'].unique()
        )
    
    with col3:
        area_filter = st.multiselect(
            "Filtra per area",
            options=df_finale['area_geografica'].unique()
        )
    
    # Applica filtri
    mask = pd.Series(True, index=df_finale.index)
    if sentiment_filter:
        mask &= df_finale['sentiment'].isin(sentiment_filter)
    if topic_filter:
        mask &= df_finale['topic'].isin(topic_filter)
    if area_filter:
        mask &= df_finale['area_geografica'].isin(area_filter)
    
    filtered_df = df_finale[mask]
    
    # Lista feedback
    st.subheader(f"📝 Feedback da gestire ({len(filtered_df)})")
    
    for idx, row in filtered_df.iterrows():
        with st.expander(
            f"#{idx} - {row['sentiment_emoji']} {row['topic']} ({row['area_geografica']})",
            expanded=row['sentiment'] == 'negativo'
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <strong>Feedback:</strong><br>{row['feedback']}<br><br>
                        <strong>Risposta suggerita:</strong><br>{row['suggested_reply']}
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <p><strong>📊 Sentiment:</strong> {row['sentiment']}</p>
                        <p><strong>📍 Area:</strong> {row['area_geografica']}</p>
                        <p><strong>📅 Data:</strong> {row['timestamp']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                email = st.text_input(
                    "✉️ Email collega",
                    key=f"email_{idx}",
                    placeholder="nome.cognome@comune.xxx.it"
                )
                
                if st.button("📨 Inoltra", key=f"send_{idx}"):
                    if email and "@" in email:
                        oggetto = f"Feedback da gestire: {row['topic']}"
                        corpo = f"""
                            Feedback ricevuto da {row['area_geografica']}
                            Topic: {row['topic']}
                            Sentiment: {row['sentiment']}
                            
                            Testo: {row['feedback']}
                            
                            Risposta suggerita:
                            {row['suggested_reply']}
                        """
                        
                        mailto_link = f"mailto:{email}?subject={oggetto}&body={corpo}"
                        st.markdown(f"[🔗 Apri nel client email]({mailto_link})")
                        
                        # Log email
                        log_email(idx, email, row['topic'])
                        st.success("✅ Email preparata con successo!")
                    else:
                        st.error("❌ Inserisci un'email valida")
                
                if st.button("✅ Segna come gestito", key=f"done_{idx}"):
                    df_finale.at[idx, 'gestito'] = True
                    df_finale.at[idx, 'data_gestione'] = datetime.now()
                    st.session_state['risultati'] = df_finale
                    st.success("✅ Feedback segnato come gestito!")

# Footer PA
st.markdown("""
    <div style="margin-top: 50px; padding: 20px; background-color: #F5F5F5; border-radius: 8px;">
        <p style="text-align: center; margin: 0;">
            🏛️ Comune di XXX - Ufficio Transizione al Digitale<br>
            📧 rtd@comune.xxx.it | 📞 XXX XXX XXXX
        </p>
    </div>
""", unsafe_allow_html=True) 