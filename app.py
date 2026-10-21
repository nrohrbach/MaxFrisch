import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Seiten-Layout konfigurieren
st.set_page_config(page_title="Theater-Prognose 2050", layout="wide")

st.title("🎭 Theaterstücke: Analyse & Regression bis 2050")
st.markdown("Interaktives Dashboard für das Schauburg-Archiv (ETH Zürich) mit automatischer Datenanbindung.")

# --- DATEN LADEN (Direkt von der ETH-URL) ---
@st.cache_data(ttl=86400) # Cacht die Daten für 24 Stunden, damit sie nicht bei jedem Klick neu heruntergeladen werden
def load_data():
    url = 'http://download.library.ethz.ch/opendata/Theaterauff%c3%bchrungen%20MF%20von%201945%20bis%20heute.xlsx'
    df = pd.read_excel(url, sheet_name='Theateraufführungen MF von 1945')
    return df

# Versuche, die Daten zu laden
try:
    with st.spinner("Lade Daten direkt von der ETH-Bibliothek..."):
        df = load_data()
    
    # 1. Datum konvertieren und Jahr extrahieren
    # Hinweis: Je nachdem wie die Spalte in der Excel heißt, ggf. anpassen. 
    # Standard laut deinem ersten Code ist 'Premierendatum'.
    df['Premierendatum'] = pd.to_datetime(df['Premierendatum'], format='%d.%m.%Y', errors='coerce')
    df['Jahr'] = df['Premierendatum'].dt.year
    
    # Alle verfügbaren Stücke ermitteln
    all_pieces = df['Stück'].dropna().unique()
    top_default = df['Stück'].value_counts().head(5).index.tolist()
    
    # Sidebar-Filter für Theaterstücke
    st.sidebar.header("⚙️ Filter & Einstellungen")
    
    # Suchfeld / Multiselect für Stücke
    selected_pieces = st.sidebar.multiselect(
        "Theaterstücke auswählen:",
        options=list(all_pieces),
        default=top_default
