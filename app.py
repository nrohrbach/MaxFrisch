import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Seiten-Layout konfigurieren
st.set_page_config(page_title="Theater-Prognose 2050", layout="wide")

st.title("🎭 Theaterstücke: Analyse & Regression bis 2050")
st.markdown("Interaktives Dashboard zur Auswertung und Zukunftsprognose von Theateraufführungen.")

# Datei-Upload in der Seitenleiste
st.sidebar.header("📁 Datenquelle")
uploaded_file = st.sidebar.file_uploader("Lade deine CSV- oder Excel-Datei hoch", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Datei einlesen
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Datum konvertieren
    df['Premierendatum'] = pd.to_datetime(df['Premierendatum'], format='%d.%m.%Y', errors='coerce')
    df['Jahr'] = df['Premierendatum'].dt.year
    
    # Alle verfügbaren Stücke ermitteln
    all_pieces = df['Stück'].dropna().unique()
    top_default = df['Stück'].value_counts().head(5).index.tolist()
    
    # Sidebar-Filter für Theaterstücke
    st.sidebar.header("⚙️ Filter")
    selected_pieces = st.sidebar.multiselect(
        "Theaterstücke auswählen:",
        options=list(all_pieces),
        default=top_default
    )
    
    if not selected_pieces:
        st.warning("Bitte wähle mindestens ein Theaterstück aus der Seitenleiste aus.")
    else:
        # Daten filtern
        df_filtered = df[df['Stück'].isin(selected_pieces)].dropna(subset=['Jahr'])
        pivot_df = df_filtered.groupby(['Jahr', 'Stück']).size().unstack(fill_value=0)
        
        if not pivot_df.empty:
            min_year = int(pivot_df.index.min())
            max_year = int(pivot_df.index.max())
            full_historical_years = range(min_year, max_year + 1)
            pivot_df = pivot_df.reindex(full_historical_years, fill_value=0)
            
            future_years = np.arange(min_year, 2051)
            
            # --- PLOT ERSTELLEN ---
            fig, ax = plt.subplots(figsize=(14, 7))
            colors = plt.cm.viridis(np.linspace(0, 1, len(selected_pieces)))
            
            prognose_daten = []
            
            for i, piece in enumerate(selected_pieces):
                if piece in pivot_df.columns:
                    y_hist = pivot_df[piece].values
                    x_hist = pivot_df.index.values
                    
                    # Regression & R^2
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_hist, y_hist)
                    r_squared = r_value ** 2
                    
                    y_pred = slope * future_years + intercept
                    y_pred_clipped = np.maximum(y_pred, 0)
                    
                    # Konfidenzband
                    n = len(x_hist)
                    dof = n - 2
                    if dof > 0:
                        t_val = stats.t.ppf(0.975, dof)
                        residuals = y_hist - (slope * x_hist + intercept)
                        s_err = np.sqrt(np.sum(residuals**2) / dof)
                        x_mean = np.mean(x_hist)
                        se_pred = s_err * np.sqrt(1/n + (future_years - x_mean)**2 / np.sum((x_hist - x_mean)**2))
                        ci_upper = np.maximum(y_pred + t_val * se_pred, 0)
                        ci_lower = np.maximum(y_pred - t_val * se_pred, 0)
                    else:
                        ci_upper = y_pred_clipped
                        ci_lower = y_pred_clipped
                        
                    val_2040 = max(0.0, slope * 2040 + intercept)
                    val_2050 = max(0.0, slope * 2050 + intercept)
                    
                    prognose_daten.append({
                        'Theaterstück': piece,
                        'R² (Modellgüte)': round(r_squared, 3),
                        'Prognose 2040': round(val_2040, 1),
                        'Prognose 2050': round(val_2050, 1)
                    })
                    
                    # Zeichnen
                    ax.plot(x_hist, y_hist, marker='.', alpha=0.3, color=colors[i])
                    ax.plot(future_years, y_pred_clipped, linestyle='--', linewidth=2, label=f'{piece} (R²={r_squared:.2f})', color=colors[i])
                    ax.fill_between(future_years, ci_lower, ci_upper, color=colors[i], alpha=0.05)
            
            ax.set_title('Jahresscharfe Entwicklung & Lineare Regression bis 2050', fontsize=14, fontweight='bold')
            ax.set_xlabel('Jahr', fontsize=12)
            ax.set_ylabel('Anzahl der Aufführungen', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            fig.tight_layout()
            
            # Im Dashboard anzeigen
            st.pyplot(fig)
            
            # --- TABELLE AUSGEBEN ---
            st.subheader("📊 Prognose & Modellgüte ($R^2$-Wert)")
            df_prognose = pd.DataFrame(prognose_daten).set_index('Theaterstück')
            st.dataframe(df_prognose, use_container_width=True)
            
        else:
            st.warning("Keine Daten für den gewählten Zeitraum gefunden.")
else:
    st.info("👈 Bitte lade links über die Seitenleiste deine Datendatei hoch, um das Dashboard zu starten.")
