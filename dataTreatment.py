import pandas as pd
import matplotlib.pyplot as plt

def score_ph(val):
    if 6.5 <= val <= 8.5:
        return 100
    elif 6 <= val <= 9:
        return 70
    else:
        return 0

def score_no3(val):
    if val < 25:
        return 100
    elif val <= 50:
        return 60
    else:
        return 0

def score_turb(val):
    if val < 1:
        return 100
    elif val <= 5:
        return 70
    elif val <= 25:
        return 40
    else:
        return 0

def score_o2(val):
    if val > 80:
        return 100
    elif val >= 60:
        return 70
    elif val >= 40:
        return 30
    else:
        return 0

def score_phyco(val):
    if val < 10:
        return 100
    elif val <= 20:
        return 50
    else:
        return 0

def compute_score(row):
    # règles bloquantes
    if row["Phycocyanine scaled"] > 20 or row["Turbidité"] > 50:
        return 0

    score = (
        score_phyco(row["Phycocyanine scaled"]) * 0.25 +
        score_turb(row["Turbidité"]) * 0.15 +
        score_o2(row["O2 Saturation"]) * 0.15 +
        score_ph(row["pH Test"]) * 0.10 +
        score_no3(row["NO3"]) * 0.05
    )

    return round(score, 1)


file = pd.read_csv("data.csv")
file = file.dropna(subset=["O2 Saturation","Turbidité","NO3","Phycocyanine scaled","pH Test"])
file["Date"] = pd.to_datetime(file["Date"], dayfirst=True, format="mixed")
file["Date_only"] = file["Date"].dt.date
# Formatage de la colonne Date en jour/mois/-année heure (ex: 26/03/-2025 15)
file["Date"] = file["Date"].dt.strftime("%d/%m/-%Y %H")
file_daily = file.groupby("Date_only")[["Turbidité"]].mean().reset_index()

# Calcul de la variation de turbidité (différence journalière)
file_daily["Variation_Turbidite"] = file_daily["Turbidité"].diff()

# Chargement du débit
df_debit = pd.read_csv("debit_simule_nancy_2025_2026.csv")
df_debit["Date"] = pd.to_datetime(df_debit["Date"]).dt.date

# Merge des deux datasets
merged = pd.merge(file_daily, df_debit, left_on="Date_only", right_on="Date", how="inner")

if not merged.empty:
    print("Corrélation de Pearson détaillée :")
    print(merged[["Variation_Turbidite", "Debit_m3_jour"]].corr())

    # Affichage du nuage de points
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.scatter(merged["Debit_m3_jour"], merged["Variation_Turbidite"], color='blue', alpha=0.7)
    ax1.set_xlabel("Débit simulé (m3/jour)")
    ax1.set_ylabel("Variation de Turbidité (NTU/jour)", color='blue')
    ax1.set_title("Corrélation entre variation de turbidité et débit du côté de Nancy")
    ax1.grid(True)
    plt.show()
else:
    print("Pas assez de dates en commun pour calculer une corrélation.")

file["Score_Baignade"] = file.apply(compute_score, axis=1)

# Graphe Pluies vs DBOeq (si DBOeq est présent et non-null)
file_dco = file.copy()
file_dco["DBOeq"] = pd.to_numeric(file_dco["DBOeq"], errors='coerce')
file_dco = file_dco.dropna(subset=["DBOeq"])
if not file_dco.empty:
    file_dco_daily = file_dco.groupby("Date_only")["DBOeq"].mean().reset_index()
    
    # Récupération des données météo pour les pluies
    from weather import get_weather_data
    weather_df = get_weather_data(days=365)
    weather_daily = weather_df.resample("D")["precipitation"].sum().reset_index()
    weather_daily["Date_only"] = weather_daily["date"].dt.date
    
    # Merge DCO et précipitations
    merged_dco_precip = pd.merge(file_dco_daily, weather_daily, on="Date_only", how="inner")
    
    # Masquer les petites précipitations pour l'affichage sans supprimer les données DBOeq
    merged_dco_precip["precip_filtered"] = merged_dco_precip["precipitation"].apply(lambda x: x if x > 10 else 0)
    
    if not merged_dco_precip.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Axe 1 : Précipitations (en barres bleues)
        ax2.bar(merged_dco_precip["Date_only"], merged_dco_precip["precip_filtered"], color='skyblue', label='Précipitations (mm)', alpha=0.6)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Précipitations (mm)", color='skyblue')
        ax2.tick_params(axis='y', labelcolor='skyblue')
        
        # Axe 2 : DCO (en ligne rouge)
        ax3 = ax2.twinx()
        ax3.plot(merged_dco_precip["Date_only"], merged_dco_precip["DBOeq"], color='darkred', marker='o', label='DBOeq', linewidth=2)
        ax3.set_ylabel("DBOeq", color='darkred')
        ax3.tick_params(axis='y', labelcolor='darkred')
        
        # Titre et légende
        plt.title("Évolution des Grosses Précipitations et de la DBOeq")
        fig2.tight_layout()
        plt.show()

if True:
    ax = plt.subplot()
    file_for_plot = file.groupby("Date_only")[["Score_Baignade"]].mean()
    ax.plot(file_for_plot.index.astype(str), file_for_plot["Score_Baignade"])
    plt.xticks(rotation=90)
    plt.title("Score de Baignade")
    plt.show()