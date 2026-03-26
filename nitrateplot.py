import pandas as pd
import matplotlib.pyplot as plt
from weather import get_weather_data

def plot_water_quality(y_column='NO3', precipitation_threshold=None):
    df = pd.read_csv('Nitrate_data.csv')

    # Convertir le Timestamp Unix en datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    if y_column not in df.columns:
        raise ValueError(f"Colonne '{y_column}' introuvable dans Nitrate_data.csv")

    # Forcer en numérique et supprimer les points sans valeur
    df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
    df = df.dropna(subset=[y_column, 'Timestamp'])
    if df.empty:
        raise ValueError(f"Aucune donnée exploitable pour la colonne '{y_column}'")

    # Événements agricoles par mois (susceptibles de réduire la qualité de l'eau)
    events = {
        1: 'Épandage lisier',
        2: 'Engrais / Épandage',
        3: 'Engrais / Labour',
        4: 'Pesticides / Engrais',
        5: 'Pesticides / Désherbage',
        6: 'Pesticides',
        7: 'Sols nus',
        8: 'Déchaumage / Travail sol',
        9: 'Récolte / Travail sol',
        10: 'Semis + engrais / Labour',
        11: 'Sols labourés',
        12: 'Épandage lisier'
    }

    # Couleurs pour les zones (dégradé rouge = impact élevé)
    colors_risk = {
        1: '#ffcccc', 2: '#ffcccc', 3: '#ff9999', 4: '#ff6666', 5: '#ff6666',
        6: '#ff9999', 7: '#ffcccc', 8: '#ff6666', 9: '#ff9999', 10: '#ff6666',
        11: '#ffcccc', 12: '#ffcccc'
    }

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(14, 6))

    # Limiter à la période des données
    date_min = df['Timestamp'].min()
    date_max = df['Timestamp'].max()

    # Récupérer la météo sur la période utile et détecter les fortes précipitations
    days = max(1, int((date_max.normalize() - date_min.normalize()).days) + 2)
    weather_df = get_weather_data(days=days)
    weather_df.index = weather_df.index.tz_localize(None)
    weather_df = weather_df[(weather_df.index >= date_min) & (weather_df.index <= date_max)]

    if not weather_df.empty:
        # Si aucun seuil n'est fourni, on garde un seuil automatique (percentile 95)
        if precipitation_threshold is None:
            precip_threshold = weather_df['precipitation'].quantile(0.95)
        else:
            precip_threshold = precipitation_threshold
        heavy_rain = weather_df[weather_df['precipitation'] >= precip_threshold]
    else:
        heavy_rain = pd.DataFrame(columns=['precipitation'])

    # Ajouter les zones colorées par mois (seulement pour la période des données)
    current_date = date_min.replace(day=1)
    while current_date <= date_max:
        month = current_date.month
        start_date = current_date
        if month == 12:
            end_date = pd.Timestamp(year=current_date.year+1, month=1, day=1)
        else:
            end_date = pd.Timestamp(year=current_date.year, month=month+1, day=1)

        # Ne tracer que la portion qui chevauche les données
        span_start = max(start_date, date_min)
        span_end = min(end_date, date_max)

        ax.axvspan(span_start, span_end, alpha=0.15, color=colors_risk[month], label=f'{month}: {events[month]}')
        current_date = end_date

    # Tracer les données de la colonne choisie
    ax.plot(df['Timestamp'], df[y_column], marker='o', linestyle='None', markersize=5,
            color='darkblue', label=y_column, zorder=10)

    # Ajouter des indicateurs bleus lors des fortes précipitations
    if not heavy_rain.empty:
        rain_marker_y = df[y_column].max() * 0.98
        ax.scatter(
            heavy_rain.index,
            [rain_marker_y] * len(heavy_rain),
            color='royalblue',
            marker='v',
            s=55,
            label='Fortes précipitations',
            zorder=12
        )

    # Déterminer la position pour le texte (un peu au-dessus du maximum des données)
    y_max_for_text = df[y_column].max() * 0.9

    # Ajouter les noms d'événements sur les zones
    current_date = date_min.replace(day=1)
    while current_date <= date_max:
        month = current_date.month
        start_date = current_date
        if month == 12:
            end_date = pd.Timestamp(year=current_date.year+1, month=1, day=1)
        else:
            end_date = pd.Timestamp(year=current_date.year, month=month+1, day=1)

        # Ne tracer que la portion qui chevauche les données
        span_start = max(start_date, date_min)
        span_end = min(end_date, date_max)

        # Ajouter le texte du mois au centre de la zone
        mid_date = span_start + (span_end - span_start) / 2
        event_text = events[month]
        ax.text(mid_date, y_max_for_text, event_text,
                rotation=0, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5),
                weight='bold')

        current_date = end_date

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(y_column, fontsize=12)
    ax.set_title(f"{y_column} en fonction du temps et activités agricoles", fontsize=14, fontweight='bold')
    ax.set_xlim(date_min, date_max)
    ax.grid(True, alpha=0.3, zorder=0)
    plt.xticks(rotation=45)

    # Légende personnalisée
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_water_quality('NO3', precipitation_threshold=5.0)
