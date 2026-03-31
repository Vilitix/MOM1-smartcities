import pandas as pd

calendar = {
    1: ['Slurry spreading'],
    2: ['Fertilizer', 'Slurry spreading'],
    3: ['Fertilizer', 'Plowing'],
    4: ['Pesticides', 'Fertilizer'],
    5: ['Pesticides', 'Weeding'],
    6: ['Pesticides'],
    7: ['Bare soils'],
    8: ['Bare soils', 'Soil work'],
    9: ['Harvest', 'Soil work'],
    10: ['Sowing', 'Fertilizer', 'Plowing'],
    11: ['Plowed soils'],
    12: ['Slurry spreading']}

def get_farming_data(csv_path='data.csv'):
    """Return farming-event indicators aligned with sensor rows.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file (default: ``data.csv``).

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``Timestamp``, ``Date`` and one binary column per
        distinct event from ``calendar``.
    """
    # Only load required columns for speed optimization
    df = pd.read_csv(csv_path, usecols=['Timestamp', 'Date'])

    required_cols = ['Timestamp', 'Date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

    out = df[['Timestamp', 'Date']].copy()

    # Parse month from Date first; fallback to Unix Timestamp when needed.
    parsed_date = pd.to_datetime(out['Date'], format='%d/%m-%y %H:%M:%S', errors='coerce')
    timestamp_dt = pd.to_datetime(out['Timestamp'], unit='s', errors='coerce')
    
    # Filter since August 2025 for performance
    mask = parsed_date >= '2025-08-01'
    out = out[mask].copy()
    parsed_date = parsed_date[mask]
    timestamp_dt = timestamp_dt[mask]
    
    month_series = parsed_date.dt.month.fillna(timestamp_dt.dt.month)

    # Pre-calculate active months for each event to avoid slow lambda applies
    event_month_map = {}
    for month, events in calendar.items():
        for event in events:
            if event not in event_month_map:
                event_month_map[event] = []
            event_month_map[event].append(month)

    # Convert month_series to integer where possible, safely.
    # We use a helper mask since month_series might have NaN
    month_int_series = month_series.fillna(0).astype(int)
    
    for event in event_month_map.keys():
        active_months = event_month_map.get(event, [])
        # Vectorized check: is the month in the list of active months for this event?
        out[event] = month_int_series.isin(active_months).astype(int)

    return out


def build_csv(input_csv_path='data.csv', output_csv_path='farming_events.csv'):
    """Build and save the farming-events CSV from the source data file.

    Parameters
    ----------
    input_csv_path : str
        Source CSV file containing at least ``Timestamp`` and ``Date``.
    output_csv_path : str
        Destination CSV path for the generated farming-events table.

    Returns
    -------
    pandas.DataFrame
        The generated DataFrame that has also been written to disk.
    """
    farming_df = get_farming_data(input_csv_path)
    farming_df.to_csv(output_csv_path, index=False)
    return farming_df

if __name__ == "__main__":
    build_csv()