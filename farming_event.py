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
    df = pd.read_csv(csv_path)

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

    # Keep event order as first encountered in the calendar dictionary.
    distinct_events = []
    seen_events = set()
    for month in sorted(calendar.keys()):
        for event in calendar[month]:
            if event not in seen_events:
                seen_events.add(event)
                distinct_events.append(event)

    for event in distinct_events:
        out[event] = month_series.apply(
            lambda month: 1 if pd.notna(month) and event in calendar.get(int(month), []) else 0
        ).astype(int)

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