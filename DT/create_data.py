from pathlib import Path
import numpy as np
import pandas as pd
from utils import load_data
from tqdm import tqdm

TEMPERATURES = [-10, -20, 0, 10, 25, 40]
DRIVE_CYCLES = ['UDDS', 'HWFET', 'LA92', 'US06']
WINDOW_SIZE = 500

def preprocess_data(
    df: pd.DataFrame
):  
    # Sort data by time
    df = df.sort_values(by='RecordingTime').reset_index(drop=True)

    # Resample data at 1Hz
    df = df.loc[::10]

    # Apply rolling window to data on voltage, current and temperature
    df['Avg_Measured_Voltage'] = df['Measured_Voltage'].rolling(WINDOW_SIZE, min_periods=1).mean()
    df['Avg_Measured_Current'] = df['Measured_Current'].rolling(WINDOW_SIZE, min_periods=1).mean()
    df['Avg_Measured_Temperature'] = df['Measured_Temperature'].rolling(WINDOW_SIZE, min_periods=1).mean()

    return df

def create_data(
    data_dir
) -> pd.DataFrame:
    
    data_dir = Path(data_dir)
    total_data = []

    for t in tqdm(TEMPERATURES):
        t_dir = data_dir / f"{t} degC"
        assert t_dir.exists(), f"{t_dir} does not exist"
        
        for d in DRIVE_CYCLES:
            for f in t_dir.glob(f"*{d}*.mat"):
                
                cur_data = load_data(f)
                cur_data = preprocess_data(cur_data)
                total_data.append(cur_data)
    
    total_data = pd.concat(total_data)
    total_data.reset_index(drop=True, inplace=True)

    return total_data


if __name__ == '__main__':
    data = create_data('../data/Turnigy Graphene')
    data.to_csv('turnigy_graphene_data.csv', index=False)