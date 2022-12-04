import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from create_data import create_data
from utils import compute_metrics
from sklearn.model_selection import train_test_split

NOMINAL_CAP = 5.0
DATA_FILE = Path('turnigy_graphene_data.csv')
FEATURES = ['Measured_Voltage', 'Measured_Current', 'Measured_Temperature',  'Avg_Measured_Voltage', 'Avg_Measured_Current', 'Avg_Measured_Temperature']
LABEL = 'Measured_SOC'
SAVE_FILE = Path('best_model.pkl')

if __name__ == '__main__':
    if not DATA_FILE.exists():
        data = create_data('../data/Turnigy Graphene')
        data.to_csv(DATA_FILE, index=False)
    else:
        data = pd.read_csv(DATA_FILE)

    data[LABEL] = (NOMINAL_CAP + data['Ah']) * 100. / NOMINAL_CAP

    train, val = train_test_split(data, test_size=0.25, random_state=42)

    x_train = train[FEATURES].values
    y_train = train[LABEL].values / 100.
    x_val = val[FEATURES].values
    y_val = val[LABEL].values / 100.

    model = XGBRegressor(n_estimators=1000)
    model.fit(x_train, y_train)

    train_metrics = compute_metrics(model.predict(x_train), y_train)
    val_metrics = compute_metrics(model.predict(x_val), y_val)

    print(f"TRAIN METRICS: {train_metrics}")
    print(f"VAL METRICS: {val_metrics}")

    pickle.dump(model, open(SAVE_FILE, "wb"))
