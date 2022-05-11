import numpy as np
import pandas as pd
from pathlib import Path
from create_data import create_data
from model import create_model, train_model
from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt

NOMINAL_CAP = 5.0
DATA_FILE = Path('turnigy_graphene_data.csv')
FEATURES = ['Measured_Voltage', 'Measured_Current', 'Measured_Temperature',  'Avg_Measured_Voltage', 'Avg_Measured_Current', 'Avg_Measured_Temperature']
LABEL = 'Measured_SOC'
EPOCHS = 100
BATCH_SIZE = 1024

if __name__ == '__main__':
    if not DATA_FILE.exists():
        data = create_data('../data/Turnigy Graphene')
    else:
        data = pd.read_csv(DATA_FILE)

    data['Measured_SOC'] = (NOMINAL_CAP + data['Ah']) * 100. / NOMINAL_CAP
    data['RecordingTime_Hours'] = data['RecordingTime'] / 3600

    print(f"Total data shape: {data.shape}")

    train, val = train_test_split(data, test_size=0.25, random_state=42)

    print(f"Train data shape: {train.shape}")
    print(f"Test data shape: {val.shape}")
    
    x_train = train[FEATURES].values
    y_train = train[LABEL].values / 100.
    x_val = val[FEATURES].values
    y_val = val[LABEL].values / 100.

    model = create_model((len(FEATURES)))
    history, model = train_model(
        model, 
        x_train, 
        x_val, 
        y_train, 
        y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    model.save('turnigy_graphene_model.h5')

    fig, axes = plt.subplots(1, 3)
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('MSE')
    axes[0].legend()

    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title('MAE')
    axes[1].legend()

    axes[2].plot(history.history['lr'], label='Learning Rate')
    axes[2].set_title('Learning Rate')
    axes[2].legend()

    plt.show()