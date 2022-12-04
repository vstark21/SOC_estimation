import scipy.io
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_data(
    mat_file: str
) -> pd.DataFrame:
    data = scipy.io.loadmat(mat_file)
    meas = data['meas']
    dtypes = meas.dtype
    meas = np.squeeze(meas).tolist()
    data_dict = {}
    for i, name in enumerate(dtypes.names):
        data_dict[name] = np.array(meas[i]).squeeze()

    data_dict = pd.DataFrame(data_dict)
    data_dict.rename(columns={
        'Time': 'RecordingTime',
        'Voltage': 'Measured_Voltage',
        'Current': 'Measured_Current',
        'Battery_Temp_degC': 'Measured_Temperature',
    }, inplace=True)
    return data_dict

def compute_metrics(preds, labels):
    return {
        'mse': mean_squared_error(preds, labels),
        'mae': mean_absolute_error(preds, labels)
    }

def simple_exponential_smoothing(arr, alpha=0.9):
    """
    Simple exponential smoothing
    """
    smoothed = np.zeros(len(arr))
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed