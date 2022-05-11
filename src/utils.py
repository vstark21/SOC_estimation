import scipy.io
import numpy as np
import pandas as pd

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
