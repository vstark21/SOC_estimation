import numpy as np
from utils import *
import matplotlib.pyplot as plt
from extended_kalman_filter import run_ekf

DATA_FILE = '../04-20-19_05.34 780_LA92_25degC_Turnigy_Graphene.mat'
BATTERY_MODEL = '../data/BatteryModel.csv'
SOC_OCV = '../data/SOC-OCV.csv'

if __name__ == '__main__':
    
    LiPoly = load_data(DATA_FILE)

    # Battery capacity in Ah taken from data
    nominal_cap = 5.
    # Calculate the SOC using coloumb counting for comparision
    LiPoly['Measured_SOC'] = (nominal_cap + LiPoly['Ah']) * 100. / nominal_cap

    # Resample data
    LiPoly = LiPoly.loc[0::10]
    
    # Discharging: +ve current, charging: -ve current
    LiPoly['Measured_Current_R'] = LiPoly['Measured_Current'] * -1

    # Converting seconds to hours
    LiPoly['RecordingTime_Hours'] = LiPoly['RecordingTime'] / 3600

    # Load battery model and soc-ocv data
    battery_model = pd.read_csv(BATTERY_MODEL)
    soc_ocv = pd.read_csv(SOC_OCV)
    
    print(battery_model.head())
    print(soc_ocv.head())

    SOC_Estimated, Vt_Estimated, Vt_Error = run_ekf(
        LiPoly, battery_model, soc_ocv
    )

    percentage_error = np.abs((LiPoly['Measured_SOC'] - SOC_Estimated * 100) / LiPoly['Measured_SOC']) * 100
    print('Mean percentage error: {:.2f}%'.format(np.mean(percentage_error)))

    # Plot the results
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(LiPoly['RecordingTime_Hours'], LiPoly['Measured_Voltage'], label='Measured Voltage', alpha=0.5)
    axes[0, 0].plot(LiPoly['RecordingTime_Hours'], Vt_Estimated, label='Estimated Voltage', alpha=0.5)
    axes[0, 0].set_ylabel('Terminal Voltage [V]')
    axes[0, 0].set_title('Measure vs. Estimated Terminal Voltage (V)')
    axes[0, 0].legend()

    axes[0, 1].plot(LiPoly['RecordingTime_Hours'], Vt_Error)
    axes[0, 1].set_ylabel('Terminal Voltage Error')
    axes[0, 1].set_title('Terminal Voltage Error')

    axes[1, 0].plot(LiPoly['RecordingTime_Hours'], LiPoly['Measured_SOC'], label='Measured SOC', alpha=0.5)
    axes[1, 0].plot(LiPoly['RecordingTime_Hours'], SOC_Estimated * 100., label='Estimated SOC using EKF', alpha=0.5)
    axes[1, 0].set_ylabel('SOC [%]')
    axes[1, 0].set_xlabel('Time [Hours]')
    axes[1, 0].set_title('Measured SOC vs. Estimated SOC using EKF')
    axes[1, 0].legend()
    
    # Plot soc error
    axes[1, 1].plot(LiPoly['RecordingTime_Hours'], (LiPoly['Measured_SOC'] - SOC_Estimated * 100.))
    axes[1, 1].set_ylabel('SOC Error [%]')
    axes[1, 1].set_xlabel('Time [Hours]')
    axes[1, 1].set_title('SOC Error')
    
    plt.show()

