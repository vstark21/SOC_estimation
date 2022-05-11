import numpy as np
from utils import *
import matplotlib.pyplot as plt
from create_data import preprocess_data
from train import FEATURES
import tensorflow as tf
from tensorflow import keras

DATA_FILE = '../data/Turnigy Graphene/25 degC/04-20-19_05.34 780_LA92_25degC_Turnigy_Graphene.mat'

if __name__ == '__main__':
    
    LiPoly = load_data(DATA_FILE)
    LiPoly = preprocess_data(LiPoly)

    # Battery capacity in Ah taken from data
    nominal_cap = 5.
    # Calculate the SOC using coloumb counting for comparision
    LiPoly['Measured_SOC'] = (nominal_cap + LiPoly['Ah']) * 100. / nominal_cap

    # Converting seconds to hours
    LiPoly['RecordingTime_Hours'] = LiPoly['RecordingTime'] / 3600

    model = keras.models.load_model('best_turnigy_graphene_model.h5')
    SOC_Estimated = model.predict(LiPoly[FEATURES].values) * 100.
    SOC_Estimated = np.squeeze(SOC_Estimated)
    SOC_Estimated = simple_exponential_smoothing(SOC_Estimated, 0.1)

    # Plot the results
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(LiPoly['RecordingTime_Hours'], LiPoly['Measured_SOC'], label='Coulomb Counting')
    axes[0].plot(LiPoly['RecordingTime_Hours'], SOC_Estimated, label='Estimated SOC using FNN')
    axes[0].set_ylabel('SOC [%]')
    axes[0].set_xlabel('Time [Hours]')
    axes[0].set_title('SOC using Coulomb Counting vs. Estimated SOC using FNN')
    axes[0].legend()
    
    # Plot soc error
    axes[1].plot(LiPoly['RecordingTime_Hours'], (LiPoly['Measured_SOC'] - SOC_Estimated))
    axes[1].set_ylabel('SOC Error [%]')
    axes[1].set_xlabel('Time [Hours]')
    axes[1].set_title('SOC Error')
    
    plt.show()

