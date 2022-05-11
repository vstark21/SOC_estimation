import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from tqdm import tqdm

def scatteredInterpolant(x, y, v):
    xi, yi = np.meshgrid(x, y)


def run_ekf(
    data: pd.DataFrame,
    bat_model: pd.DataFrame,
    soc_ocv: pd.DataFrame,
):
    Current = data['Measured_Current_R'].values
    Vt_Actual = data['Measured_Voltage'].values
    Temperature = data['Measured_Temperature'].values

    SOC_Init = 1 # Initial SOC
    X = np.array([[SOC_Init], [0], [0]]) # State space x parameter initialisation
    DeltaT = 1 # Sample time in seconds
    Qn_rated = 5. * 3600 # Ah to Amp-sec

    F_R0 = interp2d(bat_model['T'].values, bat_model.SOC.values, bat_model.R0.values)
    F_R1 = interp2d(bat_model['T'].values, bat_model.SOC.values, bat_model.R1.values)
    F_R2 = interp2d(bat_model['T'].values, bat_model.SOC.values, bat_model.R2.values)
    F_C1 = interp2d(bat_model['T'].values, bat_model.SOC.values, bat_model.C1.values)
    F_C2 = interp2d(bat_model['T'].values, bat_model.SOC.values, bat_model.C2.values)

    SOCOCV = np.polyfit(soc_ocv['SOC'].values, soc_ocv['OCV'].values, 11)
    dSOCOCV = np.polyder(SOCOCV)

    n_x = X.shape[0]
    R_x = 2.5e-5
    P_x = np.array([
        [0.025, 0, 0],
        [0, 0.01, 0],
        [0, 0, 0.01]
    ])
    Q_x = np.array([
        [1.0e-6, 0, 0],
        [0, 1.0e-5, 0],
        [0, 0, 1.0e-5]
    ])

    SOC_Estimated = []
    Vt_Estimated = []
    Vt_Error = []
    ik = len(Current)

    for k in tqdm(range(ik)):
        T = Temperature[k]
        U = Current[k]
        SOC = X[0, 0]
        V1 = X[1, 0]
        V2 = X[2, 0]

        # Evaluate the battery parameter 
        # Functions for the current temperature and SOC
        R0 = F_R0(T, SOC).squeeze()
        R1 = F_R1(T, SOC).squeeze()
        R2 = F_R2(T, SOC).squeeze()
        C1 = F_C1(T, SOC).squeeze()
        C2 = F_C2(T, SOC).squeeze()

        OCV = np.polyval(SOCOCV, SOC)

        Tau_1 = C1 * R1
        Tau_2 = C2 * R2

        a1 = np.exp(-DeltaT / Tau_1)
        a2 = np.exp(-DeltaT / Tau_2)

        b1 = R1 * (1 - np.exp(-DeltaT / Tau_1))
        b2 = R2 * (1 - np.exp(-DeltaT / Tau_2))

        Terminal_Voltage = OCV - U * R0 - V1 - V2
        eta = 1

        dOCV = np.polyval(dSOCOCV, SOC)
        C_x = np.array([[dOCV, -1, -1]])

        Error_x = np.array([[Vt_Actual[k] - Terminal_Voltage]])

        Vt_Estimated.append(Terminal_Voltage)
        SOC_Estimated.append(X[0, 0])
        Vt_Error.append(Error_x.squeeze())

        A = np.array([
            [1, 0, 0],
            [0, a1, 0],
            [0, 0, a2]
        ])
        B = np.array([[-(eta * DeltaT / Qn_rated)], [b1], [b2]])
        X = (A @ X) + (B @ np.array([[U]]))
        P_x = (A @ P_x @ A.T) + Q_x

        a = (C_x @ P_x @ C_x.T) + R_x
        KalmanGain_x = P_x @ C_x.T @ np.linalg.inv(a)
        X = X + (KalmanGain_x @ Error_x)
        P_x = (np.eye(n_x, n_x) - (KalmanGain_x @ C_x)) @ P_x

    SOC_Estimated = np.array(SOC_Estimated)
    Vt_Estimated = np.array(Vt_Estimated)
    Vt_Error = np.array(Vt_Error)

    return SOC_Estimated, Vt_Estimated, Vt_Error


