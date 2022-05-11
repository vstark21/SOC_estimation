# SOC_estimation

State of charge estimation using Extended kalman filter and Feedforward neural network.

## Usage

Download data from [here](https://data.mendeley.com/datasets/4fx8cjprxm/1) and unzip it to `data/Turnigy Graphene` directory. Then install dependencies.

```bash
pip install -r requirements.txt
```

**Running EKF**

```bash
cd EKF
python main.py
```

Running FNN

```bash
cd FNN
python main.py
```