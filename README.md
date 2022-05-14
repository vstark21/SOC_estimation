# SOC_estimation

State of charge estimation using Extended kalman filter and Feedforward neural network. You can find more details in the [presentation]("SOC_Estimation.pdf").

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

## References

- [How to Estimate Battery State of Charge using Deep Learning](https://www.youtube.com/playlist?list=PLn8PRpmsu08qEaoBNHa16bPASDDKNBQI0)

- [Overview of batteries State of Charge estimation methods](https://www.sciencedirect.com/science/article/pii/S2352146519301905/pdf?md5=e1eaca1e8655197f259bcf88b4e9e47f&pid=1-s2.0-S2352146519301905-main.pdf)

- https://data.mendeley.com/datasets/4fx8cjprxm/

- [State of Charge Estimation Using Extended Kalman Filters for Battery Management System](https://energy.stanford.edu/sites/g/files/sbiybj9971/f/taborelli_onori_ievc.pdf)

- [State of Charge Estimation based on Kalman Filter](https://in.mathworks.com/matlabcentral/fileexchange/90381-state-of-charge-estimation-function-based-on-kalman-filter)