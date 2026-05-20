# Assignment 4 - Prognostics & Health Management

Three Jupyter notebooks that walk through the PHM loop end to end: **detect → diagnose → prognose**.

- `notebooks/01_anomaly_detection.ipynb`: Autoencoder on MNIST, flags digit 9 as the anomaly class via reconstruction error.
- `notebooks/02_diagnosis.ipynb`: K-Means on raw MNIST pixels with an elbow analysis and a 10×10 confusion matrix.
- `notebooks/03_prognosis.ipynb`: SVR vs LSTM on NASA C-MAPSS FD001 turbofan data, scored with RMSE, MAE, R², and the asymmetric C-MAPSS score.

Each notebook trains its model from scratch, saves the weights to `models/`, then reloads them in later cells for evaluation and the final headline figure. Hyperparameters live in `config.py`. Shared C-MAPSS preprocessing lives in `helpers.py`.
