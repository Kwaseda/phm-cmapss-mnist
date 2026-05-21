"""
Hyperparameters and paths for the PHM notebooks.

Edit values here and rerun the notebooks.
"""

from pathlib import Path

# Paths (relative to phm-notebooks/)
NOTEBOOKS_ROOT = Path(__file__).parent
MODELS_DIR = NOTEBOOKS_ROOT / "models"
CMAPSS_DIR = NOTEBOOKS_ROOT / "data" / "CMAPSS"
MNIST_DIR = NOTEBOOKS_ROOT / "data" / "MNIST"

SEED = 42

# Part I + II shared: MNIST loading
MNIST_BATCH_SIZE = 256
MNIST_NUM_WORKERS = 0
MNIST_NORMAL_DIGITS = list(range(9))  # 0-8 are normal
MNIST_ANOMALY_DIGIT = 9
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

# Part I: Autoencoder
AE_INPUT_DIM = 28 * 28
AE_HIDDEN_1 = 256
AE_HIDDEN_2 = 64
AE_LATENT_DIM = 8 # 32
AE_EPOCHS = 20
AE_LR = 1e-3
AE_PATIENCE = 5
AE_MIN_DELTA = 1e-4
AE_VAL_RATIO = 0.1
AE_THRESHOLD_K = 1.0  # threshold = mean + K * std (assignment spec says mean + std)
AE_CHECKPOINT_NAME = "autoencoder.pt"

# Part II: K-Means on raw MNIST pixels
KMEANS_K_RANGE = list(range(2, 16))
KMEANS_N_INIT = "auto"
KMEANS_SAMPLE_SIZE = (
    10000  # subsample for the elbow sweep (raw 784-dim k-means is slow)
)

# Part III: C-MAPSS shared
CMAPSS_DATASET = "FD001"
CMAPSS_WINDOW = 50
CMAPSS_VAL_RATIO = 0.1
CMAPSS_RUL_CAP = 125

# Part III: SVR
SVR_C = 100.0
SVR_GAMMA = 0.001
SVR_EPSILON = 0.5
SVR_CHECKPOINT_NAME = "svr.pkl"

# Part III: LSTM
LSTM_BATCH_SIZE = 64
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.3943641797072
LSTM_HEAD_HIDDEN = 50
LSTM_LR = 3e-3
LSTM_WEIGHT_DECAY = 1.1038251665362905e-05
LSTM_EPOCHS = 50
LSTM_PATIENCE = 7
LSTM_MIN_DELTA = 1e-4
LSTM_CHECKPOINT_NAME = "lstm.pt"
