
import torch
import numpy as np
DATA_PATH = "/kaggle/input/datasets/sruthyks60/metooo/final_research_sequences.csv"
LABEL_PATH = "/kaggle/input/sruthyksme/anomaly_label.csv"
MAX_LEN = 100
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.2
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_CANDIDATES = np.arange(0.01, 0.99, 0.005)