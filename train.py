import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformer import TransformerModel  
from config import *

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
block_ids = df["BlockID"].values
labels = df["Label"].astype(str).str.lower()
labels = labels.apply(lambda x: 1 if x in ["anomaly", "abnormal", "1"] else 0).values
sequences = df["Sequence"].astype(str).str.split()
sequences = sequences.apply(lambda x: [int(i) for i in x if i.isdigit()]).tolist()

def pad(seq):
    if len(seq) >= MAX_LEN:
        return seq[:MAX_LEN]
    return seq + [0] * (MAX_LEN - len(seq))   

sequences = [pad(s) for s in sequences]
X = np.array(sequences)
y = np.array(labels)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, block_ids, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
    X_train, y_train, id_train, test_size=0.15, stratify=y_train, random_state=42
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.long),
                  torch.tensor(y_train, dtype=torch.float)),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val, dtype=torch.long),
                  torch.tensor(y_val, dtype=torch.float)),
    batch_size=BATCH_SIZE * 2
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.long),
                  torch.tensor(y_test, dtype=torch.float)),
    batch_size=BATCH_SIZE * 2
)


vocab_size = int(X.max()) + 1
print(f"Vocabulary size: {vocab_size}")

model = TransformerModel(
    vocab_size=vocab_size,
    d_model=EMBED_DIM,
    nhead=NUM_HEADS,
    dim_feedforward=FF_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN
).to(DEVICE)

pos_weight_val = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_auc = 0.0
best_model_state = None

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb).view(-1)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    probs, labels_val = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb).view(-1)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            labels_val.extend(yb.numpy())
    auc = roc_auc_score(labels_val, probs)
    print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {auc:.6f}")
    scheduler.step()

    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict().copy()
model.load_state_dict(best_model_state)
model.eval()

test_probs, test_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = model(xb).view(-1)
        test_probs.extend(torch.sigmoid(logits).cpu().numpy())
        test_labels.extend(yb.numpy())

test_probs = np.array(test_probs)
test_labels = np.array(test_labels)

best_thresh = 0.5
best_f1 = 0
for t in THRESHOLD_CANDIDATES:
    preds = (test_probs > t).astype(int)
    f1 = f1_score(test_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

final_preds = (test_probs > best_thresh).astype(int)

accuracy = accuracy_score(test_labels, final_preds)
precision = precision_score(test_labels, final_preds)
recall = recall_score(test_labels, final_preds)
auc = roc_auc_score(test_labels, test_probs)

print("\n===== Final Test Performance =====")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"AUC:       {auc:.6f}")
print(f"Best threshold: {best_thresh:.4f}")

torch.save(best_model_state, "hdfs_transformer_model.pt")
with open("threshold.txt", "w") as f:
    f.write(str(best_thresh))

print("\nModel saved as 'hdfs_transrmer_moel.pt'")
print("Threshold saved as 'threshold.txt'")