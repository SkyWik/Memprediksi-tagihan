import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from dataset_generator import generate_pt_maju_bersama_data
from model import PaymentRiskDNN

def train_and_evaluate():
    print("=== PROSES ANALITIK & PEMODELAN AI PT MAJU BERSAMA ===")
    # 1. Mengambil data sintetik
    df = generate_pt_maju_bersama_data(num_samples=5000)
    
    X = df.drop('is_late_30_days', axis=1).values
    y = df['is_late_30_days'].values
    
    # 2. Split Data (Simulasi Temporal Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Preprocessing (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert ke PyTorch Tensor
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # 4. Inisialisasi Model & Loss Function
    # Menghitung pos_weight untuk mengatasi imbalanced data
    pos_weight_val = (len(y_train) - sum(y_train)) / sum(y_train)
    pos_weight = torch.FloatTensor([pos_weight_val])
    
    model = PaymentRiskDNN(input_dim=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Mengatasi kelas tidak seimbang
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 5. Training Loop
    epochs = 50
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")
            
    # 6. Evaluasi Sesuai Arahan Sesi 1 (ROC-AUC, Recall, Confusion Matrix)
    model.eval()
    with torch.no_grad():
        preds_proba = model(X_test_t).numpy()
        preds_binary = (preds_proba > 0.5).astype(int)
        
    roc_auc = roc_auc_score(y_test, preds_proba)
    cm = confusion_matrix(y_test, preds_binary)
    
    print("\n=== HASIL EVALUASI MODEL DEEP LEARNING ===")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, preds_binary, target_names=['Tepat Waktu', 'Terlambat (>30 hari)']))

if __name__ == '__main__':
    train_and_evaluate()
