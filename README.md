# deeplearning-practice
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Thiết lập tính tái lập (Reproducibility)
torch.manual_seed(42)
np.random.seed(42)

# 2. Tạo tập dữ liệu chuỗi thời gian (Hàm tổng hợp xu hướng + dao động điều hòa)
def generate_time_series(n_samples=1200):
    time = np.linspace(0, 100, n_samples)
    series = np.sin(time) + 0.1 * time + np.random.normal(0, 0.05, n_samples)
    return series.astype(np.float32)

raw_data = generate_time_series()

# Chuẩn hóa dữ liệu về khoảng [0, 1]
data_min, data_max = raw_data.min(), raw_data.max()
normalized_data = (raw_data - data_min) / (data_max - data_min)

# 3. Tạo các cửa sổ thời gian (Sliding Window / Sequence Construction)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 20
X, y = create_sequences(normalized_data, SEQ_LENGTH)

# Chuyển đổi sang Torch Tensor (Batch, Seq_Len, Features)
X_tensor = torch.tensor(X).unsqueeze(-1)
y_tensor = torch.tensor(y).unsqueeze(-1)

# Chia tập huấn luyện (80%) và kiểm thử (20%)
train_size = int(len(X) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# 4. Định nghĩa Kiến trúc Mạng LSTM
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Khởi tạo mô hình
model = LSTMPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 5. Vòng lặp Huấn luyện (Training Loop)
EPOCHS = 150
BATCH_SIZE = 32

model.train()
for epoch in range(EPOCHS):
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0
    
    for i in range(0, X_train.size(0), BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_x.size(0)
        
    if (epoch + 1) % 25 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss / X_train.size(0):.6f}")

# 6. Đánh giá và Suy luận (Evaluation & Inference)
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, y_test)
    print(f"\nMSE trên tập Kiểm thử (Test Loss): {test_loss.item():.6f}")

predictions_rescaled = test_preds.numpy() * (data_max - data_min) + data_min
actuals_rescaled = y_test.numpy() * (data_max - data_min) + data_min

print("\nHoàn tất huấn luyện mô hình LSTM.")
