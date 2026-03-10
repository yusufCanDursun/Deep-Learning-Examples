import torch
import torch.nn as nn
import torch.optim as optim

# Girdiler
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
# Çıktılar
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# Çok Katmanlı Sinir Ağı Modeli
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


model = XORModel()

# Kayıp Fonksiyonu ve Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Eğitim Döngüsü
epochs = 1000
for epoch in range(epochs):
    # İleri besleme
    predictions = model(X)
    loss = criterion(predictions, y)

    # Geri yayılım
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print("\nTest Sonuçları:")
with torch.no_grad():
    final_preds = model(X)
    for i in range(len(X)):
        print(f"Girdi: {X[i].numpy()} -> Tahmin: {final_preds[i].item():.4f} (Hedef: {y[i].item()})")