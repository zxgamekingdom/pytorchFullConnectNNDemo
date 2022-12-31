import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from DirInfos import train_dataset, test_dataset
from NN import NN

train_loader = DataLoader(train_dataset, batch_size=640, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=640, shuffle=True)

epochs = 10
# device = torch_directml.device(torch_directml.default_device())
model = NN()
# model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# D:\Library\Desktop\mnist\model
if not os.path.exists(r"D:\Library\Desktop\mnist\model"):
    os.mkdir(r"D:\Library\Desktop\mnist\model")
# delete all files in the folder
for file in os.listdir(r"D:\Library\Desktop\mnist\model"):
    os.remove(os.path.join(r"D:\Library\Desktop\mnist\model", file))
# 开始计时
import time

start = time.time()
for epoch in range(epochs):
    run_loss = 0.0
    for idx, pack in enumerate(train_loader):
        images, labels = pack
        # images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        if idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {idx}, Loss: {run_loss / 10}")
            run_loss = 0.0
    print(f"Epoch {epoch} - Loss: {loss.item()}")
    # if path not exist, create it

    torch.save(model.state_dict(), f'D:\Library\Desktop\mnist\model\mnist_model_{epoch}_{loss}.pth')
# 结束计时
end = time.time()
print(f"Time: {end - start}")
