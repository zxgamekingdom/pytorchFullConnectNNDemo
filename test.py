import torch
import torch_directml
from torch.utils.data import DataLoader

import DirInfos

device = torch_directml.device(torch_directml.default_device())
# D:\Library\Desktop\mnist\modelGPU\mnist_model_9_0.1606709361076355.pth
model = torch.load(r"D:\Library\Desktop\mnist\modelGPU\mnist_model_9_0.1606709361076355.pth")
model = model.to(device)
test_loader = DataLoader(DirInfos.test_dataset, batch_size=640, shuffle=True)
correct = 0
total = 0
with torch.no_grad():
    for pack in test_loader:
        images, labels = pack
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}")
