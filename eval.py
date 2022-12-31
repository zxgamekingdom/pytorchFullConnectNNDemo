import torch
import torch_directml
from torch.utils.data import DataLoader
from torch.nn import Module
import time

import DirInfos
from CustomDataset import EvalDataset
from NN import NN

if __name__ == '__main__':
    device = torch_directml.device(torch_directml.default_device())
    model = NN()
    model = model.to(device)
    # D:\Library\Desktop\mnist\modelGPU\mnist_model_9_0.1606709361076355.pth
    model.load_state_dict(torch.load(r"D:\Library\Desktop\mnist\modelGPU\mnist_model_9_0.1606709361076355.pth"))
    # gen 0 to 100 array
    array = [i for i in range(10)]
    dataset = EvalDataset(r"D:\Library\Desktop\mnist\test", array)
    eval_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    total_time = 0
    with torch.no_grad():
        for index, pack in enumerate(eval_loader):
            images = pack
            images = images.to(device)
            start = time.time()
            outputs = model(images)
            cpu = outputs.to('cpu')
            end = time.time()
            total_time += end - start
            _, predicted = torch.max(outputs.data, 1)
            print(f"Predicted: {predicted}")
            print(f"Index: {index}")
            # end time
        print(f"Time: {total_time}")
