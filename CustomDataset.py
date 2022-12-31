import os

import torch
from PIL import Image
from torchvision import transforms


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_index):
        self.root = root_dir
        self.img_paths = []
        for i in image_index:
            self.img_paths.append(os.path.join(self.root, f"{i}.png"))
        self.images = []
        for img_path in self.img_paths:
            img = Image.open(img_path)
            self.images.append(img)

    def __getitem__(self, index):
        img = self.images[index]
        img = transforms.ToTensor()(img)
        return img

    def __len__(self):
        return len(self.images)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, extension='.tif'):
        super(CustomDataset, self).__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.extension = extension
        self.__init_datas()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # read image
        image = Image.open(self.image_paths[idx])
        # transform image
        image = transforms.ToTensor()(image)
        # get label
        label = self.labels[idx]
        return image, label

    def __init_datas(self):
        self.image_paths = []
        self.labels = []
        # read label file
        with open(self.label_dir, 'r') as f:
            # remove first line
            f.readline()
            for line in f:
                self.labels.append(int(line.split(',')[1]))
        # for each labels,index
        for index, label in enumerate(self.labels):
            # get image path
            image_path = os.path.join(self.image_dir, f'{str(index)}{self.extension}')
            self.image_paths.append(image_path)
