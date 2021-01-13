from skimage import io, img_as_ubyte
import torch
from unet import UNet
from datetime import datetime
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
import os
import random

path = "/home/dungpb/Work/ComputerVision/Code/Data"
num_epoch = 10
dev = "cpu"
if torch.cuda.is_available():
    dev = "cuda:0"

print(dev)


class LoadData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.build_dataset()

    def build_dataset(self):
        self._input_folder = os.path.join(self.data_dir, 'Image')
        self._label_folder = os.path.join(self.data_dir, 'Mask')

        self._images = [f for f in os.listdir(self._input_folder) if os.path.isfile(os.path.join(self._input_folder, f))]
        self._masks = [f for f in os.listdir(self._label_folder) if os.path.isfile(os.path.join(self._label_folder, f))]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = torch.from_numpy(img_as_ubyte(io.imread(self._input_folder + "/" + self._images[idx],  as_gray=True)))
        mask = torch.zeros(512,512)
        if self._images[idx] in self._masks:
            mask = torch.from_numpy(io.imread(self._label_folder + "/" + self._images[idx], as_gray=True))
        image = image.float().unsqueeze(0).unsqueeze(0)
        mask = mask.long()
        return image, mask


if __name__ == "__main__":
    now = datetime.now()
    time_stamp = now.strftime("%Y:%m:%d-%H:%M:%S")
    model_path = "/home/dungpb/Work/ComputerVision/Code/model" + "/" + time_stamp + ".pt"
    dataset = LoadData(path)

    model = UNet(dimensions=2)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    idx_list = [i for i, _ in enumerate(dataset)]

    l = len(dataset)

    for e in range(num_epoch):
        random.shuffle(idx_list)
        sum_loss = 0
        count = 0
        for idx in idx_list:
            model.train(True)
            optimizer.zero_grad()
            X, Y = dataset[idx]
            y = model(X)
            loss = criterion(y, Y.unsqueeze(0))
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            count += 1
            print(str(count) + "/" + str(l), end='\r')

        torch.save(model.state_dict(), model_path)
        print(sum_loss/len(dataset))

