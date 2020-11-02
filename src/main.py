import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import CPPN

img_size = 2048
scale = 2
model_size = 32
z_dim = 8
batch_size = (img_size ** 2) // 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PixelDataSet(Dataset):
    def __init__(self, img_size, z):
        super(PixelDataSet).__init__()
        self.img_size = img_size
        self.z = z
        self.input_vecs = self.make_input_vecs()

    def make_input_vecs(self):
        input_vecs = torch.zeros((self.img_size ** 2, len(self.z) + 3))
        x, y = np.indices((self.img_size, self.img_size))
        r = np.sqrt(
            ((x - 0.5 * img_size) * scale) ** 2 + ((y - 0.5 * img_size) * scale) ** 2
        )

        input_vecs[:, : len(z)] = self.z
        input_vecs[:, len(z)] = torch.Tensor(x).view(-1)
        input_vecs[:, len(z) + 1] = torch.Tensor(y).view(-1)
        input_vecs[:, len(z) + 2] = torch.Tensor(r).view(-1)

        return input_vecs

    def __getitem__(self, idx):
        return self.input_vecs[idx]

    def __len__(self):
        return self.img_size ** 2


if __name__ == "__main__":
    print("Device used:", device)

    z = torch.rand(z_dim) * scale
    cppn = CPPN(z_dim=z_dim, model_size=model_size).to(device)

    dataset = PixelDataSet(img_size, z)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    image = torch.zeros(img_size ** 2)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            out = cppn(batch.to(device))
            del batch
            image[i * batch_size : (i + 1) * batch_size] = out.view(-1)

    image = image.numpy()
    image = image.reshape(img_size, img_size)

    plt.imshow(image, cmap="gray")
    plt.show()
