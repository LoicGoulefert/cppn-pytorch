import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import CPPN

img_size = 512
scale = 2
z_dim = 16
batch_size = img_size ** 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


class PixelDataSet(Dataset):
    def __init__(self, img_size, z):
        super(PixelDataSet).__init__()
        self.img_size = img_size
        self.z = z

    def __getitem__(self, idx):
        x, y = idx // self.img_size, idx % self.img_size
        return get_input_vec(x, y, z)

    def __len__(self):
        return self.img_size ** 2


def get_input_vec(x, y, z):
    r = np.sqrt(
        ((x - 0.5 * img_size) * scale) ** 2 + ((y - 0.5 * img_size) * scale) ** 2
    )

    z = torch.Tensor(z)
    xyr = torch.Tensor([x, y, r])
    input_vec = torch.cat((z, xyr), dim=0)

    return input_vec.to(device)


if __name__ == "__main__":
    print("Device used:", device)

    z = torch.rand(z_dim) * scale
    cppn = CPPN(z_dim=z_dim).to(device)
    dataset = PixelDataSet(img_size, z)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    image = torch.zeros(img_size ** 2)

    for i, batch in tqdm(enumerate(dataloader)):
        out = cppn(batch)
        image[i * batch_size : (i + 1) * batch_size] = out.view(-1)

    image = image.detach().numpy()
    image = image.reshape(img_size, img_size)

    plt.imshow(image, cmap="gray")
    plt.show()
