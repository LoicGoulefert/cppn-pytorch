import torch
import numpy as np
import matplotlib.pyplot as plt

from model import CPPN

img_size = 256
scale = 0.1
z_dim = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_input_vec(x, y, z):
    r = np.sqrt(
        ((x - 0.5 * img_size) * scale) ** 2 + ((y - 0.5 * img_size) * scale) ** 2
    )

    z = torch.Tensor(z)
    xyr = torch.Tensor([x, y, r])
    input_vec = torch.cat((z, xyr), dim=0)

    return input_vec


if __name__ == "__main__":
    print("Device used:", device)

    z = torch.rand(z_dim) * scale
    cppn = CPPN(z_dim=z_dim).to(device)

    image = torch.zeros((img_size, img_size))
    for x in range(img_size):
        for y in range(img_size):
            input_vec = get_input_vec(x, y, z).to(device)
            image[x, y] = cppn(input_vec)

    image = image.detach().numpy()

    plt.imshow(image, cmap="gray")
    plt.show()
