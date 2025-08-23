import matplotlib.pyplot as plt
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode
from HNet.datasets_utils.data_classes import MnistRotDataset
import random
import keyboard  # Install with `pip install keyboard`

# load mnist dataset
pad = Pad((0, 0, 1, 1), fill=0)

# to reduce interpolation artifacts (e.g. when testing the model on rotated images),
# we upsample an image by a factor of 3, rotate it and finally downsample it again
resize1 = Resize(87)
resize2 = Resize(29)

totensor = ToTensor()
train_transform = Compose([
    pad,
    resize1,
    RandomRotation(180., interpolation=InterpolationMode.BILINEAR, expand=False),
    resize2,
    totensor,
])

mnist_train = MnistRotDataset(mode='train', transform=train_transform)
def show_random_image():
    idx = random.randint(0, len(mnist_train) - 1)
    image, label = mnist_train[idx]

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()

while True:
    show_random_image()
    user_input = input("Press q to continue").strip().lower()
    if user_input != "q":
        break