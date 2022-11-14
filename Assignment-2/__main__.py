# %%
from math import log
from model import GrayToColor, CAE
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# %%
computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

# Download the MNIST Dataset
dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=tensor_transform,
)
data_test_set = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=tensor_transform,
)
both = torch.utils.data.ConcatDataset([dataset, data_test_set])
# DataLoader is used to load the dataset
# for training
train_set, test_set, validate_set = torch.utils.data.random_split(
    both, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=100, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=100, shuffle=True
)
validate_loader = torch.utils.data.DataLoader(
    dataset=validate_set, batch_size=100, shuffle=True
)


# %%

# Model Initialization
mse_loss = torch.nn.MSELoss()
model = GrayToColor(
    computation_device,
    loss_function=mse_loss,
)
outputs = []
losses = []
# %%
# Validation using MSE Loss function

# Using an Adam Optimizer with lr = 0.1
adam = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)

EPOCHS = 5

# %%
outputs, test_loss = model.fit(
    EPOCHS, train_loader, optimizer=adam, logging=True, test_loader=test_loader
)

# %%
fig = plt.figure()

plt.xlabel("Iterations")
plt.ylabel("log(Loss)")

ax = fig.add_subplot(1, 2, 1)
ax.set_title("training")
ax.plot([log(y[2].detach().numpy()) for x in outputs for y in x])

ax = fig.add_subplot(1, 2, 2)
ax.set_title("testing")
ax.plot([log(y[1].detach().numpy()) for x in test_loss for y in x])

plt.show()

# %%
for run in outputs:
    for e, _index, _loss, image, reconstruction in run[-2:]:
        print(image.shape)
        image = image[0].permute((1, 2, 0))
        reconstruction = reconstruction[0].permute((1, 2, 0))

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.axis("off")
        ax.set_title("real")
        plt.imshow(image, cmap="gray")

        ax = fig.add_subplot(1, 2, 2)
        ax.axis("off")
        ax.set_title(f"fake:{e}")
        plt.imshow(reconstruction.detach().numpy())

        plt.show()

#%%
