# %%
from datetime import datetime
from math import log

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# %%
computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tensor_transform = transforms.ToTensor()

# Download the MNIST Dataset
dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=tensor_transform
)
data_test_set = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=tensor_transform
)

# DataLoader is used to load the dataset
# for training
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=True)

# %%

# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class CAE(torch.nn.Module):
    def __init__(self, device, *, loss_function=None, optimizer=None):
        super().__init__()

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.to(device)

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 12, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(12, 16, 3, padding=1),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(16, 12, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(12, 3, 3, padding=1),
        )

    def forward(self, x):
        """used to forward prop data in the model"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(
        self,
        epochs,
        data_loader,
        *,
        loss_function=None,
        optimizer=None,
        logging=False,
        prog_size=30,
    ):
        """Used to train a model given some data"""
        epoch_outputs = []
        if loss_function is None:
            loss_function = self.loss_function
        if loss_function is None:
            raise Exception("loss_function is None, loss_function must be defined")
        if optimizer is None:
            optimizer = self.optimizer
        if optimizer is None:
            raise Exception("optimizer is None, optimizer must be defined")

        load_size = len(data_loader)
        chars_in_dataset = int(log(load_size, 10)) + 1
        for epoch in range(epochs):
            output = []
            if logging:
                print(f"Epoch:{epoch+1}/{epochs}")
            for i, (data, _label) in enumerate(data_loader):
                if logging:
                    progress = ">" * (i // (load_size // prog_size))
                    print(
                        f"\r\t{{{i:{chars_in_dataset}}/{load_size}}} - [{progress:{prog_size+1}}] - {100*i/load_size:3.0f}%",
                        end="",
                    )
                data = data.to(model.device)
                reconstructed = self(data)
                loss = loss_function(reconstructed, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output.append((epoch, i, loss, data, reconstructed))

            if logging:
                avg_loss = sum([x[2] for x in output]) / len(output)
                print(
                    f"\r\t{{{load_size}/{load_size}}} - [{'#'*(prog_size+1)}] - 100% - Loss:{avg_loss:.2f}",
                )
            epoch_outputs.append(output)
        return epoch_outputs


# %%

# Model Initialization
mse_loss = torch.nn.MSELoss()
model = CAE(
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
outputs = model.fit(EPOCHS, loader, optimizer=adam, logging=True)

# %%
# Defining the Plot Style
plt.style.use("fivethirtyeight")
plt.xlabel("Iterations")
plt.ylabel("Loss")

# Plotting the last 100 values
plt.plot([y[2].detach().numpy() for x in outputs for y in x])
plt.show()

for run in outputs:
    for e, _, _, image, reconstruction in run[-5:]:
        image = image.reshape(-1, 32, 32)
        reconstruction = reconstruction.reshape(-1, 32, 32)

        fig = plt.figure()

        color_map = ["Reds", "Greens", "Blues"]
        for index, color in enumerate(color_map):
            ax = fig.add_subplot(1, 3, index + 1)
            ax.axis("off")
            ax.set_title(color[:-1])
            plt.imshow(image[index], cmap=color)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.axis("off")
        ax.set_title("real")
        plt.imshow(image[0], cmap="gray")

        ax = fig.add_subplot(1, 2, 2)
        ax.axis("off")
        ax.set_title(f"fake:{e}")
        plt.imshow(reconstruction.detach().numpy()[0], cmap="gray")

        plt.show()

#%% Save Model
stamp = datetime.now().strftime("%y%m%d-%H%M%S")
torch.save(model.state_dict(), "./models/latest.pt")
torch.save(model.state_dict(), f"./models/{stamp}.pt")
#%% Load model
model = CAE(
    computation_device,
    loss_function=mse_loss,
)
model.load_state_dict(torch.load("./models/latest.pt"))
model.eval()
