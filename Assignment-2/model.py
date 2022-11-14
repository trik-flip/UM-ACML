from datetime import datetime
from math import log
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rgb_to_grayscale


class CAE(torch.nn.Module):
    def __init__(self, device, *, loss_function=None, optimizer=None):
        super().__init__()

        self.name = CAE
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

    def test_error(
        self,
        data_loader,
        loss_function=None,
    ):
        losses = []
        if loss_function is None:
            loss_function = self.loss_function
        if loss_function is None:
            raise Exception("loss_function is None, loss_function must be defined")
        for i, (data, _label) in enumerate(data_loader):
            data = data.to(self.device)
            reconstructed = self(data)
            loss = loss_function(reconstructed, data)

            losses.append((i, loss, data, reconstructed))

        return losses

    def fit(
        self,
        epochs,
        data_loader,
        *,
        loss_function=None,
        optimizer=None,
        logging=False,
        prog_size=30,
        test_loader=None,
    ):
        """Used to train a model given some data"""
        test_loss = []
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
                data = data.to(self.device)
                reconstructed = self(data)

                # =============================
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                ax.axis("off")
                ax.set_title("real")
                plt.imshow(data.detact().numpy())

                ax = fig.add_subplot(1, 2, 2)
                ax.axis("off")
                ax.set_title(f"fake:{epoch}")
                plt.imshow(reconstructed.detach().numpy())

                plt.show()
                # =============================

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
            if test_loader is not None:
                losses = self.test_error(test_loader, loss_function)
                test_loss.append(losses)
        return epoch_outputs, test_loss


class GrayToColor(torch.nn.Module):
    def __init__(self, device, *, loss_function=None, optimizer=None) -> None:
        super().__init__()

        self.name = "GTC"
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.to(device)

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1),
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
        test_loader=None,
    ):
        test_loss = []
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
                data = data.to(self.device)

                input_data = rgb_to_grayscale(data, num_output_channels=1)
                reconstructed = self(input_data)
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
            if test_loader is not None:
                losses = self.test_error(test_loader, loss_function)
                test_loss.append(losses)
        return epoch_outputs, test_loss

    def test_error(self, data_loader, loss_function=None):
        losses = []
        if loss_function is None:
            loss_function = self.loss_function
        if loss_function is None:
            raise Exception("loss_function is None, loss_function must be defined")
        for i, (data, _label) in enumerate(data_loader):
            data = data.to(self.device)
            input_data = rgb_to_grayscale(data, num_output_channels=1)
            reconstructed = self(input_data)
            loss = loss_function(reconstructed, data)

            losses.append((i, loss, data, reconstructed))

        return losses


def save_model(model):

    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    torch.save(model.state_dict(), f"./models/{model.name}.latest.pt")
    torch.save(model.state_dict(), f"./models/{model.name}.{stamp}.pt")


def load_model(model_type):
    model = model_type(
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        loss_function=torch.nn.MSELoss(),
    )
    model.load_state_dict(torch.load(f"./models/{model.name}.latest.pt"))
    model.eval()
    return model
