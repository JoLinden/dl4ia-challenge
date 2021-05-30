from torch import nn


class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.conv_network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        self.fc = nn.Linear(in_features=12800*2, out_features=2)

    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(-1, 12800*2)
        x = self.fc(x)
        return x
