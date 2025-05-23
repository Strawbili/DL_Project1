import torch.nn as nn

class LeNet5(nn.Module):
    """
    Classic LeNet-5 architecture for handwritten digit recognition.
    Input: 1x28x28 grayscale image
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After conv/pool: feature map size = ((28-4)/2 - 4)/2 = 4 => 16*4*4
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.tanh3 = nn.Tanh()

        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()

        self.fc3 = nn.Linear(in_features=84, out_features=36)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.tanh3(x)

        x = self.fc2(x)
        x = self.tanh4(x)

        x = self.fc3(x)
        return x
