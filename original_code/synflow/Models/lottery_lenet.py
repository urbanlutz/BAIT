import torch
import torch.nn as nn
from original_code.synflow.Layers import layers
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, input_shape, num_classes=10, dense_classifier=False, pretrained=False):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            layers.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            layers.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            layers.Linear(64*14*14, 256),
            nn.ReLU(inplace=True),
            layers.Linear(256, 256),
            nn.ReLU(inplace=True),
            layers.Linear(256, num_classes),
        )

        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class LeNet_300_100(nn.Module):
    '''A LeNet fully-connected model for MNIST'''

    def __init__(self, input_shape, num_classes, dense_classifier=False, pretrained=False, init_strategy="standard"):
        super(LeNet_300_100, self).__init__()

        layers_list = []
        current_size = 784  # 28 * 28 = number of pixels in MNIST image.
        model_name = "lenet_300_100"
        plan = [int(n) for n in model_name.split('_')[1:]]
        for size in plan:
            layers_list.append(layers.Linear(current_size, size))
            current_size = size

        self.fc_layers = nn.ModuleList(layers_list)
        self.fc = layers.Linear(current_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.init_strategy = init_strategy

        # self._initialize_weights(init_strategy)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten.
        for layer in self.fc_layers:
            x = F.relu(layer(x))

        return self.fc(x)

    # def _initialize_weights(self, init_strategy):
    #     inits[init_strategy][self.modules()]
       