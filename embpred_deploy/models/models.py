import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class CustomResNet50(nn.Module):
    def __init__(self, num_classes, num_dense_layers, dense_neurons, freeze_=True, dropout_rate=0):
        """
        Parameters:
            num_classes (int): Number of output classes.
            num_dense_layers (int): Number of custom dense layers to add.
            dense_neurons (int or list): If int, all dense layers will have the same number of neurons.
                                          If list, it must have length equal to num_dense_layers.
            freeze (bool): If True, freeze the pretrained ResNet-50 layers.
            dropout_rate (float): Dropout rate to apply after each dense layer. If set to 0 or less, dropout is not used.
        """
        super(CustomResNet50, self).__init__()
        # Load the ResNet-50 model without downloading pretrained weights
        self.resnet = models.resnet50(weights=None)
        
        # Optionally freeze the ResNet-50 layers
        if freeze_:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Remove the original fully connected layer
        num_ftrs = self.resnet.fc.in_features  # In ResNet-50, this is 2048
        
        self.resnet.fc = nn.Identity()

        # If a single integer is provided for dense_neurons, replicate it for num_dense_layers layers.
        if isinstance(dense_neurons, int):
            dense_neurons = [dense_neurons] * num_dense_layers

        # Build custom dense layers based on specified parameters.
        layers = []
        input_size = num_ftrs
        if num_dense_layers > 0:
            for neurons in dense_neurons:
                layers.append(nn.Linear(input_size, neurons))  # Fully connected layer
                layers.append(nn.ReLU(inplace=True))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                input_size = neurons
        # Final output layer without any activation (CrossEntropyLoss includes softmax)
        layers.append(nn.Linear(input_size, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through ResNet-50 backbone (excluding final FC layer)
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # Flatten to a vector
        # Forward pass through custom classifier layers
        x = self.classifier(x)
        return x  # Return logits (no softmax)

class BigWNet(nn.Module):
    """
    W-Net implementation for image classification, adapted for 224 x 224 x 3 inputs.
    
    Architecture:
        1. Input layer: 224x224x3
        2. Convolutional Layer 1: 32 filters, 3x3 kernel, padding=1, ReLU activation
           followed by MaxPooling (2x2)
        3. Convolutional Layer 2: 64 filters, 3x3 kernel, padding=1, ReLU activation
           followed by MaxPooling (2x2)
        4. Convolutional Layer 3: 128 filters, 3x3 kernel, padding=1, ReLU activation
           followed by MaxPooling (2x2)
        5. Flatten layer
        6. Fully connected Layer 1: 256 units, ReLU activation and optional Dropout
        7. Fully connected Layer 2: 128 units, ReLU activation and optional Dropout
        8. Output Layer: num_classes units with softmax activation
        
    Parameters:
        num_classes (int): Number of output classes.
        dropout (bool): Whether to use dropout in the fully connected layers.
        dropout_rate (float): Dropout rate (probability of an element to be zeroed) if dropout is enabled.
    """
    def __init__(self, num_classes=12, dropout=False, dropout_rate=0.5):
        super(BigWNet, self).__init__()
        # Convolutional layers; using padding=1 keeps spatial size same when using 3x3 kernels.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 3 successive 2x2 pooling operations:
        # 224 -> 112 -> 56 -> 28
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Flattened feature size = 128 * 28 * 28
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout layer or identity based on input flag
        self.dropout = nn.Dropout(dropout_rate) if dropout else nn.Identity()

    def forward(self, x):
        # Convolution block 1
        x = self.pool(F.relu(self.conv1(x)))  # Output: [batch, 32, 112, 112]
        # Convolution block 2
        x = self.pool(F.relu(self.conv2(x)))  # Output: [batch, 64, 56, 56]
        # Convolution block 3
        x = self.pool(F.relu(self.conv3(x)))  # Output: [batch, 128, 28, 28]
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers with dropout if enabled
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        return x


class WNet(nn.Module):
    """
    W-Net implementation for image classification, adapted for 224 x 224 x 3 inputs.
    
    Architecture:
        1. Input layer: 224x224x3
        2. Convolutional Layer 1: 32 filters, 3x3 kernel, padding=1, ReLU activation
           followed by MaxPooling (2x2)
        3. Convolutional Layer 2: 64 filters, 3x3 kernel, padding=1, ReLU activation
           followed by MaxPooling (2x2)
        4. Convolutional Layer 3: 128 filters, 3x3 kernel, padding=1, ReLU activation
           followed by MaxPooling (2x2)
        5. Flatten layer
        6. Fully connected Layer 1: 256 units, ReLU activation and optional Dropout
        7. Fully connected Layer 2: 128 units, ReLU activation and optional Dropout
        8. Output Layer: num_classes units with softmax activation
        
    Parameters:
        num_classes (int): Number of output classes.
        dropout (bool): Whether to use dropout in the fully connected layers.
        dropout_rate (float): Dropout rate (probability of an element to be zeroed) if dropout is enabled.
    """
    def __init__(self, num_classes=12, dropout=False, dropout_rate=0.5, do_xavier=True):
        super(WNet, self).__init__()
        # Convolutional layers; using padding=1 keeps spatial size same when using 3x3 kernels.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        if do_xavier:
            # Initialize weights using Xavier initialization
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.conv3.weight)
        
        # 3 successive 2x2 pooling operations:
        # 224 -> 112 -> 56 -> 28
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Flattened feature size = 128 * 28 * 28
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes, )
        
        # Dropout layer or identity based on input flag
        self.dropout = nn.Dropout(dropout_rate) if dropout else nn.Identity()

    def forward(self, x):
        # Convolution block 1
        x = self.pool(F.relu(self.conv1(x)))  # Output: [batch, 32, 112, 112]
        # Convolution block 2
        x = self.pool(F.relu(self.conv2(x)))  # Output: [batch, 64, 56, 56]
        # Convolution block 3
        x = self.pool(F.relu(self.conv3(x)))  # Output: [batch, 128, 28, 28]
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers with dropout if enabled
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        
        return x



class FirstNet2D(nn.Module):

    def __init__(self, num_classes=4):
        super(FirstNet2D, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Update the in_features based on the input size and conv layers
        self.fc1 = nn.Linear(16 * 197 * 197, 120)  # Updated based on the new image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 796, 796), where N is the size of the batch
        c1 = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 398, 398) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 394, 394) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 197, 197) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 16*197*197) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 16*197*197) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Output layer: (N, 84) Tensor input, and
        # outputs a (N, num_classes) Tensor
        output = self.fc3(f6)
        return output

class BiggerNet3D(nn.Module):
    def __init__(self, num_classes=10):  # You can specify the number of classes here
        super(BiggerNet3D, self).__init__()
        # Define the first convolutional layer: 3 input channels, 8 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 8, 5)
        # Define the second convolutional layer: 8 input channels, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(8, 32, 5)
        # Define the third convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, 3)
        # Define a max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with updated sizes based on input image size (3x800x800)
        self.fc1 = nn.Linear(64 * 97 * 97, 256)  # Update input features to match the output from the conv layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply the first convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply the third convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Apply fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (no activation function here because we'll use CrossEntropyLoss, which includes softmax)
        x = self.fc3(x)
        return x

class BiggerNet3D224(nn.Module):
    def __init__(self, num_classes=10):  # You can specify the number of classes here
        super(BiggerNet3D224, self).__init__()
        # Define the first convolutional layer: 3 input channels, 8 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 8, 5)
        # Define the second convolutional layer: 8 input channels, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(8, 32, 5)
        # Define the third convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, 3)
        # Define a max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with updated sizes based on input image size (3x224x224)
        # After three convolutional layers and pooling, the image size will be reduced to 25x25
        self.fc1 = nn.Linear(64 * 25 * 25, 256)  # Updated to 64 * 25 * 25 to match new conv output
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply the first convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply the third convolution, ReLU, and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Apply fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (no activation function here because we'll use CrossEntropyLoss, which includes softmax)
        x = self.fc3(x)
        return x
    
class BiggestNet3D224(nn.Module):
    def __init__(self, num_classes=10):
        super(BiggestNet3D224, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        x = self.fc3(x)
        return x
  

class SmallerNet3D224(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallerNet3D224, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 4, 3)   # Input channels: 3, Output channels: 4, Kernel size: 3
        self.conv2 = nn.Conv2d(4, 16, 3)  # Input channels: 4, Output channels: 16, Kernel size: 3
        self.pool = nn.MaxPool2d(2, 2)     # Max pooling with 2x2 kernel

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 54 * 54, 128)  # Input features: 16*54*54
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # After conv1 -> ReLU -> pool
        x = self.pool(F.relu(self.conv2(x)))  # After conv2 -> ReLU -> pool

        # Flatten the tensor for fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer (no activation function)
        x = self.fc3(x)
        return x

class SimpleNet3D(nn.Module):
    def __init__(self, num_classes=10, batchNorm=False):  # You can specify the number of classes here
        super(SimpleNet3D, self).__init__()
        # Define the first convolutional layer: 3 input channels, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Define the first batch normalization layer
        self.bn1 = nn.BatchNorm2d(6) if batchNorm else nn.Identity()
        
        # Define the second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Define the second batch normalization layer
        self.bn2 = nn.BatchNorm2d(16) if batchNorm else nn.Identity()
        
        # Define a max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with updated sizes based on input image size (3x224x224)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Input size is calculated from conv layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Apply the first convolution, batch norm, ReLU, and max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Apply the second convolution, batch norm, ReLU, and max pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Apply fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (no activation function here because we'll use CrossEntropyLoss, which includes softmax)
        x = self.fc3(x)
        return x
    

class CustomResNet18(nn.Module):
    def __init__(self, num_classes, num_dense_layers, dense_neurons, input_shape):
        super(CustomResNet18, self).__init__()
        # Load the pretrained ResNet-18 model
        self.resnet = models.resnet18(weights=None)
        
        # Freeze all ResNet-18 layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Remove the original fully connected layer
        num_ftrs = self.resnet.fc.in_features  # Typically 512 for ResNet-18
        self.resnet.fc = nn.Linear(num_ftrs, num_ftrs)

        # Determine the feature size based on input_shape
        # This step ensures compatibility if the ResNet architecture is altered
        dummy_input = torch.zeros(1, *input_shape)
        self.resnet.eval()
        with torch.no_grad():
            features = self.resnet(dummy_input)
            if isinstance(features, torch.Tensor):
                feature_size = features.shape[1]
            else:
                raise ValueError("Unexpected feature output from ResNet backbone.")

        # If a single integer is provided for dense_neurons, replicate it for all dense layers
        if isinstance(dense_neurons, int):
            dense_neurons = [dense_neurons] * num_dense_layers

        # Define the custom dense layers dynamically based on the specified number of layers
        layers = []
        input_size = feature_size
        for i, neurons in enumerate(dense_neurons):
            layers.append(nn.Linear(input_size, neurons))  # Fully connected layer
            layers.append(nn.ReLU(inplace=True))          # ReLU activation
            layers.append(nn.Dropout(0.5))                # Dropout
            input_size = neurons  # Update input size for the next layer

        # Final output layer
        layers.append(nn.Linear(input_size, num_classes))  # Final output layer, no activation

        # Store all layers in nn.Sequential
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the ResNet backbone
        x = self.resnet(x)
        
        # Forward pass through the custom classifier layers
        x = self.classifier(x)
        
        return x

