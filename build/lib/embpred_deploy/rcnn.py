import cv2
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def extract_emb_frame_2d(embframe, model, device):# what is return type of this function? 
    return ExtractEmbFrame(embframe, embframe, embframe, model, device)[0]

def ExtractEmbFrame(r_channel, g_channel, b_channel, model, device):
    
    r_rgb = cv2.cvtColor(r_channel, cv2.COLOR_GRAY2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_tensor = transform(r_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)

    best_bbox = None
    best_score = 0
    for bbox, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
        if score > best_score:
            best_bbox = bbox
            best_score = score

    if best_bbox is None:
        
        padded_r = np.zeros((800, 800), dtype=np.uint8) # update the size
        padded_g = padded_r
        padded_b = padded_r
        
        return padded_r, padded_g, padded_b

    else:
        
        best_bbox = best_bbox.cpu().numpy()

        x_min, y_min, x_max, y_max = best_bbox.astype(int)
        cropped_r = r_channel[y_min:y_max, x_min:x_max]
        cropped_g = g_channel[y_min:y_max, x_min:x_max]
        cropped_b = b_channel[y_min:y_max, x_min:x_max]

        h, w = cropped_r.shape
    
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            pad_top = pad_bottom = 0
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            pad_left = pad_right = 0
    
        padded_r = cv2.copyMakeBorder(
            cropped_r,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  
        )
        padded_g = cv2.copyMakeBorder(
            cropped_g,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  
        )
        padded_b = cv2.copyMakeBorder(
            cropped_b,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  
        )

        return padded_r, padded_g, padded_b

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