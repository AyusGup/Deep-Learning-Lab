import streamlit as st
import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import tempfile

# Function to initialize weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Preactivation Block without Skip Connections
class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.3):
        super(PreActBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        
        return out

# ResNet-10 Model without Skip Connections
class ResNet10(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.3):
        super(ResNet10, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(128, 2, stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(256, 2, stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(512, 2, stride=2, dropout_prob=dropout_prob)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Apply weight initialization
        initialize_weights(self)
    
    def _make_layer(self, out_channels, blocks, stride, dropout_prob):
        layers = []
        layers.append(PreActBlock(self.in_channels, out_channels, stride, dropout_prob))
        self.in_channels = out_channels  # Update for next blocks
        
        for _ in range(1, blocks):
            layers.append(PreActBlock(out_channels, out_channels, dropout_prob=dropout_prob))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Define class labels
CLASS_NAMES = ["Cat", "Dog"]

# Load model correctly
device = torch.device("cpu")  # Ensure CPU usage
model = ResNet10(num_classes=2).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode

def predict_from_path(image_path):
    """Load an image from a path, predict the class, and return the result."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize
    img_array = np.transpose(img_array, (2, 0, 1))  # Change shape to (C, H, W)
    img_tensor = torch.tensor(img_array).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return CLASS_NAMES[predicted.item()]

st.title("Cat vs Dog Classifier 🐱🐶")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as temp_file:
        img.save(temp_file.name)
        temp_path = temp_file.name

    # Get prediction
    class_label = predict_from_path(temp_path)

    # Remove temporary file
    os.remove(temp_path)

    # Show result
    st.write(f"Prediction: **{class_label}** 🎯")
