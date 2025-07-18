import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# Define the model architecture
class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model
model = FeedForwardNN()
model.load_state_dict(torch.load("handwritten_digit_classification.pth", map_location=torch.device('cpu')))
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit UI
st.title("🧠 Handwritten Digit Recognizer")
st.write("Upload a handwritten digit image (preferably white digit on black background)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    st.success(f"🎯 Predicted Digit: **{predicted_class}**")
