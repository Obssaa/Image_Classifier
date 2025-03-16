import torch
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
import model_jigsaw_nested as model_jigsaw
import utils
from huggingface_hub import hf_hub_download

# Load pre-trained model
device = torch.device("cpu")  # Use CPU for inference
nb_cls = 14  # Clothing1M has 14 classes
arch = "vit_small_patch16"  # Choose model architecture

# Download model from Hugging Face Hub
repo_id = "Obsaa/jigsaw-vit-clothing1m-obsa4930"  # Replace with your Hugging Face repo ID
filename = "model.pth"
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load model
print("Loading model...")
model = model_jigsaw.create_model(arch, nb_cls).to(device)
checkpoint = torch.load(model_path, map_location=device)
utils.load_my_state_dict(model, checkpoint['net'])
model.eval()
print("Model loaded successfully!")

# Define class labels for Clothing1M
clothing_labels = [
    "T-Shirt", "Shirt", "Shawl", "Chiffon", 
    "Dress", "Underwear", "Knitwear", "Vest", "Sweater",
    "Hoodie", "Down Coat", "Jacket", "Windbreaker", "Suit"
]

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Function to make predictions
def classify_image(image):
    image = transform(image).unsqueeze(0).to(device)  # Preprocess image
    with torch.no_grad():
        _, features = model.forward_cls(image)  # Extract features
        outputs = model.head(features)  # Get classification scores
        _, predicted = torch.max(outputs, 1)  # Get predicted class index

    label = clothing_labels[predicted.item()]
    return label

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Clothing1M Image Classifier",
    description="Upload an image to classify it into one of 14 clothing categories.",
    examples=[
        ["examples/example1.jpg"],
        ["examples/example2.jpg"],
        ["examples/example3.jpg"]
    ]
)

# Run the interface
if __name__ == "__main__":
    iface.launch()
