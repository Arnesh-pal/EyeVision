import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image

# --- 1. Define Model and Class Names ---

# Define the model architecture (must be the same as during training)
# Here we are loading a pretrained ResNet50 model
model = models.resnet50(weights=None) # Not loading pretrained weights, as we will load our own state dict
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4) # Adjust the final layer to have 4 output classes

# Load the saved model weights (the state_dict)
# Make sure 'resnet50_final.pth' is in the same directory as this script, or provide the full path.
model_path = 'resnet50_final.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Use map_location for CPU compatibility

# Set the model to evaluation mode
model.eval()

# Define the class names corresponding to the model's output indices (0, 1, 2, 3)
# IMPORTANT: Replace these with your actual class names in the correct order.
# The order should match the one ImageFolder used during training (usually alphabetical).
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']


# --- 2. Define Image Transformations ---

# Create the same transformation pipeline used during training
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# --- 3. Create the Prediction Function ---

def predict(input_image: Image.Image):
    """
    Takes a PIL image, processes it, and returns a dictionary of class probabilities.
    """
    # Apply the transformations to the input image
    # The unsqueeze(0) adds a batch dimension (B, C, H, W), which the model expects
    image_tensor = data_transforms(input_image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to convert model outputs to probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Create a dictionary of class names and their probabilities
    confidences = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    return confidences


# --- 4. Set up the Gradio Interface ---

# Create the user interface with Gradio
# gr.Image(type="pil") ensures the input is a PIL Image, matching our function's expectation
# gr.Label(num_top_classes=4) will display the top 4 classes and their scores
# The `title`, `description`, and `examples` make the app more user-friendly
app_interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an eye image"),
    outputs=gr.Label(num_top_classes=4, label="Diagnosis Confidence"),
    title="Eye Disease Diagnosis",
    description="Upload a retinal image to classify it into one of four categories: Cataract, Diabetic Retinopathy, Glaucoma, or Normal. This tool uses a ResNet50 deep learning model.",
    examples=[
        # IMPORTANT: Replace these with paths to actual example images you provide
        # ['example_cataract.jpg'],
        # ['example_normal.jpg']
    ]
)

# Launch the app
if __name__ == "__main__":
    app_interface.launch()