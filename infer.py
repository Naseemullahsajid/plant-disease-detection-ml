import torch
from PIL import Image
from torchvision import transforms
from model import get_resnet


# ✅ Config
MODEL_PATH = "models/best_model.pth"

IMAGE_PATH = r"E:\project\data\test\Tomato__Target_Spot\0a3b6099-c254-4bc3-8360-53a9f558a0c4___Com.G_TgS_FL 8259.JPG"

device = "cpu"

# ✅ Classes (IMPORTANT: match training order)
classes = [
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ✅ Preprocessing (MUST match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Load model
model = get_resnet(num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Inference
with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    pred = torch.argmax(probs, dim=1).item()

print("Prediction:", classes[pred])
print("Confidence:", probs[0][pred].item())