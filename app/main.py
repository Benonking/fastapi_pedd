from fastapi import FastAPI, UploadFile, File
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
import io
import requests

app = FastAPI()

# Download the model from Azure Blob Storage once on startup
MODEL_PATH = "model.pth"
MODEL_URL = "https://mlopsws1858760940.blob.core.windows.net/azureml/LR_model_final.pth"

def download_model():
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

download_model()

# Load the model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
last_layer = model.fc
in_feat = last_layer.in_features
modified_last_layer = nn.Sequential()
modified_last_layer.append(nn.Linear(in_feat, 256))
relu = nn.ReLU()
modified_last_layer.append(relu)
modified_last_layer.append(nn.Dropout(0.4))
modified_last_layer.append(nn.Linear(256,23))
model.fc = modified_last_layer


model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=1)
    
    top_prob, top_idx = torch.max(probs, dim=0)
    if top_prob.item() > 0.4:
        return {"class": int(top_idx), "probability": top_prob.item()}
    else:
        return {"class": "Uncertain", "probability": top_prob.item()}
