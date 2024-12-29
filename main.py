from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Serve static files (Frontend)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Directory for uploaded files
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load the segmentation model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
model.eval()  # Set model to evaluation mode

def process_image(image_path: Path):
    """Process the image with the segmentation model."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))  # Resize for the model
    image_array = np.array(image).transpose(2, 0, 1) / 255.0  # Normalize
    input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).squeeze().numpy()

    prediction_image = (prediction > 0.5).astype(np.uint8) * 255
    result = Image.fromarray(prediction_image).convert("L")
    return result

@app.post("/upload/")
async def upload_files(tif: UploadFile = File(...), jpg: UploadFile = File(...)):
    tif_path = UPLOAD_DIR / tif.filename
    jpg_path = UPLOAD_DIR / jpg.filename

    with open(tif_path, "wb") as f:
        f.write(await tif.read())
    with open(jpg_path, "wb") as f:
        f.write(await jpg.read())

    result_image = process_image(tif_path)

    result_path = UPLOAD_DIR / "result.jpg"
    result_image.save(result_path)

    return FileResponse(result_path, media_type="image/jpeg", filename="result.jpg")