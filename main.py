from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image  # Assuming process_image returns a PIL Image

app = FastAPI()

# Define directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "upload"
STATIC_DIR = BASE_DIR / "static"

# Ensure necessary directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount the static folder for serving CSS and JavaScript
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def process_image(tif_path: Path) -> Image.Image:
    # Dummy function for processing images, replace with actual implementation
    # Returning a placeholder blank image for now
    return Image.new("RGB", (200, 200), color="green")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    # Serve the index.html file when visiting "/"
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(), status_code=200)

@app.post("/upload/")
async def upload_files(tif: UploadFile = File(...), jpg: UploadFile = File(...)):
    # Save uploaded files
    tif_path = UPLOAD_DIR / tif.filename
    jpg_path = UPLOAD_DIR / jpg.filename

    with open(tif_path, "wb") as f:
        f.write(await tif.read())
    with open(jpg_path, "wb") as f:
        f.write(await jpg.read())

    # Process the TIFF file
    result_image = process_image(tif_path)

    # Save the result as PNG
    result_path = UPLOAD_DIR / "result.png"
    result_image.save(result_path, format="PNG")

    return FileResponse(result_path, media_type="image/png", filename="result.png")