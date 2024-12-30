from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from process_and_save_masks import process_and_save_masks

app = FastAPI()

# Define directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "upload"
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"

# Ensure necessary directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount the static folder for serving CSS and JavaScript
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    # Serve the upload form on the homepage
    form_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Images</title>
    </head>
    <body>
        <h1>Upload Original and CAD Images</h1>
        <form action="/upload/" method="post" enctype="multipart/form-data">
            <label for="original_image">Original Image:</label>
            <input type="file" name="original_image" accept="image/*" required><br><br>
            <label for="cad_image">CAD Image:</label>
            <input type="file" name="cad_image" accept="image/*" required><br><br>
            <button type="submit">Upload and Process</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=form_html, status_code=200)

@app.post("/upload/")
async def upload_files(original_image: UploadFile = File(...), cad_image: UploadFile = File(...)):
    # Save uploaded files
    original_image_path = UPLOAD_DIR / original_image.filename
    cad_image_path = UPLOAD_DIR / cad_image.filename

    with open(original_image_path, "wb") as f:
        f.write(await original_image.read())
    with open(cad_image_path, "wb") as f:
        f.write(await cad_image.read())

    # Call the process_and_save_masks function
    process_and_save_masks(original_image_path=str(original_image_path), cad_image_path=str(cad_image_path))

    # Prepare paths for the output images
    result_origin_path = OUTPUT_DIR / "result_origin.jpg"
    result_cad_path = OUTPUT_DIR / "result_cad.jpg"

    # Generate HTML to display images side by side
    result_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Results</title>
    </head>
    <body>
        <h1>Processing Complete</h1>
        <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
            <div>
                <h2>Original Image with Mask</h2>
                <img src="/static/{result_origin_path.name}" alt="Result Origin" style="max-width: 500px;">
            </div>
            <div>
                <h2>CAD Image with Mask</h2>
                <img src="/static/{result_cad_path.name}" alt="Result CAD" style="max-width: 500px;">
            </div>
        </div>
        <br>
        <a href="/">Upload More Images</a>
    </body>
    </html>
    """
    
    # Copy output files to the static directory for serving
    result_origin_path.rename(STATIC_DIR / result_origin_path.name)
    result_cad_path.rename(STATIC_DIR / result_cad_path.name)

    return HTMLResponse(content=result_html, status_code=200)