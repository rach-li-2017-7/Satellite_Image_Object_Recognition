import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from process_and_save_masks import process_and_save_masks, load_model

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

# Load the model when the application starts
model = load_model()

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
    try:
        # Save uploaded files
        original_image_path = UPLOAD_DIR / original_image.filename
        cad_image_path = UPLOAD_DIR / cad_image.filename

        with open(original_image_path, "wb") as f:
            f.write(await original_image.read())
        with open(cad_image_path, "wb") as f:
            f.write(await cad_image.read())

        # Call the process_and_save_masks function
        process_and_save_masks(original_image_path=str(original_image_path), cad_image_path=str(cad_image_path), model=model)

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

        # Define the paths for the static output files
        static_result_origin_path = STATIC_DIR / result_origin_path.name
        static_result_cad_path = STATIC_DIR / result_cad_path.name

        # Remove existing files if they exist
        if static_result_origin_path.exists():
            os.remove(static_result_origin_path)

        if static_result_cad_path.exists():
            os.remove(static_result_cad_path)

        # Copy output files to the static directory for serving
        result_origin_path.rename(static_result_origin_path)
        result_cad_path.rename(static_result_cad_path)

        return HTMLResponse(content=result_html, status_code=200)

    except Exception as e:
        return HTMLResponse(content=f"An error occurred: {str(e)}", status_code=500)
    
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return PlainTextResponse(str(exc), status_code=500)
