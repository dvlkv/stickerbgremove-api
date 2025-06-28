from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from inference import remove_background
from PIL import Image
import io

app = FastAPI()

@app.post("/remove-bg")
def remove_bg_endpoint(file: UploadFile = File(...)):
    # Read image from request
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Process image
    result_image = remove_background(image)

    # Save result to bytes
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")
