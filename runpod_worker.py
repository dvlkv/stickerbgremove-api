import runpod
from inference import remove_background
from PIL import Image
import io
import base64

# Handler function for RunPod

def handler(event):
    # event['input'] should contain the image data
    input_data = event.get('input', {})
    image_b64 = input_data.get('image')
    if not image_b64:
        return {"error": "No image provided in input['image'] (base64-encoded)"}

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Failed to decode image: {str(e)}"}

    # Process image
    result_image = remove_background(image)

    # Save result to bytes
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    result_b64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')

    return {"result": result_b64}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 