# Imports
import torch
from torchvision import transforms
from PIL import Image
import onnxruntime
import numpy as np
from image_proc import refine_foreground

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load ONNX model
weights_file = './epoch_340.onnx'
providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
onnx_session = onnxruntime.InferenceSession(
    weights_file,
    providers=providers
)
input_name = onnx_session.get_inputs()[0].name
print(onnxruntime.get_device(), onnx_session.get_providers())

# Input Data
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_background(image: Image):
    print('Processing image ...')
    image = image.convert("RGB") if image.mode != "RGB" else image
    input_images = transform_image(image).unsqueeze(0).cpu().numpy()

    # Prediction
    pred = torch.tensor(
        onnx_session.run(None, {input_name: input_images})[-1]
    ).squeeze(0).sigmoid().cpu()

    # Show Results
    pred_pil = transforms.ToPILImage()(pred)

    image_masked = refine_foreground(image, pred_pil)
    image_masked.putalpha(pred_pil.resize(image.size))

    return image_masked