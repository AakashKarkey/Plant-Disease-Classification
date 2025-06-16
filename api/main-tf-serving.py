# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import requests
#
# app = FastAPI()
#
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
#
# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
#
# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
#
#     json_data = {
#         "instances": img_batch.tolist()
#     }
#
#     response = requests.post(endpoint, json=json_data)
#     prediction = np.array(response.json()["predictions"][0])
#
#     predicted_class = CLASS_NAMES[np.argmax(prediction)]
#     confidence = np.max(prediction)
#
#
#     return {
#         "class": predicted_class,
#         "confidence": float(confidence)
#     }
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TF Serving endpoint (make sure it's running with your model loaded)
endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize((224, 224))  # Resize to model's input
        image = np.array(image) / 255.0   # Normalize pixel values
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Add batch dimension

    json_data = {
        "instances": img_batch.tolist()  # Convert to list for JSON
    }

    try:
        response = requests.post(endpoint, json=json_data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error calling TF Serving: {e}")

    try:
        prediction = np.array(response.json()["predictions"][0])
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid response from model server")

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

