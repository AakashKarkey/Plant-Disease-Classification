#
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
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
# MODEL = tf.keras.models.load_model("../save_models/1.keras")
#
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
#         file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
#
#     predictions = MODEL.predict(img_batch)
#
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)


# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
#
# app = FastAPI()
#
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Load your Keras model once when the server starts
# MODEL = tf.keras.models.load_model("../models/2")
#
# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
#
# @app.get("/ping")
# async def ping():
#     return {"message": "Hello, I am alive"}
#
# def read_file_as_image(data) -> np.ndarray:
#     try:
#         image = Image.open(BytesIO(data)).convert("RGB")
#         image = image.resize((256, 256))  # ✅ Update this to match your model
#         image = np.array(image) / 255.0
#         return image
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
#
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)  # Add batch dimension
#
#     predictions = MODEL.predict(img_batch)
#
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = float(np.max(predictions[0]))
#
#     return {
#         "class": predicted_class,
#         "confidence": confidence
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
import tensorflow as tf

# App init
app = FastAPI()

# Allow requests from React Native dev server
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://192.168.0.100:19006",  # replace with your local IP if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (256x256 image input size)
MODEL = tf.keras.models.load_model("../models/4")  # Adjust path if needed
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize((256, 256))           # Match training size
        image = np.array(image) / 255.0             # Normalize
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    # Debug prediction
    print("Raw predictions:", predictions[0].tolist())

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

