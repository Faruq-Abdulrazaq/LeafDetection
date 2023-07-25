from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image

app = FastAPI()

model_potato = tf.keras.models.load_model("./models/potato")
model_pepper = tf.keras.models.load_model("./models/pepper")

CLASS_NAMES_POTATO = ["Early Blight", "Late Blight", "Healthy"]
CLASS_NAMES_PEPPER = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.post("/predict-potato")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = model_potato.predict(img_batch)

    predicted_class = CLASS_NAMES_POTATO[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@app.post("/predict-pepper")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = model_pepper.predict(img_batch)

    predicted_class = CLASS_NAMES_PEPPER[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)
