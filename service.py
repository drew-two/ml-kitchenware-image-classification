import numpy as np
from PIL.Image import Image as PILImage

import base64
import bentoml
from PIL import Image
from io import BytesIO
from bentoml.io import JSON
from bentoml.io import Image

from pydantic import BaseModel

DEBUG=False

IMAGE_SIZE=224
classes = [
    'cup', 
    'fork', 
    'glass', 
    'knife', 
    'plate', 
    'spoon'
]

model_ref = bentoml.keras.get("kitchenware-classification:latest")
model_runner = model_ref.to_runner()
svc = bentoml.Service("kitchenware-classification", runners=[model_runner])

def base64_decode_image(encoded_image):
    decoded_bytes = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(decoded_bytes))
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return image

def preprocess_image(image):
    image /= 127.5
    image -= 1.0
    return image

@svc.api(input=Image(), output=JSON())
async def predict_image(f: PILImage) -> "JSON":
    assert isinstance(f, PILImage)
    image = np.array(f.resize((IMAGE_SIZE, IMAGE_SIZE))).astype("float32")
    image = preprocess_image(image)

    input_arr = np.expand_dims(image, 0)  # reshape to [1, 224, 224, 3]
    prediction = await model_runner.async_run(input_arr)
    predicted_class = classes[np.argmax(prediction, axis=1)[0]]

    response = {
            'model': 'kitchenware-classifier',
            'version': 1.0,
            'prediction': {
                'prediction': predicted_class, 
                'prediction_id': 966
            },
        }

    print(response)

    return response