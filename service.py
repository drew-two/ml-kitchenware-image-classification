import numpy as np

# import json
import base64
import bentoml
from PIL import Image
from io import BytesIO
from bentoml.io import JSON

from pydantic import BaseModel

IMAGE_SIZE=244
classes = [
    'cup', 
    'fork', 
    'glass', 
    'knife', 
    'plate', 
    'spoon'
]

model_ref = bentoml.tensorflow.get("kitchenware-classification:latest")
model_runner = model_ref.to_runner()
svc = bentoml.Service("kitchenware-classification", runners=[model_runner])

def base64_decode_image(encoded_image):
    decoded_bytes = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(decoded_bytes))
    print(image.format, image.size, image.mode) 
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return image

@svc.api(input=JSON(), output=JSON())
async def classify(application_data):
    encoded_image = application_data['prediction']['image']
    image = base64_decode_image(encoded_image)
    input_arr = np.array(image)
    input_arr = np.array([input_arr])

    prediction = await model_runner.predict.async_run(input_arr)
    predicted_class = classes[np.argmax(prediction, axis=1)[0]]
    
    return {
            'model': 'kitchenware-classifier',
            'version': 1.0,
            'prediction': {
                'prediction': predicted_class, 
                'prediction_id': 966
            },
        }