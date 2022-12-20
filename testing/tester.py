import sys
import json
import base64

import requests
from deepdiff import DeepDiff

image_path = sys.argv[1]

image_bytes=[]
with open(image_path, "rb") as image:
    image_bytes = image.read()
    print(type(image_bytes))

encoded_image = base64.b64encode(image_bytes)

request = {
    "model": "kitchenware-classifier",
    "version": model_version,
    "prediction": {
        "image": encoded_image,
        "prediction_id": image_id
    },
}


url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
actual_response = requests.post(url, json=event).json()
print('actual response:')


print(json.dumps(actual_response, indent=2))

expected_response = {
    'model': 'kitchenware-classifier',
    'version': model_version,
    'prediction': {
        'prediction': "fork", 
        'prediction_id': 256
    },
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff
