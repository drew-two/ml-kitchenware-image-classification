import sys
import base64

if len(sys.argv) != 2:
    sys.exit("Usage: python make_base64.py <jpg to convert>")

image_path = sys.argv[1]

image_bytes=[]
with open(image_path, "rb") as image:
    image_bytes = image.read()
    print(type(image_bytes))

encoded_image = base64.b64encode(image_bytes)
with open("image.b64", "wb") as f:
    f.write(encoded_image)
    f.close()