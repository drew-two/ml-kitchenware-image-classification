import sys
import asyncio
import PIL.Image
import numpy as np

from bentoml.testing.utils import async_request

async def test_image(host, img_data):

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    img = PIL.Image.open(img_path)


    await async_request(
        "POST",
        url,
        data=img_bytes,
        headers={"Content-Type": "image/jpeg"},
        assert_status=200,
    )

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python test_prediction.py <path/to/image> <url>")

    img_path = sys.argv[1]
    url = sys.argv[2]

    asyncio.run(test_image(url, img_path))