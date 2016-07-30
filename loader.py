import struct
import numpy as np
import requests
import gzip
import pickle

TRAIN_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

TEST_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"


def downloader(url: str):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print("Response for", url, "is", response.status_code)
        exit(1)
    print("Downloaded", int(response.headers.get('content-length', 0)), "bytes")
    decompressed = gzip.decompress(response.raw.read())
    return decompressed


def load_data(images_url: str, labels_url: str) -> (np.array, np.array):
    images_decompressed = downloader(images_url)

    # Big endian 4 числа типа unsigned int, каждый по 4 байта
    magic, size, rows, cols = struct.unpack(">IIII", images_decompressed[:16])
    if magic != 2051:
        print("Wrong magic for", images_url, "Probably file corrupted")
        exit(2)

    image_data = np.array(np.frombuffer(images_decompressed[16:], dtype=np.dtype((np.ubyte, (rows * cols,)))) / 255,
                          dtype=np.float32)

    labels_decompressed = downloader(labels_url)
    # Big endian 2 числа типа unsigned int, каждый по 4 байта
    magic, size = struct.unpack(">II", labels_decompressed[:8])
    if magic != 2049:
        print("Wrong magic for", labels_url, "Probably file corrupted")
        exit(2)

    labels = np.frombuffer(labels_decompressed[8:], dtype=np.ubyte)

    return image_data, labels


with open("test_images.pkl", "w+b") as output:
    pickle.dump(load_data(TEST_IMAGES_URL, TEST_LABELS_URL), output)

with open("train_images.pkl", "w+b") as output:
    pickle.dump(load_data(TRAIN_IMAGES_URL, TRAIN_LABELS_URL), output)
