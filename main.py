import torch
from model import Nerf2DMLP
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

import math


# Fourier feature mapping
def frequency_encoding(x_val, y_val, n_freq):
    exp = lambda v: [math.pow(2, f) * v for f in range(0, n_freq)]

    sin_enc = lambda val: [math.sin(math.pi * v) for v in exp(val)]
    cos_enc = lambda val: [math.cos(math.pi * v) for v in exp(val)]

    return [
        item
        for enc in zip(sin_enc(x_val), cos_enc(x_val), sin_enc(y_val), cos_enc(y_val))
        for item in enc
    ]


# Normalize the raw input to be in [-1, 1]
def normalize(raw_x, raw_y, resolution_x, resolution_y):
    v_x = (float(raw_x) / resolution_x) * 2 - 1
    v_y = (float(raw_y) / resolution_y) * 2 - 1

    return (v_x, v_y)


def prepare_training_dataloader(image_array, num_freq):
    img_h, img_w, img_c = image_array.shape

    train_data_features = []
    train_data_labels = []

    for h, _ in enumerate(image_array):
        for w, val in enumerate(image_array[h]):
            v_x, v_y = normalize(w, h, img_w, img_h)
            train_input = frequency_encoding(v_x, v_y, num_freq)

            train_data_features.append(train_input)
            train_data_labels.append(val)

    feature_tensor = torch.tensor(np.array(train_data_features))
    feature_tensor = feature_tensor.to(torch.float32)

    label_tensor = torch.tensor(np.array(train_data_labels))
    label_tensor = label_tensor.to(torch.float32)

    dataset = TensorDataset(feature_tensor, label_tensor)

    return DataLoader(dataset, shuffle=True, batch_size=2048)


def prepare_inference_dataloader(image_array, num_freq):
    img_h, img_w, img_c = image_array.shape

    predict_data = []

    for h in range(img_h):
        for w in range(img_w):
            v_x, v_y = normalize(w, h, img_w, img_h)

            predict_input = frequency_encoding(v_x, v_y, num_freq)
            predict_data.append(predict_input)

    predict_input_tensor = torch.tensor(np.array(predict_data)).to(torch.float32)

    dataset = predict_input_tensor

    print("Dataset[0] in prepare_inference_dataloader: ", dataset[0])

    return DataLoader(dataset, batch_size=2048)


def generate_output_image(image_array, predictions):
    img_h, img_w, img_c = image_array.shape

    flat_tensor = torch.cat(predictions)
    output_array = flat_tensor.numpy().astype(np.uint8).reshape(img_h, img_w, img_c)

    out_image = Image.fromarray(output_array, "RGB")
    out_image.save("out_image.png")

    return out_image


def main():
    num_freq = 7

    # freq * (a pair of sin & cos) * (a pair of x and y)
    num_input = num_freq * 2 * 2

    model = Nerf2DMLP(num_input, 256, 3)
    print(model)

    print("model type", model.dtype)

    file_path = "dataset/lenna.png"

    image = Image.open(file_path)
    image_array = np.asarray(image)

    image.close()

    print(image_array.shape)
    print(image_array.dtype)
    print(image_array[0][0])

    print(" --- Loading Data --- ")

    train_dataloader = prepare_training_dataloader(image_array, num_freq)

    print(" --- Train --- ")

    trainer = pl.Trainer(limit_train_batches=20000, max_epochs=20)
    trainer.fit(model, train_dataloader)

    print(" --- Predict --- ")

    model.eval()

    inference_dataloader = prepare_inference_dataloader(image_array, num_freq)
    predictions = trainer.predict(model, inference_dataloader)

    print(" --- Output image --- ")

    out_image = generate_output_image(image_array, predictions)
    out_image.show()


if __name__ == "__main__":
    main()