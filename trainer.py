#!/usr/bin/env python
# coding: utf-8

# # Trainer
# 
# Using the DEiT small model here (DEiT Small Distilled Patch 16, Image size 244 x 244) in the interest of time and space for deployment

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from keras.applications import imagenet_utils

from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_SIZE = 9367
IMAGE_SIZE = 224
BATCH_SIZE = 8
WORKERS = 4
EPOCHS = 10

BASE_PATH='./data'

classes = [
    'cup', 
    'fork', 
    'glass', 
    'knife', 
    'plate', 
    'spoon'
]


# First, we will load the training dataframe and split it into train and validation

df_train_full = pd.read_csv('data/train.csv', dtype={'Id': str})
df_train_full['filename'] = 'data/images/' + df_train_full['Id'] + '.jpg'
df_train_full.head()


val_cutoff = int(len(df_train_full) * 0.8)
df_train = df_train_full[:val_cutoff]
df_val = df_train_full[val_cutoff:]


# ## Training

# Now let's create image generators

# These models don't have the imagenet preprocessing built in so I have to apply this
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="tf"
    )


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    dtype="float16"
)

train_generator = train_datagen.flow_from_dataframe(
    df_train,
    x_col='filename',
    y_col='label',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    dtype="float16"
)

val_generator = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)


classes = np.array(list(train_generator.class_indices.keys()))
classes


earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_accuracy',
    min_delta = 1e-4,
    patience = 3,
    mode = 'max',
    restore_best_weights = True,
    verbose = 1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'deit_s_d_p16_224_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max'
)

callbacks = [earlystopping, checkpoint]


def get_model_deit(model_url, res=IMAGE_SIZE, num_classes=len(classes)) -> tf.keras.Model:
    inputs = tf.keras.Input((res, res, 3))
    hub_module = hub.KerasLayer(model_url, trainable=False)

    base_model_layers, _ = hub_module(inputs)   # Second output in the tuple is a dictionary containing attention scores.
    outputs = keras.layers.Dense(num_classes, activation="softmax")(base_model_layers)
    
    return tf.keras.Model(inputs, outputs) 


# Warnings are normal; the pre-trained weights for the original classifications heads are being skipped.

def build_model():
    model_gcs_path = "http://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224_fe/1"
    model = get_model_deit(model_gcs_path)

    # Define the optimizer learning rate as a hyperparameter.
    learning_rate = 1e-2
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model()


history = model.fit(
    x = train_generator,
    validation_data=val_generator,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    workers=WORKERS,
    callbacks=callbacks
)


image_path = 'testing/0966.jpg'

image = tf.keras.utils.load_img(
    image_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE)
)
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
print()


print(predictions)
predicted_class = classes[np.argmax(predictions, axis=1)[0]]
predicted_class


import bentoml
bentoml.tensorflow.save_model("kitchenware-classification", model)

