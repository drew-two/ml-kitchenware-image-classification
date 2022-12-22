#!/usr/bin/env python
# coding: utf-8

# # Helper script to make bento from .h5 file
import logging, os
import bentoml
from tensorflow import keras

## Variables and settings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Loading model...")
model = keras.models.load_model(filepath='./assets/model/')

print("Packaging bento...")
bentoml_model = bentoml.keras.save_model(
    "kitchenware-classification", 
    model,
    signatures={"__call__": {"batchable": True, "batch_dim": 0}}
)

print(bentoml_model.path)