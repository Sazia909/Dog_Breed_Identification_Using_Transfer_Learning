import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import streamlit as st
import pandas as pd

labels = pd.read_csv("labels.csv")
class_names = sorted(labels["breed"].unique())

model = tf.keras.models.load_model("model/dog_breed_model.keras")

st.title("🐶 Dog Breed Identifier")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224,224))
    img = image.img_to_array(img)/255
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    breed = class_names[np.argmax(pred)]

    st.write("Predicted Breed:", breed)
