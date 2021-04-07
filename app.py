import streamlit as st
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
model = pickle.load(open('imgs.p', 'rb'))
st.title('Image Recognition')
uploaded_image = st.file_uploader("Choose an image....", type = "jpg")
if uploaded_image is not None:
  img = Image.open(uploaded_image)
  st.image(img, caption= 'Uploaded image')

  if st.button('PREDICT'):
    categories = ['Apples', 'Oranges']
    st.write('Result...')
    flat_data = []
    img = np.array(img)
    img_resized = resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    plt.imshow(img_resized)
    y_out = model.predict(flat_data)
    y_out = categories[y_out[0]]
    st.write(f' PREDICTED OUTPUT: {y_out}') 
 
