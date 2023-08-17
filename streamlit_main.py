

from predict_one_image import predict_one
import streamlit as st

import os

import matplotlib.pyplot as plt

import seaborn as sns
from PIL import Image

sns.set_theme(style="darkgrid")

sns.set()



st.title('Jurassic Park Dinosaur Detector')
def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('images_to_predict',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0

uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 

        # display the image

        display_image = Image.open(uploaded_file)

        st.image(display_image)
        st.text(os.getcwd())
        prediction = predict_one(os.path.join('images_to_predict',uploaded_file.name))
        st.text(f"The model thinks that this dinosaur is a {prediction}")
        # os.remove('images_to_predict/'+uploaded_file.name)

        # # deleting uploaded saved picture after prediction

        # drawing graphs
        # st.text(os.path.join('images_to_predict',uploaded_file.name))
        # st.text('Predictions :-')

        # fig, ax = plt.subplots()

        # ax  = sns.barplot(y = 'name',x='values', data = prediction,order = prediction.sort_values('values',ascending=False).name)

        # ax.set(xlabel='Confidence %', ylabel='Breed')

        # st.pyplot(fig)