import streamlit as st
import pandas as pd
import numpy as np
from fastai.vision.all import *


st.title('Bear Classifier')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    
    im = Image.open(uploaded_file)
    st.image(im.to_thumb(256,256))
    uploaded_file
    learn_inf = load_learner('export.pkl')

    st.write(learn_inf.predict(im))

