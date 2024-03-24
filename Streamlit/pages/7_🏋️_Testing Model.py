import streamlit as st
from ...Model import dataloader

dataloader.TrajectoryDataset

st.set_page_config(
    page_title="Testing a Model",
    page_icon="🏋️",
    layout= 'wide'
)

st.title("Tester le modèle")

tab1, tab2 = st.tabs(["Entraînement", "Évaluation"])

with tab1:
    st.markdown("On implémente ici les fonctions d'entraînement dans modèle.")

    modelParams = {
        'numShifts':50,
        'inputSize':2,
        'hiddenLayer':2,
        'latentSize':3

    }

    with st.popover("Paramètres du modèle"):
        useCustomParams = st.toggle('Utiliser des paramètres custom ?')

        if useCustomParams:
            modelParams["numShifts"] = st.slider("numShifts", 1, 50, 30)
            modelParams["hiddenLayer"] = st.slider("hiddenLayer", 0, 10, 2)
            modelParams["latentSize"] = st.slider("latentSize", 0, 10, 3)

    st.markdown("Paramètres utilisés :")

    st.write(modelParams)

    st.spinner('Test')