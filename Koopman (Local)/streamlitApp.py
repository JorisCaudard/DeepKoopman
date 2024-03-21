import streamlit as st
import preprocessingData
import streamlit.components.v1 as components
import os

def main():
    """The main function of the app.

    Calls the appropriate mode function, depending on the user's choice
    in the sidebar. The mode function that can be called is
    `DeepKoopman`.

    Returns
    -------
    None

    """

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            "Introduction",
            "Nos diapos",
            "DeepKoopman"
        ],
    )
    if app_mode == "Introduction":
        st.title("Théorie de l'opérateur de Koopman et Implémentation en Réseau de neurones")
    elif app_mode == "Nos diapos":
        presentation()
    elif app_mode == "DeepKoopman":
        DeepKoopMan()

def presentation():
    """Displays a Google Slide presentation in an iframe component.

    This function embeds a Google Slide presentation using the `iframe` component
    provided by Streamlit. The presentation is loaded from a specified URL and
    displayed with a width of 760 pixels and a height of 569 pixels.

    Returns
    -------
    None
    
    """

    components.iframe('https://docs.google.com/presentation/d/e/2PACX-1vRGr71q2lBK2Oymklg6DgEjIlWNkVfs7DqznOw_Gu9e-8VhexTUiegk2d1xtAZKvlLjbDmUX8V0e0qA/embed?start=true&loop=true&delayms=3000',width=760,height=569)

def DeepKoopMan():
    """Displays the Deep Koopman section of the application.

    Preprocesses data, allows user to select optimal or custom parameters,
    and displays the parameters of the model.

    Returns
    -------
    None

    """
    
    df, params = preprocessingData.preprocessDiscreteSpectrum(dataPath="data/DiscreteSpectrumExample_train1_x.csv")

    bestParams = params

    st.title("Théorie de l'opérateur de Koopman et Implémentation en Réseau de neurones")

    col1, col2 = st.columns(2)

    with col1:
        optimal = st.radio("Select parameters to use",
                ["Optimal", "Custom"],
                index=0,
                captions=["Use Optimal parameters & saved network", "Train a new network with custom parameters"])

    with col2:
        if optimal == 'Custom':  
            params["numShifts"] = st.slider("numShifts", 0, 50, bestParams["numShifts"])
            params["reconLam"] = st.slider("reconLam", 0, 10, 1)
            params["L1Lam"] = st.slider("L1Lam", 0, 10, 1)
            params["L2Lam"] = st.slider("L2Lam", 0, 10, 1)
            params["hiddenSize"] = st.slider("hiddenSize", 0, 10, 1)
            params["hiddenLayer"] = st.slider("hiddenLayer", 0, 10, 1)
            params["latentSize"] = st.slider("latentSize", 0, 10, 1)
        else:
            params = bestParams


    st.write("Here are the parameters of the model : ")

    st.write(params)

if __name__ == "__main__":
    main()