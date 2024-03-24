import streamlit as st
import preprocessingData
import streamlit.components.v1 as components

def main():
    """The main function of the app.

    Calls the appropriate mode function, depending on the user's choice in the sidebar. 
    The mode function that can be called are `overview`,`intro`, `architecture` ,`presentation`
    and `DeepKoopman`.

    Returns
    -------
    None

    """

    app_mode = st.sidebar.selectbox(
        "Choisissez le mode de l'application",
        [
            "Aperçu",
            "Introduction",
            "Architecture",
            "Nos diapos",
            "Deep Koopman"
        ],
    )
    if app_mode == "Aperçu":
        overview()
    elif app_mode == "Introduction":
        intro()
    elif app_mode == "Architecture":
        architecture()
    elif app_mode == "Nos diapos":
        presentation()
    elif app_mode == "DeepKoopman":
        DeepKoopMan()

def overview():
    """Displays an overview page for the project.

    This function provides an overview of the project, including the main objectives,
    key components, and the author's name.

    Returns
    -------
    None

    """

    st.title('Aperçu du projet')
    st.markdown("""
    Bienvenue sur la page d'aperçu du projet. Ce projet vise à comprendre les deux articles 
    (https://www.nature.com/articles/s41467-018-07210-0 et 
    https://proceedings.neurips.cc/paper/2017/file/3a835d3215755c435ef4fe9965a3f2a0-Paper.pdf),
    puis d'implémenter le réseau neuronal Deep Koopman.

    ## Objectifs
    Les principaux objectifs de ce projet sont les suivants :
    - Présenter de manière concise un résumé des travaux.
    - Implémenter un réseau neuronal pour estimer l'opérateur de Koopman.

    ## Auteur
    Ce projet est réalisé par:
                 
    **Huiqi Vicky ZHENG**
                
    **Joris CAUDARD**
                
    **Fanilosoan'Ivahiny La Sylviane ANDRIARIMANANA**

    Profitez de l'exploration du projet !
    """)

def intro():
    """Displays the introduction section.

    This function displays the introduction section of the project using Streamlit components such as
    `st.title`, `st.markdown`, and `st.latex`. It provides an introduction of the project's.

    Returns
    -------
    None

    """

    st.title("Deep Learning pour identifier les fonctions propres de Koopman")
    st.markdown("L'opérateur de Koopman associé à un système dynamique est défini par :")
    st.latex(r"{\cal K}{\mathbf{g}} \coloneqq {\mathbf{g}} \circ {f} \implies {\cal K}{\mathbf{g}}(x_t) = {\mathbf{g}} (x_{t+1})")
    st.markdown('''Cet opérateur nous permet de passer d'un système d'états de dimension finis non linéaire à un opérateur linéaire 
    dans un espace de dimensions infini. On s'intéresse à la recherche d'un espace de fonctions ${y_1, \dots, y_n}$ 
    fonctions propres de l'opérateur de Koopman (ainsi que les valeurs propres associées): ''')
    st.latex(r"\varphi(x_{k+1}) = \cal K\varphi(x_k) = \lambda \varphi(x_k)")

def architecture():
    """Displays the architecture section of DeepKoopman.

    This function displays the architecture section of DeepKoopman, which describes the
    neural network architecture used for identifying Koopman eigenfunctions. It includes images of the
    network architecture and explains the functionality of the autoencoder and auxiliary network.

    Returns
    -------
    None

    """

    st.title('Architecture de DeepKoopman')
    st.image("https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-018-07210-0/MediaObjects/41467_2018_7210_Fig1_HTML.png?as=webp")
    st.markdown(r'''Voici l'architecture de DeepKoopman pour identifier les fonctions propres de Koopman $\varphi(x)$''')
    st.markdown(r'''Le réseau est basé sur un autoencodeur qui est capable d'identifier les coordonnées intrinsèques 
    $y = \varphi (x)$ et de décoder ces coordonnées pour récupérer $x = \varphi^{-1}(y)$.''')
    st.image("https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-018-07210-0/MediaObjects/41467_2018_7210_Fig2_HTML.png?as=webp")
    st.markdown(r'''L'ajout d'un réseau auxiliaire sert à identifier le spectre des valeurs propres continues $\lambda$. Cela facilite 
    une réduction de dimension agressive dans l'auto-encodeur, évitant ainsi le besoin de plus hautes harmoniques de la fréquence 
    qui sont générées par la non-linéarité.''')

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
