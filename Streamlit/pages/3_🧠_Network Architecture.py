import streamlit as st

st.set_page_config(
    page_title="Network Architecture",
    page_icon="🧠",
    layout= 'wide'
)

st.title("Network architecture")

tab1, tab2 = st.tabs(["Structure théorique", "Aperçu du code"])

with tab1:
    st.header("Architecture de l'AutoEncoder DeepKoopman", divider= True)
    st.image("https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-018-07210-0/MediaObjects/41467_2018_7210_Fig1_HTML.png?as=webp")
    
    st.markdown(r'''L'architecture proposée dans le cadre de l'apprentissage de l'opérateur de Koopman se base sur une structure classique d'Auto-encoder. 
                L'objectif étant d'identifier un opérateur linéaire dans un espace d'état latent, l'espace latent de l'auto-encoder prend la forme d'une matrice, qu'on assimile dans l'implémentation python à une couche Linéaire sans biais.''')
    
    st.markdown(r'''Le réseau est basé sur un autoencodeur qui est capable d'identifier les coordonnées intrinsèques 
    $y = \varphi (x)$ et de décoder ces coordonnées pour récupérer $x = \varphi^{-1}(y)$.''')
    
    st.image("https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-018-07210-0/MediaObjects/41467_2018_7210_Fig2_HTML.png?as=webp")
    
    st.markdown(r'''L'ajout d'un réseau auxiliaire permet d'identifier le spectre des valeurs propres continues de l'opérateur $K$. Cela facilite 
    une réduction de dimension agressive dans l'auto-encodeur, évitant ainsi le besoin de plus hautes harmoniques de la fréquence 
    qui sont générées par la non-linéarité.''')

with tab2:
    st.header("Implémentation en Python", divider= True)

    st.markdown(r"L'implémentation de ce réseau simple de Koopman a été faite en utilisant les focntionnalités du module python Pytorch.")

    st.code("""class SimpleKoopmanNeuralNetwork(nn.Module):
    \"""Classe représentant un réseau de neurones simple de Koopman.
    Ce réseau de neurones prend un état en entrée et prédit la trajectoire correspondante.

    Attributes:
        params (dict): Paramètres relatifs à la structure du réseau.
        Encoder (nn.Sequential): Encoder du réseau. 
        Decoder (nn.Sequential): Decoder du réseau.
        K (nn.Linear): Matrice de l'opérateur de Koopman, identifiée à une couche nn.Linear sans biais.

    Methods:
        __init__: Initialise les paramètres du réseau de neurones.
        forward: Prédit une trajectoire à partir d'un état initial
    \"""

    def __init__(self, params):
        \"""Initialise le réseau de neurones avec la configuration spécifiée.

        Args:
            params (dict): Un dictionnaire contenant la configuration initiale du réseau de neurones.
                Ce dictionnaire doit inclure les paramètres inputSize, hiddenSize, hiddenLayer, latentSize.
        \"""
        super(SimpleKoopmanNeuralNetwork, self).__init__()
        
        self.params = params

        inputSize, hiddenSize, hiddenLayer, latentSize = self.params['inputSize'], self.params['hiddenSize'], self.params['hiddenLayer'], self.params['latentSize']

        #Input layer of Encoder
        encoderLayers = [nn.Linear(inputSize, hiddenSize), nn.ReLU()]

        ###Define Encoder Layer
        for _ in range(hiddenLayer):
            encoderLayers.append(nn.Linear(hiddenSize, hiddenSize))
            encoderLayers.append(nn.ReLU())

        #Output layer of Encoder
        encoderLayers.append(nn.Linear(hiddenSize, latentSize))

        #Creating the Encoder Network
        self.Encoder = nn.Sequential(*encoderLayers)

        #Input layer of Decoder
        decoderLayers = [nn.Linear(latentSize, hiddenSize), nn.ReLU()]

        ###Define Decoder Layer
        for _ in range(hiddenLayer):
            decoderLayers.append(nn.Linear(hiddenSize, hiddenSize))
            decoderLayers.append(nn.ReLU())

        #Output layer of Decoder
        decoderLayers.append(nn.Linear(hiddenSize, inputSize))

        #Creating the Decoder Network
        self.Decoder = nn.Sequential(*decoderLayers)

        #Simple Koopman Auto-Encoder (Without Auxiliary Network)
        self.K = nn.Linear(latentSize, latentSize, bias=False)

    def forward(self, initialInput):
        \"""Prédit une trajectoire de longueur 50 à partir d'un état initial.

        Args:
            intialInput (nn.Tensor): État initial, de dimension inputSize.

        Returns:
            trajectoryPrediction (list): Trajectoire prédite sur 50 steps
        \"""
        #Take as input a 2 dimension tensor (initial State)

        #get the first encoded state (y1)
        encodedInitialInput = self.Encoder(initialInput)

        #First element of the latent trajectory is encoded input
        latentTrajectoryPrediction = [encodedInitialInput]

        #Alongside the trajectory, we multiply by the Koopman operator
        for _ in range(49):
            latentTrajectoryPrediction.append(self.K(latentTrajectoryPrediction[-1]))

        #Decoding the trajectory
        trajectoryPrediction = []

        for latentState in latentTrajectoryPrediction:
            trajectoryPrediction.append(self.Decoder(latentState))
        
        #We output the trajectoryPrediction
        return trajectoryPrediction""")