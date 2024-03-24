import streamlit as st

st.set_page_config(
    page_title="Loss functions",
    page_icon="📉",
    layout= 'wide'
)

st.title("Loss functions")

tab1, tab2 = st.tabs(["Fonctions de perte théoriques", "Fonction de perte implémentée"])

with tab1:
    st.header("Optimisation Théorique", divider= True)
        
    st.markdown(r'''Dans le cadre théorique de l'article, les auteurs proposent l'optimisation d'une fonction de perte complexe, combinaison linéaire de différentes fonctions.''')
    
    st.subheader("Perte de reconnaissance")

    with st.container(border= True):
        st.latex(r"{\cal L}_{\mathrm{recon}} = \left\| {{\mathbf{x}}_1 - \varphi ^{ - 1}(\varphi ({\mathbf{x}}_1))} \right\|_{\mathrm{MSE}}")

    st.markdown("Cette fonction de perte permet d'assurer la reconnaissance de l'état initial après passage dans l'auto-encoder (sans opérateur de Koopman).")
    
    st.subheader("Perte de Prédiction")

    with st.container(border= True):
        st.latex(r"{\cal L}_{\mathrm{pred}} = \frac{1}{{S_p}}\sum_{m = 1}^{S_p} \left\| {{\mathbf{x}}_{m + 1} - \varphi ^{ - 1}(K^m\varphi ({\mathbf{x}}_1))} \right\|_{\mathrm{MSE}}")

    st.markdown(r"Cette fonction de perte permet de calculer la moyenne des erreurs MSE de prédictions sur une sous-trajectoires, dénotées par le paramètre $S_p$.")

    st.subheader('Perte de Linéarité')

    with st.container(border= True):
        st.latex(r"{\cal L}_{\mathrm{lin}} = \frac{1}{{T - 1}}\sum_{m = 1}^{T - 1} \left\| {\varphi ({\mathbf{x}}_{m + 1}) - K^m\varphi ({\mathbf{x}}_1)} \right\|_{\mathrm{MSE}}")

    st.markdown(r"Cette fonction de perte permet d'assurer la linéarité de l'opérateur de Koopman sur l'enesmble de la trajectoire.")


    st.subheader("Perte en norme infinie")

    with st.container(border= True):
        st.latex(r"{\cal L}_\infty = \left\| {{\mathbf{x}}_1 - \varphi ^{ - 1}(\varphi ({\mathbf{x}}_1))} \right\|_\infty + \left\| {{\mathbf{x}}_2 - \varphi ^{ - 1}(K\varphi ({\mathbf{x}}_1))} \right\|_\infty")

    st.markdown(r"Les données d'entraînement ayant étés générées par résolution d'équation différentielle, elles ne sont pas bruitées ; cette fonction de perte permet de pénaliser le plus grand écart de trajectoire après une prédiction.")

    st.subheader("Pénalisation sur les poids")

    with st.container(border= True):
        st.latex(r"\left\| {\mathbf{W}} \right\|_2^2")

    st.markdown("Enfin, les opérateurs proposent d'ajouter une pénalisation sur l'ensemble des poids du modèles, afin d'éviter l'overfitting.")


    
with tab2:
    st.header("Implémentation en Python")

    st.markdown(r"""Dans le cadre de notre implémentation, cette fonction de perte générale a été simplifiée. En effet, la matrice de l'opérateur étant définie par une couche linéaire (sans réseau auxiliaire), la linéarité est assurée par la structure même du réseau.
                De plus, la perte de reconnaissance est assimilée ici à la perte de prédiction (pour m=0). Enfin, les fonctions de pertes infinies et de pénalisation ont été écartées dans le cadre d'une première implémentation.
                Nous nous sommes donc focalisé sur la fonction de perte de prédiction dans le cadre de cette implémentation : une somme de fonctions de pertes MSE sur chaque étape de la prédiction.""")
    
    st.code("""class LossFunction(nn.Module):
    \"""Classe contenant la fonction de perte du réseau de Koopman.

    Args:
        numShifts (int): Longueur de la trajectoire sur laquelle la fonction de perte est évaluée.
    \"""
    def __init__(self, params):
        \"""Initialise la fonction de perte avec les paramètres donnés.

        Args:
            params (dict): Un dictionnaire contenant la configuration initiale de la fonction de perte.
                Ce dictionnaire doit inclure le paramètre numShifts.
        \"""
        super().__init__()

        self.numShifts = params['numShifts']

    def forward(self, targetTrajectory, predictedTrajectory):
        \"""Calcule la fonction de perte entre deux trajectoires.

        Args:
            targetTrajectory (list): Trajectoire réelle à approcher.
            predictedTrajectory (list): Trajdectoire prédite par un réseau de neurones de Koopman*

        Returns:
            lossPred (torch.Tensor): Perte calculée entre les deux trajectoires. Le torch.Tensor contient la valeur de la fonction de perte et le gradient associé. 
        \"""

        #We compute the Prediction loss
        lossPred = 0

        for m in range(self.numShifts):
            lossPred += F.mse_loss(targetTrajectory[m], predictedTrajectory[m])

        return lossPred""")