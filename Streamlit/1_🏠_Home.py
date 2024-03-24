import streamlit as st


st.set_page_config(
    page_title="Learning Koopman Operator with Deep Neural Networks",
    page_icon="🏠",
    layout= 'wide'
)

st.title('Aperçu du projet')

st.markdown("""
Bienvenue sur la page d'aperçu du projet. Ce projet vise à comprendre les deux articles 
(https://www.nature.com/articles/s41467-018-07210-0 et 
https://proceedings.neurips.cc/paper/2017/file/3a835d3215755c435ef4fe9965a3f2a0-Paper.pdf),
puis d'implémenter un Autoencoder identifiant l'opérateur de Koopman.

## Objectifs
Les principaux objectifs de ce projet sont les suivants :
- Présenter de manière concise un résumé des travaux.
- Implémenter un réseau neuronal pour estimer l'opérateur de Koopman.

## Auteurs
Ce projet est réalisé par:
                
**Huiqi Vicky ZHENG**
            
**Joris CAUDARD**
            
**Fanilosoan'Ivahiny La Sylviane ANDRIARIMANANA**
""")

