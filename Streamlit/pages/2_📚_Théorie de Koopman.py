import streamlit as st

st.set_page_config(
    page_title="Koopman Theory",
    page_icon="📚",
    layout= 'wide'
)

st.title("Théorie de Koopman")

st.write("""
         Que ce soit en physique, en biologie ou dans d'autres domaines, un Système dynamique est simplement un système qui évolue dans le temps et est régit par un ensemble d'états
         $x \in \mathbb{R}^p$ satisfaisant une équation différentielle du premier ordre :
$$\dot{x} = f(x,t)$$ où la notation $\dot{x}$ désigne la dérivée temporelle du vecteur $x$ à un instant donné $t$.""")

st.write("Dans le cas où la fonction $f$ s'avère être non-linéaire, le système devient alors plus compliqué à étudier. La théorie de Koopman propose une linéarisation possible de ce système en passant par un système d'états latents où l'équation différentielle devient alors linéaire.")

st.write("Dans notre étude, nous nous intéresserons au système dynamique suivant :")

st.latex(r"""
        \begin{equation}
    \begin{cases}
    \dot{x_1} = \mu x_1 \\
    \dot{x_2} = \lambda (x_2 - x_1^2)
    \end{cases}
\end{equation}
         """)

st.write("Dans notre cas, nous étudierons ce système dans le cas où $\lambda = -1$ et $\mu = -0.05$. Dans ce cas, le système converge vers un état stable en $(0,0)$.")

st.subheader("Opérateur de Koopman et espace latent")

st.write("Dans ce cas théorique, on connait alors de manière explicite l'opérateur de Koopman dans un espace latent de dimension finie. En effet, en En posant que $y_1 = x_1$, $y_2=x_2$ et $y_3={x_1}^2$, il est possible de linéariser le système en dimension finie (dimension 3 ici).")

st.write("On obtient alors :")

st.latex(r"""
\begin{equation*}
    \frac{\mathrm{d} }{\mathrm{d} t}\begin{pmatrix}
    y_1 \\y_2  \\ y_3
    \end{pmatrix} = \begin{pmatrix}
\mu & 0 &0 \\ 
 0& \lambda & -\lambda\\ 
0 & 0 & 2\mu
\end{pmatrix}\begin{pmatrix}
y_1\\ 
y_2\\ 
y_3
\end{pmatrix}
\end{equation*}

""")

st.write("On identifie alors de manière explicite la matrice de l'opérateur de Koopman, et on peut en déduire les valeurs propres associées.")