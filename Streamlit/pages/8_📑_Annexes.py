import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Appendices",
    page_icon="📑",
    layout= 'wide'
)

st.title("Annexes")

st.markdown("Vous retrouverez ici les différents documents annexes utilisés dans ce projet.")

tab1, tab2 = st.tabs(["Diaporama", "Documents PDF"])

with tab1:
    st.header("Diaporama")

    st.markdown("Dans cette section, vous trouverez le diaporama utilisé dans le cadre de notre préséentaiton orale.")

    components.iframe('https://docs.google.com/presentation/d/e/2PACX-1vRGr71q2lBK2Oymklg6DgEjIlWNkVfs7DqznOw_Gu9e-8VhexTUiegk2d1xtAZKvlLjbDmUX8V0e0qA/embed?start=true&loop=true&delayms=3000', height= 600)

    with open("Files/Diaporama.pdf", "rb") as diapo:
        downloadButton = st.download_button(
            label="Download as PDF",
            data = diapo,
            file_name="Diaporama.pdf",
            mime="pdf",
            help="Download this presentation as pdf",
            type='primary'
        )

with tab2:
    st.header("Additional PDF documents")

    with st.container(border= True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("[Deep learning for universal linear embeddings of nonlinear dynamics](https://www.nature.com/articles/s41467-018-07210-0)")

        with col2:
            with open("Files/Nature Article.pdf", "rb") as nature:
                downloadButton = st.download_button(
                    label="Download as PDF",
                    data = nature,
                    file_name="Nature Article.pdf",
                    mime="pdf"
                )


    with st.container(border= True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("[Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition](https://proceedings.neurips.cc/paper/2017/file/3a835d3215755c435ef4fe9965a3f2a0-Paper.pdf)")

        with col2:
            with open("Files/Proceedings Article.pdf", "rb") as proceedings:
                downloadButton = st.download_button(
                    label="Download as PDF",
                    data = proceedings,
                    file_name="Proceedings Article.pdf",
                    mime="pdf"
                )

    with st.container(border= True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Our final report")

        with col2:
            with open("Files/Report.pdf", "rb") as report:
                downloadButton = st.download_button(
                    label="Download as PDF",
                    data = report,
                    file_name="Report.pdf",
                    mime="pdf"
                )