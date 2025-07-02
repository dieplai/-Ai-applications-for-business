import streamlit as st
from pathlib import Path

def load_shared_styles():
    css_file = Path(__file__).parent.parent.parent / "static" / "styles.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
