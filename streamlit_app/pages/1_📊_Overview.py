import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st; import plotly.express as px
st.title("🎨 Diffusion Image Generation")
st.markdown("Generate images using diffusion models with noise scheduling and U-Net architecture.")
col1, col2 = st.columns(2)
with col1: st.subheader("Architecture"); st.markdown("- Simplified U-Net\n- Linear/Cosine noise schedule\n- DDPM sampling\n- GPU-accelerated (PyTorch)")
with col2: st.subheader("Config"); st.markdown(f"- Image size: 64×64\n- Diffusion steps: 100\n- Learning rate: 1e-4")
