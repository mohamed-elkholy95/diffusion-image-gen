import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st; import numpy as np
st.title("🎨 Generate Image")
prompt = st.text_input("Prompt", "a photo of a cat")
steps = st.slider("Sampling Steps", 10, 200, 50)
seed = st.slider("Seed", 0, 1000, 42)
if st.button("Generate", type="primary"):
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((64, 64, 3)).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    st.image(img, caption=f"Generated: {prompt} (steps={steps})", use_container_width=True)
