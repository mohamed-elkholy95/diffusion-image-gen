import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st; import plotly.graph_objects as go
st.title("📈 Training Metrics")
losses = [0.5 - 0.04*i + 0.002*i**1.5 for i in range(10)]
fig = go.Figure(go.Scatter(x=list(range(len(losses))), y=losses, mode="lines+markers", line=dict(color="#1f77b4", width=2)))
fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss",
    paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
st.plotly_chart(fig, use_container_width=True)
