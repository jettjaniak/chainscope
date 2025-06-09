#%% Imports and setup

import ast
from pathlib import Path
import io
import re
import asyncio
import base64
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import anthropic
import httpx
import nest_asyncio
import pandas as pd
import yaml
from dataclass_wizard import LoadMeta
from dotenv import load_dotenv
from tqdm.notebook import tqdm

from chainscope import cot_paths_eval
from chainscope.typing import SplitCotResponses, StepFaithfulness
from chainscope import cot_splitting
from chainscope import cot_faithfulness_utils

from IPython import get_ipython
from typing import Final
import plotly.graph_objects as go
from PIL import Image


ENABLE_AUTORELOAD = True  # @param {"type": "boolean"}

if ENABLE_AUTORELOAD and get_ipython() is not None:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

# TODO(arthur): Add deps to pyproject.toml:

from chainscope import typing as ctyping
from google import genai as google_genai
from openai import OpenAI
from google.genai import types as google_types

from chainscope import typing as ctyping
from chainscope.typing import CotResponses, MathDatasetParams, DefaultSamplingParams, DatasetParams

try:
    import jax  # Just for tree mapping :)
except Exception as e:
    print("TODO(arthur): FIX JAX!!!")

#%%
# Plot the thinking models comparison

# Data setup
MODELS = ["Claude", "DeepSeek", "Qwen"]
values_thinking = [
    (10/114) * 100, 
    (3/172) * 100,  
    (1/115) * 100,  
]
values_nonthinking = [
    (10/114) * 100,
    (4/79) * 100, 
    (14/51) * 100,
]

# Create figure
fig = go.Figure()

# Add thinking variants
fig.add_trace(go.Bar(
    name='Thinking (darker)',
    x=MODELS,
    y=values_thinking,
    marker_color=['#C94A3D', '#1E88E5', '#8E44AD'],
    text=[f"{v:.1f}%<br>({n} examples)" for v, n in zip(values_thinking, [10, 2, 1])],
    textposition='auto',
    textfont=dict(size=16)  # Increased font size for bar labels
))

# Add non-thinking variants
fig.add_trace(go.Bar(
    name='Non-thinking (lighter)',
    x=MODELS,
    y=values_nonthinking,
    marker_color=['#E5A59E', '#90CAF9', '#D4B0D4'],
    text=[f"{v:.1f}%<br>({n} responses)" for v, n in zip(values_nonthinking, [21, 4, 14])],
    textposition='auto',
    textfont=dict(size=16)  # Increased font size for bar labels
))

# Update layout
fig.update_layout(
    title=dict(
        text="Proportion of Correct Responses Using Unfaithful Shortcuts",
        font=dict(size=24),  # Increased title font size
        y=0.95  # Move title up slightly
    ),
    title_x=0.5,
    xaxis_title=dict(
        text="<b>Model</b>",
        font=dict(size=20)  # Increased x-axis title font size
    ),
    yaxis_title=dict(
        text="<b>Proportion of Correct Responses (%)</b>",
        font=dict(size=20)  # Increased y-axis title font size
    ),
    barmode='group',
    plot_bgcolor='white',
    width=800,
    height=600,  # Increased height to accommodate larger text
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        font=dict(size=16)  # Increased legend font size
    ),
    font=dict(size=16)  # Increased general font size
)

# Update axes
fig.update_xaxes(
    showgrid=False,
    tickfont=dict(size=16)  # Increased x-axis tick labels font size
)

# Calculate maximum y value and add 10%
max_value = max(max(values_thinking), max(values_nonthinking))
y_max = max_value * 1.2

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='lightgray',
    range=[0, y_max],
    ticksuffix="%",
    tickfont=dict(size=16)  # Increased y-axis tick labels font size
)

# Add model logos below x-axis
logo_paths = {
    "Claude": "/workspace/atc1/chainscope/assets/anthropic-logo.png",
    "DeepSeek": "/workspace/atc1/chainscope/assets/deepseek-logo.png",
    "Qwen": "/workspace/atc1/chainscope/assets/qwen-logo.png"
}

# Calculate positions for logos
bar_positions = {model: i for i, model in enumerate(MODELS)}
logo_height = 0.12  # Slightly reduced logo size to make room for larger text
y_position = -0.18  # Moved logos down slightly to avoid text overlap

for model, logo_path in logo_paths.items():
    with Image.open(logo_path) as img:
        width, height = img.size
        aspect_ratio = width / height
        logo_width = logo_height * aspect_ratio
        
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="paper",
                x=bar_positions[model],
                y=y_position,
                sizex=logo_width,
                sizey=logo_height,
                xanchor="center",
                yanchor="middle"
            )
        )

# Update layout to make room for logos and larger text
fig.update_layout(
    margin=dict(b=100, t=100, l=100, r=50)  # Increased margins all around
)

# Show plot
fig.show()

# Save plot as PDF
fig.write_image("thinking_models_comparison.pdf")

# %%
