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
from openai import OpenAI

from chainscope import typing as ctyping
from chainscope.typing import CotResponses, MathDatasetParams, DefaultSamplingParams, DatasetParams

import jax  # Just for tree mapping :-)

#%%
