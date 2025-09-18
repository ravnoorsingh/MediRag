# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from typing import Dict, Any
from camel.configs.base_config import BaseConfig

OPENAI_API_PARAMS = {
    'temperature': (float, 0.0, 2.0, 1.0),
    'top_p': (float, 0.0, 1.0, 1.0),
    'n': (int, 1, float('inf'), 1),
    'stream': (bool, False, True, False),
    'stop': (str, None, None, None),
    'max_tokens': (int, 1, float('inf'), None),
    'presence_penalty': (float, -2.0, 2.0, 0.0),
    'frequency_penalty': (float, -2.0, 2.0, 0.0),
    'logit_bias': (Dict, None, None, {}),
    'user': (str, None, None, None),
}

class ChatGPTConfig(BaseConfig):
    r"""Defines the parameters for generating chat completion."""
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Any = None
    max_tokens: int = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Dict = None
    user: str = None

class OpenSourceConfig(BaseConfig):
    r"""Defines the parameters for open source models."""
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = None