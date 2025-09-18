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
from typing import Any, Dict, Optional, Union

from camel.models.base_model import BaseModelBackend
from camel.models.gemini_model import GeminiModel
from camel.models.stub_model import StubModel
from camel.types import ModelPlatformType, ModelType
from camel.utils import BaseTokenCounter


class ModelFactory:
    r"""Factory of backend models. Only supports Gemini models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(
        model_platform: ModelPlatformType,
        model_type: Union[ModelType, str],
        model_config_dict: Dict,
        token_counter: Optional[BaseTokenCounter] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ) -> BaseModelBackend:
        r"""Creates an instance of `BaseModelBackend` for Gemini models only.

        Args:
            model_platform (ModelPlatformType): Platform from which the model
                originates. Only Gemini supported.
            model_type (Union[ModelType, str]): Model for which a backend is
                created. Only Gemini models supported.
            model_config_dict (Dict): A dictionary that will be fed into
                the backend constructor.
            token_counter (Optional[BaseTokenCounter]): Token counter to use
                for the model.
            api_key (Optional[str]): The API key for authenticating with the
                Gemini service.
            url (Optional[str]): The url to the model service.

        Raises:
            ValueError: If the model platform/type is not Gemini.

        Returns:
            BaseModelBackend: The initialized Gemini backend.
        """
        model_class: Any
        
        if isinstance(model_type, ModelType):
            if model_platform.is_gemini and model_type.is_gemini:
                model_class = GeminiModel
            elif model_type == ModelType.STUB:
                model_class = StubModel
            else:
                raise ValueError(
                    f"Only Gemini models are supported. Got platform `{model_platform}` "
                    f"and model type `{model_type}`. Please use Gemini models only."
                )
        else:
            raise ValueError(
                f"Only Gemini ModelType enum values are supported, not string model types. "
                f"Got `{model_type}`."
            )
            
        return model_class(
            model_type, model_config_dict, api_key, url, token_counter
        )