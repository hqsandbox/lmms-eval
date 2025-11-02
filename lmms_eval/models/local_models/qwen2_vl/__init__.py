# Copyright 2024 The HuggingFace Team. All rights reserved.
#
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

# Modified to use standard imports instead of lazy loading
from .configuration_qwen2_vl import (
    Qwen2VLConfig,
    Qwen2VLTextConfig,
    Qwen2VLVisionConfig,
)
from .image_processing_qwen2_vl import Qwen2VLImageProcessor
from .image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from .modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLPreTrainedModel,
    Qwen2VLModel,
)
from .processing_qwen2_vl import Qwen2VLProcessor

__all__ = [
    "Qwen2VLConfig",
    "Qwen2VLTextConfig",
    "Qwen2VLVisionConfig",
    "Qwen2VLImageProcessor",
    "Qwen2VLImageProcessorFast",
    "Qwen2VLForConditionalGeneration",
    "Qwen2VLPreTrainedModel",
    "Qwen2VLModel",
    "Qwen2VLProcessor",
]
