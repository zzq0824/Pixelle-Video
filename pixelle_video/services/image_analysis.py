# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Image Analysis Service - ComfyUI Workflow-based implementation

Uses Florence-2 or other vision models to analyze images and generate descriptions.
"""

from typing import Optional
from pathlib import Path

from loguru import logger

from pixelle_video.services.comfy_base_service import ComfyBaseService


class ImageAnalysisService(ComfyBaseService):
    """
    Image analysis service - Workflow-based

    Uses ComfyKit to execute selfhost image analysis workflows.
    Returns detailed textual descriptions of images.

    Usage:
        description = await pixelle_video.image_analysis("path/to/image.jpg")

        # Use a specific workflow
        description = await pixelle_video.image_analysis(
            "path/to/image.jpg",
            workflow="selfhost/analyse_image.json"
        )

        # List available workflows
        workflows = pixelle_video.image_analysis.list_workflows()
    """

    WORKFLOW_PREFIX = "analyse_"
    WORKFLOWS_DIR = "workflows"

    def __init__(self, config: dict, core=None):
        super().__init__(config, service_name="image_analysis", core=core)

    async def __call__(
        self,
        image_path: str,
        workflow: Optional[str] = None,
        # ComfyUI connection (optional overrides)
        comfyui_url: Optional[str] = None,
        # Additional workflow parameters
        **params
    ) -> str:
        """
        Analyze an image using workflow

        Args:
            image_path: Path to the image file (local or URL)
            workflow: Workflow filename (optional; defaults to selfhost/analyse_image.json)
            comfyui_url: ComfyUI URL (optional, overrides config)
            **params: Additional workflow parameters

        Returns:
            str: Text description of the image
        """
        # 1. Validate image path
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 2. Resolve workflow path (default to selfhost convention)
        if workflow is None:
            workflow = "selfhost/analyse_image.json"
            logger.info(f"Using default workflow: {workflow}")

        # 3. Resolve workflow (returns structured info)
        workflow_info = self._resolve_workflow(workflow=workflow)

        workflow_params = {"image": str(image_path)}
        workflow_params.update(params)
        logger.debug(f"Workflow parameters: {workflow_params}")

        try:
            kit = await self.core._get_or_create_comfykit()
            workflow_input = workflow_info["path"]
            logger.info(f"Executing selfhost workflow: {workflow_input}")
            result = await kit.execute(workflow_input, workflow_params)

            if result.status != "completed":
                error_msg = result.msg or "Unknown error"
                logger.error(f"Image analysis failed: {error_msg}")
                raise Exception(f"Image analysis failed: {error_msg}")

            # Extract text description from selfhost outputs
            # Format: {'6': {'text': ['description text']}}
            description = None
            if result.outputs:
                for _node_id, node_output in result.outputs.items():
                    if 'text' in node_output:
                        text_list = node_output['text']
                        if text_list and len(text_list) > 0:
                            description = text_list[0]
                            break

            if not description:
                logger.error(f"No text found in outputs: {result.outputs}")
                raise Exception("No description generated")

            logger.info(f"✅ Image analyzed: {description[:100]}...")
            return description

        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            raise
