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
Video Analysis Service - ComfyUI Workflow-based implementation

Uses ComfyUI workflows to analyze video content and generate descriptions.
"""

from typing import Optional
from pathlib import Path

from loguru import logger

from pixelle_video.services.comfy_base_service import ComfyBaseService


class VideoAnalysisService(ComfyBaseService):
    """
    Video analysis service - Workflow-based

    Uses ComfyKit to execute selfhost video understanding workflows.
    Returns detailed textual descriptions of video content.

    Usage:
        description = await pixelle_video.video_analysis("path/to/video.mp4")

        # Use a specific workflow
        description = await pixelle_video.video_analysis(
            "path/to/video.mp4",
            workflow="selfhost/analyse_video.json"
        )

        # List available workflows
        workflows = pixelle_video.video_analysis.list_workflows()
    """

    WORKFLOW_PREFIX = "analyse_video"
    WORKFLOWS_DIR = "workflows"

    def __init__(self, config: dict, core=None):
        super().__init__(config, service_name="video_analysis", core=core)

    async def __call__(
        self,
        video_path: str,
        workflow: Optional[str] = None,
        # ComfyUI connection (optional overrides)
        comfyui_url: Optional[str] = None,
        # Additional workflow parameters
        **params
    ) -> str:
        """
        Analyze a video using workflow

        Args:
            video_path: Path to the video file (local or URL)
            workflow: Workflow filename (optional; defaults to selfhost/analyse_video.json)
            comfyui_url: ComfyUI URL (optional, overrides config)
            **params: Additional workflow parameters

        Returns:
            str: Text description of the video content
        """
        # 1. Validate video path
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 2. Resolve workflow path (default to selfhost convention)
        if workflow is None:
            workflow = "selfhost/analyse_video.json"
            logger.info(f"Using default workflow: {workflow}")

        # 3. Resolve workflow (returns structured info)
        workflow_info = self._resolve_workflow(workflow=workflow)

        workflow_params = {"video": str(video_path)}
        workflow_params.update(params)
        logger.debug(f"Workflow parameters: {workflow_params}")

        try:
            kit = await self.core._get_or_create_comfykit()
            workflow_input = workflow_info["path"]
            logger.info(f"Executing selfhost workflow: {workflow_input}")
            result = await kit.execute(workflow_input, workflow_params)

            if result.status != "completed":
                error_msg = result.msg or "Unknown error"
                logger.error(f"Video analysis failed: {error_msg}")
                raise Exception(f"Video analysis failed: {error_msg}")

            # Extract text description (selfhost format)
            description = None
            if result.texts and len(result.texts) > 0:
                description = result.texts[0]
                logger.debug(f"Found description in result.texts: {description[:100]}...")
            elif result.outputs:
                for _node_id, node_output in result.outputs.items():
                    if 'text' in node_output:
                        text_list = node_output['text']
                        if text_list and len(text_list) > 0:
                            description = text_list[0]
                            logger.debug(f"Found description in outputs.text: {description[:100]}...")
                            break

            if not description:
                logger.error(f"No text found in result. Status: {result.status}, Outputs: {result.outputs}, Texts: {result.texts}")
                raise Exception("No description generated from video analysis")

            logger.info(f"✅ Video analyzed: {description[:100]}...")
            return description

        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            raise
