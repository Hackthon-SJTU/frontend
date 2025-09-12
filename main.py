"""
MCP模拟后端服务 - 返回预置文件
用于前端开发和测试，可无缝替换为真实MCP
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import time
import uuid
from pathlib import Path

# ==================== 初始化 ====================
app = FastAPI(title="MCP Mock Server", version="1.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要的目录结构
BASE_DIR = Path(".")
ASSETS_DIR = BASE_DIR / "assets"  # 存放预置文件的目录
OUTPUTS_DIR = BASE_DIR / "outputs"  # 输出目录

# 创建目录
for dir_path in [ASSETS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# 静态文件服务
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ==================== 请求模型（与真实MCP保持一致） ====================

class TextToImageRequest(BaseModel):
    """文生图请求 - MCP标准格式"""
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 1024
    height: Optional[int] = 768
    guidance_scale: Optional[float] = 7.5
    num_inference_steps: Optional[int] = 50
    seed: Optional[int] = -1

class ImageToVideoRequest(BaseModel):
    """图生视频请求 - MCP标准格式"""
    image_url: str
    motion_scale: Optional[float] = 1.0
    fps: Optional[int] = 30
    duration: Optional[float] = 4.0  # 固定4秒
    transition_type: Optional[str] = "none"

class VideoToAudioRequest(BaseModel):
    """视频生音频请求 - MCP标准格式"""
    video_url: str
    audio_style: Optional[str] = "ambient"
    include_sfx: Optional[bool] = True
    volume: Optional[float] = 0.8

class MergeVideosRequest(BaseModel):
    """合并视频请求 - MCP标准格式"""
    video_segments: List[str]
    audio_url: Optional[str] = None
    output_format: Optional[str] = "mp4"
    quality: Optional[str] = "high"

class WorkflowRequest(BaseModel):
    """工作流请求 - MCP标准格式"""
    prompt: str
    iterations: int = 1
    auto_enhance: Optional[bool] = True
    workflow_config: Optional[Dict[str, Any]] = None

# ==================== 预置文件配置 ====================
"""
请在 assets 目录下放置以下文件：
- sample_image.png (或 .jpg)
- sample_video_4s.mp4 (4秒视频)
- sample_audio_4s.mp3 (4秒音频)
- sample_final_video.mp4 (带音频的完整视频)
"""

MOCK_FILES = {
    "image": "assets/sample_image.png",
    "video": "assets/sample_video_4s.mp4",
    "audio": "assets/sample_audio_4s.mp3",
    "final": "assets/sample_final_video.mp4"
}

# ==================== MCP标准接口实现 ====================

@app.post("/mcp/text-to-image")
async def text_to_image(request: TextToImageRequest):
    """
    MCP标准接口：文本生成图像
    
    输入：文本提示词和生成参数
    输出：生成的图像信息
    """
    # 模拟处理延迟
    time.sleep(0.5)
    
    # 生成唯一ID
    task_id = str(uuid.uuid4())[:8]
    
    # 返回MCP标准响应
    response = {
        "status": "success",
        "task_id": f"t2i_{task_id}",
        "result": {
            "image_url": f"/assets/sample_image.png",  # 返回预置图片
            "image_id": f"img_{task_id}",
            "prompt_used": request.prompt,
            "negative_prompt_used": request.negative_prompt,
            "metadata": {
                "model": "stable-diffusion-xl",
                "version": "1.0",
                "inference_time": 3.2,
                "width": request.width,
                "height": request.height,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "seed": request.seed if request.seed != -1 else 42
            }
        },
        "timestamp": time.time()
    }
    
    return JSONResponse(content=response)

@app.post("/mcp/image-to-video")
async def image_to_video(request: ImageToVideoRequest):
    """
    MCP标准接口：图像生成视频
    
    输入：图像URL和动画参数
    输出：4秒视频片段
    """
    time.sleep(0.5)
    
    task_id = str(uuid.uuid4())[:8]
    
    response = {
        "status": "success",
        "task_id": f"i2v_{task_id}",
        "result": {
            "video_url": f"/assets/sample_video_4s.mp4",  # 返回预置4秒视频
            "video_id": f"vid_{task_id}",
            "duration": 4.0,  # 固定4秒
            "fps": request.fps,
            "metadata": {
                "model": "stable-video-diffusion",
                "version": "xt-1.1",
                "source_image": request.image_url,
                "motion_scale": request.motion_scale,
                "frames_generated": 4 * request.fps,  # 4秒 * fps
                "resolution": "1024x768",
                "codec": "h264",
                "bitrate": "5000k"
            }
        },
        "timestamp": time.time()
    }
    
    return JSONResponse(content=response)

@app.post("/mcp/video-to-audio")
async def video_to_audio(request: VideoToAudioRequest):
    """
    MCP标准接口：视频推理音频
    
    输入：视频URL和音频风格参数
    输出：匹配的音频文件
    """
    time.sleep(0.5)
    
    task_id = str(uuid.uuid4())[:8]
    
    # 视频内容分析结果（模拟）
    video_analysis = {
        "scene_type": "nature",
        "mood": "peaceful",
        "detected_objects": ["trees", "water", "sky"],
        "motion_intensity": "low",
        "color_palette": ["green", "blue", "brown"]
    }
    
    response = {
        "status": "success",
        "task_id": f"v2a_{task_id}",
        "result": {
            "audio_url": f"/assets/sample_audio_4s.mp3",  # 返回预置4秒音频
            "audio_id": f"aud_{task_id}",
            "duration": 4.0,  # 匹配视频长度
            "style": request.audio_style,
            "metadata": {
                "model": "musicgen-large",
                "version": "1.2.0",
                "source_video": request.video_url,
                "video_analysis": video_analysis,
                "include_sfx": request.include_sfx,
                "volume": request.volume,
                "sample_rate": 44100,
                "bitrate": "320k",
                "format": "mp3"
            }
        },
        "timestamp": time.time()
    }
    
    return JSONResponse(content=response)

@app.post("/mcp/merge-videos")
async def merge_videos(request: MergeVideosRequest):
    """
    MCP标准接口：合并视频片段
    
    输入：多个视频片段和音频
    输出：合成的完整视频
    """
    time.sleep(0.5)
    
    task_id = str(uuid.uuid4())[:8]
    total_duration = len(request.video_segments) * 4.0  # 每段4秒
    
    response = {
        "status": "success",
        "task_id": f"merge_{task_id}",
        "result": {
            "output_url": f"/assets/sample_final_video.mp4",  # 返回预置完整视频
            "output_id": f"final_{task_id}",
            "total_duration": total_duration,
            "segments_count": len(request.video_segments),
            "metadata": {
                "format": request.output_format,
                "quality": request.quality,
                "resolution": "1920x1080",
                "video_codec": "h264",
                "audio_codec": "aac",
                "bitrate": "8000k" if request.quality == "high" else "5000k",
                "file_size": int(total_duration * 1000000),  # 模拟文件大小
                "processing_time": 2.5
            }
        },
        "timestamp": time.time()
    }
    
    return JSONResponse(content=response)

@app.post("/mcp/workflow")
async def execute_workflow(request: WorkflowRequest):
    """
    MCP标准接口：执行完整工作流
    
    输入：提示词和迭代次数
    输出：完整的处理结果
    """
    time.sleep(1.0)
    
    workflow_id = str(uuid.uuid4())[:8]
    total_duration = request.iterations * 4.0
    
    # 生成模拟的分段结果
    segments = []
    for i in range(request.iterations):
        segments.append({
            "segment_id": f"seg_{i+1}",
            "video_url": f"/assets/sample_video_4s.mp4",
            "duration": 4.0,
            "order": i + 1
        })
    
    response = {
        "status": "success",
        "workflow_id": f"wf_{workflow_id}",
        "result": {
            "final_video_url": f"/assets/sample_final_video.mp4",
            "total_duration": total_duration,
            "segments": segments,
            "audio_url": f"/assets/sample_audio_4s.mp3",
            "intermediate_results": {
                "image_url": f"/assets/sample_image.png",
                "image_generation_time": 3.2,
                "video_generation_time": 5.0 * request.iterations,
                "audio_generation_time": 2.1,
                "merge_time": 4.5
            },
            "workflow_summary": {
                "prompt": request.prompt,
                "iterations": request.iterations,
                "total_duration_seconds": total_duration,
                "total_processing_time": 15.8,
                "models_used": {
                    "image": "stable-diffusion-xl",
                    "video": "stable-video-diffusion",
                    "audio": "musicgen-large",
                    "merge": "ffmpeg"
                }
            }
        },
        "timestamp": time.time()
    }
    
    return JSONResponse(content=response)

@app.get("/mcp/status/{task_id}")
async def get_task_status(task_id: str):
    """
    MCP标准接口：查询任务状态
    
    用于异步任务的状态轮询
    """
    # 模拟不同的任务状态
    import random
    
    statuses = ["pending", "processing", "completed"]
    current_status = random.choice(statuses)
    
    response = {
        "task_id": task_id,
        "status": current_status,
        "progress": 100 if current_status == "completed" else random.randint(0, 99),
        "message": f"Task {task_id} is {current_status}",
        "timestamp": time.time()
    }
    
    return JSONResponse(content=response)

@app.get("/")
async def root():
    """API根路径 - 返回服务信息"""
    return {
        "service": "MCP Mock Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "text_to_image": "/mcp/text-to-image",
            "image_to_video": "/mcp/image-to-video",
            "video_to_audio": "/mcp/video-to-audio",
            "merge_videos": "/mcp/merge-videos",
            "workflow": "/mcp/workflow",
            "status": "/mcp/status/{task_id}"
        },
        "note": "This is a mock server. Place sample files in 'assets' directory.",
        "required_files": {
            "image": "assets/sample_image.png",
            "video": "assets/sample_video_4s.mp4",
            "audio": "assets/sample_audio_4s.mp3",
            "final": "assets/sample_final_video.mp4"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    # 检查必需的文件是否存在
    files_status = {}
    for key, path in MOCK_FILES.items():
        files_status[key] = os.path.exists(path)
    
    all_files_exist = all(files_status.values())
    
    return {
        "status": "healthy" if all_files_exist else "warning",
        "files": files_status,
        "message": "All files ready" if all_files_exist else "Some files missing"
    }

# ==================== 启动说明 ====================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("MCP Mock Server")
    print("=" * 50)
    print("\n请确保在 'assets' 目录下放置以下文件：")
    print("1. sample_image.png - 示例图片")
    print("2. sample_video_4s.mp4 - 4秒示例视频")
    print("3. sample_audio_4s.mp3 - 4秒示例音频")
    print("4. sample_final_video.mp4 - 带音频的完整视频")
    print("\n启动服务...")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )