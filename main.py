"""
ReasoningACTION 大模型系统 - 后端API服务
Author: AI Assistant
Version: 1.0.0
Description: 基于FastAPI的多模态内容生成服务，实现文生图、图生视频、视频推理音频、音视频合成等功能
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import asyncio
import uuid
import json
import os
from pathlib import Path

# ==================== 配置和初始化 ====================
app = FastAPI(
    title="ReasoningACTION API",
    description="多模态内容生成API服务",
    version="1.0.0"
)

# CORS配置 - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要的目录
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# ==================== 数据模型定义 ====================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TextToImageRequest(BaseModel):
    """文生图请求模型"""
    prompt: str = Field(..., description="文本提示词", min_length=1, max_length=1000)
    negative_prompt: Optional[str] = Field(None, description="负面提示词")
    width: int = Field(512, description="图像宽度", ge=128, le=2048)
    height: int = Field(512, description="图像高度", ge=128, le=2048)
    guidance_scale: float = Field(7.5, description="引导系数", ge=1.0, le=20.0)
    num_inference_steps: int = Field(50, description="推理步数", ge=10, le=100)
    seed: Optional[int] = Field(None, description="随机种子")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "A beautiful sunset over mountains, oil painting style",
                "negative_prompt": "low quality, blurry",
                "width": 512,
                "height": 512,
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        }

class ImageToVideoRequest(BaseModel):
    """图生视频请求模型"""
    image_url: str = Field(..., description="输入图片URL或路径")
    motion_scale: float = Field(1.0, description="运动幅度", ge=0.1, le=5.0)
    fps: int = Field(30, description="帧率", ge=15, le=60)
    duration: float = Field(4.0, description="视频时长(秒)", min=1.0, max=10.0)
    transition_type: Optional[str] = Field("none", description="转场类型")
    
    class Config:
        schema_extra = {
            "example": {
                "image_url": "path/to/image.png",
                "motion_scale": 1.0,
                "fps": 30,
                "duration": 4.0,
                "transition_type": "fade"
            }
        }

class VideoToAudioRequest(BaseModel):
    """视频推理音频请求模型"""
    video_url: str = Field(..., description="输入视频URL或路径")
    audio_style: str = Field("ambient", description="音频风格")
    include_sfx: bool = Field(True, description="是否包含音效")
    volume: float = Field(0.8, description="音量", ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "video_url": "path/to/video.mp4",
                "audio_style": "cinematic",
                "include_sfx": True,
                "volume": 0.8
            }
        }

class VideoMergeRequest(BaseModel):
    """视频合成请求模型"""
    video_segments: List[str] = Field(..., description="视频片段URL列表")
    audio_url: Optional[str] = Field(None, description="音频文件URL")
    output_format: str = Field("mp4", description="输出格式")
    quality: str = Field("high", description="输出质量")
    
    class Config:
        schema_extra = {
            "example": {
                "video_segments": ["video1.mp4", "video2.mp4"],
                "audio_url": "audio.mp3",
                "output_format": "mp4",
                "quality": "high"
            }
        }

class WorkflowRequest(BaseModel):
    """完整工作流请求模型"""
    prompt: str = Field(..., description="初始文本提示词")
    iterations: int = Field(1, description="迭代次数", ge=1, le=20)
    auto_enhance: bool = Field(True, description="是否自动优化")
    workflow_config: Optional[Dict[str, Any]] = Field(None, description="工作流配置")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "A serene forest with flowing water",
                "iterations": 10,
                "auto_enhance": True,
                "workflow_config": {
                    "image_config": {"width": 1024, "height": 768},
                    "video_config": {"fps": 30, "motion_scale": 1.5},
                    "audio_config": {"style": "nature", "include_sfx": True}
                }
            }
        }

class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    progress: float = Field(0.0, description="进度百分比", ge=0.0, le=100.0)
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    error: Optional[str] = Field(None, description="错误信息")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

# ==================== 任务管理器 ====================

class TaskManager:
    """任务管理器 - 管理异步任务的执行和状态"""
    
    def __init__(self):
        self.tasks: Dict[str, TaskResponse] = {}
        self.lock = asyncio.Lock()
    
    async def create_task(self, task_type: str, metadata: Optional[Dict] = None) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        async with self.lock:
            self.tasks[task_id] = TaskResponse(
                task_id=task_id,
                status=TaskStatus.PENDING,
                metadata=metadata or {"type": task_type}
            )
        return task_id
    
    async def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        async with self.lock:
            if task_id in self.tasks:
                for key, value in kwargs.items():
                    if hasattr(self.tasks[task_id], key):
                        setattr(self.tasks[task_id], key, value)
                self.tasks[task_id].updated_at = datetime.now()
    
    async def get_task(self, task_id: str) -> Optional[TaskResponse]:
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    async def list_tasks(self, status: Optional[TaskStatus] = None) -> List[TaskResponse]:
        """列出任务"""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda x: x.created_at, reverse=True)

# 初始化任务管理器
task_manager = TaskManager()

# ==================== MCP模块模拟器 ====================

class MCPSimulator:
    """MCP(Model Control Protocol)模块模拟器"""
    
    @staticmethod
    async def simulate_processing(duration: float = 2.0):
        """模拟处理延迟"""
        await asyncio.sleep(duration)
    
    @staticmethod
    async def text_to_image_mcp(prompt: str, config: Dict) -> Dict:
        """模拟文生图MCP"""
        await MCPSimulator.simulate_processing(3.0)
        
        # 模拟生成图像URL
        image_id = str(uuid.uuid4())[:8]
        image_url = f"/outputs/image_{image_id}.png"
        
        return {
            "image_url": image_url,
            "prompt_used": prompt,
            "config": config,
            "metadata": {
                "model": "stable-diffusion-xl",
                "inference_time": 3.0,
                "resolution": f"{config.get('width', 512)}x{config.get('height', 512)}"
            }
        }
    
    @staticmethod
    async def image_to_video_mcp(image_url: str, config: Dict) -> Dict:
        """模拟图生视频MCP - 生成4秒视频片段"""
        await MCPSimulator.simulate_processing(5.0)
        
        video_id = str(uuid.uuid4())[:8]
        video_url = f"/outputs/video_{video_id}_4s.mp4"
        
        return {
            "video_url": video_url,
            "duration": 4.0,  # 固定4秒
            "fps": config.get("fps", 30),
            "metadata": {
                "model": "stable-video-diffusion",
                "source_image": image_url,
                "motion_scale": config.get("motion_scale", 1.0),
                "frames_generated": 4 * config.get("fps", 30)
            }
        }
    
    @staticmethod
    async def video_to_audio_mcp(video_url: str, config: Dict) -> Dict:
        """模拟视频推理音频MCP"""
        await MCPSimulator.simulate_processing(2.0)
        
        audio_id = str(uuid.uuid4())[:8]
        audio_url = f"/outputs/audio_{audio_id}.mp3"
        
        return {
            "audio_url": audio_url,
            "duration": 4.0,  # 匹配视频长度
            "style": config.get("audio_style", "ambient"),
            "metadata": {
                "model": "audio-gen-v2",
                "source_video": video_url,
                "include_sfx": config.get("include_sfx", True),
                "sample_rate": 44100
            }
        }
    
    @staticmethod
    async def merge_av_mcp(video_segments: List[str], audio_url: str, config: Dict) -> Dict:
        """模拟音视频合成MCP"""
        await MCPSimulator.simulate_processing(4.0)
        
        output_id = str(uuid.uuid4())[:8]
        total_duration = len(video_segments) * 4.0  # 每段4秒
        output_url = f"/outputs/final_{output_id}_{int(total_duration)}s.mp4"
        
        return {
            "output_url": output_url,
            "total_duration": total_duration,
            "segments_count": len(video_segments),
            "metadata": {
                "format": config.get("output_format", "mp4"),
                "quality": config.get("quality", "high"),
                "bitrate": "5000k",
                "resolution": "1920x1080",
                "audio_codec": "aac"
            }
        }

# 初始化MCP模拟器
mcp_simulator = MCPSimulator()

# ==================== API端点实现 ====================

@app.get("/")
async def root():
    """API根路径"""
    return {
        "name": "ReasoningACTION API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "text_to_image": "/api/text-to-image",
            "image_to_video": "/api/image-to-video",
            "video_to_audio": "/api/video-to-audio",
            "merge_videos": "/api/merge-videos",
            "workflow": "/api/workflow",
            "tasks": "/api/tasks"
        }
    }

@app.post("/api/text-to-image", response_model=TaskResponse)
async def text_to_image(request: TextToImageRequest, background_tasks: BackgroundTasks):
    """
    文本生成图像接口
    
    参数:
        request: 文生图请求参数
    
    返回:
        TaskResponse: 包含任务ID和状态的响应
    """
    # 创建异步任务
    task_id = await task_manager.create_task("text_to_image", {
        "prompt": request.prompt,
        "config": request.dict()
    })
    
    # 在后台执行生成任务
    async def generate_image():
        try:
            await task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0.0)
            
            # 调用MCP模块
            result = await mcp_simulator.text_to_image_mcp(
                request.prompt,
                request.dict()
            )
            
            await task_manager.update_task(
                task_id,
                status=TaskStatus.SUCCESS,
                progress=100.0,
                result=result
            )
        except Exception as e:
            await task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    background_tasks.add_task(generate_image)
    
    return await task_manager.get_task(task_id)

@app.post("/api/image-to-video", response_model=TaskResponse)
async def image_to_video(request: ImageToVideoRequest, background_tasks: BackgroundTasks):
    """
    图像生成视频接口 - 生成4秒视频片段
    
    参数:
        request: 图生视频请求参数
    
    返回:
        TaskResponse: 包含任务ID和状态的响应
    """
    task_id = await task_manager.create_task("image_to_video", {
        "image_url": request.image_url,
        "config": request.dict()
    })
    
    async def generate_video():
        try:
            await task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0.0)
            
            result = await mcp_simulator.image_to_video_mcp(
                request.image_url,
                request.dict()
            )
            
            await task_manager.update_task(
                task_id,
                status=TaskStatus.SUCCESS,
                progress=100.0,
                result=result
            )
        except Exception as e:
            await task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    background_tasks.add_task(generate_video)
    
    return await task_manager.get_task(task_id)

@app.post("/api/video-to-audio", response_model=TaskResponse)
async def video_to_audio(request: VideoToAudioRequest, background_tasks: BackgroundTasks):
    """
    视频推理生成音频接口
    
    参数:
        request: 视频转音频请求参数
    
    返回:
        TaskResponse: 包含任务ID和状态的响应
    """
    task_id = await task_manager.create_task("video_to_audio", {
        "video_url": request.video_url,
        "config": request.dict()
    })
    
    async def generate_audio():
        try:
            await task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0.0)
            
            result = await mcp_simulator.video_to_audio_mcp(
                request.video_url,
                request.dict()
            )
            
            await task_manager.update_task(
                task_id,
                status=TaskStatus.SUCCESS,
                progress=100.0,
                result=result
            )
        except Exception as e:
            await task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    background_tasks.add_task(generate_audio)
    
    return await task_manager.get_task(task_id)

@app.post("/api/merge-videos", response_model=TaskResponse)
async def merge_videos(request: VideoMergeRequest, background_tasks: BackgroundTasks):
    """
    视频片段合成接口 - 合并多个4秒片段
    
    参数:
        request: 视频合成请求参数
    
    返回:
        TaskResponse: 包含任务ID和状态的响应
    """
    task_id = await task_manager.create_task("merge_videos", {
        "segments": request.video_segments,
        "audio": request.audio_url,
        "config": request.dict()
    })
    
    async def merge_segments():
        try:
            await task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0.0)
            
            result = await mcp_simulator.merge_av_mcp(
                request.video_segments,
                request.audio_url,
                request.dict()
            )
            
            await task_manager.update_task(
                task_id,
                status=TaskStatus.SUCCESS,
                progress=100.0,
                result=result
            )
        except Exception as e:
            await task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    background_tasks.add_task(merge_segments)
    
    return await task_manager.get_task(task_id)

@app.post("/api/workflow", response_model=TaskResponse)
async def execute_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """
    执行完整工作流 - 文生图→图生视频→视频生音频→合成
    
    参数:
        request: 工作流请求参数，包含迭代次数
    
    返回:
        TaskResponse: 包含任务ID和状态的响应
    """
    task_id = await task_manager.create_task("workflow", {
        "prompt": request.prompt,
        "iterations": request.iterations,
        "total_duration": request.iterations * 4.0,  # 每次迭代4秒
        "config": request.dict()
    })
    
    async def run_workflow():
        try:
            await task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0.0)
            
            workflow_config = request.workflow_config or {}
            video_segments = []
            total_steps = request.iterations * 3 + 2  # 每次迭代3步 + 最后音频和合成2步
            current_step = 0
            
            # 步骤1: 文生图（只需要一次）
            image_config = workflow_config.get("image_config", {})
            image_result = await mcp_simulator.text_to_image_mcp(request.prompt, image_config)
            current_step += 1
            await task_manager.update_task(
                task_id, 
                progress=(current_step / total_steps) * 100
            )
            
            # 步骤2: 迭代生成视频片段
            video_config = workflow_config.get("video_config", {})
            for i in range(request.iterations):
                # 每次迭代生成一个4秒视频片段
                video_result = await mcp_simulator.image_to_video_mcp(
                    image_result["image_url"],
                    video_config
                )
                video_segments.append(video_result["video_url"])
                current_step += 1
                await task_manager.update_task(
                    task_id,
                    progress=(current_step / total_steps) * 100,
                    metadata={
                        "current_iteration": i + 1,
                        "total_iterations": request.iterations,
                        "segments_generated": len(video_segments)
                    }
                )
                
                # 可以使用上一帧的最后一帧作为下一个片段的起始图像（可选优化）
                if request.auto_enhance and i < request.iterations - 1:
                    # 模拟提取视频最后一帧
                    image_result["image_url"] = f"/outputs/frame_{i+1}.png"
            
            # 步骤3: 视频推理音频
            audio_config = workflow_config.get("audio_config", {})
            # 使用第一个视频片段推理音频风格
            audio_result = await mcp_simulator.video_to_audio_mcp(
                video_segments[0],
                audio_config
            )
            current_step += 1
            await task_manager.update_task(
                task_id,
                progress=(current_step / total_steps) * 100
            )
            
            # 步骤4: 合成最终视频
            merge_config = workflow_config.get("merge_config", {})
            final_result = await mcp_simulator.merge_av_mcp(
                video_segments,
                audio_result["audio_url"],
                merge_config
            )
            current_step += 1
            
            # 完成工作流
            await task_manager.update_task(
                task_id,
                status=TaskStatus.SUCCESS,
                progress=100.0,
                result={
                    "final_video": final_result["output_url"],
                    "total_duration": final_result["total_duration"],
                    "segments": video_segments,
                    "audio": audio_result["audio_url"],
                    "workflow_summary": {
                        "prompt": request.prompt,
                        "iterations": request.iterations,
                        "segments_count": len(video_segments),
                        "total_duration_seconds": final_result["total_duration"],
                        "processing_steps": total_steps
                    }
                }
            )
            
        except Exception as e:
            await task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    background_tasks.add_task(run_workflow)
    
    return await task_manager.get_task(task_id)

@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """
    获取任务状态
    
    参数:
        task_id: 任务ID
    
    返回:
        TaskResponse: 任务详细信息
    """
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task

@app.get("/api/tasks", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    limit: int = 10,
    offset: int = 0
):
    """
    列出所有任务
    
    参数:
        status: 可选的状态过滤
        limit: 返回数量限制
        offset: 偏移量
    
    返回:
        List[TaskResponse]: 任务列表
    """
    tasks = await task_manager.list_tasks(status)
    return tasks[offset:offset + limit]

@app.delete("/api/tasks/{task_id}")
async def cancel_task(task_id: str):
    """
    取消任务
    
    参数:
        task_id: 任务ID
    
    返回:
        成功或错误信息
    """
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task.status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        return {"message": f"Task {task_id} already completed"}
    
    await task_manager.update_task(task_id, status=TaskStatus.CANCELLED)
    return {"message": f"Task {task_id} cancelled successfully"}

# ==================== 健康检查和监控 ====================

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len([t for t in task_manager.tasks.values() if t.status == TaskStatus.PROCESSING])
    }

@app.get("/api/stats")
async def get_statistics():
    """获取系统统计信息"""
    tasks = list(task_manager.tasks.values())
    return {
        "total_tasks": len(tasks),
        "pending": len([t for t in tasks if t.status == TaskStatus.PENDING]),
        "processing": len([t for t in tasks if t.status == TaskStatus.PROCESSING]),
        "success": len([t for t in tasks if t.status == TaskStatus.SUCCESS]),
        "failed": len([t for t in tasks if t.status == TaskStatus.FAILED]),
        "cancelled": len([t for t in tasks if t.status == TaskStatus.CANCELLED])
    }

# ==================== WebSocket支持（实时更新） ====================

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/tasks/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket端点 - 实时推送任务状态更新"""
    await manager.connect(websocket)
    try:
        while True:
            # 定期发送任务状态
            task = await task_manager.get_task(task_id)
            if task:
                await websocket.send_json(task.dict())
                if task.status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==================== 启动配置 ====================

if __name__ == "__main__":
    import uvicorn
    
    # 开发环境配置
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发时启用热重载
        log_level="info"
    )
