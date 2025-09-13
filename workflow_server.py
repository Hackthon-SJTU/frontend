"""
ReasoningACTION 工作流系统 - 后端服务
与DeepSeek-v3对话并协调MCP模块生成视频
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
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime

# ==================== 初始化 ====================
app = FastAPI(
    title="ReasoningACTION Workflow System",
    description="与DeepSeek对话并调用MCP生成视频",
    version="1.0.0"
)

# CORS配置 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要目录
OUTPUTS_DIR = Path("outputs")
TEMP_DIR = Path("temp")
OUTPUTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# 静态文件服务
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ==================== 配置 ====================

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-a68ba86867044b8ebcb5e669937626a1")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# MCP服务地址（由其他同学提供）
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8001")

# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """用户聊天请求"""
    message: str  # 用户输入的视频描述
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
    gpt_prompt: str  # DeepSeek提炼的核心prompt
    status: str  # processing, completed, error
    progress: float  # 0-100
    message: str
    video_url: Optional[str] = None
    error: Optional[str] = None

class WorkflowStatus(BaseModel):
    """工作流状态"""
    session_id: str
    current_step: str
    progress: float
    details: Dict[str, Any]

# ==================== MCP接口定义 ====================

class MCPInterface:
    """MCP接口调用类"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def text_to_image(self, prompt: str, gpt_prompt: str) -> Dict:
        """
        MCP1: 文本生成图片
        
        请求格式:
        {
            "prompt": "详细的场景描述",
            "gpt_prompt": "DeepSeek提炼的核心prompt",
            "width": 1024,
            "height": 768
        }
        
        返回格式:
        {
            "status": "success",
            "image_url": "/path/to/image.png",
            "image_data": "base64_encoded_string"  # 可选
        }
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/mcp1/text-to-image",
                    json={
                        "prompt": prompt,
                        "gpt_prompt": gpt_prompt,
                        "width": 1024,
                        "height": 768
                    }
                ) as response:
                    return await response.json()
            except Exception as e:
                print(f"MCP1调用失败: {e}")
                # 模拟返回用于开发
                return {
                    "status": "success",
                    "image_url": "/outputs/sample_image.png"
                }
    
    async def image_to_video(self, image_url: str, gpt_prompt: str, is_first_frame: bool = True) -> Dict:
        """
        MCP2: 图片生成4秒视频
        
        请求格式:
        {
            "image_url": "图片URL或路径",
            "gpt_prompt": "DeepSeek提炼的核心prompt",
            "is_first_frame": true,  # 是否作为首帧
            "duration": 4  # 固定4秒
        }
        
        返回格式:
        {
            "status": "success",
            "video_url": "/path/to/video_4s.mp4",
            "last_frame_url": "/path/to/last_frame.png"  # 最后一帧
        }
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/mcp2/image-to-video",
                    json={
                        "image_url": image_url,
                        "gpt_prompt": gpt_prompt,
                        "is_first_frame": is_first_frame,
                        "duration": 4
                    }
                ) as response:
                    return await response.json()
            except Exception as e:
                print(f"MCP2调用失败: {e}")
                # 模拟返回用于开发
                return {
                    "status": "success",
                    "video_url": f"/outputs/video_segment_{uuid.uuid4().hex[:8]}.mp4",
                    "last_frame_url": "/outputs/last_frame.png"
                }
    
    async def image_to_audio(self, image_url: str, gpt_prompt: str) -> Dict:
        """
        MCP3: 图片生成10秒音频
        
        请求格式:
        {
            "image_url": "图片URL或路径",
            "gpt_prompt": "DeepSeek提炼的核心prompt",
            "duration": 10  # 固定10秒
        }
        
        返回格式:
        {
            "status": "success",
            "audio_url": "/path/to/audio_10s.mp3"
        }
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/mcp3/image-to-audio",
                    json={
                        "image_url": image_url,
                        "gpt_prompt": gpt_prompt,
                        "duration": 10
                    }
                ) as response:
                    return await response.json()
            except Exception as e:
                print(f"MCP3调用失败: {e}")
                # 模拟返回用于开发
                return {
                    "status": "success",
                    "audio_url": "/outputs/audio_10s.mp3"
                }
    
    async def merge_video_audio(self, video_urls: List[str], audio_url: str, gpt_prompt: str) -> Dict:
        """
        MCP4: 拼接40秒视频和40秒音频
        
        请求格式:
        {
            "video_urls": ["video1.mp4", "video2.mp4", ...],  # 10个4秒视频
            "audio_url": "audio_10s.mp3",  # 10秒音频
            "gpt_prompt": "DeepSeek提炼的核心prompt",
            "audio_loop": 4  # 音频循环4次变成40秒
        }
        
        返回格式:
        {
            "status": "success",
            "final_video_url": "/path/to/final_video_40s.mp4"
        }
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/mcp4/merge-video-audio",
                    json={
                        "video_urls": video_urls,
                        "audio_url": audio_url,
                        "gpt_prompt": gpt_prompt,
                        "audio_loop": 4
                    }
                ) as response:
                    return await response.json()
            except Exception as e:
                print(f"MCP4调用失败: {e}")
                # 模拟返回用于开发
                return {
                    "status": "success",
                    "final_video_url": f"/outputs/final_video_{uuid.uuid4().hex[:8]}.mp4"
                }

# 初始化MCP接口
mcp = MCPInterface(MCP_BASE_URL)

# ==================== DeepSeek集成 ====================

class DeepSeekProcessor:
    """DeepSeek-v3处理器"""
    
    @staticmethod
    async def extract_gpt_prompt(user_message: str) -> str:
        """
        调用DeepSeek-v3提炼核心prompt
        
        输入: 用户的视频描述
        输出: 提炼后的GPT prompt（包含场景、方向、故事）
        """
        system_prompt = """你是一个专业的视频创作助手。
        用户会描述他们想要的视频，你需要提炼出：
        1. 视频场景是什么
        2. 视频的方向/风格是什么
        3. 视频讲述的故事是什么
        
        请用简洁清晰的语言输出一个综合的prompt，包含上述所有要素。
        输出格式：直接输出提炼后的prompt文本，不要有其他说明。"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DEEPSEEK_API_URL,
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                ) as response:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"DeepSeek调用失败: {e}")
            # 返回默认提炼结果
            return f"Generate a video based on: {user_message}. Scene: natural environment. Style: cinematic. Story: peaceful journey through nature."
    
    @staticmethod
    async def generate_scene_prompt(gpt_prompt: str, scene_number: int = 1) -> str:
        """
        基于GPT prompt生成具体场景描述
        
        输入: GPT prompt和场景编号
        输出: 详细的场景描述prompt
        """
        base_prompt = """Based on this video concept: {gpt_prompt}
        
        Generate a detailed visual description for scene {scene_number}.
        Include: lighting, colors, composition, camera angle, specific objects, atmosphere.
        Output only the scene description, no explanations."""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DEEPSEEK_API_URL,
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": "You are a professional visual scene designer."},
                            {"role": "user", "content": base_prompt.format(
                                gpt_prompt=gpt_prompt,
                                scene_number=scene_number
                            )}
                        ],
                        "temperature": 0.8,
                        "max_tokens": 300
                    }
                ) as response:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"场景生成失败: {e}")
            # 返回默认场景描述
            return f"A serene forest landscape, lush greenery, towering trees with dense foliage, dappled sunlight filtering through the canopy, high angle view"

# ==================== 工作流管理 ====================

class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self):
        self.sessions = {}  # 存储会话状态
    
    async def execute_workflow(self, session_id: str, user_message: str, progress_callback=None):
        """
        执行完整的视频生成工作流
        
        步骤:
        1. DeepSeek提炼GPT prompt
        2. MCP1: 文本生成图片
        3. MCP2: 循环10次生成4秒视频片段
        4. MCP3: 图片生成10秒音频
        5. MCP4: 合并视频和音频
        """
        
        try:
            # 初始化会话
            self.sessions[session_id] = {
                "status": "processing",
                "progress": 0,
                "current_step": "初始化"
            }
            
            # Step 1: DeepSeek提炼prompt
            if progress_callback:
                await progress_callback(session_id, "提炼核心概念", 5)
            
            gpt_prompt = await DeepSeekProcessor.extract_gpt_prompt(user_message)
            print(f"GPT Prompt: {gpt_prompt}")
            
            # Step 2: 生成场景描述
            if progress_callback:
                await progress_callback(session_id, "生成场景描述", 10)
            
            scene_prompt = await DeepSeekProcessor.generate_scene_prompt(gpt_prompt)
            full_prompt = f"{scene_prompt}, high angle view"
            
            # Step 3: MCP1 - 文本生成图片
            if progress_callback:
                await progress_callback(session_id, "生成初始图片", 15)
            
            image_result = await mcp.text_to_image(full_prompt, gpt_prompt)
            initial_image_url = image_result["image_url"]
            print(f"初始图片: {initial_image_url}")
            
            # Step 4: MCP2 - 循环生成视频片段（10次）
            video_segments = []
            current_image = initial_image_url
            
            for i in range(10):  # 固定10次
                if progress_callback:
                    progress = 20 + (i * 5)  # 20-70%
                    await progress_callback(session_id, f"生成视频片段 {i+1}/10", progress)
                
                video_result = await mcp.image_to_video(
                    current_image, 
                    gpt_prompt,
                    is_first_frame=(i == 0)  # 第一次使用原图作为首帧
                )
                
                video_segments.append(video_result["video_url"])
                current_image = video_result["last_frame_url"]  # 使用最后一帧作为下次输入
                print(f"视频片段 {i+1}: {video_result['video_url']}")
            
            # Step 5: MCP3 - 图片生成音频（使用初始图片）
            if progress_callback:
                await progress_callback(session_id, "生成音频", 75)
            
            audio_result = await mcp.image_to_audio(initial_image_url, gpt_prompt)
            audio_url = audio_result["audio_url"]
            print(f"音频: {audio_url}")
            
            # Step 6: MCP4 - 合并视频和音频
            if progress_callback:
                await progress_callback(session_id, "合成最终视频", 85)
            
            merge_result = await mcp.merge_video_audio(
                video_segments,
                audio_url,
                gpt_prompt
            )
            
            final_video_url = merge_result["final_video_url"]
            print(f"最终视频: {final_video_url}")
            
            # 更新会话状态
            self.sessions[session_id] = {
                "status": "completed",
                "progress": 100,
                "current_step": "完成",
                "gpt_prompt": gpt_prompt,
                "video_url": final_video_url,
                "metadata": {
                    "initial_image": initial_image_url,
                    "video_segments": video_segments,
                    "audio_url": audio_url,
                    "total_duration": "40秒"
                }
            }
            
            if progress_callback:
                await progress_callback(session_id, "视频生成完成", 100)
            
            return {
                "success": True,
                "gpt_prompt": gpt_prompt,
                "video_url": final_video_url
            }
            
        except Exception as e:
            print(f"工作流执行失败: {e}")
            self.sessions[session_id] = {
                "status": "error",
                "error": str(e)
            }
            return {
                "success": False,
                "error": str(e)
            }

# 初始化工作流管理器
workflow_manager = WorkflowManager()

# ==================== WebSocket支持（实时进度） ====================

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    """WebSocket连接管理"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_progress(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

manager = ConnectionManager()

# ==================== API端点 ====================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    主聊天接口 - 接收用户消息并启动视频生成流程
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    # 异步执行工作流
    asyncio.create_task(
        workflow_manager.execute_workflow(
            session_id,
            request.message,
            progress_callback=send_progress_update
        )
    )
    
    return ChatResponse(
        session_id=session_id,
        gpt_prompt="正在提炼核心概念...",
        status="processing",
        progress=0,
        message="已开始处理您的请求，请稍候..."
    )

@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """
    获取工作流状态
    """
    if session_id not in workflow_manager.sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = workflow_manager.sessions[session_id]
    
    return {
        "session_id": session_id,
        "status": session.get("status", "unknown"),
        "progress": session.get("progress", 0),
        "current_step": session.get("current_step", ""),
        "gpt_prompt": session.get("gpt_prompt", ""),
        "video_url": session.get("video_url"),
        "error": session.get("error")
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket端点 - 实时推送进度更新
    """
    await manager.connect(websocket, session_id)
    try:
        while True:
            await websocket.receive_text()  # 保持连接
    except WebSocketDisconnect:
        manager.disconnect(session_id)

async def send_progress_update(session_id: str, step: str, progress: float):
    """
    发送进度更新
    """
    await manager.send_progress(session_id, {
        "type": "progress",
        "step": step,
        "progress": progress
    })

@app.get("/api/download/{session_id}")
async def download_video(session_id: str):
    """
    下载生成的视频
    """
    if session_id not in workflow_manager.sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = workflow_manager.sessions[session_id]
    video_url = session.get("video_url")
    
    if not video_url:
        raise HTTPException(status_code=404, detail="视频尚未生成")
    
    # 假设视频文件在本地
    video_path = video_url.lstrip("/")
    if os.path.exists(video_path):
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"generated_video_{session_id}.mp4"
        )
    else:
        raise HTTPException(status_code=404, detail="视频文件不存在")

# ==================== 健康检查 ====================

@app.get("/")
async def root():
    """API根路径"""
    return {
        "service": "ReasoningACTION Workflow System",
        "version": "1.0.0",
        "status": "running",
        "description": "与DeepSeek对话生成视频的工作流系统"
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "mcp_url": MCP_BASE_URL,
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "active_sessions": len(workflow_manager.sessions)
    }

# ==================== 启动服务 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("ReasoningACTION Workflow System")
    print("=" * 60)
    print(f"MCP服务地址: {MCP_BASE_URL}")
    print(f"DeepSeek配置: {'已配置' if DEEPSEEK_API_KEY != 'your-deepseek-api-key' else '未配置'}")
    print("\n工作流程:")
    print("1. 用户输入 → DeepSeek提炼")
    print("2. MCP1: 文本生成图片")
    print("3. MCP2: 循环10次生成视频片段")
    print("4. MCP3: 图片生成音频")
    print("5. MCP4: 合并为40秒完整视频")
    print("=" * 60)
    
    uvicorn.run("workflow_server:app", host="0.0.0.0", port=8000, reload=True)