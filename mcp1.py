"""
MCP1 - 文本生成图片服务
使用阿里云百炼平台的qwen-image模型
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import json
import os
import dashscope
import requests
from dashscope import MultiModalConversation
import uuid
from pathlib import Path
from datetime import datetime
import traceback

# ==================== 初始化FastAPI ====================
app = FastAPI(
    title="MCP1 - Text to Image Service",
    description="文本生成图片服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要的目录
OUTPUTS_DIR = Path("outputs")
TEMP_DIR = Path("temp")
OUTPUTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# 静态文件服务
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# ==================== 配置 ====================

# 阿里云API配置
ALIYUN_API_KEY = 'sk-60b22327f9a7438b99245a48ac098f1b'

# ==================== 数据模型 ====================

class TextToImageRequest(BaseModel):
    """文生图请求模型"""
    prompt: str  # 场景描述
    gpt_prompt: str  # DeepSeek提炼的核心prompt
    width: Optional[int] = 1024
    height: Optional[int] = 768

class TextToImageResponse(BaseModel):
    """文生图响应模型"""
    status: str
    image_url: str
    image_data: Optional[str] = None  # base64编码（可选）

# ==================== 核心功能 ====================

def generate_image_with_qwen(prompt: str, gpt_prompt: str, width: int = 1024, height: int = 768):
    """
    使用阿里云qwen-image模型生成图片
    
    Args:
        prompt: 场景描述
        gpt_prompt: 核心概念（用于增强生成效果）
        width: 图片宽度
        height: 图片高度
    
    Returns:
        dict: 包含图片URL和本地路径的字典
    """
    print("=" * 60)
    print(f"[MCP1] 开始生成图片")
    print(f"[MCP1] 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MCP1] 场景描述: {prompt}")
    print(f"[MCP1] GPT核心概念: {gpt_prompt}")
    print(f"[MCP1] 目标尺寸: {width}x{height}")
    print("=" * 60)
    
    # 组合prompt，将GPT prompt作为增强
    enhanced_prompt = f"{prompt}\n\nCore concept: {gpt_prompt}"
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"text": enhanced_prompt}
            ]
        }
    ]
    
    print(f"[MCP1] 调用阿里云API...")
    print(f"[MCP1] 增强后的prompt: {enhanced_prompt}")
    
    try:
        # 调用阿里云API
        response = MultiModalConversation.call(
            api_key=ALIYUN_API_KEY,
            model="qwen-image",
            messages=messages,
            result_format='message',
            stream=False,
            watermark=True,
            prompt_extend=True,
            negative_prompt='',
            size='1328*1328'  # 使用较大尺寸，后续可以缩放
        )
        
        print(f"[MCP1] API响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            print(f"[MCP1] API调用成功")
            print(f"[MCP1] 响应数据: {json.dumps(response, ensure_ascii=False, indent=2)}")
            
            # 提取图片URL
            image_url = response['output']['choices'][0]['message']['content'][0]['image']
            print(f"[MCP1] 获取到图片URL: {image_url}")
            
            # 生成唯一文件名
            image_id = str(uuid.uuid4())[:8]
            image_filename = f"generated_image_{image_id}.png"
            image_path = OUTPUTS_DIR / image_filename
            
            # 下载图片
            print(f"[MCP1] 开始下载图片...")
            download_response = requests.get(image_url)
            
            if download_response.status_code == 200:
                # 保存图片
                with open(image_path, 'wb') as f:
                    f.write(download_response.content)
                print(f"[MCP1] 图片已保存到: {image_path}")
                
                # 返回结果
                result = {
                    "remote_url": image_url,
                    "local_path": str(image_path),
                    "web_url": f"/outputs/{image_filename}"
                }
                
                print(f"[MCP1] 生成成功!")
                print(f"[MCP1] 本地路径: {result['local_path']}")
                print(f"[MCP1] Web访问路径: {result['web_url']}")
                
                return result
            else:
                error_msg = f"图片下载失败，HTTP状态码: {download_response.status_code}"
                print(f"[MCP1] 错误: {error_msg}")
                raise Exception(error_msg)
        else:
            error_msg = f"API调用失败 - HTTP状态码: {response.status_code}, 错误码: {response.code}, 错误信息: {response.message}"
            print(f"[MCP1] 错误: {error_msg}")
            print(f"[MCP1] 请参考文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            raise Exception(error_msg)
            
    except Exception as e:
        print(f"[MCP1] 异常: {str(e)}")
        print(f"[MCP1] 异常堆栈: {traceback.format_exc()}")
        raise

# ==================== API端点 ====================

@app.post("/mcp1/text-to-image", response_model=TextToImageResponse)
async def text_to_image(request: TextToImageRequest):
    """
    MCP1标准接口：文本生成图片
    
    接收文本描述和GPT prompt，生成对应的图片
    """
    print("\n" + "=" * 80)
    print(f"[MCP1 API] 收到文生图请求")
    print(f"[MCP1 API] 请求时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MCP1 API] 请求参数:")
    print(f"  - prompt: {request.prompt}")
    print(f"  - gpt_prompt: {request.gpt_prompt}")
    print(f"  - width: {request.width}")
    print(f"  - height: {request.height}")
    print("=" * 80)
    
    try:
        # 调用核心生成函数
        result = generate_image_with_qwen(
            prompt=request.prompt,
            gpt_prompt=request.gpt_prompt,
            width=request.width,
            height=request.height
        )
        
        # 构建响应
        response = TextToImageResponse(
            status="success",
            image_url=result["web_url"]
        )
        
        print(f"[MCP1 API] 响应成功")
        print(f"[MCP1 API] 返回图片URL: {response.image_url}")
        print("=" * 80 + "\n")
        
        return response
        
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        print(f"[MCP1 API] 错误: {error_msg}")
        print("=" * 80 + "\n")
        
        # 返回错误响应
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def root():
    """服务信息"""
    return {
        "service": "MCP1 - Text to Image",
        "model": "qwen-image",
        "provider": "Aliyun",
        "status": "running",
        "endpoints": {
            "text_to_image": "/mcp1/text-to-image",
            "outputs": "/outputs/",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "api_key_configured": bool(ALIYUN_API_KEY),
        "outputs_dir": str(OUTPUTS_DIR),
        "temp_dir": str(TEMP_DIR)
    }

# ==================== 测试端点 ====================

@app.post("/test")
async def test_generation():
    """测试生成功能"""
    test_request = TextToImageRequest(
        prompt="宁静的森林景观，郁郁葱葱的绿色植物，参天大树和茂密的树叶，斑驳的阳光透过树冠，一条温柔的溪流蜿蜒穿过场景，河岸两旁生机勃勃的野花和蕨类植物，宁静而未受破坏的荒野，比较高视角",
        gpt_prompt="A cinematic forest scene with natural lighting and peaceful atmosphere",
        width=1024,
        height=768
    )
    
    return await text_to_image(test_request)

# ==================== 启动服务 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("MCP1 - 文本生成图片服务")
    print("=" * 80)
    print(f"模型: qwen-image (阿里云百炼)")
    print(f"API Key: {'已配置' if ALIYUN_API_KEY else '未配置'}")
    print(f"输出目录: {OUTPUTS_DIR}")
    print(f"临时目录: {TEMP_DIR}")
    print("\n端点:")
    print("  POST /mcp1/text-to-image - 文本生成图片")
    print("  GET  /outputs/           - 访问生成的图片")
    print("  POST /test               - 测试生成")
    print("  GET  /health             - 健康检查")
    print("=" * 80 + "\n")
    
    # 启动服务，监听8001端口
    uvicorn.run("mcp1:app", host="0.0.0.0", port=8001, reload=True)