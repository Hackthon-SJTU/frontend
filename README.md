# ReasoningACTION 多模态生成系统 - 部署与使用指南

## 项目概述

ReasoningACTION 是一个基于大模型的多模态内容生成系统，实现了从文本到视频的完整创作流程：
- **文本 → 图像 → 视频 → 音频 → 音视频合成**
- 支持迭代生成（每个视频片段4秒）
- 工业级代码架构，前后端分离
- 实时任务状态追踪

## 系统架构

```
┌─────────────────────────────────────────────────┐
│                   前端 (React)                   │
│  - 用户界面                                      │
│  - 工作流控制                                    │
│  - 实时状态更新                                  │
└─────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────┐
│                后端 (FastAPI)                    │
│  - RESTful API                                  │
│  - 异步任务管理                                  │
│  - MCP模块协调                                   │
└─────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────┐
│                  MCP 模块层                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │文生图MCP │  │图生视频MCP│  │视频音频MCP│     │
│  └──────────┘  └──────────┘  └──────────┘     │
│  ┌──────────────────────────────────────┐     │
│  │        音视频合成MCP                   │     │
│  └──────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

## 快速部署

### 1. 环境要求

- Python 3.8+
- Node.js 14+ (可选，用于开发React)
- 现代浏览器（Chrome, Firefox, Safari, Edge）

### 2. 后端部署

#### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install fastapi uvicorn python-multipart aiofiles pydantic
```

#### 启动后端服务

```bash
# 开发模式（带热重载）
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. 前端部署

前端是一个独立的HTML文件，可以直接在浏览器中打开，或通过任何静态文件服务器提供服务。

#### 方式一：直接打开
- 双击 `index.html` 文件即可在浏览器中打开

#### 方式二：使用Python简单服务器
```bash
# 在前端文件目录下执行
python -m http.server 3000
```

#### 方式三：使用Node.js服务器
```bash
# 安装 http-server
npm install -g http-server

# 启动服务
http-server -p 3000
```

## API 接口文档

### 基础信息
- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`

### 核心接口

#### 1. 文本生成图像
```http
POST /api/text-to-image
```

**请求参数：**
```json
{
  "prompt": "美丽的日落山景，油画风格",
  "negative_prompt": "低质量，模糊",
  "width": 1024,
  "height": 768,
  "guidance_scale": 7.5,
  "num_inference_steps": 50,
  "seed": 42
}
```

**响应示例：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "progress": 0.0,
  "result": null,
  "error": null,
  "created_at": "2025-01-10T10:00:00",
  "updated_at": "2025-01-10T10:00:00",
  "metadata": {
    "type": "text_to_image",
    "prompt": "美丽的日落山景，油画风格"
  }
}
```

#### 2. 图像生成视频（4秒片段）
```http
POST /api/image-to-video
```

**请求参数：**
```json
{
  "image_url": "/outputs/image_abc123.png",
  "motion_scale": 1.5,
  "fps": 30,
  "duration": 4.0,
  "transition_type": "fade"
}
```

**响应：返回任务对象（同上）**

#### 3. 视频推理音频
```http
POST /api/video-to-audio
```

**请求参数：**
```json
{
  "video_url": "/outputs/video_xyz789_4s.mp4",
  "audio_style": "cinematic",
  "include_sfx": true,
  "volume": 0.8
}
```

#### 4. 视频片段合成
```http
POST /api/merge-videos
```

**请求参数：**
```json
{
  "video_segments": [
    "/outputs/video_1_4s.mp4",
    "/outputs/video_2_4s.mp4",
    "/outputs/video_3_4s.mp4"
  ],
  "audio_url": "/outputs/audio_master.mp3",
  "output_format": "mp4",
  "quality": "high"
}
```

#### 5. 完整工作流
```http
POST /api/workflow
```

**请求参数：**
```json
{
  "prompt": "宁静的森林中流淌的溪水",
  "iterations": 10,
  "auto_enhance": true,
  "workflow_config": {
    "image_config": {
      "width": 1024,
      "height": 768
    },
    "video_config": {
      "fps": 30,
      "motion_scale": 1.5
    },
    "audio_config": {
      "audio_style": "nature",
      "include_sfx": true
    }
  }
}
```

**工作流响应（完成后）：**
```json
{
  "task_id": "...",
  "status": "success",
  "progress": 100.0,
  "result": {
    "final_video": "/outputs/final_abc_40s.mp4",
    "total_duration": 40.0,
    "segments": ["video1.mp4", "video2.mp4", ...],
    "audio": "/outputs/audio.mp3",
    "workflow_summary": {
      "prompt": "宁静的森林中流淌的溪水",
      "iterations": 10,
      "segments_count": 10,
      "total_duration_seconds": 40.0,
      "processing_steps": 32
    }
  }
}
```

#### 6. 任务状态查询
```http
GET /api/tasks/{task_id}
```

#### 7. 任务列表
```http
GET /api/tasks?status=processing&limit=10&offset=0
```

## WebSocket 实时更新

系统支持WebSocket连接以获取实时任务状态更新：

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/tasks/{task_id}');

ws.onmessage = (event) => {
    const taskStatus = JSON.parse(event.data);
    console.log('Task Update:', taskStatus);
};
```

## 工作流详解

### 标准处理流程

1. **文本生成图像**
   - 输入：文本提示词
   - 输出：生成的图像URL
   - 处理时间：约3秒（模拟）

2. **图像生成视频（迭代）**
   - 输入：图像URL
   - 输出：4秒视频片段
   - 处理时间：约5秒/片段（模拟）
   - 迭代次数：用户自定义（1-20次）

3. **视频推理音频**
   - 输入：视频URL
   - 输出：匹配的音频文件
   - 处理时间：约2秒（模拟）

4. **音视频合成**
   - 输入：多个视频片段 + 音频文件
   - 输出：完整的视频文件
   - 处理时间：约4秒（模拟）

### 时长计算公式
```
总时长 = 迭代次数 × 4秒
例如：10次迭代 = 40秒视频
```

## 错误处理

所有API端点都遵循统一的错误响应格式：

```json
{
  "detail": "错误描述信息"
}
```

常见HTTP状态码：
- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源未找到
- `422`: 参数验证失败
- `500`: 服务器内部错误

## 性能优化建议

### 后端优化

1. **使用Redis缓存任务状态**
```python
# 安装: pip install redis aioredis
import aioredis

redis = await aioredis.create_redis_pool('redis://localhost')
await redis.set(f'task:{task_id}', json.dumps(task_data))
```

2. **使用消息队列（Celery）**
```python
# 安装: pip install celery
from celery import Celery

celery_app = Celery('tasks', broker='redis://localhost:6379')

@celery_app.task
def process_workflow(prompt, iterations):
    # 异步处理逻辑
    pass
```

3. **数据库持久化（PostgreSQL）**
```python
# 安装: pip install sqlalchemy asyncpg
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost/reasoning_action"
)
```

### 前端优化

1. **使用React生产构建**
```bash
npx create-react-app reasoning-action
cd reasoning-action
npm run build
```

2. **启用CDN加速**
- 将静态资源部署到CDN
- 使用图片懒加载
- 启用Gzip压缩

3. **实现虚拟滚动**
```javascript
// 对于大量任务列表
import { FixedSizeList } from 'react-window';
```

## 生产部署

### Docker容器化

创建 `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

创建 `docker-compose.yml`:
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
```

### Kubernetes部署

创建 `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reasoning-action-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reasoning-action
  template:
    metadata:
      labels:
        app: reasoning-action
    spec:
      containers:
      - name: backend
        image: reasoning-action:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## 监控和日志

### 集成Prometheus监控
```python
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('app_requests_total', 'Total requests')
request_duration = Histogram('app_request_duration_seconds', 'Request duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 结构化日志
```python
import structlog

logger = structlog.get_logger()

logger.info("workflow_started", 
            task_id=task_id, 
            prompt=prompt, 
            iterations=iterations)
```

## 安全性建议

1. **API认证**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/api/workflow")
async def workflow(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # 验证token
    pass
```

2. **请求限流**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/workflow")
@limiter.limit("10/minute")
async def workflow(request: Request):
    pass
```

3. **输入验证**
- 使用Pydantic模型严格验证输入
- 限制文件上传大小
- 过滤敏感词汇

## 故障排除

### 常见问题

1. **CORS错误**
   - 确保后端CORS配置正确
   - 检查前端请求的Origin

2. **任务超时**
   - 增加uvicorn的timeout设置
   - 实现任务重试机制

3. **内存不足**
   - 限制并发任务数量
   - 实现任务队列

## 联系支持

- 项目仓库：[GitHub链接]
- 问题反馈：[Issue页面]
- 技术文档：[Wiki页面]

---

**版本**: 1.0.0  
**最后更新**: 2025-01-10  
**作者**: ReasoningACTION Team