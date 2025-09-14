# ReasoningACTION Open-world 4D - 快速启动指南

## 系统架构
```
用户 ←→ 前端(React) ←→ 后端(FastAPI) ←→ DeepSeek-v3
                            ↓
                    MCP模块（其他同学提供）
                    - MCP1: 文本→图片
                    - MCP2: 图片→视频(4s)
                    - MCP3: 图片→音频(10s)  
                    - MCP4: 合成(40s)
```

## 文件结构
```
project/
├── workflow_server.py    # 后端服务（工作流协调）
├── index.html           # 前端界面
├── MCP_API_规范.md      # MCP接口规范（给其他同学）
├── README.md           # 本文档
└── outputs/            # 生成的文件目录（自动创建）
```

## 环境要求
- Python 3.8+
- 现代浏览器（Chrome/Firefox/Safari）

## 快速启动

### 1. 安装依赖
```bash
pip install fastapi uvicorn aiohttp python-multipart
```

### 2. 配置环境变量
```bash
# Linux/Mac
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export MCP_BASE_URL="http://localhost:8001"  # MCP服务地址

# Windows
set DEEPSEEK_API_KEY=your-deepseek-api-key
set MCP_BASE_URL=http://localhost:8001
```

### 3. 启动后端服务
```bash
python workflow_server.py
```
服务将在 http://localhost:8000 启动

### 4. 打开前端
在浏览器中打开 `index.html` 文件

## 使用流程

1. **输入描述**: 在聊天框输入视频描述
   - 例如："生成一个森林场景的流式视频"

2. **AI处理**: 系统自动执行以下步骤
   - DeepSeek提炼核心概念
   - 生成初始图片
   - 生成10个4秒视频片段
   - 生成10秒音频
   - 合成40秒完整视频

3. **下载结果**: 视频生成完成后可以预览和下载

## 开发模式

### 无MCP服务时的测试
后端代码已包含模拟返回，即使MCP服务未启动也可以测试工作流：
- 会返回模拟的文件路径
- 不会生成实际的视频文件
- 可以测试完整的工作流程

### 调试模式
```bash
# 启动时开启热重载
uvicorn workflow_server:app --reload --host 0.0.0.0 --port 8000
```

## API端点

### 主要端点
- `POST /api/chat` - 发送消息开始生成
- `GET /api/status/{session_id}` - 查询生成状态
- `GET /api/download/{session_id}` - 下载生成的视频
- `WS /ws/{session_id}` - WebSocket实时进度

### 健康检查
- `GET /` - 服务信息
- `GET /health` - 健康状态

## 配置说明

### DeepSeek配置
需要有效的DeepSeek API Key才能使用AI提炼功能。
如未配置，系统会使用默认的提炼结果。

### MCP服务配置
MCP服务地址默认为 `http://localhost:8001`
确保MCP服务已按照规范实现4个接口：
- `/mcp1/text-to-image`
- `/mcp2/image-to-video`
- `/mcp3/image-to-audio`
- `/mcp4/merge-video-audio`

## 生成规格
- **视频总时长**: 40秒
- **视频片段**: 10个×4秒
- **音频时长**: 10秒（循环4次）
- **视频分辨率**: 1024×768
- **输出格式**: MP4

## 常见问题

### Q: MCP服务未启动怎么办？
A: 系统会使用模拟数据，可以测试工作流但不会生成实际文件。

### Q: 生成过程需要多长时间？
A: 完整流程约2-3分钟，取决于MCP服务的处理速度。

### Q: 可以修改视频时长吗？
A: 当前版本固定为40秒（10个4秒片段），修改需要调整代码中的常量。

### Q: 支持哪些视频场景？
A: 取决于MCP模型能力，建议描述自然场景、城市景观等常见场景。

## 技术栈
- **后端**: FastAPI, Python异步
- **前端**: React, TailwindCSS
- **通信**: REST API, WebSocket
- **AI**: DeepSeek-v3

## 注意事项
1. 这是一个Demo系统，没有用户认证和限额功能
2. 生成的文件保存在`outputs`目录
3. 建议定期清理`outputs`目录以节省空间
4. WebSocket用于实时进度更新，确保防火墙不阻止

## 联系支持
如有问题，请查看MCP API规范文档或联系开发团队。
