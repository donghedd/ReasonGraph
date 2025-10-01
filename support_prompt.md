# ReasonGraph Packycode 接入问题排查请求

## 项目背景
- Flask + SQLAlchemy + Flask-Login 的推理平台，路径：/mnt/d/Users/SY/Desktop/ReasonGraph
- 配置文件 `config.py` 提供模型列表、默认参数
- 推理流程：`templates/index.html` -> `/process` (app.py) -> `ReasoningService` -> `PackycodeAPI`

## 关键代码
1. **Packycode API 客户端**：`api_base.py`
   - 类 `PackycodeAPI(BaseAPI)`，默认 `base_url=https://codex-api.packycode.com/v1`
   - 请求 `/responses` 时 JSON 载荷：
     ```json
     {
       "model": "gpt-5-codex",
       "input": "<formatted_prompt>",
       "stream": true,
       "max_output_tokens": <max_tokens>,
       "response_format": {"type": "text"},
       "reasoning": {"effort": "high"}
     }
     ```
   - 采用 `requests.Session`，`stream=True` 读取 SSE，解析 `data:` 行。
   - `MODEL_ALIASES` 仅映射 `gpt-5-codex` 及 low/medium/high。

2. **推理服务**：`reasoning_service.py`
   - 将前端输入 (question、prompt_format、max_tokens 等) 传给 `PackycodeAPI.generate_response`。

3. **前端与接口**：
   - `/process` (app.py) 校验参数后入库并执行推理；`question` 字段从前端文本框取得。

4. **环境配置**：`.env`
   - 已添加 `PACKYCODE_BASE_URL/ PACKYCODE_WIRE_API=responses/ PACKYCODE_REASONING_EFFORT=high`；Headers/Cookies 保持 `{}`。

## 现象
- 选择 Packycode + `gpt-5-codex-low` 在 `/process` 调用时，多次返回：
  - `{"detail": "Instructions are required"}` 或 `"Instructions are not valid"`
- 历史错误中也出现：403 Cloudflare、`"Stream must be set to true"`, `"Unsupported model"`
- 当前代码已满足 stream/模型/endpoint 要求，但 Packycode 依旧认为 `instructions` 不合法。

## 请求
- 请协助确定 Packycode Codex `/responses` 接口所需的精确 JSON schema：
  - 是否需要数组结构、分段 `instructions`、`input_text` 等字段？
  - 是否必须附带 `tools`/`metadata`/`input_format`？
  - `stream` 模式是否要求 `chunk` 结构？
- 若 `/responses` 与 `wire_api=responses` 仍要求其他参数，请提供示例请求/返回。
- 如果推荐改用 `wire_api=chat`，请说明 message 格式或 system prompt 约束。

## 当前文件
- `api_base.py` (PackycodeAPI class)
- `config.py` (Packycode 模型列表)
- `reasoning_service.py`
- `app.py`
- `.env`

请基于上述信息给出调整方法，以便 Packycode Codex 推理成功。
