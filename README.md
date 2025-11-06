# Y 平面字幕相似度判定算法原型（v3.0）

FastAPI 原型后台，用于验证 Y 平面字幕相似度判定算法 v3.0。服务在运行时按需加载 `frames/yuv` 下的 `.yuv` 帧，基于给定 ROI 与字幕亮度先验计算两帧是否为同一段字幕。

算法版本文档（每版一句话）：

- v1.0（docs/v1.0.md）：以亮度/边缘与投影相似度为主的基础融合，适合对比度稳定、文本密度中等的场景。
- v1.1.patch（docs/v1.1.patch.md）：在 v1.0 基础上加入背景抑制、Masked‑SSIM、加权 POC、容错 IoU 与自适应权重，增强 ΔY 大与背景差异场景。
- v2.0（docs/v2.0.md）：文本中心化与位移容忍；背景抑制 + 自适应软掩码 + 峰表特征 + SPS/EOH + 去偏归一与自适应融合，应对 ΔY 大与背景变化。
- v3.0（docs/v3.0.md）：行带锚点 + 网格前景密度图，结合稀疏锚点与自适应融合，显著降低负样本底噪并提升密/稀文本鲁棒性。
- v4.0（docs/v4.0.md）：轻量稀疏几何法；候选前景 + 边缘稀疏化 + 双向部分 Chamfer/Hausdorff + 笔画宽轻惩罚，易调参、适合稀疏文本。
- v5.0（docs/v5.0.md）：OpenCV 工程化方案；形态学/边缘/距离变换 + 模板匹配/ORB，便于 C++/Python 直接落地。

## 原型后台实现与使用说明

本仓库在 `subtitle_contrast_prototype` 包内提供了 FastAPI 服务，核心功能包括：

- 列出 `frames/yuv` 目录下的 `.yuv` 帧；
- 仅在访问 `/frames/{name}/image` 时即时读取帧并输出 Y 平面 PNG 预览；
- 计算字幕相似度，输入两帧名称、字幕 ROI、亮度均值 μ_sub、亮度波动 ΔY 与平移搜索半径。

### 启动方式

```bash
uv pip install -e .
python main.py  # 等价于 uvicorn 启动
# 或者
# uvicorn subtitle_contrast_prototype.api:app
```

### 环境变量

| 变量                 | 默认值             | 说明 |
|---------------------|-------------------|------|
| `DATA_ROOT`         | `frames/yuv`      | YUV 源目录 |
| `YUV_WIDTH`         | 自动推断           | 帧宽度（像素，可手动覆盖） |
| `YUV_HEIGHT`        | 自动推断           | 帧高度（像素，可手动覆盖） |
| `YUV_FORMAT`        | 自动推断           | `y_only` / `yuv420` / `yuv422` |
| `HOST` / `PORT`     | `127.0.0.1` / `8000` | 启动地址与端口 |
| `RELOAD`            | `false`           | 设置为 `true` 可开启热加载（开发用途） |
| `YUV_SEARCH_RADIUS` | `3`               | ROI 内平移搜索半径（像素） |
| `YUV_EDGE_LAMBDA`   | `0.6`             | 亮度与边缘融合权重 |

> 未显式提供宽高/格式时，程序会根据首个 `.yuv` 文件字节数在 `y_only` / `yuv420` / `yuv422` 中推断。推断过程中仅在启动时读取一次原始数据。

### REST 接口概要

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET /health` | 查看当前配置（分辨率、格式、算法版本等） |
| `GET /frames` | 返回帧列表（文件名、字节数、推断宽高与格式） |
| `GET /frames/{name}/image` | 输出指定帧的 Y 平面 `PNG` 预览 |
| `POST /compare` | 计算字幕相似度 |

`POST /compare` 请求示例：

```json
{
  "frame_a": "12.yuv",
  "frame_b": "13.yuv",
  "roi": {"x": 320, "y": 880, "width": 640, "height": 180},
  "mu_sub": 235,
  "delta_y": 15,
  "search_radius": 3
}
```

响应示例字段：

- `score`：融合相似度（0~1）；
- `confidence`：与 score 同步的置信度；
- `decision`：`same` / `different` / `uncertain`；
- `dx`, `dy`：ROI 内平移偏移；
- `roi`：服务端实际裁剪后的 ROI（避免越界）；
- `metrics`：核心分支指标（如网格 IoU、容错 IoU、密度峰表、稀疏锚点匹配等归一化结果）；
- `details`：调试信息（亮度对齐参数、ρ/H、自适应权重、峰值表、哈希比特等）。
