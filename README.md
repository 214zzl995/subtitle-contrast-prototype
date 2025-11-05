# Y 平面字幕相似度判定算法设计（v1.0）

## 0. 原型后台实现与使用说明

本仓库在 `subtitle_contrast_prototype` 包内提供了一个 FastAPI 原型后台，实现了：

- 列出 `frames/yuv` 目录所有 `.yuv` 帧；
- 仅在请求 `/frames/{name}/image` 时按需读取文件并将 **Y 平面** 转成 `PNG`（预览用，不涉及 `webp`）；
- 根据 README 描述的思路计算字幕相似度，输入两个帧名、字幕 ROI、已知亮度均值 μ_sub 及容忍波动 ΔY。

### 启动方式

```bash
uv pip install -e .
python main.py  # 等价于 uvicorn 启动
# 或者
# uvicorn subtitle_contrast_prototype.api:app
```

可用环境变量：

| 变量              | 默认值          | 说明 |
|------------------|----------------|------|
| `DATA_ROOT`      | `frames/yuv`   | YUV 源目录 |
| `YUV_WIDTH`      | 自动推断        | 帧宽度（像素，可手动覆盖） |
| `YUV_HEIGHT`     | 自动推断        | 帧高度（像素，可手动覆盖） |
| `YUV_FORMAT`     | 自动推断        | `y_only`/`yuv420`/`yuv422` |
| `HOST` / `PORT`  | `127.0.0.1`/`8000` | 启动地址与端口 |
| `RELOAD`         | `false`        | 设为 `true` 启动热加载（开发用途） |
| `YUV_SEARCH_RADIUS` | `3`        | ROI 内平移搜索半径 |
| `YUV_EDGE_LAMBDA`  | `0.6`        | 亮度 vs 边缘概率融合权重 |

> 未显式提供宽高/格式时，程序会根据首个 `.yuv` 文件字节数在 `y_only` / `yuv420` / `yuv422` 中择优推断，若数据异常请手动指定。
> 推断过程中会计算多个候选分辨率的亮度梯度与宽高比，仅在启动时读取一次原始数据，不会在正常服务过程中占用额外内存。

### REST 接口概要

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET /health` | 查看当前配置（格式、宽高等） |
| `GET /frames` | 返回帧列表（文件名、字节数、推断宽高与格式） |
| `GET /frames/{name}/image` | 输出指定帧的 `PNG` 预览（基于 Y 平面） |
| `POST /compare` | 计算字幕相似度 |

`POST /compare` 示例：

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

响应字段：

- `score`：融合相似度（0~1）；
- `confidence`：与 score 同步的置信度（供前端直接展示）；
- `decision`：`same` / `different` / `uncertain`；
- `dx`, `dy`: ROI 内平移偏移；
- `roi`：服务端实际裁剪的 ROI（防止越界）；
- `metrics`：归一化后的核心指标（IoU、Dice、SSIM、投影、POC）；
- `details`：原始对齐峰值、PSR、亮度增益/偏置等调试信息。

---

## 算法设计文档

完整的 `v1.0` 算法说明（目标、总体思路、详细步骤、伪代码、参数与测试计划等）已迁移至 [`docs/v1.0.md`](docs/v1.0.md)。若需查阅或引用具体算法内容，请访问该文档。
