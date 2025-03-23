# 视频处理工具

这个工具可以对视频进行多种处理，包括放大视频、对齐裁剪、添加马赛克以及应用各种防检测效果。

## 特性

- **视频放大**：将视频按指定倍数放大
- **智能裁剪**：支持多种对齐方式（左下、右下、随机底部、居中等）
- **高级马赛克**：提供可调节的马赛克密度和块大小
- **防检测处理**：多种图像特征扰动效果，有效规避图搜图检测
- **批量处理**：支持批量处理多个视频文件
- **易用性**：提供简易处理脚本，使用预设配置快速处理视频
- **详细日志**：记录处理过程中的各种信息，便于分析和调试
- **日志分析**：提供日志查看和分析工具，便于了解处理过程

## 安装

1. 确保已安装Python 3.6或更高版本
2. 安装必要的依赖库:

```bash
pip install opencv-python numpy
```

## 使用方法

### 简易处理（推荐）

使用`easy_process.py`可以快速处理视频，支持多种预设效果级别:

```bash
# 使用默认设置（中等强度防检测）处理视频
python easy_process.py

# 指定输入和输出文件
python easy_process.py --input your_video.mp4 --output result.mp4

# 选择效果级别
python easy_process.py --preset light   # 轻度防检测
python easy_process.py --preset medium  # 中度防检测（默认）
python easy_process.py --preset strong  # 强力防检测
python easy_process.py --preset extreme # 极致防检测（会明显影响视觉质量）

# 自定义效果组合
python easy_process.py --custom --effects noise,texture,watermark,color

# 指定日志级别和日志文件
python easy_process.py --log-level debug --log-file my_process.log
```

### 单个视频处理

使用`video_processor.py`可以获得更多控制选项:

```bash
# 基本用法
python video_processor.py input.mp4 output.mp4

# 自定义参数
python video_processor.py input.mp4 output.mp4 --scale 1.2 --align right_bottom --block-size 20 --density 200

# 启用防检测效果
python video_processor.py input.mp4 output.mp4 --anti-detection --noise --texture --watermark

# 使用随机效果组合
python video_processor.py input.mp4 output.mp4 --anti-detection --random-effects --random-count 4

# 调整效果参数
python video_processor.py input.mp4 output.mp4 --anti-detection --noise --noise-strength 0.04 --watermark --watermark-strength 0.03

# 启用详细日志记录
python video_processor.py input.mp4 output.mp4 --log-level debug --log-file detailed_process.log
```

### 批量处理

使用`batch_processor.py`可以批量处理多个视频:

```bash
# 默认处理resources目录下所有视频
python batch_processor.py

# 指定输入输出目录
python batch_processor.py --input-dir videos --output-dir processed

# 启用防检测效果
python batch_processor.py --anti-detection --noise --texture --color

# 启用详细日志记录
python batch_processor.py --log-level debug
```

### 日志查看和分析

使用`log_viewer.py`可以查看和分析处理日志:

```bash
# 查看最新的日志文件
python log_viewer.py

# 查看指定的日志文件
python log_viewer.py path/to/logfile.log

# 列出所有可用的日志文件
python log_viewer.py --list

# 过滤显示特定级别的日志
python log_viewer.py --level INFO

# 过滤显示特定模块的日志
python log_viewer.py --module video_effects

# 搜索包含特定关键词的日志
python log_viewer.py --keyword "噪点"

# 只显示日志摘要
python log_viewer.py --summary

# 分析防检测效果的使用情况
python log_viewer.py --analyze

# 显示所有日志条目
python log_viewer.py --all

# 组合使用多个过滤条件
python log_viewer.py --level INFO --module video_processor --keyword "处理完成"
```

## 防检测效果说明

本工具提供多种针对图像搜索检测的处理效果:

1. **帧级噪点 (--noise)**  
   在每一帧上添加几乎不可见的随机噪点，强度可调

2. **纹理覆盖 (--texture)**  
   在整个画面上叠加微弱的随机纹理，透明度可调

3. **局部几何变形 (--distortion)**  
   对画面中不同区域应用微小的波纹变形效果

4. **亮度波动 (--brightness)**  
   为每一帧添加微小的亮度变化模式

5. **隐形水印 (--watermark)**  
   添加像素级别的隐形水印，影响图像特征

6. **边缘调整 (--edge)**  
   微调视频中的边缘锐度，可增强或减弱

7. **色彩调整 (--color)**  
   对RGB颜色空间进行微小的非线性调整

8. **透视变形 (--perspective)**  
   对整个帧应用极其轻微的透视变换

## 日志系统

工具内置了完善的日志系统，可以详细记录处理过程中的各种信息:

### 日志级别

- **debug**: 最详细的日志级别，记录所有信息，包括每一个处理步骤和参数
- **info**: 标准日志级别，记录关键处理步骤和重要参数（默认级别）
- **warning**: 只记录警告和错误信息
- **error**: 只记录错误信息

### 日志位置

默认情况下，日志文件将保存在以下位置:

- 单个视频处理: 在输出视频所在目录的`logs`子目录下
- 批量处理: 在输出目录的`logs`子目录下

### 查看日志

有两种方式查看日志:

1. **使用任何文本编辑器打开日志文件**:  
   日志内容包括时间戳、日志级别、模块名称和具体信息。

2. **使用内置的日志查看工具**:  
   ```bash
   python log_viewer.py
   ```
   该工具提供更多功能，如摘要显示、过滤、搜索和防检测效果分析等。

### 日志分析功能

使用`log_viewer.py`的`--analyze`选项可以分析日志中记录的防检测效果使用情况:

- 统计每种效果的使用次数和占比
- 分析处理过的视频数量
- 统计处理的总帧数
- 查看各种日志级别的分布情况

### 日志命令行参数

所有脚本都支持以下与日志相关的参数:

- `--log-file`: 指定日志文件的位置，如果不指定则自动生成
- `--log-level`: 指定日志记录的级别，可选值：debug、info、warning、error

## 参数调整

### 马赛克参数

- `--block-size`: 马赛克方块大小 (默认: 15)
- `--density`: 马赛克密度，值越大越清晰 (范围: 1-4000, 默认: 100)

### 防检测效果参数

- `--noise-strength`: 噪点强度 (0.01-0.05, 默认: 0.03)
- `--texture-opacity`: 纹理不透明度 (0.03-0.10, 默认: 0.05)
- `--distortion-strength`: 变形强度 (0.1-0.5, 默认: 0.3)
- `--watermark-strength`: 水印强度 (0.01-0.05, 默认: 0.02)
- `--edge-amount`: 边缘调整量 (0.2-0.5, 默认: 0.3)
- `--perspective-strength`: 透视变换强度 (0.005-0.02, 默认: 0.01)

## 配置文件

可以通过编辑`config.py`文件来修改默认配置，避免每次在命令行中指定参数。

## 常见问题

### 如何选择合适的防检测效果?

- 如果需要最小的视觉影响，使用 `--preset light` 或只启用 `noise`、`watermark` 和 `color` 效果
- 如果需要最强的防检测效果，使用 `--preset extreme` 或启用所有效果并调高强度参数
- 建议在实际使用前测试不同效果的组合和强度，找到最适合您需求的配置

### 如何保留原视频的音频?

默认情况下，处理后的视频不包含音频。如果需要保留音频，请使用FFmpeg工具进行后处理:

```bash
ffmpeg -i processed.mp4 -i original.mp4 -c:v copy -map 0:v:0 -map 1:a:0 -shortest output_with_audio.mp4
```

### 如何提高处理速度?

- 增加并行处理的工作进程数: `--workers 8`
- 减小视频分辨率或降低帧率
- 使用更快的存储设备
- 禁用不必要的效果

### 如何查看详细的处理日志?

使用debug日志级别可以查看最详细的处理信息:

```bash
python video_processor.py input.mp4 output.mp4 --log-level debug
```

或者使用日志查看工具分析日志:

```bash
python log_viewer.py --level DEBUG
``` 