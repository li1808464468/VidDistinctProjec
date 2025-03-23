#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频处理工具的配置文件
可以在这里设置默认参数，避免每次在命令行中指定
"""

# 默认放大倍数
DEFAULT_SCALE = 1.1

# 默认对齐方式
# 可选值:
# - 'left_bottom': 左下角对齐
# - 'right_bottom': 右下角对齐
# - 'random_bottom': 随机底部对齐
# - 'center': 居中对齐
# - 'left_top': 左上角对齐
# - 'right_top': 右上角对齐
DEFAULT_ALIGN_MODE = 'random_bottom'

# 默认马赛克方块大小
DEFAULT_MOSAIC_BLOCK_SIZE = 15

# 默认马赛克密度（像素频率）：值越大越清晰，值越小越模糊（范围：1-4000）
DEFAULT_MOSAIC_DENSITY = 3000

# 默认模糊强度
DEFAULT_BLUR_STRENGTH = 15

# 默认视频质量 (CRF值, 0-51, 越小质量越高)
DEFAULT_VIDEO_QUALITY = 23

# 默认马赛克区域 [x1, y1, x2, y2]
# 默认为None，表示使用全屏马赛克区域 [0, 0, width, height]
DEFAULT_MOSAIC_REGION = None

# 默认模糊区域 [x1, y1, x2, y2]
DEFAULT_BLUR_REGION = None

# 默认线程数/进程数
DEFAULT_WORKERS = 4

# 默认是否保留音频
DEFAULT_KEEP_AUDIO = True

# 默认是否递归处理子目录
DEFAULT_RECURSIVE = True

# 默认文件匹配模式
DEFAULT_FILE_PATTERN = "*.mp4"

# 默认视频输入目录（相对路径）
DEFAULT_INPUT_DIR = "resources"

# 默认视频输出目录（相对路径）
DEFAULT_OUTPUT_DIR = "output"

# 默认防检测效果设置
# 各效果默认开关状态
DEFAULT_ANTI_DETECTION_ENABLED = True   # 整体开关
DEFAULT_NOISE_ENABLED = False           # 帧级噪点
DEFAULT_TEXTURE_ENABLED = False         # 纹理覆盖
DEFAULT_DISTORTION_ENABLED = False      # 局部几何变形
DEFAULT_BRIGHTNESS_ENABLED = False      # 亮度波动
DEFAULT_WATERMARK_ENABLED = False       # 隐形水印
DEFAULT_EDGE_ENABLED = False            # 边缘调整
DEFAULT_COLOR_ENABLED = False           # 色彩空间调整
DEFAULT_PERSPECTIVE_ENABLED = False     # 透视变换

# 各效果默认参数
DEFAULT_NOISE_STRENGTH = 0.03           # 噪点强度 (0.01-0.05)
DEFAULT_TEXTURE_OPACITY = 0.05          # 纹理不透明度 (0.03-0.10)
DEFAULT_DISTORTION_STRENGTH = 0.3       # 变形强度 (0.1-0.5)
DEFAULT_WATERMARK_STRENGTH = 0.02       # 水印强度 (0.01-0.05)
DEFAULT_EDGE_STRENGTHEN = True          # 是否增强边缘 (True/False)
DEFAULT_EDGE_AMOUNT = 0.3               # 边缘调整量 (0.2-0.5)
DEFAULT_PERSPECTIVE_STRENGTH = 0.01     # 透视变换强度 (0.005-0.02)

# 随机效果数量（当使用随机效果组合时）
DEFAULT_RANDOM_EFFECTS_COUNT = 3

# 日志配置
DEFAULT_LOG_TO_FILE = False  # 默认不输出日志到文件
DEFAULT_LOG_LEVEL = 'info'   # 默认日志级别

# 用户自定义配置
# 可以根据需要修改以下配置来适应特定项目
USER_CONFIG = {
    # 示例：修改默认放大倍数为1.2
    # "scale": 1.2,
    
    # 示例：修改默认对齐方式为右下角
    # "align_mode": "right_bottom",
    
    # 示例：设置默认马赛克区域(人脸区域示例)
    # "mosaic_region": [100, 50, 300, 250],
    
    # 示例：修改默认马赛克块大小和密度
    # "mosaic_block_size": 20,
    # "mosaic_density": 30,
    
    # 示例：修改默认输入输出目录
    # "input_dir": "my_videos",
    # "output_dir": "processed_videos",
    
    # 示例：设置递归处理选项
    # "recursive": True,
    
    # 示例：启用防检测效果
    # "anti_detection_enabled": True,
    # "noise_enabled": True,
    # "texture_enabled": True,
    # "noise_strength": 0.02,
    
    # 示例：修改日志设置
    # "log_to_file": True,        # 启用日志文件输出
    # "log_level": "debug",       # 设置日志级别为debug
}

# 应用用户自定义配置
if "scale" in USER_CONFIG:
    DEFAULT_SCALE = USER_CONFIG["scale"]
    
if "align_mode" in USER_CONFIG:
    DEFAULT_ALIGN_MODE = USER_CONFIG["align_mode"]
    
if "mosaic_block_size" in USER_CONFIG:
    DEFAULT_MOSAIC_BLOCK_SIZE = USER_CONFIG["mosaic_block_size"]
    
if "mosaic_density" in USER_CONFIG:
    DEFAULT_MOSAIC_DENSITY = USER_CONFIG["mosaic_density"]
    
if "blur_strength" in USER_CONFIG:
    DEFAULT_BLUR_STRENGTH = USER_CONFIG["blur_strength"]
    
if "video_quality" in USER_CONFIG:
    DEFAULT_VIDEO_QUALITY = USER_CONFIG["video_quality"]
    
if "mosaic_region" in USER_CONFIG:
    DEFAULT_MOSAIC_REGION = USER_CONFIG["mosaic_region"]
    
if "blur_region" in USER_CONFIG:
    DEFAULT_BLUR_REGION = USER_CONFIG["blur_region"]
    
if "workers" in USER_CONFIG:
    DEFAULT_WORKERS = USER_CONFIG["workers"]
    
if "keep_audio" in USER_CONFIG:
    DEFAULT_KEEP_AUDIO = USER_CONFIG["keep_audio"]
    
if "recursive" in USER_CONFIG:
    DEFAULT_RECURSIVE = USER_CONFIG["recursive"]
    
if "file_pattern" in USER_CONFIG:
    DEFAULT_FILE_PATTERN = USER_CONFIG["file_pattern"]

if "input_dir" in USER_CONFIG:
    DEFAULT_INPUT_DIR = USER_CONFIG["input_dir"]
    
if "output_dir" in USER_CONFIG:
    DEFAULT_OUTPUT_DIR = USER_CONFIG["output_dir"]

# 应用用户自定义防检测效果设置
if "anti_detection_enabled" in USER_CONFIG:
    DEFAULT_ANTI_DETECTION_ENABLED = USER_CONFIG["anti_detection_enabled"]

if "noise_enabled" in USER_CONFIG:
    DEFAULT_NOISE_ENABLED = USER_CONFIG["noise_enabled"]

if "texture_enabled" in USER_CONFIG:
    DEFAULT_TEXTURE_ENABLED = USER_CONFIG["texture_enabled"]

if "distortion_enabled" in USER_CONFIG:
    DEFAULT_DISTORTION_ENABLED = USER_CONFIG["distortion_enabled"]

if "brightness_enabled" in USER_CONFIG:
    DEFAULT_BRIGHTNESS_ENABLED = USER_CONFIG["brightness_enabled"]

if "watermark_enabled" in USER_CONFIG:
    DEFAULT_WATERMARK_ENABLED = USER_CONFIG["watermark_enabled"]

if "edge_enabled" in USER_CONFIG:
    DEFAULT_EDGE_ENABLED = USER_CONFIG["edge_enabled"]

if "color_enabled" in USER_CONFIG:
    DEFAULT_COLOR_ENABLED = USER_CONFIG["color_enabled"]

if "perspective_enabled" in USER_CONFIG:
    DEFAULT_PERSPECTIVE_ENABLED = USER_CONFIG["perspective_enabled"]

if "noise_strength" in USER_CONFIG:
    DEFAULT_NOISE_STRENGTH = USER_CONFIG["noise_strength"]

if "texture_opacity" in USER_CONFIG:
    DEFAULT_TEXTURE_OPACITY = USER_CONFIG["texture_opacity"]

if "distortion_strength" in USER_CONFIG:
    DEFAULT_DISTORTION_STRENGTH = USER_CONFIG["distortion_strength"]

if "watermark_strength" in USER_CONFIG:
    DEFAULT_WATERMARK_STRENGTH = USER_CONFIG["watermark_strength"]

if "edge_strengthen" in USER_CONFIG:
    DEFAULT_EDGE_STRENGTHEN = USER_CONFIG["edge_strengthen"]

if "edge_amount" in USER_CONFIG:
    DEFAULT_EDGE_AMOUNT = USER_CONFIG["edge_amount"]

if "perspective_strength" in USER_CONFIG:
    DEFAULT_PERSPECTIVE_STRENGTH = USER_CONFIG["perspective_strength"]

if "random_effects_count" in USER_CONFIG:
    DEFAULT_RANDOM_EFFECTS_COUNT = USER_CONFIG["random_effects_count"]

# 应用日志配置
if "log_to_file" in USER_CONFIG:
    DEFAULT_LOG_TO_FILE = USER_CONFIG["log_to_file"]
    
if "log_level" in USER_CONFIG:
    DEFAULT_LOG_LEVEL = USER_CONFIG["log_level"] 