import cv2
import numpy as np
import argparse
import os
import time
import random
import logging
from config import (
    DEFAULT_SCALE, 
    DEFAULT_ALIGN_MODE, 
    DEFAULT_MOSAIC_BLOCK_SIZE,
    DEFAULT_MOSAIC_DENSITY,
    DEFAULT_MOSAIC_REGION,
    DEFAULT_ANTI_DETECTION_ENABLED,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_TEXTURE_ENABLED,
    DEFAULT_DISTORTION_ENABLED,
    DEFAULT_BRIGHTNESS_ENABLED,
    DEFAULT_WATERMARK_ENABLED,
    DEFAULT_EDGE_ENABLED,
    DEFAULT_COLOR_ENABLED,
    DEFAULT_PERSPECTIVE_ENABLED,
    DEFAULT_NOISE_STRENGTH,
    DEFAULT_TEXTURE_OPACITY,
    DEFAULT_DISTORTION_STRENGTH,
    DEFAULT_WATERMARK_STRENGTH,
    DEFAULT_EDGE_STRENGTHEN,
    DEFAULT_EDGE_AMOUNT,
    DEFAULT_PERSPECTIVE_STRENGTH,
    DEFAULT_RANDOM_EFFECTS_COUNT,
    DEFAULT_LOG_TO_FILE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_KEEP_AUDIO
)

# 导入新的视频效果模块
from video_effects import VideoEffects

# 配置日志
logger = logging.getLogger('video_processor')
logger.setLevel(logging.INFO)

# 创建文件处理器，将日志写入文件
def setup_logging(log_file=None, console_level=logging.INFO, file_level=logging.DEBUG, log_to_file=DEFAULT_LOG_TO_FILE):
    """设置日志配置"""
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 如果启用日志输出到文件且指定了日志文件，创建文件处理器
    if log_to_file and log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志将保存到: {log_file}")
    elif log_file and not log_to_file:
        logger.info("日志文件输出已禁用，仅输出到控制台")

def apply_mosaic(frame, region, block_size=15, density=20):
    """
    对视频帧的指定区域应用马赛克效果
    
    参数:
        frame: 输入视频帧
        region: 马赛克区域 [x1, y1, x2, y2]
        block_size: 马赛克方块大小
        density: 马赛克密度（像素频率），值越大越清晰，值越小越模糊（范围：1-4000）
    
    返回:
        处理后的帧
    """
    x1, y1, x2, y2 = region
    roi = frame[y1:y2, x1:x2].copy()
    h, w = roi.shape[:2]
    
    logger.debug(f"应用马赛克: 区域={region}, 方块大小={block_size}, 密度={density}")
    
    # 根据密度参数调整下采样和上采样的程度
    # 密度值范围: 1-4000，密度越小，采样因子越小，马赛克效果越模糊
    # 扩展映射范围以支持更高的密度值
    
    if density <= 100:
        # 1-100 映射到 0.01-1.0
        sample_factor = max(0.01, min(1.0, density / 100.0))
    elif density <= 1000:
        # 101-1000 映射到 1.0-5.0
        sample_factor = 1.0 + min(4.0, (density - 100) / 225.0)
    else:
        # 1001-4000 映射到 5.0-20.0
        sample_factor = 5.0 + min(15.0, (density - 1000) / 200.0)
    
    logger.debug(f"马赛克采样因子: {sample_factor:.4f}")
    
    # 计算降采样尺寸，处理超高密度值
    # 当密度值很大时，增加采样尺寸以提高清晰度
    down_h = max(1, int(h * sample_factor))
    down_w = max(1, int(w * sample_factor))
    
    # 根据不同密度范围使用不同的限制方式
    if density <= 100:
        # 低密度：严格限制，确保明显的马赛克效果
        down_h = min(down_h, max(1, int(h / block_size)))
        down_w = min(down_w, max(1, int(w / block_size)))
    elif density <= 1000:
        # 中等密度：适中限制
        min_h = max(1, int(h / (block_size * 1.5)))
        min_w = max(1, int(w / (block_size * 1.5)))
        down_h = max(down_h, min_h)
        down_w = max(down_w, min_w)
    else:
        # 高密度：几乎无限制，接近原始图像
        min_h = max(1, int(h / (block_size / 2)))
        min_w = max(1, int(w / (block_size / 2)))
        down_h = max(down_h, min_h)
        down_w = max(down_w, min_w)
        
        # 对于超高密度，限制最大尺寸，避免内存问题
        max_dim = 4000  # 最大尺寸限制
        if down_h > max_dim or down_w > max_dim:
            scale = max_dim / max(down_h, down_w)
            down_h = int(down_h * scale)
            down_w = int(down_w * scale)
    
    logger.debug(f"马赛克降采样尺寸: {down_w}x{down_h}")
    
    # 进行下采样和上采样，形成马赛克效果
    # 对于高密度值，使用不同的插值方法以获得更平滑的效果
    if density > 1000:
        small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_CUBIC)
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 将处理后的区域放回原帧
    processed = frame.copy()
    processed[y1:y2, x1:x2] = mosaic
    return processed

def get_crop_coordinates(align_mode, width, height, new_width, new_height):
    """
    根据对齐方式计算裁剪坐标
    
    参数:
        align_mode: 对齐方式，可选值：
            'left_bottom': 左下角对齐
            'right_bottom': 右下角对齐
            'random_bottom': 随机底部对齐
            'center': 居中对齐
            'left_top': 左上角对齐
            'right_top': 右上角对齐
        width: 原始宽度
        height: 原始高度
        new_width: 放大后宽度
        new_height: 放大后高度
    
    返回:
        裁剪区域的左上角坐标 (x, y)
    """
    # 默认左下角对齐
    x = 0
    y = new_height - height
    
    if align_mode == 'right_bottom':
        # 右下角对齐
        x = new_width - width
        y = new_height - height
    elif align_mode == 'random_bottom':
        # 为防止每帧抖动，使用静态变量存储随机选择的x坐标
        if not hasattr(get_crop_coordinates, 'random_x'):
            # 首次调用时随机选择对齐方式
            if random.choice([True, False]):
                # 选择左下角对齐
                get_crop_coordinates.random_x = 0
                logger.debug("随机底部对齐: 选择左下角对齐")
            else:
                # 选择右下角对齐
                get_crop_coordinates.random_x = new_width - width
                logger.debug("随机底部对齐: 选择右下角对齐")
        
        # 使用存储的x坐标
        x = get_crop_coordinates.random_x
        y = new_height - height
    elif align_mode == 'center':
        # 居中对齐
        x = (new_width - width) // 2
        y = (new_height - height) // 2
    elif align_mode == 'left_top':
        # 左上角对齐
        x = 0
        y = 0
    elif align_mode == 'right_top':
        # 右上角对齐
        x = new_width - width
        y = 0
    # 左下角对齐是默认值，不需要额外处理
    
    logger.debug(f"裁剪坐标: 对齐方式={align_mode}, 坐标=({x}, {y})")
    return x, y

def get_anti_detection_config(args):
    """
    根据命令行参数构建防检测效果配置
    
    参数:
        args: 解析后的命令行参数
        
    返回:
        防检测效果配置字典
    """
    config = {}
    
    # 如果整体开关关闭，直接返回空配置
    if not args.anti_detection:
        logger.info("防检测效果已禁用")
        return config
    
    # 检查各个效果的开关
    if args.random_effects:
        # 使用随机效果组合
        effects = ['noise', 'texture', 'distortion', 'brightness', 
                   'watermark', 'edge', 'color', 'perspective']
        selected = random.sample(effects, min(args.random_count, len(effects)))
        
        logger.info(f"使用随机效果组合，选择了 {len(selected)} 个效果: {', '.join(selected)}")
        
        for effect in effects:
            config[effect] = effect in selected
    else:
        # 使用指定效果
        config['noise'] = args.noise
        config['texture'] = args.texture
        config['distortion'] = args.distortion
        config['brightness'] = args.brightness
        config['watermark'] = args.watermark
        config['edge'] = args.edge
        config['color'] = args.color
        config['perspective'] = args.perspective
        
        enabled = [effect for effect, enabled in config.items() if enabled]
        logger.info(f"使用指定效果组合: {', '.join(enabled) if enabled else '无'}")
    
    # 设置效果参数
    if config.get('noise', False):
        config['noise_strength'] = args.noise_strength
        logger.debug(f"噪点效果强度: {args.noise_strength}")
    
    if config.get('texture', False):
        config['texture_opacity'] = args.texture_opacity
        logger.debug(f"纹理不透明度: {args.texture_opacity}")
    
    if config.get('distortion', False):
        config['distortion_strength'] = args.distortion_strength
        logger.debug(f"变形强度: {args.distortion_strength}")
    
    if config.get('watermark', False):
        config['watermark_strength'] = args.watermark_strength
        logger.debug(f"水印强度: {args.watermark_strength}")
    
    if config.get('edge', False):
        config['edge_strengthen'] = args.edge_strengthen
        config['edge_amount'] = args.edge_amount
        mode = "增强" if args.edge_strengthen else "减弱"
        logger.debug(f"边缘调整: 模式={mode}, 强度={args.edge_amount}")
    
    if config.get('perspective', False):
        config['perspective_strength'] = args.perspective_strength
        logger.debug(f"透视变换强度: {args.perspective_strength}")
    
    return config

def process_video(input_path, output_path, scale=DEFAULT_SCALE, align_mode=DEFAULT_ALIGN_MODE, 
                 mosaic_regions=DEFAULT_MOSAIC_REGION, block_size=DEFAULT_MOSAIC_BLOCK_SIZE,
                 density=DEFAULT_MOSAIC_DENSITY, 
                 anti_detection=DEFAULT_ANTI_DETECTION_ENABLED,
                 effects_config=None, seed=None, log_file=None, log_level=logging.INFO, 
                 log_to_file=DEFAULT_LOG_TO_FILE, keep_audio=DEFAULT_KEEP_AUDIO):
    """
    处理视频：放大、按指定方式对齐裁剪、添加马赛克，应用防检测效果
    
    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径
        scale: 放大倍数
        align_mode: 对齐方式，可选值：'left_bottom', 'right_bottom', 'random_bottom'等
        mosaic_regions: 需要马赛克的区域列表，每个元素为 [x1, y1, x2, y2]
        block_size: 马赛克方块大小
        density: 马赛克密度（像素频率），值越大越清晰，值越小越模糊（范围：1-4000）
        anti_detection: 是否启用防检测效果
        effects_config: 防检测效果配置字典
        seed: 随机数种子，用于确保效果的一致性
        log_file: 日志文件路径，如果为None则只输出到控制台
        log_level: 日志级别，默认为INFO
        log_to_file: 是否输出日志到文件，默认为配置中的值
        keep_audio: 是否保留原视频的音频，默认为True
    """
    if log_file is None and log_to_file:
        log_dir = os.path.join(os.path.dirname(output_path), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f"{os.path.basename(output_path)}.log")
    
    setup_logging(log_file, console_level=log_level, file_level=logging.DEBUG, log_to_file=log_to_file)
    
    logger.info("-" * 50)
    logger.info(f"开始处理视频: {input_path} -> {output_path}")
    logger.info(f"处理参数: 放大倍数={scale}, 对齐方式={align_mode}, 马赛克块大小={block_size}, 马赛克密度={density}")
    if keep_audio:
        logger.info("将保留原视频音频")
    else:
        logger.info("不保留原视频音频")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        logger.error(f"错误：输入文件 '{input_path}' 不存在")
        return False
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"错误：无法打开视频文件 '{input_path}'")
        return False
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"视频信息: {width}x{height}, {fps}fps, 共{frame_count}帧")
    
    # 初始化防检测效果处理器
    video_effects = None
    
    # 添加防检测效果处理
    if anti_detection and effects_config:
        enabled_effects = [k for k, v in effects_config.items() if v and not k.endswith('_strength') and not k.endswith('_opacity') and not k.endswith('_amount')]
        if enabled_effects:
            logger.info(f"启用防检测效果: {', '.join(enabled_effects)}")
            # 初始化视频效果处理器
            video_effects = VideoEffects(seed=seed, log_level=log_level)
            if seed is not None:
                logger.info(f"使用固定随机种子: {seed}")
            else:
                logger.info("使用自动生成的随机种子")
        else:
            logger.warning("未启用任何防检测效果")
            anti_detection = False
    
    # 计算放大后的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)
    logger.info(f"放大后尺寸: {new_width}x{new_height}")
    
    # 设置输出视频编码器和格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'avc1' 或其他编码器
    
    # 创建一个临时文件用于保存无音频的处理后视频
    temp_output_path = output_path + '.temp.mp4'
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        logger.error(f"错误：无法创建输出视频文件 '{temp_output_path}'")
        cap.release()
        return False
    
    # 如果没有指定马赛克区域，则创建一个全屏区域
    if mosaic_regions is None:
        mosaic_regions = [[0, 0, width, height]]  # 全屏马赛克
        logger.info("使用全屏马赛克模式")
    else:
        logger.info(f"使用指定马赛克区域: {mosaic_regions}")
    
    start_time = time.time()
    frame_index = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 只在大的间隔记录进度日志（每100帧或开头/结尾）
            if frame_index % 100 == 0 or frame_index == frame_count - 1:
                logger.info(f"处理进度: {frame_index}/{frame_count} ({frame_index/frame_count*100:.1f}%)")
            elif frame_index % 10 == 0:
                # 较小间隔的进度只在DEBUG级别记录
                logger.debug(f"处理帧: {frame_index}/{frame_count}")
            
            # 1. 放大视频
            enlarged = cv2.resize(frame, (new_width, new_height))
            
            # 2. 根据对齐方式裁剪
            x, y = get_crop_coordinates(align_mode, width, height, new_width, new_height)
            cropped = enlarged[y:y+height, x:x+width]
            
            # 3. 应用马赛克效果到指定区域
            processed = cropped.copy()
            for region in mosaic_regions:
                processed = apply_mosaic(processed, region, block_size, density)
            
            # 4. 应用防检测效果
            if anti_detection and video_effects and effects_config:
                processed = video_effects.process_frame(processed, effects_config)
            
            # 写入输出视频
            out.write(processed)
            
            # 显示进度
            frame_index += 1
            if frame_index % 100 == 0 or frame_index == frame_count:
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / frame_index) * (frame_count - frame_index) if frame_index > 0 else 0
                print(f"处理进度: {frame_index}/{frame_count} 帧 ({frame_index/frame_count*100:.1f}%) - 已用时间: {elapsed_time:.1f}秒 - 预计剩余: {eta:.1f}秒")
    
    except Exception as e:
        logger.error(f"处理视频时发生错误: {str(e)}", exc_info=True)
        cap.release()
        out.release()
        return False
    
    # 释放资源
    cap.release()
    out.release()
    
    # 如果需要保留音频，使用ffmpeg合并视频和音频
    if keep_audio:
        try:
            import subprocess
            import shlex
            
            logger.info("正在合并视频和音频...")
            
            # 检查临时输出文件是否有效
            if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) < 1000:
                logger.error(f"临时视频文件不存在或大小异常: {temp_output_path}")
                raise Exception("临时视频文件无效")
                
            # 检查输入文件是否包含音频流
            has_audio = False
            try:
                cmd_check = f'ffprobe -i "{input_path}" -show_streams -select_streams a -loglevel error'
                if os.name == 'nt':
                    check_process = subprocess.Popen(cmd_check, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    check_process = subprocess.Popen(shlex.split(cmd_check), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = check_process.communicate()
                has_audio = check_process.returncode == 0 and stdout.strip()
                if not has_audio:
                    logger.warning("输入视频不包含音频流，将输出无音频视频")
                    os.rename(temp_output_path, output_path)
                    return True
            except Exception as e:
                logger.warning(f"检查音频流时出错: {str(e)}")
            
            # 构建ffmpeg命令
            cmd = f'ffmpeg -y -i "{temp_output_path}" -i "{input_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest "{output_path}"'
            
            logger.debug(f"执行命令: {cmd}")
            
            # 在Windows系统上不需要shlex处理
            if os.name == 'nt':
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            stdout, stderr = process.communicate()
            stderr_text = stderr.decode('utf-8', errors='ignore')
            
            if process.returncode != 0:
                logger.warning(f"合并音频时出现警告: {stderr_text}")
                
                # 尝试使用不同的ffmpeg命令重试
                logger.info("尝试使用替代方法合并音频...")
                retry_cmd = f'ffmpeg -y -i "{temp_output_path}" -i "{input_path}" -c:v copy -c:a aac -map 0:v -map 1:a? "{output_path}"'
                logger.debug(f"重试命令: {retry_cmd}")
                
                if os.name == 'nt':
                    retry_process = subprocess.Popen(retry_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    retry_process = subprocess.Popen(shlex.split(retry_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                retry_stdout, retry_stderr = retry_process.communicate()
                
                if retry_process.returncode != 0:
                    logger.warning(f"重试合并音频也失败: {retry_stderr.decode('utf-8', errors='ignore')}")
                    # 如果ffmpeg失败，使用没有音频的视频
                    logger.warning("音频处理失败，将使用无音频的视频")
                    os.rename(temp_output_path, output_path)
                else:
                    logger.info("使用替代方法成功合并视频和音频")
                    # 删除临时文件
                    os.remove(temp_output_path)
            else:
                logger.info("视频和音频合并成功")
                # 删除临时文件
                os.remove(temp_output_path)
                
        except Exception as e:
            logger.error(f"合并音频时发生错误: {str(e)}")
            # 如果出错，使用没有音频的视频
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 1000:
                os.rename(temp_output_path, output_path)
                logger.info("由于音频处理错误，使用无音频的视频")
            else:
                logger.error("处理失败：临时视频文件无效")
                return False
    else:
        # 如果不需要保留音频，直接重命名临时文件
        os.rename(temp_output_path, output_path)
    
    total_time = time.time() - start_time
    logger.info(f"视频处理完成! 总用时: {total_time:.1f}秒")
    logger.info(f"输出文件: {output_path}")
    logger.info("-" * 50)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='视频处理工具: 放大、裁剪、马赛克和防检测效果')
    parser.add_argument('input', help='输入视频文件路径')
    parser.add_argument('output', help='输出视频文件路径')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE, help=f'视频放大倍数 (默认: {DEFAULT_SCALE})')
    parser.add_argument('--align', choices=['left_bottom', 'right_bottom', 'random_bottom', 'center', 'left_top', 'right_top'], 
                        default=DEFAULT_ALIGN_MODE, help=f'对齐方式 (默认: {DEFAULT_ALIGN_MODE})')
    parser.add_argument('--block-size', type=int, default=DEFAULT_MOSAIC_BLOCK_SIZE, 
                        help=f'马赛克方块大小 (默认: {DEFAULT_MOSAIC_BLOCK_SIZE})')
    parser.add_argument('--density', type=int, default=DEFAULT_MOSAIC_DENSITY,
                        help=f'马赛克密度，值越大越清晰 (范围: 1-4000, 默认: {DEFAULT_MOSAIC_DENSITY})')
    parser.add_argument('--region', type=str, help='马赛克区域，格式: x1,y1,x2,y2 (如: 100,100,300,300)')
    parser.add_argument('--keep-audio', action='store_true', default=DEFAULT_KEEP_AUDIO,
                        help=f'保留原视频音频 (默认: {DEFAULT_KEEP_AUDIO})')
    parser.add_argument('--no-audio', action='store_false', dest='keep_audio',
                        help='不保留原视频音频')
    parser.add_argument('--log-file', type=str, help='日志文件路径 (默认为输出视频文件名.log)')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], default=DEFAULT_LOG_LEVEL,
                        help=f'日志级别 (默认: {DEFAULT_LOG_LEVEL})')
    parser.add_argument('--log-to-file', action='store_true', default=DEFAULT_LOG_TO_FILE,
                        help='启用日志文件输出 (默认: %(default)s)')
    parser.add_argument('--no-log-to-file', action='store_false', dest='log_to_file',
                        help='禁用日志文件输出')
    
    # 添加防检测效果相关参数
    anti_detection_group = parser.add_argument_group('防检测效果选项')
    anti_detection_group.add_argument('--anti-detection', action='store_true', default=DEFAULT_ANTI_DETECTION_ENABLED,
                                     help='启用防检测效果')
    anti_detection_group.add_argument('--random-effects', action='store_true', 
                                     help='使用随机效果组合而不是单独指定')
    anti_detection_group.add_argument('--random-count', type=int, default=DEFAULT_RANDOM_EFFECTS_COUNT,
                                     help=f'随机选择的效果数量 (默认: {DEFAULT_RANDOM_EFFECTS_COUNT})')
    anti_detection_group.add_argument('--seed', type=int, help='随机数种子，用于确保效果的一致性')
    
    # 各效果开关
    anti_detection_group.add_argument('--noise', action='store_true', default=DEFAULT_NOISE_ENABLED,
                                     help='启用帧级噪点效果')
    anti_detection_group.add_argument('--texture', action='store_true', default=DEFAULT_TEXTURE_ENABLED,
                                     help='启用纹理覆盖效果')
    anti_detection_group.add_argument('--distortion', action='store_true', default=DEFAULT_DISTORTION_ENABLED,
                                     help='启用局部几何变形效果')
    anti_detection_group.add_argument('--brightness', action='store_true', default=DEFAULT_BRIGHTNESS_ENABLED,
                                     help='启用亮度波动效果')
    anti_detection_group.add_argument('--watermark', action='store_true', default=DEFAULT_WATERMARK_ENABLED,
                                     help='启用隐形水印效果')
    anti_detection_group.add_argument('--edge', action='store_true', default=DEFAULT_EDGE_ENABLED,
                                     help='启用边缘调整效果')
    anti_detection_group.add_argument('--color', action='store_true', default=DEFAULT_COLOR_ENABLED,
                                     help='启用色彩空间调整效果')
    anti_detection_group.add_argument('--perspective', action='store_true', default=DEFAULT_PERSPECTIVE_ENABLED,
                                     help='启用透视变换效果')
    
    # 各效果参数
    anti_detection_group.add_argument('--noise-strength', type=float, default=DEFAULT_NOISE_STRENGTH,
                                     help=f'噪点强度 (0.01-0.05, 默认: {DEFAULT_NOISE_STRENGTH})')
    anti_detection_group.add_argument('--texture-opacity', type=float, default=DEFAULT_TEXTURE_OPACITY,
                                     help=f'纹理不透明度 (0.03-0.10, 默认: {DEFAULT_TEXTURE_OPACITY})')
    anti_detection_group.add_argument('--distortion-strength', type=float, default=DEFAULT_DISTORTION_STRENGTH,
                                     help=f'变形强度 (0.1-0.5, 默认: {DEFAULT_DISTORTION_STRENGTH})')
    anti_detection_group.add_argument('--watermark-strength', type=float, default=DEFAULT_WATERMARK_STRENGTH,
                                     help=f'水印强度 (0.01-0.05, 默认: {DEFAULT_WATERMARK_STRENGTH})')
    anti_detection_group.add_argument('--edge-strengthen', action='store_true', default=DEFAULT_EDGE_STRENGTHEN,
                                     help=f'是否增强边缘 (默认: {"增强" if DEFAULT_EDGE_STRENGTHEN else "减弱"})')
    anti_detection_group.add_argument('--no-edge-strengthen', action='store_false', dest='edge_strengthen',
                                     help='减弱边缘而不是增强')
    anti_detection_group.add_argument('--edge-amount', type=float, default=DEFAULT_EDGE_AMOUNT,
                                     help=f'边缘调整量 (0.2-0.5, 默认: {DEFAULT_EDGE_AMOUNT})')
    anti_detection_group.add_argument('--perspective-strength', type=float, default=DEFAULT_PERSPECTIVE_STRENGTH,
                                     help=f'透视变换强度 (0.005-0.02, 默认: {DEFAULT_PERSPECTIVE_STRENGTH})')
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    
    # 设置日志
    setup_logging(args.log_file, console_level=log_level, file_level=logging.DEBUG, log_to_file=args.log_to_file)
    
    logger.info(f"启动视频处理工具，命令行参数: {vars(args)}")
    
    # 处理马赛克区域参数
    mosaic_regions = DEFAULT_MOSAIC_REGION
    if args.region:
        try:
            coords = list(map(int, args.region.split(',')))
            if len(coords) == 4:
                mosaic_regions = [coords]
                logger.info(f"使用自定义马赛克区域: {coords}")
            else:
                logger.warning("马赛克区域格式错误，应为'x1,y1,x2,y2'。将使用默认区域。")
        except ValueError:
            logger.warning("马赛克区域参数无效，应为整数。将使用默认区域。")
    
    # 获取防检测效果配置
    effects_config = get_anti_detection_config(args)
    
    # 调用处理函数
    success = process_video(
        args.input, 
        args.output, 
        scale=args.scale, 
        align_mode=args.align, 
        mosaic_regions=mosaic_regions, 
        block_size=args.block_size,
        density=args.density,
        anti_detection=args.anti_detection,
        effects_config=effects_config,
        seed=args.seed,
        log_file=args.log_file,
        log_level=log_level,
        log_to_file=args.log_to_file,
        keep_audio=args.keep_audio
    )
    
    if not success:
        logger.error("视频处理失败")
        return 1
    
    logger.info("视频处理成功完成")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 