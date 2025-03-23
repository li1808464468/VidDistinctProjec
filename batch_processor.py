#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import glob
import logging
from video_processor import process_video, get_anti_detection_config, setup_logging
from concurrent.futures import ProcessPoolExecutor
import time
import fnmatch
from config import (
    DEFAULT_SCALE, 
    DEFAULT_ALIGN_MODE, 
    DEFAULT_MOSAIC_BLOCK_SIZE,
    DEFAULT_MOSAIC_DENSITY,
    DEFAULT_MOSAIC_REGION,
    DEFAULT_WORKERS,
    DEFAULT_FILE_PATTERN,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
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
    DEFAULT_KEEP_AUDIO,
    DEFAULT_RECURSIVE
)

# 配置日志
logger = logging.getLogger('batch_processor')

def process_single_video(params):
    """处理单个视频文件的包装函数，用于多进程调用"""
    input_file, output_file, scale, align_mode, mosaic_regions, block_size, density, anti_detection, effects_config, seed, log_file, log_to_file, keep_audio = params
    
    try:
        # 为每个处理任务创建单独的日志文件
        video_log_file = log_file.replace('.log', f'_{os.path.basename(input_file)}.log') if log_file and log_to_file else None
        
        print(f"开始处理: {input_file} -> {output_file}")
        
        # 创建视频特定的日志级别设置
        video_log_level = logging.INFO  # 默认使用INFO级别，这样每帧的日志不会记录
        
        success = process_video(
            input_file, 
            output_file, 
            scale, 
            align_mode, 
            mosaic_regions, 
            block_size, 
            density,
            anti_detection,
            effects_config,
            seed,
            video_log_file,
            log_level=video_log_level,
            log_to_file=log_to_file,
            keep_audio=keep_audio
        )
        return success, input_file, output_file
    except Exception as e:
        # 捕获并记录异常，确保不会影响其他视频的处理
        if log_file and log_to_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"处理 {input_file} 时发生错误: {str(e)}\n")
        print(f"处理 {input_file} 时发生错误: {str(e)}")
        return False, input_file, output_file

def batch_process(input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR, 
                  file_pattern=DEFAULT_FILE_PATTERN, scale=DEFAULT_SCALE, 
                  align_mode=DEFAULT_ALIGN_MODE, mosaic_regions=DEFAULT_MOSAIC_REGION, 
                  block_size=DEFAULT_MOSAIC_BLOCK_SIZE, density=DEFAULT_MOSAIC_DENSITY,
                  workers=DEFAULT_WORKERS, force_overwrite=False,
                  anti_detection=DEFAULT_ANTI_DETECTION_ENABLED,
                  effects_config=None, seed=None, log_file=None, log_level=logging.INFO,
                  log_to_file=DEFAULT_LOG_TO_FILE, keep_audio=DEFAULT_KEEP_AUDIO,
                  recursive=DEFAULT_RECURSIVE):
    """
    批量处理目录中的视频文件
    
    参数:
        input_dir: 输入视频目录
        output_dir: 输出视频目录
        file_pattern: 文件匹配模式（如 *.mp4）
        scale: 放大倍数
        align_mode: 对齐方式，可选值：'left_bottom', 'right_bottom', 'random_bottom'等
        mosaic_regions: 马赛克区域列表
        block_size: 马赛克方块大小
        density: 马赛克密度，值越大越清晰（范围：1-4000）
        workers: 并行处理的工作进程数（0或1表示串行处理）
        force_overwrite: 是否强制覆盖已存在的输出文件
        anti_detection: 是否启用防检测效果
        effects_config: 防检测效果配置字典
        seed: 随机数种子
        log_file: 日志文件路径
        log_level: 日志级别
        log_to_file: 是否输出日志到文件
        keep_audio: 是否保留原视频的音频
        recursive: 是否递归处理子目录
    """
    # 设置日志
    if log_file is None and log_to_file:
        log_dir = os.path.join(output_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f"batch_process_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    setup_logging(log_file, console_level=log_level, file_level=logging.DEBUG, log_to_file=log_to_file)
    
    logger.info("-" * 60)
    logger.info(f"开始批量处理视频: {input_dir} -> {output_dir}")
    logger.info(f"处理参数: 放大倍数={scale}, 对齐方式={align_mode}, 马赛克块大小={block_size}, 马赛克密度={density}")
    logger.info(f"文件匹配模式: {file_pattern}, 并行工作进程: {workers}, 递归处理: {'是' if recursive else '否'}")
    if log_to_file:
        logger.info(f"日志文件: {log_file}")
    else:
        logger.info("日志文件输出已禁用")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    # 查找所有匹配的视频文件（根据recursive参数决定是否递归子目录）
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    input_files = []
    
    # 首先尝试使用用户提供的文件模式
    if file_pattern != DEFAULT_FILE_PATTERN:
        if recursive:
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if fnmatch.fnmatch(file, file_pattern):
                        input_files.append(os.path.join(root, file))
        else:
            # 不递归，只在当前目录中搜索
            for file in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, file)) and fnmatch.fnmatch(file, file_pattern):
                    input_files.append(os.path.join(input_dir, file))
    
    # 如果没有找到匹配文件，则搜索所有支持的视频格式
    if not input_files:
        if recursive:
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in supported_extensions):
                        input_files.append(os.path.join(root, file))
        else:
            # 不递归，只在当前目录中搜索
            for file in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in supported_extensions):
                    input_files.append(file_path)
    
    if not input_files:
        if recursive:
            logger.warning(f"在 '{input_dir}' 及其子目录中没有找到匹配的视频文件")
        else:
            logger.warning(f"在 '{input_dir}' 目录中没有找到匹配的视频文件")
        return
    
    logger.info(f"找到 {len(input_files)} 个视频文件需要处理")
    
    if anti_detection:
        enabled_effects = [k for k, v in effects_config.items() if v and not k.endswith('_strength') and not k.endswith('_opacity') and not k.endswith('_amount')]
        if enabled_effects:
            logger.info(f"启用防检测效果: {', '.join(enabled_effects)}")
            if seed is not None:
                logger.info(f"使用固定随机种子: {seed}")
            else:
                logger.info("为每个视频使用唯一随机种子")
    
    # 准备处理参数
    process_params = []
    skipped_files = []
    
    for input_file in input_files:
        # 保持输出目录结构与输入目录结构一致
        rel_path = os.path.relpath(input_file, input_dir)
        output_file = os.path.join(output_dir, rel_path)
        output_dir_path = os.path.dirname(output_file)
        
        # 确保输出子目录存在
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
            logger.debug(f"创建输出子目录: {output_dir_path}")
        
        # 如果输出文件已存在，检查是否覆盖
        if os.path.exists(output_file) and not force_overwrite:
            logger.warning(f"输出文件已存在: {output_file}")
            response = input("是否覆盖? (y/n): ").lower()
            if response != 'y':
                logger.info(f"跳过处理: {input_file}")
                skipped_files.append(input_file)
                continue
        
        # 为每个视频文件设置唯一的种子，保证随机效果不同
        file_seed = seed
        if seed is None and anti_detection:
            filename = os.path.basename(input_file)
            file_seed = hash(filename) % 10000
            logger.debug(f"为 {filename} 生成随机种子: {file_seed}")
            
        process_params.append((
            input_file, 
            output_file, 
            scale, 
            align_mode, 
            mosaic_regions, 
            block_size, 
            density,
            anti_detection,
            effects_config,
            file_seed,
            log_file,
            log_to_file,
            keep_audio
        ))
    
    if not process_params:
        logger.warning("没有需要处理的文件")
        return
    
    if skipped_files:
        logger.info(f"跳过 {len(skipped_files)} 个文件")
    
    start_time = time.time()
    success_count = 0
    failed_files = []
    
    # 根据workers参数决定是否使用多进程
    if workers > 1:
        logger.info(f"使用 {workers} 个工作进程并行处理...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_single_video, process_params))
            
            for success, input_file, output_file in results:
                if success:
                    success_count += 1
                    logger.info(f"成功处理: {input_file} -> {output_file}")
                else:
                    logger.error(f"处理失败: {input_file}")
                    failed_files.append(input_file)
    else:
        logger.info("使用单进程顺序处理...")
        for params in process_params:
            success, input_file, output_file = process_single_video(params)
            if success:
                success_count += 1
            else:
                failed_files.append(input_file)
    
    total_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"批量处理完成!")
    logger.info(f"总共处理: {len(process_params)} 个文件")
    logger.info(f"成功处理: {success_count} 个文件")
    
    if failed_files:
        logger.warning(f"处理失败: {len(failed_files)} 个文件")
        for file in failed_files:
            logger.warning(f"  - {file}")
    
    if skipped_files:
        logger.info(f"跳过处理: {len(skipped_files)} 个文件")
    
    avg_time = total_time / len(process_params) if process_params else 0
    logger.info(f"总用时: {total_time:.1f} 秒, 平均每个文件: {avg_time:.1f} 秒")
    if log_to_file:
        logger.info(f"日志文件: {log_file}")
    logger.info("-" * 60)

def main():
    parser = argparse.ArgumentParser(description='批量视频处理工具')
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR, 
                       help=f'输入视频目录 (默认: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, 
                       help=f'输出视频目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--pattern', default=DEFAULT_FILE_PATTERN, 
                       help=f'文件匹配模式 (默认: {DEFAULT_FILE_PATTERN})')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE, 
                       help=f'视频放大倍数 (默认: {DEFAULT_SCALE})')
    parser.add_argument('--align', choices=['left_bottom', 'right_bottom', 'random_bottom', 'center', 'left_top', 'right_top'], 
                       default=DEFAULT_ALIGN_MODE, help=f'对齐方式 (默认: {DEFAULT_ALIGN_MODE})')
    parser.add_argument('--block-size', type=int, default=DEFAULT_MOSAIC_BLOCK_SIZE, 
                       help=f'马赛克方块大小 (默认: {DEFAULT_MOSAIC_BLOCK_SIZE})')
    parser.add_argument('--density', type=int, default=DEFAULT_MOSAIC_DENSITY,
                       help=f'马赛克密度，值越大越清晰 (范围: 1-4000, 默认: {DEFAULT_MOSAIC_DENSITY})')
    parser.add_argument('--region', type=str, help='马赛克区域，格式: x1,y1,x2,y2')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, 
                       help=f'并行处理的工作进程数 (默认: {DEFAULT_WORKERS}, 设为0或1表示串行处理)')
    parser.add_argument('--force', action='store_true', help='强制覆盖已存在的输出文件')
    parser.add_argument('--keep-audio', action='store_true', default=DEFAULT_KEEP_AUDIO,
                       help=f'保留原视频音频 (默认: {DEFAULT_KEEP_AUDIO})')
    parser.add_argument('--no-audio', action='store_false', dest='keep_audio',
                       help='不保留原视频音频')
    parser.add_argument('--recursive', action='store_true', default=DEFAULT_RECURSIVE,
                       help='递归处理子目录中的视频文件')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                       help='不递归处理子目录')
    parser.add_argument('--log-file', type=str, help='日志文件路径 (默认自动生成)')
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
    
    # 验证密度值的范围
    density = max(1, min(4000, args.density))
    if density != args.density:
        print(f"警告: 密度值已调整为有效范围内的值: {density}")
    
    # 解析马赛克区域
    mosaic_regions = DEFAULT_MOSAIC_REGION
    if args.region:
        try:
            coords = list(map(int, args.region.split(',')))
            if len(coords) == 4:
                mosaic_regions = [coords]
            else:
                print("错误：马赛克区域格式不正确，应为 x1,y1,x2,y2")
                return
        except ValueError:
            print("错误：马赛克区域应为整数")
            return
    
    # 获取防检测效果配置
    effects_config = get_anti_detection_config(args)
    
    # 批量处理视频
    batch_process(
        args.input_dir, 
        args.output_dir, 
        args.pattern, 
        args.scale, 
        args.align,
        mosaic_regions, 
        args.block_size,
        density,
        args.workers,
        args.force,
        args.anti_detection,
        effects_config,
        args.seed,
        args.log_file,
        log_level,
        args.log_to_file,
        args.keep_audio,
        args.recursive
    )

if __name__ == "__main__":
    main() 