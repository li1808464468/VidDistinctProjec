#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简易视频处理脚本，使用预设配置快速处理视频
"""

import os
import argparse
import logging
import time
import fnmatch
import glob
from concurrent.futures import ProcessPoolExecutor
from video_processor import process_video, get_anti_detection_config, setup_logging
from config import (
    DEFAULT_SCALE,
    DEFAULT_ALIGN_MODE,
    DEFAULT_MOSAIC_BLOCK_SIZE,
    DEFAULT_MOSAIC_DENSITY,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_LOG_TO_FILE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_KEEP_AUDIO,
    DEFAULT_WORKERS,
    DEFAULT_RECURSIVE
)
from video_effects import VideoEffects

# 设置日志记录器
logger = logging.getLogger('easy_process')

def process_single_video(params):
    """处理单个视频文件的包装函数，用于多进程调用"""
    input_file, output_file, scale, align_mode, block_size, density, anti_detection, effects_config, log_file, log_level, log_to_file, keep_audio = params
    
    try:
        # 为每个处理任务创建单独的日志文件
        video_log_file = log_file.replace('.log', f'_{os.path.basename(input_file)}.log') if log_file and log_to_file else None
        
        print(f"开始处理: {input_file} -> {output_file}")
        
        success = process_video(
            input_file, 
            output_file, 
            scale, 
            align_mode, 
            None,  # 使用默认的全屏马赛克区域 
            block_size, 
            density,
            anti_detection,
            effects_config,
            None,  # 不使用固定种子
            video_log_file,
            log_level=log_level,
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

def main():
    parser = argparse.ArgumentParser(description='简易视频处理工具 - 快速应用马赛克和防检测效果')
    parser.add_argument('--input', help=f'输入视频文件路径 (默认使用 {DEFAULT_INPUT_DIR} 目录下所有mp4文件)')
    parser.add_argument('--output-dir', help=f'输出视频目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--preset', choices=['light', 'medium', 'strong', 'extreme'], default='medium',
                       help='预设效果级别: light(轻度), medium(中度), strong(强度), extreme(极致) (默认: medium)')
    
    # 添加基本参数组
    basic_group = parser.add_argument_group('基本参数')
    basic_group.add_argument('--scale', type=float, default=DEFAULT_SCALE, 
                            help=f'视频放大倍数 (默认: {DEFAULT_SCALE})')
    basic_group.add_argument('--align', choices=['left_bottom', 'right_bottom', 'random_bottom', 'center', 'left_top', 'right_top'], 
                            default=DEFAULT_ALIGN_MODE, help=f'对齐方式 (默认: {DEFAULT_ALIGN_MODE})')
    basic_group.add_argument('--density', type=int, default=DEFAULT_MOSAIC_DENSITY,
                            help=f'马赛克密度，值越大越清晰 (默认: {DEFAULT_MOSAIC_DENSITY})')
    basic_group.add_argument('--keep-audio', action='store_true', default=DEFAULT_KEEP_AUDIO,
                            help=f'保留原视频音频 (默认: {DEFAULT_KEEP_AUDIO})')
    basic_group.add_argument('--no-audio', action='store_false', dest='keep_audio',
                            help='不保留原视频音频')
    basic_group.add_argument('--batch', action='store_true', help='批量处理所有视频文件')
    basic_group.add_argument('--workers', type=int, default=DEFAULT_WORKERS, 
                           help=f'并行处理的工作进程数 (默认: {DEFAULT_WORKERS}, 设为0或1表示串行处理)')
    basic_group.add_argument('--recursive', action='store_true', default=DEFAULT_RECURSIVE,
                           help='递归处理子目录中的视频文件')
    basic_group.add_argument('--no-recursive', action='store_false', dest='recursive',
                           help='不递归处理子目录')
    basic_group.add_argument('--force', action='store_true', help='强制覆盖已存在的输出文件')
    
    # 添加防检测参数组
    anti_group = parser.add_argument_group('防检测参数')
    anti_group.add_argument('--no-anti-detection', action='store_true', help='禁用防检测效果')
    anti_group.add_argument('--custom', action='store_true', help='使用自定义效果组合而不是预设')
    anti_group.add_argument('--effects', type=str, 
                           help='自定义效果组合，逗号分隔，可选: noise,texture,distortion,brightness,watermark,edge,color,perspective')
    
    # 添加日志相关参数
    log_group = parser.add_argument_group('日志参数')
    log_group.add_argument('--log-file', type=str, help='日志文件路径 (默认自动生成)')
    log_group.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], default=DEFAULT_LOG_LEVEL,
                          help=f'日志级别 (默认: {DEFAULT_LOG_LEVEL})')
    log_group.add_argument('--log-to-file', action='store_true', default=DEFAULT_LOG_TO_FILE,
                          help='启用日志文件输出 (默认: %(default)s)')
    log_group.add_argument('--no-log-to-file', action='store_false', dest='log_to_file',
                          help='禁用日志文件输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    
    # 确定输出目录
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置日志文件
    log_file = args.log_file
    if not log_file and args.log_to_file:
        log_dir = os.path.join(output_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f"easy_process_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    # 初始化日志
    setup_logging(log_file, console_level=log_level, file_level=logging.DEBUG, log_to_file=args.log_to_file)
    
    logger.info("-" * 60)
    logger.info("启动简易视频处理工具")
    
    # 确定是单文件处理还是批量处理
    if args.input and not args.batch:
        # 单文件处理模式
        input_files = [args.input]
        logger.info("单文件处理模式")
    else:
        # 批量处理模式
        logger.info("批量处理模式")
        input_dir = DEFAULT_INPUT_DIR
        
        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            logger.error(f"错误: 默认输入目录 '{input_dir}' 不存在")
            return 1
        
        # 查找所有视频文件
        supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        input_files = []
        
        if args.recursive:
            # 递归搜索所有子目录
            for root, _, files in os.walk(input_dir):
                for file in files:
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in supported_extensions):
                        input_files.append(os.path.join(root, file))
        else:
            # 只搜索当前目录
            for file in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in supported_extensions):
                    input_files.append(file_path)
        
        if not input_files:
            logger.error(f"错误: 在 '{input_dir}' {'及其子目录' if args.recursive else '目录'} 中未找到视频文件")
            return 1
    
    logger.info(f"找到 {len(input_files)} 个视频文件需要处理")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"预设效果级别: {args.preset if not args.custom else '自定义'}")
    logger.info(f"处理参数: 放大倍数={args.scale}, 对齐方式={args.align}, 马赛克密度={args.density}")
    if args.log_to_file:
        logger.info(f"日志文件: {log_file}")
    else:
        logger.info("日志文件输出已禁用")
    
    # 构建防检测效果配置
    anti_detection = not args.no_anti_detection
    
    # 模拟命令行参数以使用get_anti_detection_config函数
    class Args:
        pass
    
    config_args = Args()
    config_args.anti_detection = anti_detection
    config_args.random_effects = False
    
    # 根据预设或自定义配置设置效果
    if args.custom and args.effects:
        # 使用用户自定义的效果组合
        effects_list = [e.strip().lower() for e in args.effects.split(',')]
        valid_effects = ['noise', 'texture', 'distortion', 'brightness', 'watermark', 'edge', 'color', 'perspective']
        
        logger.info(f"使用自定义效果组合: {', '.join(e for e in effects_list if e in valid_effects)}")
        
        # 设置每个效果的开关
        for effect in valid_effects:
            setattr(config_args, effect, effect in effects_list)
            
        # 设置默认的效果参数
        config_args.noise_strength = 0.03
        config_args.texture_opacity = 0.05
        config_args.distortion_strength = 0.3
        config_args.watermark_strength = 0.02
        config_args.edge_strengthen = True
        config_args.edge_amount = 0.3
        config_args.perspective_strength = 0.01
        
    else:
        # 使用预设配置
        if args.preset == 'light':
            # 轻度效果配置 - 微妙变化，几乎不可见
            logger.info("使用轻度效果预设配置")
            config_args.noise = True
            config_args.color = True
            config_args.brightness = False
            config_args.texture = False
            config_args.distortion = False
            config_args.watermark = True
            config_args.edge = False
            config_args.perspective = False
            
            config_args.noise_strength = 0.01
            config_args.watermark_strength = 0.01
            
        elif args.preset == 'medium':
            # 中度效果配置 - 默认，平衡视觉质量和防检测效果
            logger.info("使用中度效果预设配置")
            config_args.noise = True
            config_args.color = True
            config_args.brightness = True
            config_args.texture = True
            config_args.distortion = False
            config_args.watermark = True
            config_args.edge = False
            config_args.perspective = True
            
            config_args.noise_strength = 0.02
            config_args.texture_opacity = 0.03
            config_args.watermark_strength = 0.02
            config_args.perspective_strength = 0.005
            
        elif args.preset == 'strong':
            # 强度效果配置 - 明显的防检测效果，但不过分影响视觉
            logger.info("使用强度效果预设配置")
            config_args.noise = True
            config_args.color = True
            config_args.brightness = True
            config_args.texture = True
            config_args.distortion = True
            config_args.watermark = True
            config_args.edge = True
            config_args.perspective = True
            
            config_args.noise_strength = 0.03
            config_args.texture_opacity = 0.05
            config_args.distortion_strength = 0.2
            config_args.watermark_strength = 0.03
            config_args.edge_strengthen = True
            config_args.edge_amount = 0.25
            config_args.perspective_strength = 0.008
            
        elif args.preset == 'extreme':
            # 极致效果配置 - 最大程度防检测，但会明显影响视觉质量
            logger.info("使用极致效果预设配置")
            config_args.noise = True
            config_args.color = True
            config_args.brightness = True
            config_args.texture = True
            config_args.distortion = True
            config_args.watermark = True
            config_args.edge = True
            config_args.perspective = True
            
            config_args.noise_strength = 0.05
            config_args.texture_opacity = 0.08
            config_args.distortion_strength = 0.4
            config_args.watermark_strength = 0.04
            config_args.edge_strengthen = True
            config_args.edge_amount = 0.4
            config_args.perspective_strength = 0.015
    
    # 获取最终的效果配置
    effects_config = get_anti_detection_config(config_args)
    
    # 记录防检测配置信息
    if anti_detection:
        enabled_effects = [k for k, v in effects_config.items() if v and not k.endswith('_strength') and not k.endswith('_opacity') and not k.endswith('_amount')]
        if enabled_effects:
            logger.info(f"启用防检测效果: {', '.join(enabled_effects)}")
        else:
            logger.warning("未启用任何防检测效果")
    else:
        logger.info("防检测效果已禁用")
    
    # 设置视频处理时的日志级别，防止每帧日志在INFO级别记录
    video_log_level = logging.INFO
    
    # 准备处理参数
    process_params = []
    skipped_files = []
    
    for input_file in input_files:
        # 保持输出目录结构与输入目录结构一致
        rel_path = os.path.relpath(input_file, DEFAULT_INPUT_DIR if input_file.startswith(DEFAULT_INPUT_DIR) else os.path.dirname(input_file))
        output_file = os.path.join(output_dir, rel_path)
        output_dir_path = os.path.dirname(output_file)
        
        # 确保输出子目录存在
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        
        # 如果输出文件已存在，检查是否覆盖
        if os.path.exists(output_file) and not args.force:
            logger.warning(f"输出文件已存在: {output_file}")
            response = input("是否覆盖? (y/n): ").lower()
            if response != 'y':
                logger.info(f"跳过处理: {input_file}")
                skipped_files.append(input_file)
                continue
        
        process_params.append((
            input_file,
            output_file,
            args.scale,
            args.align,
            DEFAULT_MOSAIC_BLOCK_SIZE,
            args.density,
            anti_detection,
            effects_config,
            log_file,
            video_log_level,
            args.log_to_file,
            args.keep_audio
        ))
    
    if not process_params:
        logger.warning("没有需要处理的文件")
        return 1
    
    if skipped_files:
        logger.info(f"跳过 {len(skipped_files)} 个文件")
    
    start_time = time.time()
    success_count = 0
    failed_files = []
    
    # 根据workers参数决定是否使用多进程
    workers = args.workers
    if len(process_params) == 1:
        workers = 1  # 单文件时强制使用单进程
    
    if workers > 1 and len(process_params) > 1:
        logger.info(f"使用 {workers} 个工作进程并行处理...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_single_video, process_params))
            
            for success, input_file, output_file in results:
                if success:
                    success_count += 1
                    logger.info(f"成功处理: {input_file} -> {output_file}")
                else:
                    logger.error(f"处理失败: {input_file}")
                    failed_files.append((input_file, output_file))
    else:
        logger.info("使用单进程顺序处理...")
        for params in process_params:
            success, input_file, output_file = process_single_video(params)
            if success:
                success_count += 1
                logger.info(f"成功处理: {input_file} -> {output_file}")
            else:
                logger.error(f"处理失败: {input_file}")
                failed_files.append((input_file, output_file))
    
    # 尝试单独重新处理失败的文件
    if failed_files and workers > 1:
        logger.info(f"尝试使用单进程模式重新处理 {len(failed_files)} 个失败的文件...")
        retry_success = 0
        
        for input_file, output_file in failed_files:
            logger.info(f"重新处理: {input_file}")
            # 使用与之前相同的参数，但强制单进程处理
            retry_params = (
                input_file,
                output_file,
                args.scale,
                args.align,
                DEFAULT_MOSAIC_BLOCK_SIZE,
                args.density,
                anti_detection,
                effects_config,
                log_file,
                video_log_level,
                args.log_to_file,
                args.keep_audio
            )
            retry_success_flag, _, _ = process_single_video(retry_params)
            if retry_success_flag:
                retry_success += 1
                success_count += 1
                logger.info(f"重试成功: {input_file} -> {output_file}")
                # 从失败列表中移除
                failed_files.remove((input_file, output_file))
            else:
                logger.error(f"重试仍然失败: {input_file}")
        
        if retry_success > 0:
            logger.info(f"重试处理成功: {retry_success}/{len(failed_files) + retry_success} 个文件")
    
    total_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"批量处理完成!")
    logger.info(f"总共处理: {len(process_params)} 个文件")
    logger.info(f"成功处理: {success_count} 个文件")
    
    if failed_files:
        logger.warning(f"处理失败: {len(failed_files)} 个文件")
        for input_file, output_file in failed_files:
            logger.warning(f"  - {input_file}")
    
    if skipped_files:
        logger.info(f"跳过处理: {len(skipped_files)} 个文件")
    
    avg_time = total_time / len(process_params) if process_params else 0
    logger.info(f"总用时: {total_time:.1f} 秒, 平均每个文件: {avg_time:.1f} 秒")
    if args.log_to_file:
        logger.info(f"日志文件: {log_file}")
    logger.info("-" * 60)
    
    return 0 if success_count == len(process_params) else 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 