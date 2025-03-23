#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os
import time
import subprocess
import tempfile
import shutil
import random
from concurrent.futures import ThreadPoolExecutor
from config import (
    DEFAULT_SCALE,
    DEFAULT_ALIGN_MODE,
    DEFAULT_MOSAIC_BLOCK_SIZE,
    DEFAULT_BLUR_STRENGTH,
    DEFAULT_VIDEO_QUALITY,
    DEFAULT_MOSAIC_REGION,
    DEFAULT_BLUR_REGION,
    DEFAULT_WORKERS,
    DEFAULT_KEEP_AUDIO
)

def apply_mosaic(frame, region, block_size=DEFAULT_MOSAIC_BLOCK_SIZE):
    """
    对视频帧的指定区域应用马赛克效果
    
    参数:
        frame: 输入视频帧
        region: 马赛克区域 [x1, y1, x2, y2]
        block_size: 马赛克方块大小
    
    返回:
        处理后的帧
    """
    x1, y1, x2, y2 = region
    roi = frame[y1:y2, x1:x2].copy()
    h, w = roi.shape[:2]
    
    # 按块大小对图像进行分块处理
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            if i_end > i and j_end > j:
                block = roi[i:i_end, j:j_end]
                color = np.mean(block, axis=(0, 1))
                roi[i:i_end, j:j_end] = color
    
    # 将处理后的区域放回原帧
    processed = frame.copy()
    processed[y1:y2, x1:x2] = roi
    return processed

def apply_blur(frame, region, blur_strength=DEFAULT_BLUR_STRENGTH):
    """
    对视频帧的指定区域应用高斯模糊
    
    参数:
        frame: 输入视频帧
        region: 模糊区域 [x1, y1, x2, y2]
        blur_strength: 模糊强度
    
    返回:
        处理后的帧
    """
    x1, y1, x2, y2 = region
    roi = frame[y1:y2, x1:x2].copy()
    
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
    
    # 将模糊区域放回原帧
    processed = frame.copy()
    processed[y1:y2, x1:x2] = blurred
    return processed

def get_crop_coordinates(align_mode, width, height, new_width, new_height):
    """
    根据对齐方式计算裁剪坐标
    
    参数:
        align_mode: 对齐方式，可选值：'left_bottom', 'right_bottom', 'random_bottom'
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
        # 随机左右下角对齐
        max_x = new_width - width
        x = random.randint(0, max_x)
        y = new_height - height
    # 左下角对齐是默认值，不需要额外处理
    
    return x, y

def process_frame(frame, scale=DEFAULT_SCALE, align_mode=DEFAULT_ALIGN_MODE, 
                  mosaic_regions=DEFAULT_MOSAIC_REGION, blur_regions=DEFAULT_BLUR_REGION, 
                  mosaic_block_size=DEFAULT_MOSAIC_BLOCK_SIZE, 
                  blur_strength=DEFAULT_BLUR_STRENGTH):
    """
    处理单个视频帧
    
    参数:
        frame: 输入视频帧
        scale: 放大倍数
        align_mode: 对齐方式
        mosaic_regions: 马赛克区域列表
        blur_regions: 模糊区域列表
        mosaic_block_size: 马赛克块大小
        blur_strength: 模糊强度
    
    返回:
        处理后的帧
    """
    # 获取原始尺寸
    height, width = frame.shape[:2]
    
    # 计算放大后的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 1. 放大视频
    enlarged = cv2.resize(frame, (new_width, new_height))
    
    # 2. 根据对齐方式裁剪
    x, y = get_crop_coordinates(align_mode, width, height, new_width, new_height)
    cropped = enlarged[y:y+height, x:x+width]
    
    # 3. 应用马赛克效果到指定区域
    processed = cropped.copy()
    
    # 应用马赛克
    if mosaic_regions:
        for region in mosaic_regions:
            processed = apply_mosaic(processed, region, mosaic_block_size)
    
    # 应用模糊
    if blur_regions:
        for region in blur_regions:
            processed = apply_blur(processed, region, blur_strength)
    
    return processed

def process_video_with_ffmpeg(input_path, output_path, temp_dir, 
                             scale=DEFAULT_SCALE, align_mode=DEFAULT_ALIGN_MODE, 
                             mosaic_regions=DEFAULT_MOSAIC_REGION, blur_regions=DEFAULT_BLUR_REGION,
                             mosaic_block_size=DEFAULT_MOSAIC_BLOCK_SIZE, 
                             blur_strength=DEFAULT_BLUR_STRENGTH, quality=DEFAULT_VIDEO_QUALITY, 
                             keep_audio=DEFAULT_KEEP_AUDIO, num_threads=DEFAULT_WORKERS):
    """
    使用OpenCV处理视频帧并用FFmpeg合成（保留音频）
    
    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径
        temp_dir: 临时文件目录
        scale: 放大倍数
        align_mode: 对齐方式，可选值：'left_bottom', 'right_bottom', 'random_bottom'
        mosaic_regions: 马赛克区域列表
        blur_regions: 模糊区域列表
        mosaic_block_size: 马赛克方块大小
        blur_strength: 模糊强度
        quality: 输出视频质量 (CRF值, 0-51, 越小质量越高)
        keep_audio: 是否保留原视频音频
        num_threads: 并行处理的线程数
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件 '{input_path}' 不存在")
        return False
    
    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{input_path}'")
        return False
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, 共{frame_count}帧")
    print(f"处理配置: 放大倍数={scale}, 对齐方式={align_mode}")
    
    # 如果没有指定马赛克区域，则创建一个默认区域（右上角）
    if mosaic_regions is None and blur_regions is None:
        mosaic_regions = [[int(width*0.7), 0, width, int(height*0.3)]]
    
    # 创建帧处理函数
    def process_frame_task(frame_index):
        # 设置视频位置并读取帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return False
        
        # 处理帧
        processed = process_frame(
            frame, scale, align_mode, mosaic_regions, blur_regions, 
            mosaic_block_size, blur_strength
        )
        
        # 保存处理后的帧
        output_file = os.path.join(temp_dir, f"frame_{frame_index:08d}.png")
        cv2.imwrite(output_file, processed)
        
        return True
    
    start_time = time.time()
    
    # 使用线程池处理帧
    print(f"使用 {num_threads} 个线程处理视频帧...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_frame_task, range(frame_count)))
    
    # 关闭视频读取
    cap.release()
    
    processing_time = time.time() - start_time
    print(f"视频帧处理完成，用时: {processing_time:.1f}秒")
    print("正在使用FFmpeg合成视频...")
    
    # 使用FFmpeg将帧合成为视频
    frame_pattern = os.path.join(temp_dir, "frame_%08d.png")
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', frame_pattern,
    ]
    
    # 如果保留音频，添加音频输入
    if keep_audio:
        ffmpeg_cmd.extend(['-i', input_path])
    
    # 添加输出选项
    ffmpeg_cmd.extend([
        '-c:v', 'libx264',
        '-crf', str(quality),
        '-preset', 'medium',
        '-pix_fmt', 'yuv420p',
    ])
    
    # 如果保留音频，添加音频映射
    if keep_audio:
        ffmpeg_cmd.extend([
            '-map', '0:v',
            '-map', '1:a',
            '-c:a', 'aac',
            '-b:a', '192k',
        ])
    
    # 最后添加输出文件
    ffmpeg_cmd.append(output_path)
    
    # 执行FFmpeg命令
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"视频合成完成，输出到: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg错误: {e}")
        return False
    
    total_time = time.time() - start_time
    print(f"总处理用时: {total_time:.1f}秒")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='高级视频处理工具: 保留音频, 多线程处理')
    parser.add_argument('input', help='输入视频文件路径')
    parser.add_argument('output', help='输出视频文件路径')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE, 
                        help=f'视频放大倍数 (默认: {DEFAULT_SCALE})')
    parser.add_argument('--align', choices=['left_bottom', 'right_bottom', 'random_bottom'], 
                        default=DEFAULT_ALIGN_MODE, help=f'对齐方式 (默认: {DEFAULT_ALIGN_MODE})')
    parser.add_argument('--mosaic-block', type=int, default=DEFAULT_MOSAIC_BLOCK_SIZE, 
                        help=f'马赛克方块大小 (默认: {DEFAULT_MOSAIC_BLOCK_SIZE})')
    parser.add_argument('--blur-strength', type=int, default=DEFAULT_BLUR_STRENGTH, 
                        help=f'模糊强度 (默认: {DEFAULT_BLUR_STRENGTH})')
    parser.add_argument('--mosaic-region', type=str, help='马赛克区域，格式: x1,y1,x2,y2')
    parser.add_argument('--blur-region', type=str, help='模糊区域，格式: x1,y1,x2,y2')
    parser.add_argument('--quality', type=int, default=DEFAULT_VIDEO_QUALITY, 
                        help=f'输出视频质量 (CRF值, 0-51, 越小质量越高, 默认: {DEFAULT_VIDEO_QUALITY})')
    parser.add_argument('--no-audio', action='store_true', help='不保留原视频音频')
    parser.add_argument('--threads', type=int, default=DEFAULT_WORKERS, 
                        help=f'处理线程数 (默认: {DEFAULT_WORKERS})')
    
    args = parser.parse_args()
    
    # 解析马赛克区域
    mosaic_regions = DEFAULT_MOSAIC_REGION
    if args.mosaic_region:
        try:
            coords = [int(x) for x in args.mosaic_region.split(',')]
            if len(coords) == 4:
                mosaic_regions = [coords]
            else:
                print("错误：马赛克区域格式不正确，应为 x1,y1,x2,y2")
                return
        except ValueError:
            print("错误：马赛克区域应为整数")
            return
    
    # 解析模糊区域
    blur_regions = DEFAULT_BLUR_REGION
    if args.blur_region:
        try:
            coords = [int(x) for x in args.blur_region.split(',')]
            if len(coords) == 4:
                blur_regions = [coords]
            else:
                print("错误：模糊区域格式不正确，应为 x1,y1,x2,y2")
                return
        except ValueError:
            print("错误：模糊区域应为整数")
            return
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="video_process_")
    
    try:
        # 处理视频
        success = process_video_with_ffmpeg(
            args.input, args.output, temp_dir,
            args.scale, args.align, mosaic_regions, blur_regions,
            args.mosaic_block, args.blur_strength,
            args.quality, not args.no_audio, args.threads
        )
        
        if success:
            print("视频处理成功完成！")
        else:
            print("视频处理失败。")
    finally:
        # 清理临时文件
        print("清理临时文件...")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # 检查ffmpeg是否安装
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("错误: ffmpeg未安装。请安装ffmpeg后再运行此脚本。")
        print("可以从 https://ffmpeg.org/download.html 下载，或使用包管理器安装：")
        print("  - Ubuntu/Debian: sudo apt install ffmpeg")
        print("  - macOS: brew install ffmpeg")
        print("  - Windows: 请下载ffmpeg并添加到系统PATH")
        exit(1)
    
    main() 