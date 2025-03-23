#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志查看和分析工具，用于查看视频处理的日志文件。
"""

import os
import sys
import glob
import re
import argparse
import time
from collections import defaultdict
from datetime import datetime

def parse_log_line(line):
    """解析单行日志内容"""
    # 标准日志格式: 2023-05-15 14:30:45,123 - module_name - LEVEL - message
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)'
    match = re.match(pattern, line)
    
    if match:
        timestamp_str, module, level, message = match.groups()
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            return {
                'timestamp': timestamp,
                'module': module,
                'level': level,
                'message': message
            }
        except ValueError:
            pass
    
    return None

def load_log_file(log_file, filter_level=None, modules=None, keyword=None):
    """
    加载并解析日志文件
    
    参数:
        log_file: 日志文件路径
        filter_level: 过滤的日志级别 (如 'INFO', 'DEBUG' 等)
        modules: 过滤的模块名列表
        keyword: 过滤的关键词
        
    返回:
        解析后的日志条目列表
    """
    entries = []
    
    if not os.path.exists(log_file):
        print(f"错误: 日志文件 '{log_file}' 不存在")
        return entries
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            entry = parse_log_line(line)
            if not entry:
                continue
            
            # 应用过滤条件
            if filter_level and entry['level'] != filter_level:
                continue
                
            if modules and entry['module'] not in modules:
                continue
                
            if keyword and keyword.lower() not in entry['message'].lower():
                continue
            
            entries.append(entry)
    
    return entries

def find_logs(log_dir=None, pattern='*.log'):
    """查找日志文件"""
    if not log_dir:
        # 默认查找在output/logs目录
        log_dir = os.path.join('output', 'logs')
        
        # 如果默认目录不存在，查找当前目录下的logs目录
        if not os.path.exists(log_dir):
            log_dir = 'logs'
            
            # 如果还不存在，使用当前目录
            if not os.path.exists(log_dir):
                log_dir = '.'
    
    log_files = glob.glob(os.path.join(log_dir, pattern))
    return sorted(log_files, key=os.path.getmtime, reverse=True)

def display_log_summary(entries):
    """显示日志摘要"""
    if not entries:
        print("没有找到符合条件的日志条目")
        return
    
    print(f"共找到 {len(entries)} 条日志条目")
    
    # 统计各级别日志数量
    level_counts = defaultdict(int)
    module_counts = defaultdict(int)
    
    for entry in entries:
        level_counts[entry['level']] += 1
        module_counts[entry['module']] += 1
    
    # 显示级别统计
    print("\n日志级别统计:")
    for level, count in level_counts.items():
        print(f"  {level}: {count} 条")
    
    # 显示模块统计
    print("\n模块统计:")
    for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {module}: {count} 条")
    
    # 显示时间范围
    if entries:
        start_time = entries[0]['timestamp']
        end_time = entries[-1]['timestamp']
        duration = end_time - start_time
        print(f"\n时间范围: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"持续时间: {duration}")

def display_log_entries(entries, limit=None, show_all=False):
    """显示日志条目内容"""
    if not entries:
        return
    
    # 如果指定了限制且不显示全部，则截取相应数量的条目
    if limit and not show_all:
        if limit < len(entries):
            print(f"\n显示 {limit}/{len(entries)} 条日志条目:")
            # 显示前后各一半
            half = limit // 2
            first_entries = entries[:half]
            last_entries = entries[-half:] if limit > 1 else []
            
            # 显示前半部分
            for entry in first_entries:
                print(f"[{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {entry['level']} <{entry['module']}>: {entry['message']}")
            
            # 如果有中间部分被省略，显示省略提示
            if len(entries) > limit:
                print(f"\n... 省略了 {len(entries) - limit} 条日志 ...\n")
            
            # 显示后半部分
            for entry in last_entries:
                print(f"[{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {entry['level']} <{entry['module']}>: {entry['message']}")
            
            return
    
    # 显示全部日志条目
    print(f"\n显示全部 {len(entries)} 条日志:")
    for entry in entries:
        print(f"[{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {entry['level']} <{entry['module']}>: {entry['message']}")

def analyze_effects(entries):
    """分析防检测效果的使用情况"""
    effects = defaultdict(int)
    videos_processed = set()
    frame_count = 0
    
    # 正则表达式匹配应用的效果
    effect_pattern = r'已应用效果: (.+)'
    video_pattern = r'开始处理视频: (.+) -> (.+)'
    frame_pattern = r'处理帧 #(\d+)'
    
    for entry in entries:
        message = entry['message']
        
        # 匹配视频处理
        video_match = re.search(video_pattern, message)
        if video_match:
            input_file, output_file = video_match.groups()
            videos_processed.add(input_file)
            continue
        
        # 匹配帧计数
        frame_match = re.search(frame_pattern, message)
        if frame_match:
            frame_index = int(frame_match.group(1))
            frame_count = max(frame_count, frame_index + 1)
            continue
        
        # 匹配应用的效果
        effect_match = re.search(effect_pattern, message)
        if effect_match:
            effect_list = effect_match.group(1)
            # 分割效果列表并计数
            for effect in effect_list.split(', '):
                # 提取效果名称
                effect_name = effect.split('(')[0] if '(' in effect else effect
                effects[effect_name] += 1
    
    # 显示分析结果
    if effects:
        print("\n防检测效果使用分析:")
        for effect, count in sorted(effects.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / max(1, frame_count)) * 100
            print(f"  {effect}: 使用 {count} 次 ({percentage:.1f}% 的帧)")
    
    if videos_processed:
        print(f"\n处理的视频文件数: {len(videos_processed)}")
    
    if frame_count:
        print(f"处理的总帧数: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='视频处理日志查看和分析工具')
    parser.add_argument('log_file', nargs='?', help='要查看的日志文件路径 (如不指定则自动查找最新日志)')
    parser.add_argument('--dir', '-d', help='日志文件所在目录')
    parser.add_argument('--level', '-l', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='过滤显示指定级别的日志')
    parser.add_argument('--module', '-m', action='append', help='过滤显示指定模块的日志 (可多次使用)')
    parser.add_argument('--keyword', '-k', help='过滤包含指定关键词的日志')
    parser.add_argument('--limit', '-n', type=int, default=20, help='限制显示的日志条数 (默认: 20)')
    parser.add_argument('--all', '-a', action='store_true', help='显示所有日志条目')
    parser.add_argument('--summary', '-s', action='store_true', help='只显示日志摘要')
    parser.add_argument('--analyze', '-A', action='store_true', help='分析日志中的防检测效果使用情况')
    parser.add_argument('--list', action='store_true', help='列出发现的日志文件')
    
    args = parser.parse_args()
    
    # 如果要列出日志文件
    if args.list:
        log_files = find_logs(args.dir)
        if log_files:
            print(f"找到 {len(log_files)} 个日志文件:")
            for i, log_file in enumerate(log_files):
                file_time = datetime.fromtimestamp(os.path.getmtime(log_file)).strftime('%Y-%m-%d %H:%M:%S')
                file_size = os.path.getsize(log_file) / 1024  # KB
                print(f"{i+1}. {os.path.basename(log_file)} ({file_time}, {file_size:.1f} KB)")
        else:
            print("未找到任何日志文件")
        return
    
    # 确定要查看的日志文件
    log_file = args.log_file
    if not log_file:
        log_files = find_logs(args.dir)
        if not log_files:
            print("未找到任何日志文件")
            return
        log_file = log_files[0]
        print(f"使用最新的日志文件: {log_file}")
    
    # 加载日志
    print(f"正在加载日志文件: {log_file}")
    entries = load_log_file(log_file, args.level, args.module, args.keyword)
    
    # 显示摘要
    display_log_summary(entries)
    
    # 分析防检测效果
    if args.analyze:
        analyze_effects(entries)
    
    # 如果不是只显示摘要，则显示日志条目
    if not args.summary:
        display_log_entries(entries, args.limit, args.all)
    
    print("\n日志查看完成。")

if __name__ == "__main__":
    main() 