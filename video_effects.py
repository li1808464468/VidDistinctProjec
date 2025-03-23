#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import math
import logging
from typing import Tuple, List, Optional, Callable, Dict, Any

# 配置日志记录
logger = logging.getLogger('video_effects')

class VideoEffects:
    """视频特效处理类，提供各种用于规避图像搜索检测的效果"""
    
    def __init__(self, seed: Optional[int] = None, log_level: int = logging.INFO):
        """
        初始化视频效果处理器
        
        参数:
            seed: 随机数种子，用于确保效果的一致性或随机性
            log_level: 日志级别，默认为INFO
        """
        # 配置日志级别
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.frame_count = 0
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        logger.info(f"初始化视频效果处理器: 随机种子={self.seed}")
        
        # 为亮度波动生成随机参数
        self.brightness_phase = random.uniform(0, 2 * math.pi)
        self.brightness_freq = random.uniform(0.01, 0.05)
        self.brightness_amp = random.uniform(0.02, 0.05)
        
        # 为颜色波动生成随机参数
        self.color_phases = [random.uniform(0, 2 * math.pi) for _ in range(3)]
        self.color_freqs = [random.uniform(0.01, 0.03) for _ in range(3)]
        self.color_amps = [random.uniform(0.01, 0.03) for _ in range(3)]
        
        # 为变形波动生成随机参数
        self.distort_phase = random.uniform(0, 2 * math.pi)
        self.distort_freq = random.uniform(0.005, 0.02)
        self.distort_amp = random.uniform(1.0, 3.0)
        
        # 生成隐形水印模板
        self.watermark = None
        self._generate_watermark_template(64, 64)
        
        logger.debug("参数初始化完成")
    
    def _generate_watermark_template(self, width: int, height: int):
        """生成随机的隐形水印模板"""
        self.watermark = np.random.randint(0, 3, (height, width, 3), dtype=np.int8) - 1
        logger.debug(f"生成隐形水印模板: {width}x{height}")
    
    def apply_frame_noise(self, frame: np.ndarray, strength: float = 0.03) -> np.ndarray:
        """
        添加随机噪点到帧
        
        参数:
            frame: 输入视频帧
            strength: 噪声强度 (0.0-1.0)，推荐范围: 0.01-0.05
            
        返回:
            处理后的帧
        """
        logger.debug(f"应用帧级噪点: 强度={strength:.4f}")
        noise = np.random.normal(0, strength * 255, frame.shape).astype(np.int16)
        noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_frame
    
    def apply_texture_overlay(self, frame: np.ndarray, opacity: float = 0.05) -> np.ndarray:
        """
        在帧上叠加随机纹理
        
        参数:
            frame: 输入视频帧
            opacity: 纹理不透明度 (0.0-1.0)，推荐范围: 0.03-0.10
            
        返回:
            处理后的帧
        """
        logger.debug(f"应用纹理覆盖: 不透明度={opacity:.4f}")
        texture = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
        # 应用高斯模糊使纹理更自然
        texture = cv2.GaussianBlur(texture, (9, 9), 3)
        
        # 混合原始帧和纹理
        result = cv2.addWeighted(frame, 1.0, texture, opacity, 0)
        return result
    
    def apply_local_distortion(self, frame: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        对帧应用局部几何变形
        
        参数:
            frame: 输入视频帧
            strength: 变形强度 (0.0-1.0)，推荐范围: 0.1-0.5
            
        返回:
            处理后的帧
        """
        height, width = frame.shape[:2]
        logger.debug(f"应用局部几何变形: 强度={strength:.4f}, 帧大小={width}x{height}")
        
        # 创建变形映射网格
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)
        
        # 根据帧计数生成动态变化的变形
        phase = self.distort_phase + self.frame_count * self.distort_freq
        
        # 计算变形映射
        for y in range(height):
            for x in range(width):
                # 添加基于正弦波的扭曲
                offset_x = strength * math.sin(y / 30.0 + phase) * self.distort_amp
                offset_y = strength * math.cos(x / 30.0 + phase) * self.distort_amp
                
                map_x[y, x] = x + offset_x
                map_y[y, x] = y + offset_y
        
        # 应用变形
        distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        return distorted
    
    def apply_brightness_fluctuation(self, frame: np.ndarray) -> np.ndarray:
        """
        为帧添加轻微的亮度波动
        
        参数:
            frame: 输入视频帧
            
        返回:
            处理后的帧
        """
        # 基于正弦波计算亮度调整
        phase = self.brightness_phase + self.frame_count * self.brightness_freq
        adjustment = 1.0 + self.brightness_amp * math.sin(phase)
        
        logger.debug(f"应用亮度波动: 调整因子={adjustment:.4f}")
        
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 调整V通道（亮度）
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * adjustment, 0, 255)
        
        # 转回BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
    
    def apply_invisible_watermark(self, frame: np.ndarray, strength: float = 0.02) -> np.ndarray:
        """
        添加几乎不可见的水印
        
        参数:
            frame: 输入视频帧
            strength: 水印强度 (0.0-1.0)，推荐范围: 0.01-0.05
            
        返回:
            处理后的帧
        """
        if self.watermark is None:
            logger.warning("无法应用水印：水印模板未初始化")
            return frame
        
        height, width = frame.shape[:2]
        wm_height, wm_width = self.watermark.shape[:2]
        
        logger.debug(f"应用隐形水印: 强度={strength:.4f}, 水印大小={wm_width}x{wm_height}")
        
        # 将水印调整到适合目标帧的大小
        if height != wm_height or width != wm_width:
            # 创建足够大的水印
            tiled_wm = np.tile(
                self.watermark, 
                (max(1, height // wm_height + 1), 
                 max(1, width // wm_width + 1), 
                 1)
            )
            watermark = tiled_wm[:height, :width, :]
        else:
            watermark = self.watermark
        
        # 应用水印
        watermarked = frame.astype(np.float32)
        watermarked += watermark.astype(np.float32) * strength * 255
        watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
        return watermarked
    
    def apply_edge_adjustment(self, frame: np.ndarray, strengthen: bool = True, amount: float = 0.3) -> np.ndarray:
        """
        增强或削弱图像边缘
        
        参数:
            frame: 输入视频帧
            strengthen: True为增强边缘，False为减弱边缘
            amount: 调整量 (0.0-1.0)，推荐范围: 0.2-0.5
            
        返回:
            处理后的帧
        """
        mode = "增强" if strengthen else "减弱"
        logger.debug(f"应用边缘调整: 模式={mode}, 强度={amount:.4f}")
        
        # 使用拉普拉斯边缘检测
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        
        # 归一化边缘强度
        edge_intensity = amount * (laplacian / 128.0 if strengthen else -laplacian / 128.0)
        
        # 应用到原始帧
        result = np.clip(frame.astype(np.float32) + edge_intensity, 0, 255).astype(np.uint8)
        return result
    
    def apply_color_space_adjustment(self, frame: np.ndarray) -> np.ndarray:
        """
        对RGB颜色空间进行微小的非线性调整
        
        参数:
            frame: 输入视频帧
            
        返回:
            处理后的帧
        """
        logger.debug(f"应用色彩空间调整: 帧计数={self.frame_count}")
        
        # 分离RGB通道
        b, g, r = cv2.split(frame.astype(np.float32))
        
        # 计算每个通道的动态调整
        adjustments = []
        for i, (channel, phase, freq, amp) in enumerate(zip([b, g, r], self.color_phases, self.color_freqs, self.color_amps)):
            adjustment = 1.0 + amp * math.sin(phase + self.frame_count * freq)
            adjustments.append(adjustment)
            
            # 非线性调整（弱gamma变换）
            gamma = 1.0 + 0.05 * (adjustment - 1.0)
            channel_adjusted = np.power(channel / 255.0, gamma) * 255.0
            
            if i == 0:
                b = channel_adjusted
            elif i == 1:
                g = channel_adjusted
            else:
                r = channel_adjusted
        
        logger.debug(f"色彩调整因子: R={adjustments[2]:.4f}, G={adjustments[1]:.4f}, B={adjustments[0]:.4f}")
        
        # 合并通道
        result = cv2.merge([np.clip(b, 0, 255).astype(np.uint8),
                           np.clip(g, 0, 255).astype(np.uint8),
                           np.clip(r, 0, 255).astype(np.uint8)])
        return result
    
    def apply_perspective_distortion(self, frame: np.ndarray, strength: float = 0.01) -> np.ndarray:
        """
        应用微小的透视变换
        
        参数:
            frame: 输入视频帧
            strength: 变形强度 (0.0-1.0)，推荐范围: 0.005-0.02
            
        返回:
            处理后的帧
        """
        height, width = frame.shape[:2]
        
        # 计算扰动参数（基于帧计数和随机种子变化）
        perturbation = strength * 30 * math.sin(self.frame_count * 0.01 + self.distort_phase)
        
        logger.debug(f"应用透视变换: 强度={strength:.4f}, 扰动量={perturbation:.4f}像素")
        
        # 定义原始的四个顶点
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        
        # 定义变换后的四个顶点（加入微小扰动）
        pts2 = np.float32([[0 + perturbation, 0 + perturbation], 
                           [width - perturbation, 0], 
                           [0, height - perturbation], 
                           [width - perturbation, height - perturbation]])
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        # 应用透视变换
        result = cv2.warpPerspective(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        return result
    
    def process_frame(self, frame: np.ndarray, effects_config: Dict[str, Any]) -> np.ndarray:
        """
        应用所有启用的效果处理一帧
        
        参数:
            frame: 输入视频帧
            effects_config: 效果配置字典，键为效果名称，值为效果参数（如果为None或False则不应用）
            
        返回:
            处理后的帧
        """
        processed = frame.copy()
        
        # 记录帧处理信息（仅在DEBUG级别记录）
        logger.debug(f"处理帧 #{self.frame_count}，尺寸: {frame.shape[1]}x{frame.shape[0]}")
        
        applied_effects = []
        
        # 帧级噪点
        if effects_config.get('noise', False):
            strength = effects_config.get('noise_strength', 0.03)
            processed = self.apply_frame_noise(processed, strength)
            applied_effects.append(f"噪点(强度={strength:.3f})")
        
        # 纹理覆盖
        if effects_config.get('texture', False):
            opacity = effects_config.get('texture_opacity', 0.05)
            processed = self.apply_texture_overlay(processed, opacity)
            applied_effects.append(f"纹理(不透明度={opacity:.3f})")
        
        # 局部几何变形
        if effects_config.get('distortion', False):
            strength = effects_config.get('distortion_strength', 0.3)
            processed = self.apply_local_distortion(processed, strength)
            applied_effects.append(f"变形(强度={strength:.3f})")
        
        # 亮度波动
        if effects_config.get('brightness', False):
            processed = self.apply_brightness_fluctuation(processed)
            applied_effects.append("亮度波动")
        
        # 隐形水印
        if effects_config.get('watermark', False):
            strength = effects_config.get('watermark_strength', 0.02)
            processed = self.apply_invisible_watermark(processed, strength)
            applied_effects.append(f"水印(强度={strength:.3f})")
        
        # 边缘调整
        if effects_config.get('edge', False):
            strengthen = effects_config.get('edge_strengthen', True)
            amount = effects_config.get('edge_amount', 0.3)
            processed = self.apply_edge_adjustment(processed, strengthen, amount)
            mode = "增强" if strengthen else "减弱"
            applied_effects.append(f"边缘{mode}(量={amount:.3f})")
        
        # 色彩调整
        if effects_config.get('color', False):
            processed = self.apply_color_space_adjustment(processed)
            applied_effects.append("色彩调整")
        
        # 透视变换
        if effects_config.get('perspective', False):
            strength = effects_config.get('perspective_strength', 0.01)
            processed = self.apply_perspective_distortion(processed, strength)
            applied_effects.append(f"透视(强度={strength:.3f})")
        
        if applied_effects:
            # 只在DEBUG级别记录每帧应用的效果
            logger.debug(f"帧 #{self.frame_count} 已应用效果: {', '.join(applied_effects)}")
        else:
            logger.warning("未应用任何效果")
        
        # 增加帧计数
        self.frame_count += 1
        
        return processed
    
    def get_random_config(self, enable_count: int = 3) -> Dict[str, Any]:
        """
        获取随机效果配置
        
        参数:
            enable_count: 启用的效果数量
            
        返回:
            效果配置字典
        """
        # 可用效果列表
        effects = ['noise', 'texture', 'distortion', 'brightness', 
                  'watermark', 'edge', 'color', 'perspective']
        
        # 随机选择指定数量的效果
        selected = random.sample(effects, min(enable_count, len(effects)))
        
        logger.info(f"随机选择了 {len(selected)} 个效果: {', '.join(selected)}")
        
        # 创建配置字典
        config = {effect: effect in selected for effect in effects}
        
        # 为启用的效果生成随机参数
        if config['noise']:
            config['noise_strength'] = random.uniform(0.01, 0.05)
            logger.debug(f"随机噪点强度: {config['noise_strength']:.4f}")
        
        if config['texture']:
            config['texture_opacity'] = random.uniform(0.03, 0.10)
            logger.debug(f"随机纹理不透明度: {config['texture_opacity']:.4f}")
        
        if config['distortion']:
            config['distortion_strength'] = random.uniform(0.1, 0.5)
            logger.debug(f"随机变形强度: {config['distortion_strength']:.4f}")
        
        if config['watermark']:
            config['watermark_strength'] = random.uniform(0.01, 0.05)
            logger.debug(f"随机水印强度: {config['watermark_strength']:.4f}")
        
        if config['edge']:
            config['edge_strengthen'] = random.choice([True, False])
            config['edge_amount'] = random.uniform(0.2, 0.5)
            mode = "增强" if config['edge_strengthen'] else "减弱"
            logger.debug(f"随机边缘调整: 模式={mode}, 强度={config['edge_amount']:.4f}")
        
        if config['perspective']:
            config['perspective_strength'] = random.uniform(0.005, 0.02)
            logger.debug(f"随机透视变换强度: {config['perspective_strength']:.4f}")
        
        return config 