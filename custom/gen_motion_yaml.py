#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""自动生成motion YAML配置文件"""

import os
import yaml
import argparse
import numpy as np
from pathlib import Path


def scan_motions(src_dir, base_dir=None):
    """扫描目录中的motion文件并生成配置
    
    Args:
        src_dir: 源数据目录
        base_dir: YAML中motion路径的基准目录，默认为src_dir的basename
    """
    motions = []
    idx = 1
    
    if base_dir is None:
        base_dir = Path(src_dir).name
    
    for root, _, files in os.walk(src_dir):
        for file in sorted(files):
            if file.endswith('.npz') and not file.endswith(('stagei.npz', 'shape.npz')):
                filepath = os.path.join(root, file)
                
                try:
                    data = np.load(filepath)
                    
                    # 获取FPS
                    fps = data.get('mocap_framerate', data.get('mocap_frame_rate', 30))
                    if isinstance(fps, np.ndarray):
                        fps = float(fps.item())
                    else:
                        fps = float(fps)
                    
                    # 计算时长
                    poses = data.get('poses', data.get('body_pose', None))
                    if poses is not None:
                        end_time = float(poses.shape[0] / fps)
                    else:
                        continue
                    
                    # 生成相对路径的motion文件名
                    rel_path = os.path.relpath(filepath, src_dir)
                    motion_file = rel_path.replace('.npz', '.motion') \
                                         .replace('-', '_') \
                                         .replace(' ', '_') \
                                         .replace('(', '_') \
                                         .replace(')', '_')
                    
                    # 添加motion配置
                    motions.append({
                        'file': motion_file,
                        'fps': fps,
                        'idx': idx,
                        'sub_motions': [{'timings': {'start': 0.0, 'end': end_time}}],
                        'weight': 1.0
                    })
                    idx += 1
                    
                except Exception as e:
                    print(f"警告: 无法处理 {filepath}: {e}")
                    continue
    
    return {'motions': motions}


def main():
    parser = argparse.ArgumentParser(description='自动生成motion YAML配置')
    parser.add_argument('src_dir', help='源数据目录')
    parser.add_argument('--output', '-o', help='输出YAML文件路径')
    parser.add_argument('--base-dir', help='YAML中motion路径的基准目录')
    
    args = parser.parse_args()
    
    # 扫描并生成配置
    yaml_data = scan_motions(args.src_dir, args.base_dir)
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        # 默认输出到data/yaml_files目录
        dir_name = Path(args.src_dir).name.lower()
        output_path = f'data/yaml_files/amass_smpl_train_{dir_name}.yaml'
    
    # 创建目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存YAML
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"已生成: {output_path}")
    print(f"包含 {len(yaml_data['motions'])} 个motion")
    
    return output_path


if __name__ == '__main__':
    main()
