#!/usr/bin/env python3
"""
保存静态场景到.pt文件，自动处理格式转换

根据目标模拟器自动转换：
- isaacgym: OBJ → URDF（引用OBJ）
- isaaclab: OBJ → USDA（完整USD资产）
- newton/genesis: 直接使用OBJ

示例：
python custom/save_static_scene.py \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --output AMASS/tiaoma_scene.pt \
    --simulator isaacgym \
    --collision convex_decomposition
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom.scene_utils import create_static_mesh_scene, convert_obj_to_urdf, convert_obj_to_usda
from protomotions.components.scene_lib import SceneLib


def main():
    parser = argparse.ArgumentParser(description="保存静态场景（自动转换格式）")
    parser.add_argument("--obj-file", type=str, required=True, help="OBJ文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出.pt文件路径")
    parser.add_argument("--simulator", type=str, default="isaacgym", 
                       choices=["isaacgym", "isaaclab", "newton", "genesis"],
                       help="目标模拟器")
    parser.add_argument("--translation", type=float, nargs=3, default=[0.0, 0.0, 0.0], 
                       help="位置(x y z)")
    parser.add_argument("--rotation", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0],
                       help="旋转四元数(x y z w)")
    parser.add_argument("--fix-base-link", action="store_true", default=True,
                       help="固定基座")
    parser.add_argument("--density", type=float, default=1000.0, help="密度")
    
    # Isaac Gym 碰撞配置
    parser.add_argument("--collision", type=str, default="convex_hull",
                       choices=["convex_hull", "convex_decomposition", "none"],
                       help="碰撞模式(isaacgym):\n"
                            "  convex_hull - 单一凸包(默认,最快但不精确)\n"
                            "  convex_decomposition - V-HACD凸分解(精确,适合复杂形状)\n"
                            "  none - 无碰撞(仅可视化)")
    parser.add_argument("--vhacd-resolution", type=int, default=100000,
                       help="V-HACD体素分辨率(默认100000,越高越精确但越慢)")
    parser.add_argument("--vhacd-max-hulls", type=int, default=10,
                       help="V-HACD最大凸包数量(默认10)")
    parser.add_argument("--vhacd-max-vertices", type=int, default=64,
                       help="V-HACD每个凸包最大顶点数(默认64)")
    
    # Isaac Lab 碰撞配置
    parser.add_argument("--mass", type=float, default=None, help="质量（仅isaaclab）")
    parser.add_argument("--collision-isaaclab", type=str, default="meshSimplification",
                       choices=["none", "meshSimplification", "convexHull", "convexDecomposition"],
                       help="碰撞近似（仅isaaclab）")
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.obj_file):
        print(f"错误: 文件不存在: {args.obj_file}")
        return 1
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== 保存静态场景 ===")
    print(f"输入: {args.obj_file}")
    print(f"模拟器: {args.simulator}")
    print(f"碰撞模式: {args.collision}")
    
    # 根据模拟器转换格式
    mesh_path = args.obj_file
    if args.obj_file.endswith(".obj"):
        if args.simulator == "isaacgym":
            # isaacgym使用URDF引用OBJ
            print("生成URDF（引用OBJ）...")
            mesh_path = convert_obj_to_urdf(args.obj_file)
            
            # 显示碰撞配置信息
            if args.collision == "convex_hull":
                print("  碰撞: 单一凸包（默认）")
            elif args.collision == "convex_decomposition":
                print(f"  碰撞: V-HACD凸分解")
                print(f"    - 分辨率: {args.vhacd_resolution}")
                print(f"    - 最大凸包数: {args.vhacd_max_hulls}")
                print(f"    - 每凸包最大顶点数: {args.vhacd_max_vertices}")
            else:
                print("  碰撞: 无碰撞（仅可视化）")
                
        elif args.simulator == "isaaclab":
            # isaaclab转换为USDA
            print("转换OBJ→USDA...")
            mesh_path = convert_obj_to_usda(
                args.obj_file,
                mass=args.mass,
                collision_approximation=args.collision_isaaclab
            )
        else:
            # newton/genesis直接使用OBJ
            print("使用OBJ（无需转换）")
    
    print(f"网格: {mesh_path}")
    print(f"输出: {args.output}")
    
    # 准备V-HACD参数
    vhacd_params = None
    if args.simulator == "isaacgym" and args.collision == "convex_decomposition":
        vhacd_params = {
            "resolution": args.vhacd_resolution,
            "max_convex_hulls": args.vhacd_max_hulls,
            "max_num_vertices_per_ch": args.vhacd_max_vertices,
        }
    
    # 创建场景
    scene = create_static_mesh_scene(
        mesh_path=mesh_path,
        translation=tuple(args.translation),
        rotation=tuple(args.rotation),
        fix_base_link=args.fix_base_link,
        density=args.density,
        collision_mode=args.collision if args.simulator == "isaacgym" else "convex_hull",
        vhacd_params=vhacd_params,
    )
    
    # 保存场景
    SceneLib.save_scenes_to_file([scene], args.output)
    
    print(f"✓ 场景已保存: {args.output}")
    print(f"  - 网格格式: {Path(mesh_path).suffix}")
    print(f"  - 使用: --scenes-file {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
