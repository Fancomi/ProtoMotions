#!/usr/bin/env python3
"""
场景工具模块 - 简化静态场景创建

提供便捷函数用于：
1. 创建静态OBJ场景（用于动画对齐验证和训练）
2. 集成到motion_libs_visualizer和训练流程中
3. 自动格式转换（isaacgym→URDF, isaaclab→USDA）
"""

import os
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from protomotions.components.scene_lib import (
    Scene,
    MeshSceneObject,
    ObjectOptions,
    SceneLib,
    SceneLibConfig,
)


def create_static_mesh_scene(
    mesh_path: str,
    translation: tuple = (0.0, 0.0, 0.0),
    rotation: tuple = (0.0, 0.0, 0.0, 1.0),  # xyzw quaternion
    fix_base_link: bool = True,
    density: float = 1000.0,
    collision_mode: str = "convex_hull",  # 新增：碰撞模式
    vhacd_params: Optional[Dict[str, Any]] = None,  # 新增：V-HACD参数
) -> Scene:
    """
    创建包含单个静态网格的场景
    
    Args:
        mesh_path: 网格文件路径（OBJ/URDF/USDA等）
        translation: 位置(x, y, z)
        rotation: 四元数旋转(x, y, z, w)
        fix_base_link: 是否固定（静态物体用True）
        density: 密度
        collision_mode: 碰撞模式，可选：
            - "convex_hull": 单一凸包（默认，最快但最不精确）
            - "convex_decomposition": V-HACD凸分解（精确，适合复杂形状）
            - "none": 无碰撞（仅可视化）
        vhacd_params: V-HACD参数字典（仅当collision_mode="convex_decomposition"时有效）
            例如：{"resolution": 100000, "max_convex_hulls": 10}
    
    Returns:
        Scene对象
    
    Notes:
        Isaac Gym碰撞体说明：
        - 默认：mesh会被转换为单一凸包（convex hull）
        - V-HACD：将mesh分解成多个凸包，更精确地表示复杂形状
        - 静态物体理论上可用三角网格，但Isaac Gym在asset加载时统一处理
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh文件不存在: {mesh_path}")
    
    # 配置碰撞选项
    vhacd_enabled = (collision_mode == "convex_decomposition")
    
    # 默认V-HACD参数
    default_vhacd_params = {
        "resolution": 100000,
        "max_convex_hulls": 10,
        "max_num_vertices_per_ch": 64,
    }
    
    if vhacd_params:
        default_vhacd_params.update(vhacd_params)
    
    # 构建options字典
    options_dict = {
        "fix_base_link": fix_base_link,
        "density": density,
        "vhacd_enabled": vhacd_enabled,
    }
    
    # 添加V-HACD参数
    if vhacd_enabled:
        options_dict["vhacd_params"] = default_vhacd_params
    
    mesh_obj = MeshSceneObject(
        object_path=mesh_path,
        translation=translation,
        rotation=rotation,
        options=ObjectOptions(**options_dict),
    )
    
    return Scene(objects=[mesh_obj], offset=(0.0, 0.0))


def create_static_mesh_scene_lib(
    mesh_path: str,
    num_envs: int,
    device: str = "cuda:0",
    translation: tuple = (0.0, 0.0, 0.0),
    rotation: tuple = (0.0, 0.0, 0.0, 1.0),
    terrain=None,
    collision_mode: str = "convex_hull",
    vhacd_params: Optional[Dict[str, Any]] = None,
) -> SceneLib:
    """
    创建包含静态网格的SceneLib（适用于多环境）
    
    Args:
        mesh_path: 网格文件路径
        num_envs: 环境数量
        device: 设备
        translation: 网格位置
        rotation: 网格旋转（四元数xyzw）
        terrain: 地形对象（可选）
        collision_mode: 碰撞模式（见create_static_mesh_scene）
        vhacd_params: V-HACD参数（见create_static_mesh_scene）
    
    Returns:
        SceneLib对象
    """
    scene = create_static_mesh_scene(
        mesh_path=mesh_path,
        translation=translation,
        rotation=rotation,
        collision_mode=collision_mode,
        vhacd_params=vhacd_params,
    )
    
    config = SceneLibConfig(
        scene_file=None,
        pointcloud_samples_per_object=None,
    )
    
    return SceneLib(
        config=config,
        num_envs=num_envs,
        scenes=[scene],
        device=device,
        terrain=terrain,
    )


def convert_obj_to_urdf(
    obj_file: str,
    output_urdf: Optional[str] = None,
) -> str:
    """
    生成URDF文件引用OBJ（用于isaacgym）
    
    Args:
        obj_file: OBJ文件路径
        output_urdf: 输出URDF路径（None则自动生成）
    
    Returns:
        URDF文件路径
    
    Notes:
        - URDF中的<mesh>标签在Isaac Gym中会被转换为凸包或V-HACD分解
        - 具体行为由ObjectOptions中的vhacd_enabled控制
    """
    obj_path = Path(obj_file)
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ文件不存在: {obj_file}")
    
    # 确定输出路径
    if output_urdf is None:
        urdf_path = obj_path.with_suffix(".urdf")
    else:
        urdf_path = Path(output_urdf)
    
    # 如果URDF已存在，跳过
    if urdf_path.exists():
        print(f"URDF已存在，跳过: {urdf_path}")
        return str(urdf_path)
    
    # 生成URDF内容（引用OBJ）
    obj_filename = obj_path.name
    urdf_content = f"""<?xml version="1.0"?>
<robot name="object">
    <link name="object">
        <visual>
            <origin xyz="0 0 0"/>
            <geometry>
                <mesh filename="{obj_filename}"/>
            </geometry>
            <material name="mat">
                <color rgba="0.5 0.5 0.5 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0"/>
            <geometry>
                <mesh filename="{obj_filename}"/>
            </geometry>
        </collision>
    </link>
</robot>
"""
    
    # 写入URDF文件
    urdf_path.write_text(urdf_content)
    print(f"✓ 生成URDF: {urdf_path}")
    return str(urdf_path)


def convert_obj_to_usda(
    obj_file: str,
    mass: Optional[float] = None,
    collision_approximation: str = "meshSimplification",
    output_usda: Optional[str] = None,
) -> str:
    """
    转换OBJ为USDA（用于isaaclab，动态导入避免依赖）
    
    Args:
        obj_file: OBJ文件路径
        mass: 质量（None则自动计算）
        collision_approximation: 碰撞近似
        output_usda: 输出USDA路径（None则自动生成）
    
    Returns:
        USDA文件路径
    """
    obj_path = Path(obj_file)
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ文件不存在: {obj_file}")
    
    # 确定输出路径
    if output_usda is None:
        usda_path = obj_path.with_suffix(".usda")
    else:
        usda_path = Path(output_usda)
    
    # 如果USDA已存在，跳过
    if usda_path.exists():
        print(f"USDA已存在，跳过: {usda_path}")
        return str(usda_path)
    
    # 动态导入isaaclab（避免启动时依赖）
    try:
        from isaaclab.app import AppLauncher
        from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
        from isaaclab.sim.schemas import schemas_cfg
    except ImportError as e:
        raise ImportError(f"isaaclab未安装或无法导入: {e}")
    
    # 启动IsaacLab
    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app
    
    try:
        # 配置质量和刚体
        if mass is not None:
            mass_props = schemas_cfg.MassPropertiesCfg(mass=mass)
            rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
        else:
            mass_props = None
            rigid_props = None
        
        # 配置碰撞
        collision_props = schemas_cfg.CollisionPropertiesCfg(
            collision_enabled=collision_approximation != "none"
        )
        
        # 执行转换
        mesh_cfg = MeshConverterCfg(
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            asset_path=str(obj_path),
            force_usd_conversion=True,
            usd_dir=str(obj_path.parent),
            usd_file_name=usda_path.name,
            make_instanceable=False,
            collision_approximation=collision_approximation,
        )
        MeshConverter(mesh_cfg)
        
        print(f"✓ 生成USDA: {usda_path}")
        return str(usda_path)
    
    finally:
        simulation_app.close()


# 快速创建函数
def quick_static_scene(obj_path: str, **kwargs) -> Scene:
    """快速创建静态场景的便捷函数"""
    return create_static_mesh_scene(obj_path, **kwargs)
