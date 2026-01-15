#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""USD工具测试示例

演示如何使用usd_utils读取和操作USD文件。
"""

import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom.usd_utils import USDReader, USDWriter, USD_AVAILABLE, convert_usd_to_obj


def test_read_usd():
    """测试读取USD文件"""
    print("\n=== 测试读取USD文件 ===")
    
    if not USD_AVAILABLE:
        print("❌ USD库未安装，跳过测试")
        print("安装方法: pip install usd-core")
        return False
    
    usd_file = "examples/data/grab_teapot_pour/teapot.usda"
    
    if not os.path.exists(usd_file):
        print(f"❌ 测试文件不存在: {usd_file}")
        return False
    
    try:
        reader = USDReader(usd_file)
        print(f"✓ 成功打开文件: {usd_file}")
        
        # 获取所有prim
        all_prims = reader.get_prim_paths()
        print(f"✓ 找到 {len(all_prims)} 个prims")
        
        # 获取mesh
        meshes = reader.get_prim_paths(prim_type='Mesh')
        print(f"✓ 找到 {len(meshes)} 个meshes")
        
        if meshes:
            mesh_path = meshes[0]
            print(f"\n分析mesh: {mesh_path}")
            
            # 读取顶点
            vertices = reader.get_mesh_vertices(mesh_path)
            if vertices is not None:
                print(f"  - 顶点数: {len(vertices)}")
                print(f"  - 顶点范围: [{vertices.min():.3f}, {vertices.max():.3f}]")
            
            # 读取面
            faces = reader.get_mesh_faces(mesh_path)
            if faces is not None:
                print(f"  - 面索引数: {len(faces)}")
                print(f"  - 三角形数: {len(faces) // 3}")
            
            # 读取包围盒
            bbox = reader.get_bounding_box(mesh_path)
            if bbox:
                min_pt, max_pt = bbox
                size = max_pt - min_pt
                print(f"  - 包围盒最小点: [{min_pt[0]:.3f}, {min_pt[1]:.3f}, {min_pt[2]:.3f}]")
                print(f"  - 包围盒最大点: [{max_pt[0]:.3f}, {max_pt[1]:.3f}, {max_pt[2]:.3f}]")
                print(f"  - 尺寸: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
            
            # 读取变换
            transform = reader.get_transform(mesh_path)
            if transform is not None:
                print(f"  - 变换矩阵形状: {transform.shape}")
        
        reader.close()
        print("\n✓ 读取测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_usd():
    """测试创建USD文件"""
    print("\n=== 测试创建USD文件 ===")
    
    if not USD_AVAILABLE:
        print("❌ USD库未安装，跳过测试")
        return False
    
    output_dir = "custom/output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_cube.usda")
    
    try:
        writer = USDWriter(output_file, create_new=True)
        print(f"✓ 创建新文件: {output_file}")
        
        # 创建立方体顶点
        vertices = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], 
            [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], 
            [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        # 立方体面（12个三角形）
        faces = np.array([
            0, 1, 2, 0, 2, 3,  # 底面
            4, 5, 6, 4, 6, 7,  # 顶面
            0, 1, 5, 0, 5, 4,  # 前面
            2, 3, 7, 2, 7, 6,  # 后面
            0, 3, 7, 0, 7, 4,  # 左面
            1, 2, 6, 1, 6, 5,  # 右面
        ], dtype=np.int32)
        
        # 创建mesh
        writer.create_mesh("/World/Cube", vertices, faces)
        print("✓ 创建立方体mesh")
        
        # 设置变换
        writer.set_transform("/World/Cube", 
                           translation=(0, 0, 1),
                           rotation=(0, 0, 45),
                           scale=(1, 1, 1))
        print("✓ 设置变换")
        
        # 保存
        writer.save()
        writer.close()
        print(f"✓ 保存文件: {output_file}")
        
        # 验证文件
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✓ 文件大小: {file_size} bytes")
            print("\n✓ 创建测试通过")
            return True
        else:
            print("❌ 文件未成功创建")
            return False
            
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convert_to_obj():
    """测试USD转OBJ"""
    print("\n=== 测试USD转OBJ ===")
    
    if not USD_AVAILABLE:
        print("❌ USD库未安装，跳过测试")
        return False
    
    usd_file = "examples/data/grab_teapot_pour/teapot.usda"
    output_dir = "custom/output"
    os.makedirs(output_dir, exist_ok=True)
    obj_file = os.path.join(output_dir, "teapot_converted.obj")
    
    if not os.path.exists(usd_file):
        print(f"❌ 源文件不存在: {usd_file}")
        return False
    
    try:
        success = convert_usd_to_obj(usd_file, obj_file)
        
        if success and os.path.exists(obj_file):
            file_size = os.path.getsize(obj_file)
            print(f"✓ 转换成功: {obj_file}")
            print(f"✓ 文件大小: {file_size} bytes")
            
            # 统计OBJ文件内容
            vertex_count = 0
            face_count = 0
            with open(obj_file, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        vertex_count += 1
                    elif line.startswith('f '):
                        face_count += 1
            
            print(f"✓ 顶点数: {vertex_count}")
            print(f"✓ 面数: {face_count}")
            
            # 使用trimesh验证OBJ文件
            try:
                import trimesh
                mesh = trimesh.load_mesh(obj_file)
                print(f"✓ trimesh加载成功")
                print(f"  - trimesh顶点数: {len(mesh.vertices)}")
                print(f"  - trimesh面数: {len(mesh.faces)}")
                print(f"  - 是否watertight: {mesh.is_watertight}")
            except Exception as e:
                print(f"⚠ trimesh加载失败: {e}")
            
            print("\n✓ 转换测试通过")
            return True
        else:
            print("❌ 转换失败")
            return False
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("USD工具测试")
    print("=" * 60)
    
    results = []
    
    # 测试读取
    results.append(("读取USD", test_read_usd()))
    
    # 测试创建
    results.append(("创建USD", test_create_usd()))
    
    # 测试转换
    results.append(("USD转OBJ", test_convert_to_obj()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")
    
    return all(r for _, r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)