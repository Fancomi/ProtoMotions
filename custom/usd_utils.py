# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""USD/USDA文件读写工具

提供USD格式模型的属性读取、修改和保存功能。
使用pxr.Usd库进行操作，支持几何、材质、变换等属性的访问。
"""

import os
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

try:
    from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    print("Warning: pxr (USD) not available. Install with: pip install usd-core")


class USDReader:
    """USD文件读取器
    
    提供USD文件的基础读取功能，包括几何、变换、材质等属性。
    """
    
    def __init__(self, file_path: str):
        """初始化USD读取器
        
        Args:
            file_path: USD/USDA文件路径
            
        Raises:
            ImportError: 如果pxr库未安装
            FileNotFoundError: 如果文件不存在
        """
        if not USD_AVAILABLE:
            raise ImportError("pxr library required. Install: pip install usd-core")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"USD file not found: {file_path}")
        
        self.file_path = file_path
        self.stage = Usd.Stage.Open(file_path)
        if not self.stage:
            raise RuntimeError(f"Failed to open USD stage: {file_path}")
    
    def get_prim_paths(self, prim_type: Optional[str] = None) -> List[str]:
        """获取所有prim路径
        
        Args:
            prim_type: 过滤特定类型的prim（如'Mesh', 'Xform'等），None返回所有
            
        Returns:
            prim路径列表
        """
        paths = []
        for prim in self.stage.Traverse():
            if prim_type is None or prim.GetTypeName() == prim_type:
                paths.append(str(prim.GetPath()))
        return paths
    
    def get_mesh_vertices(self, prim_path: str) -> Optional[np.ndarray]:
        """获取mesh顶点坐标
        
        Args:
            prim_path: mesh prim的路径
            
        Returns:
            顶点坐标数组 (N, 3)，失败返回None
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            return None
        
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        if not points_attr:
            return None
        
        points = points_attr.Get()
        return np.array(points, dtype=np.float32) if points else None
    
    def get_mesh_faces(self, prim_path: str) -> Optional[np.ndarray]:
        """获取mesh面索引
        
        Args:
            prim_path: mesh prim的路径
            
        Returns:
            面索引数组，失败返回None
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            return None
        
        mesh = UsdGeom.Mesh(prim)
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        return np.array(face_vertex_indices, dtype=np.int32) if face_vertex_indices else None
    
    def get_transform(self, prim_path: str) -> Optional[np.ndarray]:
        """获取prim的变换矩阵
        
        Args:
            prim_path: prim路径
            
        Returns:
            4x4变换矩阵，失败返回None
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim:
            return None
        
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None
        
        local_transform = xformable.GetLocalTransformation()
        return np.array(local_transform, dtype=np.float32).reshape(4, 4)
    
    def get_bounding_box(self, prim_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取prim的包围盒
        
        Args:
            prim_path: prim路径
            
        Returns:
            (min_point, max_point)元组，每个为(3,)数组，失败返回None
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim:
            return None
        
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
        bbox = bbox_cache.ComputeWorldBound(prim)
        
        if not bbox:
            return None
        
        range_obj = bbox.ComputeAlignedRange()
        min_point = np.array(range_obj.GetMin(), dtype=np.float32)
        max_point = np.array(range_obj.GetMax(), dtype=np.float32)
        
        return min_point, max_point
    
    def get_metadata(self, prim_path: str = "/") -> Dict[str, Any]:
        """获取prim的元数据
        
        Args:
            prim_path: prim路径，默认为根节点
            
        Returns:
            元数据字典
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim:
            return {}
        
        metadata = {}
        for key in prim.GetAllMetadata():
            metadata[key] = prim.GetMetadata(key)
        
        return metadata
    
    def close(self):
        """关闭stage"""
        self.stage = None


class USDWriter:
    """USD文件写入器
    
    提供USD文件的创建和修改功能。
    """
    
    def __init__(self, file_path: str, create_new: bool = True):
        """初始化USD写入器
        
        Args:
            file_path: 输出USD/USDA文件路径
            create_new: True创建新文件，False打开已有文件修改
            
        Raises:
            ImportError: 如果pxr库未安装
        """
        if not USD_AVAILABLE:
            raise ImportError("pxr library required. Install: pip install usd-core")
        
        self.file_path = file_path
        
        if create_new:
            self.stage = Usd.Stage.CreateNew(file_path)
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"USD file not found: {file_path}")
            self.stage = Usd.Stage.Open(file_path)
        
        if not self.stage:
            raise RuntimeError(f"Failed to create/open USD stage: {file_path}")
    
    def create_mesh(self, prim_path: str, vertices: np.ndarray,
                   faces: np.ndarray, normals: Optional[np.ndarray] = None) -> bool:
        """创建mesh prim
        
        Args:
            prim_path: mesh路径
            vertices: 顶点坐标 (N, 3)
            faces: 面索引数组
            normals: 法线向量 (N, 3)，可选
            
        Returns:
            成功返回True
        """
        mesh = UsdGeom.Mesh.Define(self.stage, prim_path)
        
        # 设置顶点 - 转换为Python原生float避免类型问题
        points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
        mesh.GetPointsAttr().Set(points)
        
        # 设置面 - 转换为Python原生int
        face_indices = [int(idx) for idx in faces]
        mesh.GetFaceVertexIndicesAttr().Set(face_indices)
        
        # 设置面顶点数（假设都是三角形）
        face_vertex_counts = [3] * (len(faces) // 3)
        mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        
        # 设置法线
        if normals is not None:
            normals_vec = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals]
            mesh.GetNormalsAttr().Set(normals_vec)
        
        return True
    
    def set_transform(self, prim_path: str, translation: Optional[Tuple[float, float, float]] = None,
                     rotation: Optional[Tuple[float, float, float]] = None,
                     scale: Optional[Tuple[float, float, float]] = None) -> bool:
        """设置prim变换
        
        Args:
            prim_path: prim路径
            translation: 平移 (x, y, z)
            rotation: 旋转（欧拉角，度） (rx, ry, rz)
            scale: 缩放 (sx, sy, sz)
            
        Returns:
            成功返回True
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim:
            return False
        
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return False
        
        if translation:
            xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
        
        if rotation:
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        
        if scale:
            xformable.AddScaleOp().Set(Gf.Vec3f(*scale))
        
        return True
    
    def set_metadata(self, prim_path: str, key: str, value: Any) -> bool:
        """设置prim元数据
        
        Args:
            prim_path: prim路径
            key: 元数据键
            value: 元数据值
            
        Returns:
            成功返回True
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim:
            return False
        
        prim.SetMetadata(key, value)
        return True
    
    def save(self):
        """保存USD文件"""
        self.stage.Save()
    
    def close(self):
        """关闭stage"""
        self.stage = None


def convert_usd_to_obj(usd_path: str, obj_path: str) -> bool:
    """将USD文件转换为OBJ格式
    
    简单转换，仅提取第一个mesh的几何信息。
    
    Args:
        usd_path: 输入USD文件路径
        obj_path: 输出OBJ文件路径
        
    Returns:
        成功返回True
    """
    if not USD_AVAILABLE:
        return False
    
    try:
        # 打开USD文件
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"Failed to open USD file: {usd_path}")
            return False
        
        # 查找第一个mesh
        mesh_prim = None
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_prim = prim
                break
        
        if not mesh_prim:
            print(f"No mesh found in {usd_path}")
            return False
        
        mesh = UsdGeom.Mesh(mesh_prim)
        
        # 获取顶点
        points_attr = mesh.GetPointsAttr()
        if not points_attr:
            return False
        points = points_attr.Get()
        
        # 获取面索引和面顶点数
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        
        if not face_vertex_indices or not face_vertex_counts:
            return False
        
        # 写入OBJ
        with open(obj_path, 'w') as f:
            # 写入顶点
            for v in points:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # 写入面 - 正确处理每个面的顶点数
            idx = 0
            for count in face_vertex_counts:
                # OBJ索引从1开始
                face_indices = [face_vertex_indices[idx + i] + 1 for i in range(count)]
                f.write(f"f {' '.join(map(str, face_indices))}\n")
                idx += count
        
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# 示例用法
if __name__ == "__main__":
    # 读取示例
    if USD_AVAILABLE:
        usd_file = "examples/data/armchair.usda"
        
        if os.path.exists(usd_file):
            print(f"Reading {usd_file}...")
            reader = USDReader(usd_file)
            
            # 获取所有prim
            all_prims = reader.get_prim_paths()
            print(f"Found {len(all_prims)} prims")
            
            # 获取mesh
            meshes = reader.get_prim_paths(prim_type='Mesh')
            print(f"Found {len(meshes)} meshes: {meshes}")
            
            # 获取第一个mesh的信息
            if meshes:
                mesh_path = meshes[0]
                vertices = reader.get_mesh_vertices(mesh_path)
                bbox = reader.get_bounding_box(mesh_path)
                
                if vertices is not None:
                    print(f"Mesh {mesh_path}: {len(vertices)} vertices")
                
                if bbox:
                    print(f"Bounding box: min={bbox[0]}, max={bbox[1]}")
            
            reader.close()
        else:
            print(f"File not found: {usd_file}")
    else:
        print("USD library not available")