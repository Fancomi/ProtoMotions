# Custom USD Utils

USD/USDA文件读写工具模块，用于ProtoMotions项目中的模型属性操作和接触仿真开发。

## 功能特性

- **USDReader**: 读取USD文件的几何、变换、包围盒等属性
- **USDWriter**: 创建和修改USD文件
- **格式转换**: USD转OBJ等基础转换功能

## 安装依赖

```bash
pip install usd-core
```

## 使用示例

### 读取USD文件

```python
from custom.usd_utils import USDReader

# 打开USD文件
reader = USDReader("examples/data/armchair.usda")

# 获取所有mesh路径
meshes = reader.get_prim_paths(prim_type='Mesh')
print(f"Found meshes: {meshes}")

# 读取第一个mesh的顶点和面
if meshes:
    vertices = reader.get_mesh_vertices(meshes[0])
    faces = reader.get_mesh_faces(meshes[0])
    print(f"Vertices shape: {vertices.shape}")
    print(f"Faces shape: {faces.shape}")

# 获取包围盒（用于碰撞检测）
bbox = reader.get_bounding_box(meshes[0])
if bbox:
    min_point, max_point = bbox
    print(f"Bounding box: min={min_point}, max={max_point}")

# 获取变换矩阵
transform = reader.get_transform(meshes[0])
print(f"Transform matrix:\n{transform}")

reader.close()
```

### 创建USD文件

```python
from custom.usd_utils import USDWriter
import numpy as np

# 创建新USD文件
writer = USDWriter("output/new_model.usda", create_new=True)

# 创建简单立方体mesh
vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
], dtype=np.float32)

faces = np.array([
    0, 1, 2, 0, 2, 3,  # 前面
    4, 5, 6, 4, 6, 7,  # 后面
    # ... 其他面
], dtype=np.int32)

writer.create_mesh("/World/Cube", vertices, faces)

# 设置变换
writer.set_transform("/World/Cube", 
                    translation=(0, 0, 1),
                    rotation=(0, 0, 45),
                    scale=(1, 1, 1))

# 保存文件
writer.save()
writer.close()
```

### USD转OBJ

```python
from custom.usd_utils import convert_usd_to_obj

# 转换USD为OBJ格式
success = convert_usd_to_obj(
    "examples/data/armchair.usda",
    "output/armchair.obj"
)
print(f"Conversion {'succeeded' if success else 'failed'}")
```

## 与scene_lib.py的关系

当前`protomotions/components/scene_lib.py`中的`MeshSceneObject`使用trimesh加载USDA文件时，会将路径替换为`.obj`/`.stl`/`.ply`：

```python
# scene_lib.py 第265-268行
obj_path = (
    self.object_path.replace(".urdf", ".obj")
    .replace(".usda", ".obj")  # 这里丢失了USD原生信息
    .replace(".usd", ".obj")
)
```

**问题**：这种方式丢失了USD的场景图、材质、动画等信息。

**建议改进方向**（后续开发）：
1. 对于支持USD的仿真器（如IsaacLab），直接使用USD格式
2. 对于不支持的仿真器，使用本工具提取几何信息
3. 接触仿真可以利用`get_bounding_box()`获取碰撞包围盒

## API参考

### USDReader

- `get_prim_paths(prim_type=None)`: 获取所有prim路径
- `get_mesh_vertices(prim_path)`: 获取mesh顶点 (N, 3)
- `get_mesh_faces(prim_path)`: 获取mesh面索引
- `get_transform(prim_path)`: 获取4x4变换矩阵
- `get_bounding_box(prim_path)`: 获取包围盒 (min, max)
- `get_metadata(prim_path)`: 获取元数据字典

### USDWriter

- `create_mesh(prim_path, vertices, faces, normals=None)`: 创建mesh
- `set_transform(prim_path, translation, rotation, scale)`: 设置变换
- `set_metadata(prim_path, key, value)`: 设置元数据
- `save()`: 保存文件

## 注意事项

1. **依赖**: 需要安装`usd-core`库
2. **性能**: 大型mesh读取可能较慢，建议缓存结果
3. **格式**: 支持`.usd`（二进制）和`.usda`（文本）格式
4. **线程安全**: 当前实现非线程安全，多线程使用需加锁

## 后续扩展方向

- [ ] 支持材质读写
- [ ] 支持动画关键帧
- [ ] 批量转换工具
- [ ] 与trimesh集成
- [ ] 碰撞体自动生成