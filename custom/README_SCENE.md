# 静态场景集成指南

为ProtoMotions添加静态OBJ场景支持，用于动画对齐验证和训练。

## 快速开始

### 完整工作流程

```bash
# 1. 生成动画配置
python custom/gen_motion_yaml.py AMASS/TIAOMA/20260115 \
    --output data/yaml_files/amass_smpl_train_tiaoma.yaml

# 2. 转换动画数据
python data/scripts/convert_amass_to_motionlib.py \
    AMASS/TIAOMA/20260115 \
    AMASS/TIAOMA_ \
    --humanoid-type smpl \
    --motion-config data/yaml_files/amass_smpl_train_tiaoma.yaml

# 3. 可视化验证对齐
python custom/visualize_motion_with_scene.py \
    --motion-file AMASS/TIAOMA_/amass_smpl_train_tiaoma.pt \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --robot smpl \
    --simulator isaacgym \
    --num-envs 1

# 4. 保存场景（自动转换格式）
python custom/save_static_scene.py \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --output AMASS/tiaoma_scene.pt \
    --simulator isaacgym

# 5. 训练
python protomotions/train_agent.py \
    --robot-name smpl \
    --simulator isaaclab \
    --experiment-path custom/experiment_with_scene.py \
    --experiment-name smpl_tiaoma_scene \
    --motion-file AMASS/TIAOMA_/amass_smpl_train_tiaoma.pt \
    --scenes-file AMASS/tiaoma_scene.pt \
    --num-envs 4096

# 6. 推理
python protomotions/inference_agent.py \
    --checkpoint results/smpl_tiaoma_scene/last.ckpt \
    --simulator isaaclab \
    --scenes-file AMASS/tiaoma_scene.pt
```

## 核心文件

| 文件 | 功能 |
|------|------|
| [`scene_utils.py`](scene_utils.py) | 场景工具库（格式转换） |
| [`visualize_motion_with_scene.py`](visualize_motion_with_scene.py) | 可视化验证对齐 |
| [`save_static_scene.py`](save_static_scene.py) | 保存场景文件 |
| [`experiment_with_scene.py`](experiment_with_scene.py) | 训练配置模板 |

## 格式转换说明

`save_static_scene.py`根据目标模拟器自动转换：

| 模拟器 | 输入 | 输出 | 说明 |
|--------|------|------|------|
| **isaacgym** | OBJ | URDF | 生成URDF引用OBJ |
| **isaaclab** | OBJ | USDA | 完整USD资产（含碰撞） |
| **newton** | OBJ | OBJ | 直接使用 |
| **genesis** | OBJ | OBJ | 直接使用 |

## 进阶用法

### 调整OBJ位置/旋转

```bash
python custom/save_static_scene.py \
    --obj-file scene.obj \
    --output scene.pt \
    --simulator isaacgym \
    --translation 0 0 0.5 \
    --rotation 0 0 0 1
```

### 多个OBJ场景

```python
from protomotions.components.scene_lib import Scene, MeshSceneObject, ObjectOptions, SceneLib

objs = [
    MeshSceneObject("obj1.urdf", translation=(0,0,0), options=ObjectOptions(fix_base_link=True)),
    MeshSceneObject("obj2.urdf", translation=(1,0,0), options=ObjectOptions(fix_base_link=True))
]
scene = Scene(objects=objs)
SceneLib.save_scenes_to_file([scene], "multi_scene.pt")
```

### 使用Python API

```python
from custom.scene_utils import create_static_mesh_scene_lib

scene_lib = create_static_mesh_scene_lib(
    mesh_path="scene.urdf",  # 或.usda/.obj
    num_envs=4096,
    device="cuda:0",
    translation=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0, 1.0)
)
```

## 常见问题

**Q: 为什么需要转换格式？**  
A: 
- isaacgym需要URDF格式
- isaaclab优先使用USDA格式（更完整的物理属性）
- `save_static_scene.py`自动处理

**Q: 动画和OBJ不对齐？**  
A: 
1. 使用`visualize_motion_with_scene.py`检查
2. 使用`--translation`和`--rotation`调整
3. 确保OBJ已与动画坐标系对齐

**Q: URDF和USDA有什么区别？**  
A:
- URDF：轻量，引用OBJ文件
- USDA：完整USD资产，包含碰撞几何和物理属性

**Q: 如何指定物理属性？**  
A: isaaclab可用`--mass`和`--collision`参数：
```bash
python custom/save_static_scene.py \
    --obj-file scene.obj \
    --output scene.pt \
    --simulator isaaclab \
    --mass 10.0 \
    --collision convexHull
```

## API参考

### scene_utils.py

```python
# 创建场景
create_static_mesh_scene(
    mesh_path: str,
    translation: tuple = (0, 0, 0),
    rotation: tuple = (0, 0, 0, 1),
    fix_base_link: bool = True,
    density: float = 1000.0
) -> Scene

# 创建SceneLib
create_static_mesh_scene_lib(
    mesh_path: str,
    num_envs: int,
    device: str = "cuda:0",
    translation: tuple = (0, 0, 0),
    rotation: tuple = (0, 0, 0, 1),
    terrain = None
) -> SceneLib

# OBJ → URDF（isaacgym）
convert_obj_to_urdf(
    obj_file: str,
    output_urdf: Optional[str] = None
) -> str

# OBJ → USDA（isaaclab）
convert_obj_to_usda(
    obj_file: str,
    mass: Optional[float] = None,
    collision_approximation: str = "meshSimplification",
    output_usda: Optional[str] = None
) -> str
```

### SceneLib

```python
# 保存场景
SceneLib.save_scenes_to_file(scenes: List[Scene], filepath: str)

# 加载场景
SceneLib._load_scenes_from_file(filepath: str) -> List[Scene]
```

## 技术实现

### 自动格式转换

#### isaacgym (OBJ → URDF)
生成轻量URDF文件引用OBJ：
```xml
<robot name="object">
    <link name="object">
        <visual><geometry><mesh filename="scene.obj"/></geometry></visual>
        <collision><geometry><mesh filename="scene.obj"/></geometry></collision>
    </link>
</robot>
```

#### isaaclab (OBJ → USDA)
使用IsaacLab的MeshConverter转换为完整USD资产（动态导入，避免启动时依赖）。

#### newton/genesis
直接使用OBJ文件，无需转换。

### 设计特点

- ✅ **自动转换** - 根据simulator智能选择格式
- ✅ **兼容性** - 动态导入isaaclab，不影响isaacgym
- ✅ **简洁API** - 单函数创建和保存
- ✅ **工程化** - 充分复用SceneLib/MotionLib
- ✅ **鲁棒性** - 跳过已存在文件，避免重复转换

## 参考

- 官方文档: https://nvlabs.github.io/ProtoMotions/
- Tutorial 5: `examples/tutorial/5_motion_manager.py`（动态物体）
- SceneLib: `protomotions/components/scene_lib.py`
