# Isaac Gym 碰撞体完整指南

## 📋 目录

1. [问题背景](#问题背景)
2. [核心问题解答](#核心问题解答)
3. [快速使用](#快速使用)
4. [碰撞体生成机制](#碰撞体生成机制)
5. [完整工作流](#完整工作流)
6. [参数调优指南](#参数调优指南)
7. [常见问题FAQ](#常见问题faq)
8. [技术细节](#技术细节)
9. [参考资料](#参考资料)

---

## 问题背景

在使用ProtoMotions + Isaac Gym时，你发现URDF中的`<collision>`标签设置为`<mesh>`后，可视化碰撞体却显示为**凸包（Convex Hull）**，而不是原始网格。

---

## 核心问题解答

### ❓ 在Isaac Gym中，碰撞体是如何产生的？

**答案**：Isaac Gym基于PhysX物理引擎，碰撞体生成规则如下：

1. **URDF中的 `<mesh>`** → 自动转换为**凸形状**（Convex Shapes）
   - **原因**：PhysX要求动态刚体必须使用凸形状
   - **即使物体是静态的**，asset加载时也会统一按此规则处理

2. **两种凸形状模式**：
   - `vhacd_enabled=False`（默认）→ **单一凸包**（Convex Hull）- 你看到的情况
   - `vhacd_enabled=True` → **V-HACD凸分解**（多个凸包组合）- 更精确的解决方案

### ❓ 碰撞体生成方式可选吗？

**答案**：✅ **可以！** 通过 `ObjectOptions.vhacd_enabled` 和 `vhacd_params` 控制。

---

## 快速使用

### 方法1️⃣：命令行（推荐）

#### V-HACD模式（精确碰撞，适合跳马等复杂形状）

```bash
python custom/save_static_scene.py \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --output AMASS/tiaoma_scene.pt \
    --simulator isaacgym \
    --collision convex_decomposition \
    --vhacd-resolution 200000 \
    --vhacd-max-hulls 15
```

#### 单一凸包模式（默认，适合简单几何体）

```bash
python custom/save_static_scene.py \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --output AMASS/tiaoma_scene.pt \
    --simulator isaacgym \
    --collision convex_hull
```

#### 无碰撞模式（仅可视化）

```bash
python custom/save_static_scene.py \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --output AMASS/tiaoma_scene.pt \
    --simulator isaacgym \
    --collision none
```

### 方法2️⃣：代码中使用

```python
from custom.scene_utils import create_static_mesh_scene
from protomotions.components.scene_lib import SceneLib

# V-HACD模式
scene = create_static_mesh_scene(
    mesh_path="AMASS/tiaoma_fbx_amass_transformed.urdf",
    collision_mode="convex_decomposition",
    vhacd_params={
        "resolution": 200000,
        "max_convex_hulls": 15,
        "max_num_vertices_per_ch": 64,
    }
)

SceneLib.save_scenes_to_file([scene], "AMASS/tiaoma_scene.pt")
```

### 碰撞体可视化验证

运行可视化后：
1. **按 `V` 键** → 切换碰撞体显示/隐藏
2. **按 `O` 键** → 切换摄像机目标
3. **按 `Q` 键** → 退出程序

对比原始模型和碰撞体形状，如果差异太大，调整V-HACD参数后重新生成。

---

## 碰撞体生成机制

### Isaac Gym 的限制

Isaac Gym基于NVIDIA PhysX物理引擎，有以下限制：

1. **动态刚体**：只能使用凸形状（Convex Shapes）
   - 单一凸包（Convex Hull）
   - V-HACD凸分解（多个凸包组合）

2. **静态刚体**：理论上可以使用三角网格（Triangle Mesh）
   - 但在asset加载阶段，Isaac Gym不知道物体将来是动态还是静态
   - 因此**统一按照可能用于动态物体的规则处理**

3. **URDF中的mesh标签处理**：
   ```xml
   <collision>
       <geometry>
           <mesh filename="model.obj"/>
       </geometry>
   </collision>
   ```
   被加载后会自动转换为：
   - **默认**：单一凸包（最快，但对复杂形状不精确）← 你看到的
   - **启用V-HACD**：多个凸包组合（更精确，但计算慢）← 解决方案

### Isaac Gym vs Isaac Lab 对比

| 特性 | Isaac Gym | Isaac Lab |
|------|-----------|-----------|
| 格式 | URDF | USD/USDA |
| 默认行为 | 单一凸包 | mesh简化 |
| 配置方式 | `vhacd_enabled` + `vhacd_params` | `collision_approximation` |
| 凸分解工具 | V-HACD | USD内置 |
| 碰撞选项 | convex_hull / convex_decomposition / none | none / meshSimplification / convexHull / convexDecomposition |

---

## 完整工作流

### 跳马场景示例

```bash
# 1. 生成motion配置YAML
python custom/gen_motion_yaml.py AMASS/TIAOMA \
    --output data/yaml_files/amass_smpl_train_tiaoma.yaml

# 2. 转换motion数据（SMPL → ProtoMotions格式）
python data/scripts/convert_amass_to_motionlib.py \
    AMASS/TIAOMA AMASS/TIAOMA_ \
    --humanoid-type smpl \
    --motion-config data/yaml_files/amass_smpl_train_tiaoma.yaml

# 3. 转换scene数据（OBJ → URDF + V-HACD配置）
python custom/save_static_scene.py \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --output AMASS/tiaoma_scene.pt \
    --simulator isaacgym \
    --collision convex_decomposition \
    --vhacd-resolution 200000 \
    --vhacd-max-hulls 15

# 4. 可视化验证（按V键查看碰撞体）
python custom/visualize_motion_with_scene.py \
    --motion-file AMASS/TIAOMA_/amass_smpl_train_tiaoma.pt \
    --robot-name smpl \
    --simulator isaacgym \
    --num-envs 1 \
    --scenes-file AMASS/tiaoma_scene.pt \
    --experiment-path custom/experiment_with_scene.py \
    --experiment-name motion_scene_vis
```

---

## 参数调优指南

### 推荐配置表

| 场景类型 | `--collision` | `--vhacd-resolution` | `--vhacd-max-hulls` | 说明 |
|---------|--------------|---------------------|-------------------|------|
| **简单几何体** | `convex_hull` | - | - | 盒子、球体等，默认即可 |
| **快速测试** | `convex_decomposition` | `50000` | `5` | 快但不精确 |
| **平衡质量** | `convex_decomposition` | `100000` | `10` | 日常使用推荐 |
| **跳马（复杂）** | `convex_decomposition` | `200000` | `15` | ⭐ 高精度，推荐 |
| **极高质量** | `convex_decomposition` | `500000` | `20` | 很慢，谨慎使用 |
| **仅可视化** | `none` | - | - | 无物理碰撞 |

### V-HACD参数详解

#### `--vhacd-resolution` (体素分辨率)
- **作用**：控制V-HACD算法的精度
- **默认值**：100000
- **建议范围**：50000-500000
- **影响**：
  - 越高 → 碰撞体越精确，但计算越慢
  - 越低 → 计算快，但可能丢失细节

#### `--vhacd-max-hulls` (最大凸包数)
- **作用**：生成的凸包最大数量
- **默认值**：10
- **建议范围**：5-20（建议不超过20）
- **影响**：
  - 越多 → 形状越精确，但运行时碰撞检测略慢
  - 越少 → 碰撞检测快，但可能简化形状

#### `--vhacd-max-vertices` (每凸包最大顶点数)
- **作用**：每个凸包的顶点数限制
- **默认值**：64
- **建议范围**：32-256
- **影响**：影响每个凸包的复杂度

### 调优流程

1. **初次测试**：使用默认参数
   ```bash
   --collision convex_decomposition
   ```

2. **查看效果**：运行可视化，按`V`键查看碰撞体

3. **根据情况调整**：
   - 碰撞体太简化 → 增加 `resolution` 和 `max_hulls`
   - 加载太慢 → 减少 `resolution` 和 `max_hulls`
   - 碰撞体有明显缝隙 → 增加 `max_vertices`

4. **重新生成**：用新参数重新运行 `save_static_scene.py`

### 跳马场景推荐配置

基于跳马的复杂曲面特性，推荐：

```bash
python custom/save_static_scene.py \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --output AMASS/tiaoma_scene.pt \
    --simulator isaacgym \
    --collision convex_decomposition \
    --vhacd-resolution 200000 \
    --vhacd-max-hulls 15 \
    --vhacd-max-vertices 64
```

**理由**：
- 跳马有复杂曲面，单一凸包会严重简化形状
- V-HACD可以用15个凸包较好地逼近真实形状
- 200000分辨率在精度和速度间取得平衡

---

## 常见问题FAQ

### Q1: 为什么URDF的collision mesh变成了凸包？

**A**：Isaac Gym基于PhysX物理引擎，该引擎要求：
- **动态物体必须使用凸形状**（性能和稳定性考虑）
- 即使是静态物体，Isaac Gym在asset加载时不区分动态/静态
- 因此**统一按凸形状处理**

这不是bug，而是PhysX的设计限制。

### Q2: 为什么不能直接使用三角网格（Triangle Mesh）？

**A**：虽然静态物体理论上支持三角网格，但：
1. Isaac Gym在load_asset阶段不知道物体未来是动态还是静态
2. 为了保证灵活性（物体可能后续变为动态），统一处理为凸形状
3. PhysX对动态物体的三角网格碰撞支持有限

### Q3: 如何获得更精确的碰撞？

**A**：使用V-HACD凸分解：
```bash
--collision convex_decomposition --vhacd-resolution 200000 --vhacd-max-hulls 15
```

V-HACD将复杂mesh分解成多个凸包，用多个简单凸形状组合逼近原始形状。

### Q4: V-HACD会影响运行速度吗？

**A**：
- **加载阶段**：V-HACD计算会让scene加载变慢（几秒到几十秒）
- **运行时**：碰撞检测速度与凸包数量相关
  - 15个凸包通常不会明显影响性能
  - 如果超过30个凸包，可能影响实时性能

### Q5: 如何找到最佳参数？

**A**：迭代调优流程：
1. 用默认参数生成scene
2. 运行可视化，按`V`键查看碰撞体
3. 根据精度调整参数（见[参数调优指南](#参数调优指南)）
4. 重新生成scene文件
5. 重复2-4直到满意

### Q6: 官方demo为什么可以，我的不行？

**A**：官方vaulting demo使用的是**简化后的几何体**（简单凸形状），所以默认单一凸包就足够了。

你的跳马模型如果是从真实扫描或建模得来，几何复杂度高，需要V-HACD多个凸包来逼近。

### Q7: Isaac Lab的碰撞配置怎么做？

**A**：Isaac Lab使用USD格式，配置方式不同：

```bash
python custom/save_static_scene.py \
    --obj-file model.obj \
    --output scene.pt \
    --simulator isaaclab \
    --collision-isaaclab convexDecomposition \
    --mass 100.0
```

Isaac Lab支持的碰撞近似：
- `none`：无碰撞
- `meshSimplification`：网格简化（默认）
- `convexHull`：凸包
- `convexDecomposition`：凸分解

### Q8: 如何判断碰撞体是否合适？

**A**：可视化验证：
1. 按`V`键显示碰撞体
2. 对比原始模型
3. 检查关键接触区域（如跳马顶部）精度是否足够
4. 运行动画，观察角色与物体交互是否自然

---

## 技术细节

### ObjectOptions完整参数

```python
from protomotions.components.scene_lib import ObjectOptions

options = ObjectOptions(
    # 基础配置
    fix_base_link=True,      # 静态物体固定
    density=1000.0,          # 密度（kg/m³）
    
    # 碰撞配置
    vhacd_enabled=True,      # 启用V-HACD凸分解
    
    # V-HACD详细参数
    vhacd_params={
        "resolution": 100000,              # 体素分辨率
        "max_convex_hulls": 10,            # 最大凸包数
        "max_num_vertices_per_ch": 64,     # 每凸包最大顶点数
        "concavity": 0.001,                # 凹度阈值（越小越精确）
        "alpha": 0.05,                     # 对称性偏好权重
        "beta": 0.05,                      # 革命对称性偏好权重
        "min_volume_per_ch": 0.0001,       # 最小凸包体积
        "pca": 0,                          # PCA归一化（0=禁用）
        "mode": 0,                         # 模式（0=体素，1=四面体）
    }
)
```

### Isaac Gym资产加载流程

```
OBJ文件
  ↓
URDF生成 (convert_obj_to_urdf)
  ↓
  <collision>
    <mesh filename="model.obj"/>
  </collision>
  ↓
gym.load_asset(sim, asset_root, asset_file, AssetOptions)
  ↓
检查 AssetOptions.vhacd_enabled
  ↓
  ├─ vhacd_enabled=False (默认)
  │    ↓
  │  生成单一凸包 (Convex Hull)
  │    ↓
  │  简化的碰撞体 ← 你之前看到的
  │
  └─ vhacd_enabled=True
       ↓
     运行V-HACD算法
       ↓
     生成多个凸包 (15个)
       ↓
     精确的碰撞体 ← 解决方案
```

### 代码实现示例

完整的scene创建代码：

```python
from custom.scene_utils import create_static_mesh_scene
from protomotions.components.scene_lib import SceneLib

# 创建V-HACD配置的场景
scene = create_static_mesh_scene(
    mesh_path="AMASS/tiaoma_fbx_amass_transformed.urdf",
    translation=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0, 1.0),  # xyzw quaternion
    fix_base_link=True,
    density=1000.0,
    collision_mode="convex_decomposition",
    vhacd_params={
        "resolution": 200000,
        "max_convex_hulls": 15,
        "max_num_vertices_per_ch": 64,
    }
)

# 保存为scene文件
SceneLib.save_scenes_to_file([scene], "AMASS/tiaoma_scene.pt")

print("✓ Scene保存成功")
print("  按V键可视化碰撞体")
```

### ProtoMotions集成

在实验配置中使用：

```python
# custom/experiment_with_scene.py
def scene_lib_config(args):
    from protomotions.components.scene_lib import SceneLibConfig
    
    return SceneLibConfig(
        scene_file=args.scenes_file,  # 使用生成的scene.pt
        pointcloud_samples_per_object=None,
    )
```

---

## 参考资料

### 相关链接

- **V-HACD算法**：[GitHub - V-HACD](https://github.com/kmammou/v-hacd)
- **PhysX文档**：[NVIDIA PhysX Collision Shapes](https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/guide/Manual/Geometry.html)
- **ProtoMotions**：[官方文档](https://nvlabs.github.io/ProtoMotions/)
- **Isaac Gym**：[NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)

### 相关文件

项目中的相关文件：
- [`custom/scene_utils.py`](scene_utils.py) - 场景创建工具
- [`custom/save_static_scene.py`](save_static_scene.py) - 场景保存脚本
- [`custom/visualize_motion_with_scene.py`](visualize_motion_with_scene.py) - 可视化工具
- [`protomotions/simulator/isaacgym/simulator.py`](../protomotions/simulator/isaacgym/simulator.py) - Isaac Gym模拟器实现

### V-HACD论文

> Mamou, K., & Ghorbel, F. (2009). "A simple and efficient approach for 3D mesh approximate convex decomposition". 
> IEEE International Conference on Image Processing (ICIP).

---

## 总结

### 关键要点

1. ✅ **URDF中的mesh在Isaac Gym中必然变成凸形状**（这是PhysX的要求，不是bug）
2. ✅ **解决办法不是改回mesh，而是用V-HACD生成多个凸包逼近原始形状**
3. ✅ **碰撞体生成方式可以通过 `--collision` 参数控制**
4. ✅ **官方demo使用简化几何体，你的真实模型需要V-HACD精确碰撞**
5. ✅ **按`V`键可视化碰撞体，迭代调优参数直到满意**

### 快速命令

跳马场景一键生成（推荐配置）：

```bash
python custom/save_static_scene.py \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --output AMASS/tiaoma_scene.pt \
    --simulator isaacgym \
    --collision convex_decomposition \
    --vhacd-resolution 200000 \
    --vhacd-max-hulls 15
```

---

**文档版本**：v1.0  
**最后更新**：2026-01-16
