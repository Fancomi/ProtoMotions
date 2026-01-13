# ProtoMotions 教程详解与交互指南

本文档详细解释了 `examples/tutorial` 目录下所有示例程序的作用，并提供了关于如何设置仿真人与物体交互的详细指南。

## 第一部分：示例程序详解

### 0. 创建仿真器 (`0_create_simulator.py`)
**作用**：这是最基础的入门程序，展示了如何启动物理仿真器并加载一个机器人。
**关键步骤**：
1.  **导入仿真器**：使用 `import_simulator_before_torch` 确保在导入 PyTorch 之前正确加载 IsaacGym 或 IsaacLab。
2.  **配置机器人**：创建 `RobotConfig`，定义机器人的资产路径（URDF/MJCF/USD）和关节名称映射。
3.  **配置仿真器**：创建 `SimulatorConfig`，设置物理参数（如 dt, 重力, 设备）。
4.  **创建地形**：使用 `TerrainConfig` 创建一个简单的平坦地形。
5.  **初始化**：实例化 `Simulator` 类并运行仿真循环。
6.  **运行循环**：在 `while` 循环中发送随机动作 (`actions`) 驱动机器人。

### 1. 添加复杂地形 (`1_add_terrain.py`)
**作用**：展示如何在仿真环境中生成非平坦的复杂地形。
**关键步骤**：
1.  **配置复杂地形**：使用 `ComplexTerrainConfig`。
2.  **设置地形比例**：通过 `terrain_proportions` 参数定义不同地形类型的比例（如 20% 平地, 10% 楼梯, 10% 斜坡等）。
3.  **采样位置**：使用 `terrain.sample_valid_locations` 确保机器人出生在有效的地形表面上，而不是卡在地下或空中。

### 2. 加载不同机器人 (`2_load_robot.py`)
**作用**：展示框架如何支持多种形态的机器人（如 G1, SMPL Humanoid）。
**关键步骤**：
1.  **工厂模式**：使用 `robot_config(args.robot)` 工厂函数根据名称加载配置。
2.  **通用接口**：无论机器人是 52 个关节的 SMPL 还是其他人形机器人，`Simulator` 类都通过统一的接口处理状态 (`root_pos`, `dof_pos`)。

### 3. 场景创建与物体加载 (`3_scene_creation.py`)
**作用**：展示如何在环境中添加静态或动态物体（如桌子、椅子、大象）。
**关键步骤**：
1.  **定义物体属性**：使用 `ObjectOptions` 设置物体的物理属性（质量、摩擦力、是否固定基座 `fix_base_link`）。
2.  **创建物体实例**：使用 `MeshSceneObject` 加载网格文件 (.obj/.usd/.urdf) 或 `BoxSceneObject` 创建几何体。
3.  **组装场景**：使用 `Scene` 类将多个物体组合在一起。
4.  **场景库**：使用 `SceneLib` 管理场景，它可以将场景复制到每一个并行环境中。

### 4. 基础环境封装 (`4_basic_environment.py`)
**作用**：引入 `BaseEnv` 类，这是强化学习（RL）的标准接口封装。
**关键步骤**：
1.  **环境配置**：使用 `EnvConfig` 定义环境参数（如最大步数）。
2.  **观测配置**：使用 `HumanoidObsConfig` 定义机器人需要观测哪些信息（如关节位置、速度、地形高度）。
3.  **RL 循环**：展示了标准的 `env.step(actions)` 接口，它返回 `obs` (观测), `rewards` (奖励), `dones` (结束标志)。这是训练算法与仿真交互的桥梁。

### 5. 运动管理与重播 (`5_motion_manager.py`)
**作用**：展示如何加载动作捕捉数据（Motion Capture Data）并在仿真中进行运动学重播。
**关键步骤**：
1.  **加载动作库**：使用 `MotionLib` 加载 `.motion` 文件（包含骨骼的旋转和平移序列）。
2.  **加载动态场景数据**：加载与动作对应的物体运动数据（如 `.npy` 文件记录的茶壶轨迹）。
3.  **运动管理器**：配置 `MimicMotionManagerConfig`。
    *   `init_start_prob=1.0`：表示每次重置都从动作的第 0 帧开始（适合演示）。
    *   `sync_motion=True`：开启“运动学同步”模式，机器人不受物理引擎力矩控制，而是直接被“瞬移”到目标姿态。这用于验证数据是否正确。

### 6. 模仿学习环境 (`6_mimic_environment.py`)
**作用**：这是训练的核心环境。它结合了物理仿真和参考动作，用于让机器人学习模仿动作。
**关键步骤**：
1.  **Mimic 配置**：使用 `MimicEnvConfig`。
    *   `sync_motion=False`：关闭同步，开启物理控制。机器人必须自己用力矩去跟踪参考动作。
2.  **模仿观测**：配置 `MimicObsConfig`，让机器人“看到”参考动作的未来姿态 (`mimic_target_pose`)、当前相位 (`mimic_phase`) 等。
3.  **奖励计算**：虽然此文件主要展示环境，但它为计算“动作与参考动作的误差”做好了准备。

### 7. DeepMimic 训练 (`7_deepmimic.py`)
**作用**：展示如何构建并训练一个 PPO 智能体（Agent）来完成模仿任务。
**关键步骤**：
1.  **奖励函数**：定义 `reward_config`，包含位置误差 (`gt_rew`)、旋转误差 (`gr_rew`) 等组件。
2.  **网络架构**：配置 Actor（策略网络）和 Critic（价值网络）。使用 `MLPWithConcatConfig` 将观测数据拼接后输入多层感知机。
3.  **PPO 代理**：创建 `PPOAgent`，设置学习率、Batch Size 等超参数。
4.  **开始训练**：调用 `agent.fit()` 开始训练循环。

---

## 第二部分：如何设置物体交互

在 ProtoMotions 中，让仿真人与物体交互（例如：人坐椅子、人倒茶），通常不是通过编写“如果距离小于X则吸附”的脚本规则实现的，而是通过 **模仿学习 (Imitation Learning)** 实现的。

### 核心逻辑
1.  **数据驱动**：你需要一段“人与物体交互”的参考数据（Motion Data）。
2.  **物理跟随**：训练机器人通过物理力矩去尽可能贴合这段参考数据。
3.  **物理交互**：当机器人学会了贴合数据，它自然就会做出“坐下”或“抓取”的动作。如果物理引擎检测到碰撞，交互就发生了。

### 具体实施步骤

#### 步骤 1：准备数据
你需要两个文件：
*   **机器人动作文件 (.motion)**：记录了每一帧机器人的关节角度和根节点位置。
*   **物体定义/数据**：
    *   如果是**静态物体**（如椅子）：只需要物体的 3D 模型文件（.urdf/.usd/.obj）和它相对于世界原点的位置。
    *   如果是**动态物体**（如被拿起的茶壶）：需要物体每一帧的轨迹数据（通常存为 .npy），或者在仿真中将其设为自由刚体并依靠摩擦力抓取（这很难，通常使用 `Link` 约束或模仿物体轨迹）。

#### 步骤 2：定义场景 (参考 `3_scene_creation.py`)
在代码中创建一个 `Scene`，将物体放置在与参考动作匹配的位置。

```python
# 定义椅子
chair_options = ObjectOptions(fix_base_link=True, ...) # 固定在地面
chair = MeshSceneObject(
    object_path="chair.urdf",
    translation=(0.0, 0.5, 0.0), # 必须与动作数据中的椅子位置一致
    rotation=(0.0, 0.0, 0.0, 1.0)
)

# 创建场景，关联到动作ID 0
scene = Scene(objects=[chair], humanoid_motion_id=0)
```

#### 步骤 3：配置模仿环境 (参考 `6_mimic_environment.py`)
配置 `MimicEnv`，关键是启用 **Mimic Observations**（模仿观测）。

```python
env_config = MimicEnvConfig(
    sync_motion=False,  # 必须为 False，表示物理仿真
    mimic_obs=MimicObsConfig(
        enabled=True,
        mimic_target_pose=MimicTargetPoseConfig(
            enabled=True,
            future_steps=2  # 让机器人“看到”未来几帧的动作，预判交互
        )
    ),
    # ...
)
```

#### 步骤 4：定义奖励函数 (参考 `7_deepmimic.py`)
为了让机器人学会交互，你需要奖励它“像参考动作那样移动”。

```python
reward_config={
    # 姿态模仿奖励：鼓励关节角度与参考动作一致
    "gr_rew": RewardComponentConfig(function=rotation_error_exp, ...),
    # 关键点位置奖励：鼓励手脚位置与参考动作一致
    "gt_rew": RewardComponentConfig(function=mean_squared_error_exp, ...),
}
```
*注意：对于交互任务，通常不需要额外的“接触奖励”，因为如果机器人完美模仿了“屁股落在椅子上”的参考动作，物理引擎自然会产生接触力支撑机器人。*

#### 步骤 5：训练
运行 PPO 训练。
*   **初期**：机器人会摔倒，无法对准椅子。
*   **中期**：机器人学会了走到椅子前。
*   **后期**：机器人学会了转身并坐下。

### 总结：怎么做交互？
1.  **不要**试图写代码控制机器人“伸手 -> 闭合手指 -> 移动”。
2.  **要**准备一段“人坐椅子”的动作捕捉数据。
3.  **要**在仿真场景的对应位置放一把椅子。
4.  **要**训练机器人模仿这段动作。

只要参考动作里人是坐着的，且仿真里的椅子位置是对的，训练出来的机器人就会真的坐在椅子上。

---

## 第三部分：常见问题解答 (FAQ)

### Q1: 为什么 `0_create_simulator.py` 中的机器人加载后会抽搐？
**A:** 机器人并没有静止，而是因为代码中显式地发送了**随机噪声动作**。
在代码的 `while` 循环中：
```python
actions = torch.randn(simulator_cfg.num_envs, robot_cfg.number_of_actions, device=device)
simulator.step(actions)
```
`torch.randn` 生成了正态分布的随机数，这些随机数被当作目标位置（PD Control Targets）发送给机器人。因此，机器人是在疯狂地试图执行这些随机变化的指令，导致看起来像是在抽搐。如果去掉这两行或发送全 0 动作，机器人就会受重力影响倒地。

### Q2: `4_basic_environment.py` 相对于 `0_create_simulator.py` 学到了什么？
**A:** 教程 4 的核心是引入了 **`BaseEnv` 抽象层**。
*   **教程 0** 直接操作 `Simulator` 类，你需要手动处理物理步进。
*   **教程 4** 使用 `BaseEnv` 类，这是强化学习（RL）的标准封装。它自动处理了：
    1.  **观测计算 (`get_obs`)**：自动获取并标准化机器人的状态。
    2.  **自动重置 (`reset`)**：当环境超时或机器人倒地时，自动重置该环境。
    3.  **RL 接口**：提供标准的 `step()` 函数，返回 `(obs, reward, done, info)`，这是连接 PPO 等算法的标准格式。

### Q3: `5_motion_manager.py` 中的 Policy 是训好的吗？它是如何拿起杯子的？
**A:** 这里的 Policy **不是训练好的**，甚至可以说没有 Policy。
*   **Kinematic Playback (运动学重播)**：注意配置中的 `sync_motion=True`。这意味着物理引擎的动力学计算被覆盖了，机器人被强制“瞬移”到参考动作的每一帧位置。
*   **杯子的移动**：杯子并没有被机器人“拿起”。杯子的运动轨迹是**预先录制好的**（从 `.npy` 文件加载），并在仿真中同步播放。
*   **意义**：这个教程展示的是如何**加载和验证数据**。在开始训练之前，你需要确保你的参考动作和物体轨迹在视觉上是匹配的。

### Q4: `7_deepmimic.py` 和 `protomotions/inference_agent.py` 有什么不同？
**A:** 它们的角色完全不同：
*   **`7_deepmimic.py` (训练脚本)**：
    *   用于**从头开始训练**一个模型。
    *   它定义了 PPO 算法、网络结构、奖励函数。
    *   运行它会生成 Checkpoint（模型权重文件）。
*   **`protomotions/inference_agent.py` (推理/测试脚本)**：
    *   用于**加载已经训练好的模型**并查看效果。
    *   它不进行反向传播（不学习），只运行前向推理。
    *   通常用于在训练完成后，录制视频或评估模型性能。

---

## 第四部分：进阶技术问答 (Deep Dive)

### 0. 如何切换对 env_ids 的观察? 怎么重置视角?
*   **切换视角**：在仿真窗口中，按键盘上的 **`O`** 键（字母 O）。
    *   它会循环切换关注的目标（Environment 0 -> Environment 1 -> ...）。
    *   如果场景中有物体，它也会在机器人和物体之间循环。
*   **重置视角**：通常没有直接的“重置”键，但你可以通过鼠标拖拽调整，或者按 `O` 键切换到下一个环境再切回来。

### 1. Actions 是绝对位置还是相对位置？为什么全 0 会倒地？
*   **相对位置 (PD Target)**：Actions 通常是相对于**默认姿态 (Default Pose)** 的偏移量。
    *   公式：`Target_Pos = Default_Pos + Action * Scale`
*   **全 0 不会倒地**：如果发送全 0 动作 (`actions = 0`)，意味着 `Target_Pos = Default_Pos`。机器人会尽力保持其默认姿态（通常是 T-Pose 或站立姿态）。
    *   **为什么你看到跪姿/平衡？** 因为默认姿态通常是站立的，PD 控制器会产生力矩去维持这个姿态，对抗重力。如果 PD 参数（刚度 Stiffness）不够大，或者默认姿态本身就是跪姿，机器人就会呈现该姿态。它不会像布娃娃一样瘫软倒地，除非你把 PD 增益设为 0。

### 2. `terrain_proportions` 是什么意思？地形是完全参数化定义的吗？
*   **含义**：这是一个列表，定义了不同类型地形在总地形中所占的百分比。
    *   例如 `[0.2, 0.1, ...]` 表示 20% 是平滑斜坡，10% 是粗糙斜坡。
    *   顺序对应：`[smooth slope, rough slope, stairs up, stairs down, discrete, stepping, poles, flat]`。
*   **参数化**：是的，这些地形是程序化生成的（Procedural Generation）。
*   **自定义接口**：支持。`TerrainConfig` 中有 `terrain_path` 和 `load_terrain` 选项。你可以加载自定义的高度图（Heightmap）或网格（Mesh），但这通常需要更高级的配置。

### 3. 我能从主循环获得什么样的力学数据？
*   **位置/速度**：`dof_pos` (关节角度), `dof_vel` (关节角速度)。
*   **力 (Forces)**：
    *   **内力 (Motor Forces)**：`simulator.get_dof_forces()`。这是电机为了达到目标位置而输出的力矩。
    *   **外力 (Contact Forces)**：`simulator.get_bodies_contact_buf()`。这是地面或物体施加在机器人身体上的接触力。
*   **关系**：`Action` -> `PD Controller` -> `Torque (Force)` -> `Physics Engine` -> `New State (pos, vel)`。

### 4. `env.step(actions)` 返回值具体是什么？
*   **`obs` (Dict)**：当前的观测数据（如关节位置、目标差异）。它和 `get_obs()` 的内容是一样的，只是 `step` 返回的是执行动作**之后**的新状态。
*   **`rewards` (Tensor)**：当前这一步获得的奖励值（标量）。定义在 `EnvConfig.reward_config` 中。
*   **`dones` (Tensor)**：布尔值，表示环境是否结束。
*   **`terminated` (Tensor)**：表示是否因为“失败”而结束（如摔倒）。
*   **`extras` (Dict)**：额外信息（如日志）。
*   **结束标志**：定义在 `EnvConfig` 中，通常包括：
    *   **超时**：达到 `max_episode_length`。
    *   **摔倒**：根节点高度低于阈值。
    *   **误差过大**：模仿任务中，与参考动作差异过大。
*   **结束了会怎么样？** `BaseEnv` 会自动调用 `reset()` 重置该环境，你不需要手动处理。

### 5. 交互物件的 USDA 怎么制作？
*   **制作流程**：
    1.  **建模**：使用 Blender, Maya 等 DCC 软件制作 3D 模型。
    2.  **导出**：导出为 `.obj` 或 `.fbx`。
    3.  **转换**：使用 NVIDIA 的工具（如 `usd_converter` 或 Isaac Sim）将模型转换为 `.usd` 或 `.usda` 格式。
    4.  **物理属性**：在 USDA 文件中添加物理属性（刚体 API, 碰撞体 API）。
*   **流水线**：通常是 `Blender -> FBX -> Isaac Sim (Import) -> Save as USD`。

### 6. 是否支持绳子？手持物？
*   **绳子**：支持（通过 Deformable Body），但计算量大且配置复杂，通常不建议在初级 RL 中使用。
*   **手持物**：
    *   **Link (约束)**：最简单的方法。在 XML/URDF 中将物体通过 `Fixed Joint` 绑在手上。
    *   **Friction (摩擦)**：很难。靠手指摩擦力抓取在仿真中非常不稳定。
    *   **双手持物**：可以，只要动作数据是双手持物的，且物理约束设置正确。

### 7. Fabric 是什么？
*   **作用**：`Fabric` 是 PyTorch Lightning 的一个组件，用于**加速和简化分布式训练**。
*   **用途**：它处理了多 GPU / 多节点训练时的繁琐细节（如设备分配、梯度同步、混合精度训练）。在 `7_deepmimic.py` 中，它用于管理 PPO 的训练循环，使其能轻易扩展到多卡训练。

### 8. `agent.fit()` 能加进去 Tutorial 0-6 的操作吗？
*   **可以**。`agent.fit()` 内部会调用 `env.step()`。
*   如果你想在训练过程中加入自定义操作（如动态改变地形、随机推机器人一把），你需要**修改 Environment 类**（即继承 `Mimic` 类并重写 `step` 或 `pre_physics_step` 方法），而不是修改 `agent.fit()`。

### 9. 日志项详解
```text
Motion Tracking:
  - Motion IDs: [0, 0, 0, 0]          # 4个环境都在模仿第0号动作
  - Motion times: [8.6, 8.8, ...]     # 每个环境当前的动作播放进度（秒）

Imitation Learning:
  - Rewards: [0., 0., ...]            # 当前步的奖励（0说明还没开始学或配置有误）
  - Average reward: 0.000             # 平均奖励
  - Dones: 0 environments             # 当前步没有环境重置
  - Actions range: [-2.7, 2.5]        # 神经网络输出的动作范围（通常在-1到1或更大）

Observations:
  - 'max_coords_obs': [4, 358]        # 机器人自身状态（关节角度、速度等），358维
  - 'historical_...': [4, 716]        # 历史状态（过去几帧的状态），用于捕捉速度/加速度信息
  - 'terrain': [4, 256]               # 地形高度图（16x16=256），机器人脚下的地形
  - 'mimic_phase': [4, 2]             # 动作相位（sin(phase), cos(phase)），表示动作进行到哪了
  - 'mimic_time_left': [4, 1]         # 动作剩余时间
  - 'mimic_target_poses': [4, 433]    # 目标姿态（参考动作的下一帧），机器人要努力变成的样子