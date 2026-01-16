#!/usr/bin/env python3
"""
使用静态场景可视化动画 - 直接基于env_kinematic_playback.py

示例：
python custom/visualize_motion_with_scene.py \
    --motion-file AMASS/TIAOMA_/amass_smpl_train_tiaoma.pt \
    --robot-name smpl \
    --simulator isaacgym \
    --num-envs 1 \
    --experiment-path custom/experiment_with_scene.py \
    --scenes-file AMASS/tiaoma_scene.pt
"""

def create_parser():
    """创建参数解析器"""
    import argparse
    parser = argparse.ArgumentParser(description="调试：使用静态场景可视化动画")
    
    # 必需参数
    parser.add_argument("--robot-name", type=str, required=True, help="机器人名称")
    parser.add_argument("--simulator", type=str, required=True, help="模拟器类型")
    parser.add_argument("--num-envs", type=int, default=1, help="环境数量")
    parser.add_argument("--motion-file", type=str, required=True, help="动画文件路径")
    parser.add_argument("--experiment-path", type=str, required=True, help="实验配置路径")
    
    # 可选参数
    parser.add_argument("--scenes-file", type=str, default=None, help="场景文件路径")
    parser.add_argument("--experiment-name", type=str, default="motion_scene_vis", help="实验名称")
    parser.add_argument("--headless", action="store_true", default=False, help="无头模式")
    parser.add_argument("--cpu-only", action="store_true", default=False, help="仅使用CPU")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    
    return parser


# 解析参数（在导入torch之前）
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# 导入模拟器（必须在torch之前）
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# 现在可以安全导入torch等
from pathlib import Path  # noqa: E402
import logging  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import importlib.util  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger(__name__)


def main():
    global parser, args
    
    device = torch.device("cuda:0" if not args.cpu_only else "cpu")
    
    # 加载实验模块
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        raise FileNotFoundError(f"实验文件不存在: {experiment_path}")
    
    spec = importlib.util.spec_from_file_location("experiment_module", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)
    
    args = parser.parse_args()
    
    print("\n=== 🐛 调试：动画场景可视化 ===")
    print(f"实验路径: {args.experiment_path}")
    print(f"机器人: {args.robot_name}")
    print(f"模拟器: {args.simulator}")
    print(f"环境数量: {args.num_envs}")
    print(f"动画文件: {args.motion_file}")
    print(f"场景文件: {args.scenes_file}")
    print(f"设备: {device}")
    
    # IsaacLab特殊处理
    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(device)}
        app_launcher = AppLauncher(app_launcher_flags)
        extra_simulator_params["simulation_app"] = app_launcher.app
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # 从实验模块构建配置
    from protomotions.utils.config_builder import build_standard_configs
    
    print("\n=== 🔨 从实验构建配置 ===")
    
    terrain_config_fn = getattr(experiment_module, "terrain_config")
    scene_lib_config_fn = getattr(experiment_module, "scene_lib_config")
    motion_lib_config_fn = getattr(experiment_module, "motion_lib_config")
    env_config_fn = getattr(experiment_module, "env_config")
    configure_robot_and_simulator_fn = getattr(
        experiment_module, "configure_robot_and_simulator", None
    )
    
    configs = build_standard_configs(
        args=args,
        terrain_config_fn=terrain_config_fn,
        scene_lib_config_fn=scene_lib_config_fn,
        motion_lib_config_fn=motion_lib_config_fn,
        env_config_fn=env_config_fn,
        configure_robot_and_simulator_fn=configure_robot_and_simulator_fn,
        agent_config_fn=None,
    )
    
    robot_config = configs["robot"]
    simulator_config = configs["simulator"]
    terrain_config = configs["terrain"]
    scene_lib_config = configs["scene_lib"]
    motion_lib_config = configs["motion_lib"]
    env_config = configs["env"]
    
    # 🐛 调试输出：检查机器人配置
    print("\n=== 🔍 调试：机器人配置检查 ===")
    print(f"控制类型: {robot_config.control.control_type}")
    print(f"禁用重力: {robot_config.asset.disable_gravity}")
    print(f"固定基座: {robot_config.asset.fix_base_link}")
    print(f"自碰撞: {robot_config.asset.self_collisions}")
    
    # 检查控制信息
    if hasattr(robot_config.control, 'override_control_info') and robot_config.control.override_control_info:
        print("控制参数覆盖:")
        for pattern, info in robot_config.control.override_control_info.items():
            print(f"  {pattern}: stiffness={info.stiffness}, damping={info.damping}")
    else:
        print("⚠️  警告：未发现控制参数覆盖！")
    
    # 启用运动学播放模式
    print("\n=== ⚙️  设置运动学播放模式 ===")
    print(f"设置前 sync_motion: {getattr(env_config, 'sync_motion', 'N/A')}")
    env_config.sync_motion = True
    env_config.show_terrain_markers = False
    print(f"设置后 sync_motion: {env_config.sync_motion}")
    
    print("\n=== 🏗️  创建环境 ===")
    
    # 创建组件
    from protomotions.utils.component_builder import build_all_components
    
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        save_dir=None,
        **extra_simulator_params,
    )
    
    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]
    
    # 创建环境
    from protomotions.envs.base_env.env import BaseEnv
    
    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )
    
    print("环境创建成功")
    print(f"动画库: {env.motion_lib.num_motions()} 个动画")
    print(f"场景库: {env.scene_lib.num_scenes()} 个场景")
    
    # 🐛 调试输出：检查环境配置
    print("\n=== 🔍 调试：环境配置检查 ===")
    print(f"env.sync_motion: {getattr(env, 'sync_motion', 'N/A')}")
    print(f"motion_manager 类型: {type(env.motion_manager).__name__ if env.motion_manager else 'None'}")
    
    # 重置环境
    print("\n=== 🔄 重置环境 ===")
    env.reset()
    
    # 🐛 调试输出：检查初始状态
    print("\n=== 🔍 调试：初始状态检查 ===")
    if env.motion_manager is not None:
        print(f"Motion IDs: {env.motion_manager.motion_ids}")
        print(f"Motion times: {env.motion_manager.motion_times}")
        
        # 获取参考姿态
        ref_state = env.motion_lib.get_motion_state(
            env.motion_manager.motion_ids,
            env.motion_manager.motion_times
        )
        print(f"参考根位置: {ref_state.root_pos[0]}")
        
        # 获取当前仿真状态
        sim_state = env.simulator.get_robot_state()
        print(f"仿真根位置: {sim_state.root_pos[0]}")
        print(f"位置差异: {torch.norm(sim_state.root_pos[0] - ref_state.root_pos[0]).item():.6f}")
    
    # 运行仿真循环
    print("\n=== 🎬 开始运动学播放 ===")
    print("控制:")
    print("  L - 开始/停止录制")
    print("  O - 切换摄像机目标")
    print("  Q - 退出")
    
    try:
        step_count = 0
        max_position_error = 0.0
        
        while env.is_simulation_running():
            actions = torch.zeros(
                env.num_envs, robot_config.number_of_actions, device=device
            )
            obs, rewards, dones, terminated, infos = env.step(actions)
            step_count += 1
            
            # 🐛 调试：每100步检查一次姿态误差
            if step_count % 100 == 0 and env.motion_manager is not None:
                ref_state = env.motion_lib.get_motion_state(
                    env.motion_manager.motion_ids,
                    env.motion_manager.motion_times
                )
                sim_state = env.simulator.get_robot_state()
                
                pos_error = torch.norm(sim_state.root_pos - ref_state.root_pos, dim=-1).mean().item()
                body_pos_error = torch.norm(
                    sim_state.rigid_body_pos - ref_state.rigid_body_pos, dim=-1
                ).mean().item()
                
                max_position_error = max(max_position_error, body_pos_error)
                print("="*40)
                print(f"\nStep {step_count}:", end="")
                print(f" 根位置误差: {pos_error:.6f}", end="")
                print(f" 身体位置误差: {body_pos_error:.6f}", end="")
                print(f" 最大误差: {max_position_error:.6f}", end="")
                print(f" 奖励均值: {rewards.mean().item():.4f}", end="")
                print(f" 重置: {dones.sum().item()}")

                print("ref_state pos:", ref_state.rigid_body_pos)
                
                if body_pos_error > 0.1:
                    print("⚠️  警告：身体位置误差过大！运动学同步可能未生效。")
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        env.close()
    
    print("\n=== 📊 播放完成 ===")
    print(f"总步数: {step_count}")
    print(f"最大身体位置误差: {max_position_error:.6f}")
    if max_position_error > 0.1:
        print("❌ 检测到严重的姿态跟踪问题！")
        print("   可能原因：")
        print("   1. 机器人控制参数配置不当（刚度/阻尼过高）")
        print("   2. sync_motion 模式未生效")
        print("   3. 物理模拟干扰运动学播放")
    else:
        print("✅ 姿态跟踪正常")


if __name__ == "__main__":
    main()
