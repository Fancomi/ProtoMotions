# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
带场景的Mimic实验配置模板
基于examples/experiments/mimic/mlp.py，添加场景支持
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.mimic.config import MimicEnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
import argparse


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """配置机器人添加接触传感器和运动学播放模式"""
    from protomotions.robot_configs.base import ControlInfo, ControlType
    
    # 检查是否需要运动学播放模式（用于可视化）
    is_kinematic_playback = hasattr(args, 'experiment_name') and 'vis' in args.experiment_name.lower()
    
    if is_kinematic_playback:
        print("🎬 启用运动学播放模式（纯可视化）")
        
        # 修改控制参数：极低刚度+零扭矩控制
        robot_cfg.control.control_type = ControlType.TORQUE
        robot_cfg.control.override_control_info = {
            ".*": ControlInfo(
                stiffness=1,      # 极低刚度，避免物理干扰
                damping=1,        # 极低阻尼
                effort_limit=500,
                velocity_limit=100
            ),
        }
        
        # 修改资产属性：禁用物理效果
        robot_cfg.asset.disable_gravity = True
        robot_cfg.asset.fix_base_link = False
        robot_cfg.asset.self_collisions = False
        
        # 添加接触传感器
        robot_cfg.update_fields(
            contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
        )
        
        # 重新初始化控制信息（因为我们修改了override_control_info）
        robot_cfg.control.initialize_control_info(robot_cfg.asset)
    else:
        print("🏋️ 启用标准训练模式（物理模拟）")
        # 标准训练配置：正常接触传感器
        robot_cfg.update_fields(
            contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
        )


def terrain_config(args: argparse.Namespace):
    """地形配置"""
    from protomotions.components.terrains.config import TerrainConfig
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """场景库配置 - 支持从命令行参数加载场景文件"""
    from protomotions.components.scene_lib import SceneLibConfig
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """动作库配置"""
    from protomotions.components.motion_lib import MotionLibConfig
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> MimicEnvConfig:
    """环境配置 - 标准Mimic配置"""
    from protomotions.envs.mimic.config import (
        MimicEarlyTerminationEntry,
        MimicObsConfig,
        MimicMotionManagerConfig,
    )
    from protomotions.envs.obs.config import FuturePoseType, MimicTargetPoseConfig
    from protomotions.envs.base_env.config import RewardComponentConfig
    from protomotions.envs.obs.config import HumanoidObsConfig, ActionHistoryConfig
    from protomotions.envs.utils.rewards import (
        mean_squared_error_exp,
        rotation_error_exp,
        power_consumption_sum,
        norm,
        contact_mismatch_sum,
        impact_force_penalty,
    )

    mimic_early_termination = [
        MimicEarlyTerminationEntry(
            mimic_early_termination_key="max_joint_err",
            mimic_early_termination_thresh=0.5,
            less_than=False,
        )
    ]

    reward_config = {
        "action_smoothness": RewardComponentConfig(
            function=norm,
            variables={"x": "current_actions - previous_actions"},
            weight=-0.02,
        ),
        "gt_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_pos",
                "ref_x": "ref_state.rigid_body_pos",
                "coefficient": "-100.0",
            },
            weight=0.5,
        ),
        "gr_rew": RewardComponentConfig(
            function=rotation_error_exp,
            variables={
                "q": "current_state.rigid_body_rot",
                "ref_q": "ref_state.rigid_body_rot",
                "coefficient": "-5.0",
            },
            weight=0.3,
        ),
        "gv_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_vel",
                "ref_x": "ref_state.rigid_body_vel",
                "coefficient": "-0.5",
            },
            weight=0.1,
        ),
        "gav_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_ang_vel",
                "ref_x": "ref_state.rigid_body_ang_vel",
                "coefficient": "-0.1",
            },
            weight=0.1,
        ),
        "rh_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_pos[:, 0, 2]",
                "ref_x": "ref_state.rigid_body_pos[:, 0, 2]",
                "coefficient": "-100.0",
            },
            weight=0.2,
        ),
        "pow_rew": RewardComponentConfig(
            function=power_consumption_sum,
            variables={
                "dof_forces": "current_state.dof_forces",
                "dof_vel": "current_state.dof_vel",
                "use_torque_squared": "False",
            },
            weight=-1e-5,
            min_value=-0.5,
            zero_during_grace_period=True,
        ),
        "contact_match_rew": RewardComponentConfig(
            function=contact_mismatch_sum,
            variables={
                "sim_contacts": "current_state.rigid_body_contacts",
                "ref_contacts": "ref_state.rigid_body_contacts",
            },
            indices_subset=["all_left_foot_bodies", "all_right_foot_bodies"],
            weight=-0.2,
            zero_during_grace_period=True,
        ),
        "contact_force_change_rew": RewardComponentConfig(
            function=impact_force_penalty,
            variables={
                "current_forces": "current_contact_force_magnitudes",
                "previous_forces": "prev_contact_force_magnitudes",
            },
            indices_subset=["all_left_foot_bodies", "all_right_foot_bodies"],
            weight=-1e-5,
            min_value=-0.5,
            zero_during_grace_period=True,
        ),
    }

    # 检查是否是可视化模式
    is_visualization = hasattr(args, 'experiment_name') and 'vis' in args.experiment_name.lower()
    
    config = MimicEnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        humanoid_obs=HumanoidObsConfig(
            action_history=ActionHistoryConfig(
                enabled=True,
                num_historical_steps=1,
            ),
        ),
        reward_config=reward_config,
        mimic_early_termination=mimic_early_termination,
        mimic_bootstrap_on_episode_end=True,
        mimic_obs=MimicObsConfig(
            enabled=True,
            mimic_target_pose=MimicTargetPoseConfig(
                enabled=True, type=FuturePoseType.MAX_COORDS, with_velocities=True
            ),
        ),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=1.0 if is_visualization else 0.2,  # 可视化时总是从头开始
            resample_on_reset=True,
        ),
    )
    
    # 可视化模式：启用运动学同步
    if is_visualization:
        print("🎬 环境配置：启用运动学播放 (sync_motion=True)")
        config.sync_motion = True
        config.show_terrain_markers = False
    
    return config


def agent_config(
    robot_config: RobotConfig, env_config: MimicEnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Agent配置"""
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses", "historical_previous_actions"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "historical_previous_actions",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
            output_activation="tanh",
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "historical_previous_actions"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "historical_previous_actions",
            ],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            eval_metric_keys=[
                "gt_err",
                "gr_err",
                "gr_err_degrees",
                "lr_err_degrees",
                "gt_rew",
                "gr_rew",
                "pow_rew",
                "contact_force_change_rew",
            ],
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True
        ),
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    args: argparse.Namespace,
):
    """推理时的配置覆盖"""
    if env_cfg is not None:
        if hasattr(env_cfg, "mimic_early_termination"):
            env_cfg.mimic_early_termination = None
        if hasattr(env_cfg, "max_episode_length"):
            env_cfg.max_episode_length = 1000000
        if hasattr(env_cfg, "motion_manager"):
            if hasattr(env_cfg.motion_manager, "resample_on_reset"):
                env_cfg.motion_manager.resample_on_reset = True
            if hasattr(env_cfg.motion_manager, "init_start_prob"):
                env_cfg.motion_manager.init_start_prob = 1.0
