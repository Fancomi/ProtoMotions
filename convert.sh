# =====================
# 测试数据
# python data/scripts/convert_amass_to_motionlib.py \
# /home/baidu/Documents/workspace/ProtoMotions/AMASS/SSM \
# /home/baidu/Documents/workspace/ProtoMotions/AMASS/SSM_ \
# --humanoid-type smpl \
# --motion-config data/yaml_files/amass_smpl_train_small.yaml

# python examples/motion_libs_visualizer.py \
#     --motion_files ./AMASS/SSM_/amass_smpl_train_small.pt \
#     --robot smpl \
#     --simulator isaaclab

# # =====================
# # 太极数据
# python data/scripts/convert_amass_to_motionlib.py \
# /home/baidu/Documents/workspace/ProtoMotions/AMASS/TAIJI \
# /home/baidu/Documents/workspace/ProtoMotions/AMASS/TAIJI_ \
# --humanoid-type smpl \
# --motion-config data/yaml_files/amass_smpl_train_taiji.yaml


# python examples/motion_libs_visualizer.py \
# --motion_files ./AMASS/TAIJI_/amass_smpl_train_taiji.pt \
# --robot smpl \
# --simulator isaaclab

# =====================
# 跳马
ROOT_DIR=$PWD/AMASS
SRC_DIR=$ROOT_DIR/TIAOMA
DST_DIR=$ROOT_DIR/TIAOMA_
YAML_FILE=data/yaml_files/amass_smpl_train_tiaoma.yaml

# 自动生成YAML配置
python custom/gen_motion_yaml.py "$SRC_DIR" --output "$YAML_FILE"

# 转换motion数据
python data/scripts/convert_amass_to_motionlib.py \
"$SRC_DIR" "$DST_DIR" \
--humanoid-type smpl \
--motion-config "$YAML_FILE"

# 转换scene数据（使用V-HACD凸分解以获得更精确的碰撞）
# 🔧 碰撞配置说明：
#   - convex_hull: 单一凸包（默认，快但不精确）
#   - convex_decomposition: V-HACD凸分解（精确，适合复杂形状如跳马）
#   - none: 无碰撞（仅可视化）
python custom/save_static_scene.py \
    --obj-file AMASS/tiaoma_fbx_amass_transformed.obj \
    --output AMASS/tiaoma_scene.pt \
    --simulator isaacgym \
    --collision convex_decomposition \
    --vhacd-resolution 200000 \
    --vhacd-max-hulls 15 \
    --vhacd-max-vertices 64

# 💡 提示：在可视化时按 V 键查看碰撞体
# 如果碰撞体不够精确，可以调整参数：
#   --vhacd-resolution: 体素分辨率（50000-500000，越高越精确但越慢）
#   --vhacd-max-hulls: 最大凸包数（5-20，越多越精确但影响性能）

# 可视化（运动学播放模式）
python custom/visualize_motion_with_scene.py \
    --motion-file AMASS/TIAOMA_/amass_smpl_train_tiaoma.pt \
    --robot-name smpl \
    --simulator isaacgym \
    --num-envs 1 \
    --scenes-file AMASS/tiaoma_scene.pt \
    --experiment-path custom/experiment_with_scene.py \
    --experiment-name motion_scene_vis


# python examples/motion_libs_visualizer.py \
# --motion_files AMASS/TIAOMA_/amass_smpl_train_tiaoma.pt \
# --robot smpl \
# --simulator isaacgym


