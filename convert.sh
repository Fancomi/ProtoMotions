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

# =====================
# 太极数据
python data/scripts/convert_amass_to_motionlib.py \
/home/baidu/Documents/workspace/ProtoMotions/AMASS/TAIJI \
/home/baidu/Documents/workspace/ProtoMotions/AMASS/TAIJI_ \
--humanoid-type smpl \
--motion-config data/yaml_files/amass_smpl_train_taiji.yaml


python examples/motion_libs_visualizer.py \
--motion_files ./AMASS/TAIJI_/amass_smpl_train_taiji.pt \
--robot smpl \
--simulator isaaclab


