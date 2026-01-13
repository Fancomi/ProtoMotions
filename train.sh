
# # mimic
# # source /home/baidu/env_isaaclab/bin/activate
# python protomotions/train_agent.py \
#     --robot-name smpl \
#     --simulator isaaclab \
#     --experiment-path examples/experiments/mimic/mlp.py \
#     --experiment-name smpl_amass_taiji \
#     --motion-file AMASS/TAIJI_/amass_smpl_train_taiji.pt \
#     --num-envs 4096 \
#     --batch-size 4096 \
#     --ngpu 1

# 基于预训练
python protomotions/train_agent.py \
--robot-name smpl \
--simulator isaaclab \
--experiment-path examples/experiments/mimic/mlp.py \
--experiment-name smpl_finetune_taiji \
--motion-file AMASS/TAIJI_/amass_smpl_train_taiji.pt \
--checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt \
--num-envs 4096 \
--batch-size 4096 #\
# --use-wandb


# # ADD
# # source /home/baidu/env_isaaclab/bin/activate
# python protomotions/train_agent.py \
#     --robot-name smpl \
#     --simulator isaaclab \
#     --experiment-path examples/experiments/add/mlp.py \
#     --experiment-name smpl_amass_taiji_add \
#     --motion-file AMASS/TAIJI_/amass_smpl_train_taiji.pt \
#     --num-envs 4096 \
#     --batch-size 4096 \
#     --ngpu 1


# # AMP (非跟踪)
# # source /home/baidu/env_isaaclab/bin/activate
# python protomotions/train_agent.py \
#     --robot-name smpl \
#     --simulator isaaclab \
#     --experiment-path examples/experiments/amp/mlp.py \
#     --experiment-name smpl_amass_taiji_amp \
#     --motion-file AMASS/TAIJI_/amass_smpl_train_taiji.pt \
#     --num-envs 4096 \
#     --batch-size 4096 \
#     --ngpu 1