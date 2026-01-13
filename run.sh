
# # =============================================
# # DEFAULT MODEL
python protomotions/inference_agent.py \
--checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt \
--motion-file AMASS/TAIJI_/amass_smpl_train_taiji.pt \
--simulator isaaclab


# =============================================
# source /home/baidu/env_isaaclab/bin/activate
# python protomotions/inference_agent.py \
#     --checkpoint results/smpl_amass_taiji/last.ckpt \
#     --motion-file AMASS/TAIJI_/amass_smpl_train_taiji.pt \
#     --simulator isaaclab

# python protomotions/inference_agent.py \
#     --checkpoint results/smpl_amass_taiji_amp/last.ckpt \
#     --motion-file AMASS/TAIJI_/amass_smpl_train_taiji.pt \
#     --simulator isaaclab

# python protomotions/inference_agent.py \
#     --checkpoint results/smpl_amass_taiji_add/last.ckpt \
#     --motion-file AMASS/TAIJI_/amass_smpl_train_taiji.pt \
#     --simulator isaaclab


# python protomotions/inference_agent.py \
#     --checkpoint results/smpl_finetune_taiji/last.ckpt \
#     --motion-file AMASS/TAIJI_/amass_smpl_train_taiji.pt \
#     --simulator isaaclab

#     