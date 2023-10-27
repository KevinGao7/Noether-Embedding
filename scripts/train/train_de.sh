. ./scripts/env_vars.sh
python main.py \
    --exp_name=train_de \
    --data_path=${data_path} \
    --cfg_embedding_file=${de} \
    --cfg_event_file=${event_baseline} \
    --cfg_loss_file=${loss_baseline} \
    --cfg_opt_file=${opt_baseline} \
    --dataset=GDELT \
    --seed=1000
