. ./scripts/env_vars.sh

python main.py \
    --exp_name=train_ne \
    --data_path=${data_path} \
    --cfg_embedding_file=${ne} \
    --cfg_event_file=${event_cos} \
    --cfg_loss_file=${loss_mse} \
    --cfg_opt_file=${opt_ne} \
    --dataset=ICEWS14_forecasting \
    --seed=1000
