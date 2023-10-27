. ./scripts/env_vars.sh
. ./scripts/model_path.sh

python main.py \
    --exp_name=tnt-14 \
    --data_path=${data_path} \
    --model_path=./log/wandb/run-20230511_204557-n9etw907/files \
    --cfg_embedding_file=${tnt} \
    --cfg_event_file=${event_cos} \
    --cfg_ta_file=${ta_corr} \
    --cfg_eval_file=${both} \
    --dataset=ICEWS14_forecasting \
    --seed=1000