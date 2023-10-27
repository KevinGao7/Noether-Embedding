. ./scripts/env_vars.sh
. ./scripts/model_path.sh

python main.py \
    --exp_name=ne-gdelt \
    --data_path=${data_path} \
    --model_path=${ne_gdelt_path} \
    --cfg_embedding_file=${ne} \
    --cfg_event_file=${event_cos} \
    --cfg_ta_file=${ta_corr} \
    --cfg_eval_file=${both} \
    --dataset=GDELT \
    --seed=1000