. ./scripts/env_vars.sh
. ./scripts/model_path.sh

python main.py \
    --exp_name=te-18 \
    --data_path=${data_path} \
    --model_path=${te_18_path} \
    --cfg_embedding_file=${te} \
    --cfg_event_file=${event_cos} \
    --cfg_ta_file=${ta_corr} \
    --cfg_eval_file=${both} \
    --dataset=ICEWS18_forecasting \
    --seed=1000