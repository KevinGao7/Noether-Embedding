. ./scripts/env_vars.sh
. ./scripts/model_path.sh

python main.py \
    --exp_name=de-14 \
    --data_path=${data_path} \
    --model_path=${de_14_path} \
    --cfg_embedding_file=${de} \
    --cfg_event_file=${event_cos} \
    --cfg_ta_file=${ta_corr} \
    --cfg_eval_file=${both} \
    --dataset=ICEWS14_forecasting \
    --seed=1000