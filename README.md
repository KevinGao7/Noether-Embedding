# Noether Embedding

This repository contains the code for "Noether Embedding: Efficient Learning of Temporal Regularities" (NeurIPs 2023). The code provides training and testing functionalities for temporal knowledge graph datasets, including ICEWS14, ICEWS18, and GDELT.

The code for baselines has been adopted from the following repositories: 
- [TNTComplEx](https://github.com/facebookresearch/tkbc)
- [BoxTE](https://github.com/JohannesMessner/BoxTE)
- [TASTER](https://github.com/ZERONE00/TASTER)
- [DE-Simple](https://github.com/BorealisAI/de-simple)
- [ATISE](https://github.com/soledad921/ATISE)
- [TERO](https://github.com/soledad921/ATISE)

The settings for ICEWS14 and ICEWS18 are the same as [xERTE](https://github.com/TemporalKGTeam/xERTE), while the settings for GDELT are the same as [RE-Net](https://github.com/INK-USC/RE-Net).

## Environment Setup
Ensure that you have Python 3.8 installed. To set up the environment, run the following command using pip:
```
pip install -r requirements.txt
```

## Training and Testing
To train Noether Embedding (NE) or other baselines, execute the following command:
```
python main.py \
    --exp_name=${exp_name} \ # Set manually
    --data_path=${data_path} \ # Specify the data directory that contains ICEWS14_forecasting, ICEWS18_forecasting, and GDELT
    --cfg_embedding_file=${embedding_config_path} \ # Specify the embedding method: NE or any baseline. Default is in ./cfgs/default/embedding
    --cfg_event_file=${event_score_config_path} \ # Specify the $f$ function with two options: `ne` or `baseline`. Default is in ./cfgs/default/component
    --cfg_loss_file=${loss_config_path} \ # Specify the loss function with two options: `mse` or `baseline`. Default is in ./cfgs/default/component
    --cfg_opt_file=${optim_config_path} \ # Default is in ./cfgs/default/optimization
    --dataset=${dataset_name} \ # Specify either ICEWS14_forecasting, ICEWS18_forecasting or GDELT
    --seed=${rand_seed}              
```

To evaluate the trained models, use the following command:
```
python main.py \
    --exp_name=${exp_name} \
    --data_path=${data_path} \
    --model_path=${saved_model_directories} \ # Specify the directory that contains model checkpoints, such as "NE_0.pth", "NE_1.pth", etc. (the number indicates different checkpoints in repeated trials under the same settings)
    --cfg_embedding_file=${embedding_config_path} \
    --cfg_event_file=${event_score_config_path} \
    --cfg_ta_file=${tr_score_config_path} \ # Specify the $g(\tau)$ or $g'(\tau)$ used in evaluation. Default is in ./cfgs/default/component
    --cfg_eval_file=${task_config_path} \ # Default is in ./cfgs/default/evaluation, "both" means evaluating on both detection and query 
    --dataset=${dataset_name} \
    --seed=${rand_seed}
```

Alternatively, you can use the script files provided in the `scripts/` directory. Use the `train` script for training and the `test` script for evaluation. Ensure that the datasets are located in `./data`. Before evaluation, make sure to set the corresponding model directories in `scripts/model_path.sh`, for example: `ne_14_path="log/wandb/.../files"`.
