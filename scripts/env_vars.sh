#!/bin/sh
data_path=./data

# Main
ne=cfgs/default/embedding/ne.yaml
ne_gdelt=cfgs/default/embedding/ne-gdelt.yaml
ne_demo=cfgs/demo/ne.yaml
de=cfgs/default/embedding/de.yaml
te=cfgs/default/embedding/tero.yaml
tnt=cfgs/default/embedding/tntcomplex.yaml
at=cfgs/default/embedding/atise.yaml
box=cfgs/default/embedding/boxte.yaml
taster=cfgs/default/embedding/taster.yaml
ne_linear=cfgs/ab-linear/ne.yaml

query=cfgs/default/evaluation/query.yaml
detection=cfgs/default/evaluation/detection.yaml
mining=cfgs/default/evaluation/mining.yaml
mining_general=cfgs/default/evaluation/mining_general.yaml
demo=cfgs/default/evaluation/demo.yaml
both=cfgs/default/evaluation/both.yaml

opt_ne=cfgs/default/optimization/offline-ne.yaml
opt_baseline=cfgs/default/optimization/offline-baseline.yaml
opt_baseline_small=cfgs/default/optimization/offline-baseline-small.yaml
opt_taster=cfgs/default/optimization/offline-taster.yaml

# Ab-embedding

ab_emb_query_24=cfgs/ab-embedding/evaluation/query_24.yaml
ab_emb_query_48=cfgs/ab-embedding/evaluation/query_48.yaml
ab_emb_query_96=cfgs/ab-embedding/evaluation/query_96.yaml
ab_emb_query_192=cfgs/ab-embedding/evaluation/query_192.yaml
ab_emb_query_144=cfgs/ab-embedding/evaluation/query_144.yaml
ab_emb_query_288=cfgs/ab-embedding/evaluation/query_288.yaml

ab_emb_detection_24=cfgs/ab-embedding/evaluation/detection_24.yaml
ab_emb_detection_48=cfgs/ab-embedding/evaluation/detection_48.yaml
ab_emb_detection_96=cfgs/ab-embedding/evaluation/detection_96.yaml
ab_emb_detection_192=cfgs/ab-embedding/evaluation/detection_192.yaml
ab_emb_detection_144=cfgs/ab-embedding/evaluation/detection_144.yaml
ab_emb_detection_288=cfgs/ab-embedding/evaluation/detection_288.yaml

ab_emb_opt_24=cfgs/ab-embedding/optimization/offline_24.yaml
ab_emb_opt_48=cfgs/ab-embedding/optimization/offline_48.yaml
ab_emb_opt_96=cfgs/ab-embedding/optimization/offline_96.yaml
ab_emb_opt_144=cfgs/ab-embedding/optimization/offline_144.yaml
ab_emb_opt_192=cfgs/ab-embedding/optimization/offline_192.yaml
ab_emb_opt_288=cfgs/ab-embedding/optimization/offline_288.yaml
ab_emb_opt_384=cfgs/ab-embedding/optimization/offline_384.yaml
ab_emb_opt_480=cfgs/ab-embedding/optimization/offline_480.yaml
ab_emb_opt_960=cfgs/ab-embedding/optimization/offline_960.yaml
ab_emb_opt_var=cfgs/ab-embedding/optimization/offline_variable.yaml

# Ab-param
ab_param_ne_d_1=cfgs/ab-param/embedding/ne_d_1.yaml
ab_param_ne_d_5=cfgs/ab-param/embedding/ne_d_5.yaml
ab_param_ne_d_10=cfgs/ab-param/embedding/ne_d_10.yaml
ab_param_ne_d_25=cfgs/ab-param/embedding/ne_d_25.yaml
ab_param_ne_d_50=cfgs/ab-param/embedding/ne_d_50.yaml
ab_param_ne_d_100=cfgs/ab-param/embedding/ne_d_100.yaml
ab_param_ne_d_200=cfgs/ab-param/embedding/ne_d_200.yaml
ab_param_ne_d_500=cfgs/ab-param/embedding/ne_d_500.yaml
ab_param_ne_d_600=cfgs/ab-param/embedding/ne_d_600.yaml
ab_param_ne_d_800=cfgs/ab-param/embedding/ne_d_800.yaml

ab_param_ne_omg_1=cfgs/ab-param/embedding/ne_omg_1.yaml
ab_param_ne_omg_5=cfgs/ab-param/embedding/ne_omg_5.yaml
ab_param_ne_omg_10=cfgs/ab-param/embedding/ne_omg_10.yaml
ab_param_ne_omg_50=cfgs/ab-param/embedding/ne_omg_50.yaml
ab_param_ne_omg_100=cfgs/ab-param/embedding/ne_omg_100.yaml
ab_param_ne_omg_200=cfgs/ab-param/embedding/ne_omg_200.yaml
ab_param_ne_omg_400=cfgs/ab-param/embedding/ne_omg_400.yaml
ab_param_ne_omg_600=cfgs/ab-param/embedding/ne_omg_600.yaml
ab_param_ne_omg_800=cfgs/ab-param/embedding/ne_omg_800.yaml
ab_param_ne_omg_900=cfgs/ab-param/embedding/ne_omg_900.yaml
ab_param_ne_omg_950=cfgs/ab-param/embedding/ne_omg_950.yaml
ab_param_ne_omg_990=cfgs/ab-param/embedding/ne_omg_990.yaml
ab_param_ne_omg_1000=cfgs/ab-param/embedding/ne_omg_1000.yaml
ab_param_ne_omg_1010=cfgs/ab-param/embedding/ne_omg_1010.yaml
ab_param_ne_omg_1050=cfgs/ab-param/embedding/ne_omg_1050.yaml
ab_param_ne_omg_1200=cfgs/ab-param/embedding/ne_omg_1200.yaml
ab_param_ne_omg_1400=cfgs/ab-param/embedding/ne_omg_1400.yaml

# Ab-Event-Norm
ab_event_norm_ne_norm=cfgs/ab-event-norm/ne-norm.yaml
ab_event_norm_ne_nonorm=cfgs/ab-event-norm/ne-nonorm.yaml

ab_event_norm_event_score_nonorm=cfgs/ab-event-norm/event-nonorm.yaml

# Componet
event_cos=cfgs/default/component/event_cos.yaml
event_baseline=cfgs/default/component/event_baseline.yaml

loss_mse=cfgs/default/component/loss_mse.yaml
loss_baseline=cfgs/default/component/loss_baseline.yaml

ta_direct=cfgs/default/component/ta_direct.yaml
ta_corr=cfgs/default/component/ta_corr.yaml