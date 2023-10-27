import argparse
import torch
from torch import optim
import numpy as np
import wandb

from tqdm.auto import tqdm
from copy import deepcopy

from ne.datasets import *
from ne.optimizers import *
from ne.models import *
from ne.regularizers import N3, Lambda3
from ne.utils import *
from ne.evaluation import *
from ne.config import cfg, cfg_from_yaml_file


def get_model(cfg, info_dict, ckpt):
    if cfg.MODEL_NAME == 'search':
        return None

    model_type = {
        'NE': NE,
        'TNTComplEx': TNTComplEx,
        'DE_SimplE': DE_SimplE,
        'TeRo': TeRo,
        'ATISE': ATISE,
        'BoxTE': BoxTE,
        'TASTER': TASTER
    }[cfg.MODEL_NAME]

    if cfg.MODEL_NAME == 'NE':
        model = model_type(cfg, info_dict)
    else:
        model = model_type(cfg)

    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt), strict=True)
    
    return model.to('cuda')


def train_model(cfg, model, train_data, save_name, train_time_range=None):
    ## Train Data
    if train_data is None or train_data.shape[0] == 0:
        return
    np.random.shuffle(train_data)
    examples = torch.from_numpy(train_data).cuda()
    
    ## Train Range
    if train_time_range is None:
        train_time_range = (train_data[:, 3].min(), train_data[:, 3].max() + 1)
    
    ## Optimizer, Scheduler and Loss
    opt = {
        'Adagrad': optim.Adagrad(model.parameters(), **(cfg.OPTIMIZER.CONFIG))
    }[cfg.OPTIMIZER.NAME]

    scheduler = {
        'StepLR': optim.lr_scheduler.StepLR(opt, **(cfg.SCHEDULER.CONFIG))
    }[cfg.SCHEDULER.NAME]

    time_loss = {
        'MSE': MSELoss(cfg.EVENT_POS, cfg.EVENT_NEG),
        'CE': CrossEntropyLoss(),
        'baseline': None
    }[cfg.LOSS_MODE]

    for epoch in range(cfg.EPOCH):
        print(f'epoch: {epoch}')
        model.train()

        if cfg.LOSS_MODE == 'MSE' or cfg.MODEL_NAME == 'NE':
            optimizer = NEOptimizer(
                model, opt,
                time_loss = time_loss,
                batch_size = cfg.BATCH_SIZE,
                clip = cfg.GRAD_CLIP
            )
            loss_epoch = optimizer.epoch(examples, train_time_range, first_epoch=(epoch == 0))
        elif cfg.MODEL_NAME == 'TNTComplEx':
            emb_reg = N3(cfg.EMB_REG)
            time_reg = Lambda3(cfg.TIME_REG)
            optimizer = TKBCOptimizer(
                model, emb_reg, time_reg,
                opt, batch_size = cfg.BATCH_SIZE
            )
            loss_epoch = optimizer.epoch(examples)
        elif cfg.MODEL_NAME == 'DE_SimplE':
            optimizer = DEOptimizer(
                model, opt, batch_size = cfg.BATCH_SIZE
            )
            loss_epoch = optimizer.epoch(examples)
        elif cfg.MODEL_NAME == 'TeRo':
            optimizer = TEOptimizer(
                'TeRo', model, opt, batch_size = cfg.BATCH_SIZE
            )
            loss_epoch = optimizer.epoch(examples)
        elif cfg.MODEL_NAME == 'ATISE':
            optimizer = TEOptimizer(
                'ATISE', model, opt, batch_size = cfg.BATCH_SIZE
            )
            loss_epoch = optimizer.epoch(examples)
        elif cfg.MODEL_NAME == 'BoxTE':
            optimizer = BoxOptimizer(model, opt, cfg)
            loss_epoch = optimizer.epoch(examples)
        elif cfg.MODEL_NAME == 'TASTER':
            optimizer = TasterOptimizer(model, opt, cfg)
            loss_epoch = optimizer.epoch(examples)
        if epoch % 20 == 0:
            torch.save(deepcopy(model.state_dict()), f'''{wandb.run.dir}/{save_name}.pth''')
        if epoch % 10 == 0:
            print(save_name)

        if loss_epoch <= cfg.BREAK_LOSS_THD:
            break

        scheduler.step()

    torch.save(deepcopy(model.state_dict()), f'''{wandb.run.dir}/{save_name}.pth''')
    print(f'model saved @ {wandb.run.dir}/{save_name}.pth')


def train_online(cfg, model, train_data):
    time_max = train_data[:, 3].max()
    online_interval = cfg.ONLINE_SETTING.ONLINE_INTERVAL
    for t in tqdm(range(0, time_max, online_interval)):
    # for t in tqdm(range(0, time_max)):
        print(f'Interval: [{t}, {t + online_interval})')
        idx_list = (t <= train_data[:, 3]) * (train_data[:, 3] < t + online_interval)
        l, r = t, min(t + online_interval, time_max + 1)
        train_model(cfg, model, train_data[idx_list], train_time_range=(l, r))

        torch.save(deepcopy(model.state_dict()), f'{wandb.run.dir}/{cfg.MODEL_NAME}-({l},{r}).pth')


def optimize(cfg, model, train_data, save_name):
    if 'EPOCH' in cfg and cfg.EPOCH > 0:
        train_model(cfg, model, train_data, save_name)

def evaluate(cfg, model, dataset):
    tal = TemporalAssociationLearning(cfg, dataset)
    tal.evaluate(model)


def parse_config():
    parser = argparse.ArgumentParser(description="Running args for Noether Embeddings")
    parser.add_argument('--project_name', type=str, default="NIPS", help="Config file path")
    parser.add_argument('--exp_name', type=str, default="dumb", help="Config file path")
    parser.add_argument('--data_path', type=str, help="Data directory path")
    parser.add_argument('--model_path', type=str, default=None, help='Pytorch checkpoint path')
    parser.add_argument('--cfg_embedding_file', type=str, default=None, help="Embedding config file path")
    parser.add_argument('--cfg_event_file', type=str, default=None, help="Event score config file path")
    parser.add_argument('--cfg_ta_file', type=str, default=None, help="TA score config file path")
    parser.add_argument('--cfg_loss_file', type=str, default=None, help="Loss config file path")
    parser.add_argument('--cfg_eval_file', type=str, default=None, help="Evaluation config file path")
    parser.add_argument('--cfg_opt_file', type=str, default=None, help="Optimization config file path")
    parser.add_argument('--dataset', type=str, default='ICEWS14_forecasting', help="Dataset config file path")
    parser.add_argument('--seed', type=int, default=1000, help="Running seed")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_embedding_file, cfg)
    cfg_from_yaml_file(args.cfg_event_file,     cfg)
    cfg_from_yaml_file(args.cfg_ta_file,        cfg)
    cfg_from_yaml_file(args.cfg_eval_file,      cfg)
    cfg_from_yaml_file(args.cfg_loss_file,      cfg)
    cfg_from_yaml_file(args.cfg_opt_file,       cfg)

    cfg['DATASET'] = args.dataset
    cfg['SEED'] = args.seed

    return args, cfg


def init_wandb(args, cfg):
    os.environ['WANDB_DIR']='./log'
    if not os.path.exists('./log'):
        os.mkdir('./log')
    wandb.login()
    wandb.init(project=args.project_name,
                name=args.exp_name,
                config=cfg,
                settings=wandb.Settings(start_method="fork"))
    print(f'WANDB config: {wandb.config}')
    print(f"WANDB dir: {wandb.run.dir}")


if __name__ == '__main__':
    ## Init
    args, cfg = parse_config()
    init_wandb(args, cfg)
    set_random_seed(args.seed)

    ## Dataset
    dataset = TemporalDataset(name=args.dataset, root_path=args.data_path)

    if cfg.MODEL_NAME == 'BoxTE' or cfg.MODEL_NAME == 'TASTER':
        dataset.prep_for_special_baselines()

    ## Model
    train_data = dataset.get_train()
    info_dict =  {
        'spo': count_triples(train_data)['tri_to_idx'],
        'sp': count_sp(train_data),
        'po': count_po(train_data)
    }
    cfg.update({
        'sizes': get_shape(train_data),
        'entity_ids': dataset.get_ent_id_list(),
        'relation_ids': dataset.get_rel_id_list(),
        'timestamps': dataset.get_timestamps(),
        'dataset': dataset
    })

    print('sizes: ', cfg.sizes)
    print('spo:', len(list(info_dict['spo'])))
    print('train data:', train_data.shape, 'time_min, time_max:', train_data[:, 3].min(), train_data[:, 3].max())

    
    ## Optimization
    if 'EPOCH' in cfg:
        for i in range(cfg.REPEAT_TIMES):
            model = get_model(cfg, info_dict, args.model_path)
            optimize(cfg, model, train_data, f'{cfg.MODEL_NAME}_{i}')
            print(f'model saved @ {wandb.run.dir}')

            if 'TASK_NAME' in cfg:
                if cfg.TASK_NAME == 'both':
                    task_list = ['query', 'detection']
                else:
                    task_list = [cfg.TASK_NAME]
                for task_name in task_list:
                    cfg.TASK_NAME = task_name
                    evaluate(cfg, model, dataset)
    elif 'TASK_NAME' in cfg:
        if cfg.TASK_NAME == 'both':
            task_list = ['query', 'detection']
        else:
            task_list = [cfg.TASK_NAME]
        for i in range(cfg.REPEAT_TIMES):
            model = get_model(cfg, info_dict, f'{args.model_path}/{cfg.MODEL_NAME}_{i}.pth')
            for task_name in task_list:
                cfg.TASK_NAME = task_name
                evaluate(cfg, model, dataset)

    wandb.finish()