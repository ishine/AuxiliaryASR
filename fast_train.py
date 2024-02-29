import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import yaml
from torch.utils.tensorboard import SummaryWriter

from trainer import Trainer

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

import logging
from logging import StreamHandler

def main(rank: int, world_size: int, config):
    ddp_setup(rank, world_size)

    #dataset, model, optimizer = load_train_objs()

    batch_size = config.get('batch_size', 10)
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)

    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        dataset_config=config.get('dataset_params', {}))

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      dataset_config=config.get('dataset_params', {}))

    model = build_model(model_params=config['model_params'] or {})

    criterion = build_criterion(critic_params={
        'ctc': {'blank': blank_index},
    })

    optimizer, scheduler = build_optimizer(
        {"params": model.parameters(), "optimizer_params":{}, "scheduler_params": scheduler_params})

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    trainer = Trainer(model=model,
      criterion=criterion,
      optimizer=optimizer,
      scheduler=scheduler,
      save_freq = save_freq,
      train_dataloader=train_dataloader,
      val_dataloader=val_dataloader,
      log_dir=config.get('log_dir', '/'),
      logger=logger)

    trainer.train(epochs)

    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--config_path', type=str, help='configuration file path')
    args = parser.parse_args()

    config_path = args.config_path
    config = yaml.safe_load(open(config_path))
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.mkdir(log_dir)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    '''file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)'''

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config), nprocs=world_size)