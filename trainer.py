# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import calc_wer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *

class Trainer(object):
    def __init__(self,
         model=None,
         criterion=None,
         optimizer=None,
         scheduler=None,
         config={},
         gpu_id=0,
         save_freq = 5,
         logger=logger,
         train_dataloader=None,
         val_dataloader=None,
         initial_steps=0,
         log_dir=None,
         initial_epochs=0):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.og_model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.gpu_id = gpu_id
        self.finish_train = False
        self.log_dir = log_dir
        self.logger = logger
        self.save_freq = save_freq
        self.fp16_run = False

        if self.gpu_id == 0:
            self.writer = SummaryWriter(log_dir + "/tensorboard")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """

        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._load(state_dict["model"], self.model)

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

            # overwrite schedular argument parameters
            state_dict["scheduler"].update(**self.config.get("scheduler_params", {}))
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    @staticmethod
    def get_image(arrs):
        pil_images = []
        height = 0
        width = 0
        for arr in arrs:
            uint_arr = (((arr - arr.min()) / (arr.max() - arr.min())) * 255).astype(np.uint8)
            pil_image = Image.fromarray(uint_arr)
            pil_images.append(pil_image)
            height += uint_arr.shape[0]
            width = max(width, uint_arr.shape[1])

        palette = Image.new('L', (width, height))
        curr_heigth = 0
        for pil_image in pil_images:
            palette.paste(pil_image, (0, curr_heigth))
            curr_heigth += pil_image.size[1]

        return palette

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            train_results = self._train_epoch(epoch)
            eval_results = self._eval_epoch(epoch)
            results = train_results.copy()
            results.update(eval_results)

            if self.gpu_id == 0:
                print(results)

                logger.info('--- epoch %d ---' % epoch)
                for key, value in results.items():
                    if isinstance(value, float):
                        logger.info('%-15s: %.4f' % (key, value))
                        self.writer.add_scalar(key, value, epoch)
                    else:
                        for v in value:
                            self.writer.add_figure('eval_attn', plot_image(v), epoch)

                if (epoch % self.save_freq) == 0:
                    self.save_checkpoint(osp.join(self.log_dir, 'epoch_%05d.pth' % epoch))

    def run(self, batch):
        self.optimizer.zero_grad()
        batch = [b.to(self.gpu_id) for b in batch]
        text_input, text_input_length, mel_input, mel_input_length = batch
        mel_input_length = mel_input_length // (2 ** self.og_model.n_down)
        future_mask = self.og_model.get_future_mask(
            mel_input.size(2)//(2**self.og_model.n_down), unmask_future_steps=0).to(self.gpu_id)
        mel_mask = self.og_model.length_to_mask(mel_input_length)
        text_mask = self.og_model.length_to_mask(text_input_length)
        ppgs, s2s_pred, s2s_attn = self.model(
            mel_input, src_key_padding_mask=mel_mask, text_input=text_input)

        loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                      text_input, mel_input_length, text_input_length)

        loss_s2s = 0
        for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
            loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_s2s /= text_input.size(0)

        loss = loss_ctc + loss_s2s
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
        self.optimizer.step()
        self.scheduler.step()
        return {'loss': loss.item(),
                'ctc': loss_ctc.item(),
                's2s': loss_s2s.item()}

    def _train_epoch(self, epoch):
        train_losses = defaultdict(list)
        self.model.train()
        self.train_dataloader.sampler.set_epoch(epoch)
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]", disable=self.gpu_id == 0), 1):
            losses = self.run(batch)
            for key, value in losses.items():
                train_losses["train/%s" % key].append(value)

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        train_losses['train/learning_rate'] = self._get_lr()
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self, epoch):
        self.val_dataloader.sampler.set_epoch(epoch)
        self.model.eval()
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]", disable=self.gpu_id == 0), 1):
            batch = [b.to(self.gpu_id) for b in batch]
            text_input, text_input_length, mel_input, mel_input_length = batch
            mel_input_length = mel_input_length // (2 ** self.og_model.n_down)
            future_mask = self.og_model.get_future_mask(
                mel_input.size(2)//(2**self.og_model.n_down), unmask_future_steps=0).to(self.gpu_id)
            mel_mask = self.og_model.length_to_mask(mel_input_length)
            text_mask = self.og_model.length_to_mask(text_input_length)
            ppgs, s2s_pred, s2s_attn = self.model(
                mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
            loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                          text_input, mel_input_length, text_input_length)
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= text_input.size(0)
            loss = loss_ctc + loss_s2s

            eval_losses["eval/ctc"].append(loss_ctc.item())
            eval_losses["eval/s2s"].append(loss_s2s.item())
            eval_losses["eval/loss"].append(loss.item())

            _, amax_ppgs = torch.max(ppgs, dim=2)
            wers = [calc_wer(target[:text_length],
                             pred[:mel_length],
                             ignore_indexes=list(range(5))) \
                    for target, pred, text_length, mel_length in zip(
                            text_input.cpu(), amax_ppgs.cpu(), text_input_length.cpu(), mel_input_length.cpu())]
            eval_losses["eval/wer"].extend(wers)

            _, amax_s2s = torch.max(s2s_pred, dim=2)
            acc = [torch.eq(target[:length], pred[:length]).float().mean().item() \
                   for target, pred, length in zip(text_input.cpu(), amax_s2s.cpu(), text_input_length.cpu())]
            eval_losses["eval/acc"].extend(acc)

            if eval_steps_per_epoch <= 2:
                eval_images["eval/image"].append(
                    self.get_image([s2s_attn[0].cpu().numpy()]))

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses