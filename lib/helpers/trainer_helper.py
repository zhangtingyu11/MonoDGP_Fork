import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc


def _tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)


def _to_device_nonblocking(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: _to_device_nonblocking(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_to_device_nonblocking(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device_nonblocking(item, device) for item in value]
    return value


class CudaBatchPrefetcher:
    """Keep one training batch ready while the current batch is computing."""

    def __init__(self, iterator, device, copy_stream):
        self.iterator = iterator
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CudaBatchPrefetcher requires a CUDA device')
        self.copy_stream = copy_stream
        expected_device = (self.device.index if self.device.index is not None
                           else torch.cuda.current_device())
        if self.copy_stream.device.index != expected_device:
            raise ValueError('CUDA prefetch stream is on the wrong device')
        self._next_batch = None
        self._next_host_batch = None
        self._next_ready = None
        self._retained_host_batches = []
        self._preload()

    def _preload(self):
        try:
            host_batch = next(self.iterator)
        except StopIteration:
            self._next_batch = None
            self._next_host_batch = None
            self._next_ready = None
            return

        if not isinstance(host_batch, (tuple, list)) or len(host_batch) != 4:
            raise ValueError('training batch must contain four fields')
        transferable = host_batch[:3]
        leaves = list(_tensor_leaves(transferable))
        if not leaves or not all(tensor.is_pinned() for tensor in leaves):
            raise RuntimeError(
                'CUDA prefetch requires every transferred CPU tensor to be pinned')

        with torch.cuda.stream(self.copy_stream):
            moved = _to_device_nonblocking(transferable, self.device)
            ready = torch.cuda.Event(blocking=False)
            ready.record(self.copy_stream)
        self._next_batch = (*moved, host_batch[3])
        self._next_host_batch = host_batch
        self._next_ready = ready

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self._next_ready)
        batch = self._next_batch
        host_batch = self._next_host_batch
        ready = self._next_ready
        for tensor in _tensor_leaves(batch[:3]):
            tensor.record_stream(current_stream)

        # The CPU source must outlive its asynchronous host-to-device copy.
        self._retained_host_batches.append((host_batch, ready))
        if len(self._retained_host_batches) > 2:
            _, old_ready = self._retained_host_batches.pop(0)
            old_ready.synchronize()

        self._preload()
        return batch

    def close(self):
        """Finish outstanding copies and release epoch-local references."""
        if self._next_ready is not None:
            self._next_ready.synchronize()
        for _, ready in self._retained_host_batches:
            ready.synchronize()
        self._retained_host_batches.clear()
        self._next_batch = None
        self._next_host_batch = None
        self._next_ready = None
        self.iterator = None


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None
        self.use_cuda_batch_prefetch = bool(
            cfg.get('use_cuda_batch_prefetch', False))
        if self.use_cuda_batch_prefetch and self.device.type != 'cuda':
            raise RuntimeError('CUDA batch prefetch is enabled without CUDA')
        # One stream is reused for the entire Trainer lifetime. Creating one
        # stream per epoch previously caused allocator growth across epochs.
        self.cuda_batch_copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.use_cuda_batch_prefetch else None)

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        
    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
               
                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                    ckpt_name)

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    results = self.tester.inference()
                    cur_result = self.tester.evaluate(results)
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        batch_source = self.train_loader
        prefetched = self.use_cuda_batch_prefetch
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        if prefetched:
            batch_source = CudaBatchPrefetcher(
                iter(self.train_loader), self.device,
                copy_stream=self.cuda_batch_copy_stream)
        try:
            for batch_idx, (inputs, calibs, targets, info) in enumerate(batch_source):
                if not prefetched:
                    inputs = inputs.to(self.device)
                    calibs = calibs.to(self.device)
                    for key in targets.keys():
                        targets[key] = targets[key].to(self.device)
                img_sizes = targets['img_size']
                targets = self.prepare_targets(targets, inputs.shape[0])
                ##dn
                dn_args = None
                if self.cfg["use_dn"]:
                    dn_args=(targets, self.cfg['scalar'], self.cfg['label_noise_scale'], self.cfg['box_noise_scale'], self.cfg['num_patterns'])
                ###
                # train one batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
                mask_dict=None
                #ipdb.set_trace()
                detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

                weight_dict = self.detr_loss.weight_dict
                detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
                detr_losses = sum(detr_losses_dict_weighted)

                detr_losses_dict = misc.reduce_dict(detr_losses_dict)
                detr_losses_dict_log = {}
                detr_losses_log = 0
                for k in detr_losses_dict.keys():
                    if k in weight_dict:
                        detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                        detr_losses_log += detr_losses_dict_log[k]
                detr_losses_dict_log["loss_detr"] = detr_losses_log
                epoch_loss_sum += detr_losses_log
                epoch_batch_count += 1

                flags = [True] * 5
                if batch_idx % 30 == 0:
                    print("----", batch_idx, "----")
                    print("%s: %.2f, " %("loss_detr", detr_losses_dict_log["loss_detr"]))
                    for key, val in detr_losses_dict_log.items():
                        if key == "loss_detr":
                            continue
                        if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                            if flags[int(key[-1])]:
                                print("")
                                flags[int(key[-1])] = False
                        print("%s: %.2f, " %(key, val), end="")
                    print("")
                    print("")

                detr_losses.backward()
                self.optimizer.step()

                progress_bar.update()
        finally:
            if prefetched:
                batch_source.close()
        progress_bar.close()
        return {
            'batch_count': epoch_batch_count,
            'mean_loss': (epoch_loss_sum / epoch_batch_count
                          if epoch_batch_count else float('nan')),
        }

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
                if key == 'depth_map':
                    target_dict[key] = val[bz]
                if key == 'obj_region':
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list
