import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.nets.MPCDNetBase import get_mpcdnetbase_from_plans
from nnunetv2.nets.MPCDNet import MPCDNet

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from nnunetv2.training.dataloading.pseudo_data_loader_2d import PseudoDataLoader2D
from multiprocessing.synchronize import Barrier

from contextlib import ExitStack
from collections import defaultdict
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

class MPCDNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), epochs: int = 1000, barrier: Barrier = None, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, epochs)
        # Each GPU trains a different model, and the models do not share parameters.
        # self.is_ddp = False
        os.makedirs(join(self.output_folder, f'checkpoints'), exist_ok=True)
        self._best_ema_epoch = 0
        self.barrier = barrier
        self.fn_fp_loss = None  # -> self.initialize
        self.val_dice_note = defaultdict(int)
        self.dice_score = GeneralizedDiceScore(num_classes=1, input_format="one-hot")
        self.mean_iou = MeanIoU(num_classes=1, per_class=True, input_format='one-hot')

        # for the fn fp
        self.dataset_json['labels'].update({'fn': 2, 'fp': 3})
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
    
    def do_split(self):
        splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        if not isfile(splits_file):
            raise RuntimeError("Cannot run training without splits_final.json. Run the split generation first.")
        else:
            self.print_to_log_file("Using splits from existing split file:", splits_file)
            splits = load_json(splits_file)
            self.print_to_log_file(f"The split file contains {len(splits)} splits.")

        self.print_to_log_file("Desired fold for training: %d" % self.local_rank)
        if self.local_rank < len(splits):
            tr_keys = splits[self.local_rank]['train']
            val_keys = splits[self.local_rank]['val']
            self.print_to_log_file("This split has %d training and %d validation cases."
                                    % (len(tr_keys), len(val_keys)))
        else:
            raise RuntimeError("Cannot run training with the specified fold. The split file does not contain "
                                "enough splits.")
        if any([i in val_keys for i in tr_keys]):
            self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                    'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys
    
    def _build_inject_loss(self, loss):
        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        if len(arch_init_kwargs['kernel_sizes'][0]) == 2:
            s_network = get_mpcdnetbase_from_plans(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                      num_input_channels, num_output_channels, True, enable_deep_supervision)
            t_network = get_mpcdnetbase_from_plans(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                      num_input_channels, num_output_channels, True, enable_deep_supervision)
            model = MPCDNet(s_network, t_network, m=0.98)
        else:
            raise NotImplementedError("Only 2D models are supported")
        
        print("UMambaBot: {}".format(model))

        return model

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.fn_fp_loss = self._build_inject_loss(torch.nn.L1Loss())
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        pseudo_target = batch['pseudo_target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            pseudo_target = [i.to(self.device, non_blocking=True) for i in pseudo_target]
        else:
            target = target.to(self.device, non_blocking=True)
            pseudo_target = pseudo_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            bg_gt_pred_col = []
            fn_pred_col = []
            fp_pred_col = []
            bg_gt_target_col = []
            fn_target_col = []
            fp_target_col = []
            for _idx, o in enumerate(output):
                bg_gt_pred = o[:, [0, 1], ...]
                gt_pred = o[:, [1], ...]
                fn_pred = o[:, [2], ...]
                fp_pred = o[:, [3], ...]
                bg_gt_pred_col.append(bg_gt_pred)
                fn_pred_col.append(fn_pred)
                fp_pred_col.append(fp_pred)

                t = pseudo_target[_idx]
                # bg_gt_target = torch.nn.functional.one_hot(t.squeeze(1).to(torch.long), num_classes=2).permute(0, 3, 1, 2).to(t.dtype)
                bg_gt_target = t
                fn_target = ((gt_pred <= 0) & (t > 0)).to(t.dtype)  # fn
                fp_target = ((gt_pred > 0) & (t <= 0)).to(t.dtype)  # fp
                bg_gt_target_col.append(bg_gt_target)
                fn_target_col.append(fn_target)
                fp_target_col.append(fp_target)

            bg_gt_l = self.loss(bg_gt_pred_col, bg_gt_target_col)
            fn_l = self.fn_fp_loss(fn_pred_col, fn_target_col)
            fp_l = self.fn_fp_loss(fp_pred_col, fp_target_col)
            l = bg_gt_l * 0.9 + (fn_l + fp_l) * 0.1

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        pseudo_target = batch['pseudo_target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            pseudo_target = [i.to(self.device, non_blocking=True) for i in pseudo_target]
        else:
            target = target.to(self.device, non_blocking=True)
            pseudo_target = pseudo_target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            bg_gt_pred_col = []
            fn_pred_col = []
            fp_pred_col = []
            bg_gt_target_col = []
            fn_target_col = []
            fp_target_col = []
            for _idx, o in enumerate(output):
                bg_gt_pred = o[:, [0, 1], ...]
                gt_pred = o[:, [1], ...]
                fn_pred = o[:, [2], ...]
                fp_pred = o[:, [3], ...]
                bg_gt_pred_col.append(bg_gt_pred)
                fn_pred_col.append(fn_pred)
                fp_pred_col.append(fp_pred)

                t = pseudo_target[_idx]
                # bg_gt_target = torch.nn.functional.one_hot(t.squeeze(1).to(torch.long), num_classes=2).permute(0, 3, 1, 2).to(t.dtype)
                bg_gt_target = t
                fn_target = ((gt_pred <= 0) & (t > 0)).to(t.dtype)  # fn
                fp_target = ((gt_pred > 0) & (t <= 0)).to(t.dtype)  # fp
                bg_gt_target_col.append(bg_gt_target)
                fn_target_col.append(fn_target)
                fp_target_col.append(fp_target)

            bg_gt_l = self.loss(bg_gt_pred_col, bg_gt_target_col)
            fn_l = self.fn_fp_loss(fn_pred_col, fn_target_col)
            fp_l = self.fn_fp_loss(fp_pred_col, fp_target_col)
            l = bg_gt_l * 0.9 + (fn_l + fp_l) * 0.1

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
            pseudo_target = pseudo_target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    class DI_Helpers(object):
        def __init__(self, cls, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.cls = cls
        
        def __call__(self, *args, **kwargs):
            return self.cls(*args, *self.args, **kwargs, **self.kwargs)

    def on_train_start(self):
        # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
        # which may not be present  when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders(loader_2d_cls=self.DI_Helpers(PseudoDataLoader2D, pseudo_data_dir=f'{self.output_folder}/tmp_pseudo_data'))

        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)), verify_npy=True)
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")
    
    def on_validation_epoch_start(self):
        self.network.eval()

        self.print_to_log_file("Generating psuedo data ...")

        # generate pseudo data
        data_generator = self.dataloader_val.generator
        for cases in data_generator.squence_indices_generator():
            batch_data = []
            batch_seg = []
            batch_ori_shape = []
            batch_properties = []
            for case_key in cases:
                data, seg, properties = data_generator._data.load_case(case_key)
                batch_data.append(data)
                batch_seg.append(seg)
                batch_ori_shape.append(data.shape[-2:])
                batch_properties.append(properties)
            
            batch_ori_shape = np.array(batch_ori_shape)
            batch_proper_size = data_generator.gen_proper_size_from_max_size(np.max(batch_ori_shape))
            
            pseudo_seg = []
            for i, (data, seg, properties) in enumerate(zip(batch_data, batch_seg, batch_properties)):
                # filter out all classes that are not present here
                eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == data_generator.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    len(eligible_classes_or_regions) > 0 else None

                if selected_class_or_region is not None:
                    selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
                else:
                    selected_slice = np.random.choice(len(data[0]))

                data = data[:, selected_slice]
                seg = seg[:, selected_slice]

                proper_data = data_generator.shape_to_proper_size(data, batch_proper_size)

                raw_data = np.expand_dims(proper_data, axis=0)

                _raw_output = self.network(torch.tensor(raw_data).to(self.device))

                raw_output = _raw_output[0].detach().cpu()
                pseudo_seg.append((torch.softmax(raw_output[:, [0, 1], ...], 1)[0,1,...] > 0.5).long())
            pseudo_seg = torch.stack(pseudo_seg)
            pseudo_seg = pseudo_seg.detach().cpu().numpy().astype(np.int16)
            for i, (case_key, s_seg, seg, ori_shape) in enumerate(
                zip(cases, pseudo_seg, batch_seg, batch_ori_shape)):
                _s_seg = s_seg[..., :ori_shape[-2], :ori_shape[-1]]
                m = self.mean_iou(torch.tensor(_s_seg.reshape(1, 1, *_s_seg.shape)), torch.tensor(seg))
                if self.val_dice_note[case_key] < m:
                    self.val_dice_note[case_key] = m
                    with ExitStack() as stack:
                        np.save(stack.enter_context(open(join(data_generator.pseudo_data_dir, f'{case_key}.npy'), 'wb+')),
                                s_seg[..., :ori_shape[-2], :ori_shape[-1]], allow_pickle=False)
            del pseudo_seg
            if self.current_epoch % 100 == 0 or self.current_epoch in list(range(0, 100, 2)):
                dest_dir = f'{data_generator.pseudo_data_dir}_epoch{self.current_epoch}'
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copytree(data_generator.pseudo_data_dir, dest_dir, dirs_exist_ok=True)
            
            if self.barrier is not None:
                self.barrier.wait()

    def on_epoch_end(self):
            self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

            self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
            self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
            self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                                self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
            self.print_to_log_file(
                f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

            # handling periodic checkpointing
            current_epoch = self.current_epoch
            if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
                self.save_checkpoint(join(self.output_folder, f'checkpoints', f'epoch_{current_epoch}.pth'))

            # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
            if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
                best_checkpoint = join(self.output_folder, f'checkpoints', 'checkpoint_best.pth')
                if current_epoch >= self.num_epochs / 10 and os.path.exists(best_checkpoint):
                    shutil.copy2(best_checkpoint, join(self.output_folder, f'checkpoints', f'checkpoint_best_{self._best_ema_epoch}.pth'))
                self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
                self._best_ema_epoch = current_epoch
                self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
                self.save_checkpoint(best_checkpoint)

            if self.local_rank == 0:
                self.logger.plot_progress_png(self.output_folder)

            self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if not self.disable_checkpointing:
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod

            checkpoint = {
                'network_weights': mod.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                'logging': self.logger.get_checkpoint(),
                '_best_ema': self._best_ema,
                'current_epoch': self.current_epoch + 1,
                'init_args': self.my_init_kwargs,
                'trainer_name': self.__class__.__name__,
                'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
            }
            torch.save(checkpoint, filename)
        else:
            self.print_to_log_file('No checkpoint written, checkpointing is disabled')
