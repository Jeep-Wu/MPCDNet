import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
import os


class PseudoDataLoader2D(nnUNetDataLoaderBase):
    def __init__(self, data: nnUNetDataset, batch_size: int, patch_size: tuple, final_patch_size: tuple,
                    label_manager, oversample_foreground_percent: float = 0.0, sampling_probabilities=None, pad_sides=None,
                    probabilistic_oversampling=False, transforms=None, pseudo_data_dir='./tmp_pseudo_data'):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent,
                         sampling_probabilities, pad_sides, probabilistic_oversampling, transforms)
        self.pseudo_data_dir = os.path.abspath(pseudo_data_dir)
        os.makedirs(self.pseudo_data_dir, exist_ok=True)
    
    def squence_indices_generator(self, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.batch_size
        for i in range(0, len(self.indices), chunk_size):
            yield self.indices[i:i + chunk_size]

    @staticmethod
    def gen_proper_size_from_max_size(max_size: int):
        # return 2 ** int(np.ceil(np.log2(max_size)))
        return 128 * int(np.ceil(max_size / 128))

    @staticmethod
    def gen_proper_size_from_shape(input_shape: np.ndarray):
        max_ori_size = np.max(input_shape[-2:])
        return PseudoDataLoader2D.gen_proper_size_from_max_size(max_ori_size)
    
    @staticmethod
    def shape_to_proper_size(data: np.ndarray, proper_size=None, return_size=False):
        if proper_size is None:
            proper_size = PseudoDataLoader2D.gen_proper_size_from_shape(data.shape)
        proper_shape = np.array(data.shape)
        proper_shape[-2:] = proper_size
        proper_data = np.zeros(proper_shape, dtype=data.dtype)
        proper_data[..., :data.shape[-2], :data.shape[-1]] = data[..., :data.shape[-2], :data.shape[-1]]
        if return_size:
            return proper_data, proper_size
        else:
            return proper_data

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.uint8)
        pseudo_seg_all = np.zeros(self.seg_shape, dtype=np.uint8)
        case_properties = []
        ori_shape = []

        for j, current_key in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(current_key)
            case_properties.append(properties)
            ori_shape.append(data.shape[-2:])

            pseudo_seg_path = os.path.join(self.pseudo_data_dir, f'{current_key}.npy')
            if os.path.exists(pseudo_seg_path):
                try:
                    with open(pseudo_seg_path, 'rb') as f:
                        tmp_seg = np.load(f, allow_pickle=False).astype(np.uint8)
                    # tmp_seg = np.load(pseudo_seg_path, allow_pickle=False).astype(np.uint8)
                    pseudo_seg = np.zeros(seg.shape, dtype=np.uint8)
                    pseudo_seg[..., :seg.shape[-2], :seg.shape[-1]] = tmp_seg[..., :seg.shape[-2], :seg.shape[-1]]
                except EOFError as e:
                    print(f'{pseudo_seg_path}:', e)
                    pseudo_seg = np.copy(seg)
                except ValueError as e:
                    print(f'{pseudo_seg_path}:', e)
                    pseudo_seg = np.copy(seg)
            else:
                pseudo_seg = np.copy(seg)

            # select a class/region first, then a slice where this class is present, then crop to that area
            if not force_fg:
                if self.has_ignore:
                    selected_class_or_region = self.annotated_classes_key if (
                            len(properties['class_locations'][self.annotated_classes_key]) > 0) else None
                else:
                    selected_class_or_region = None
            else:
                # filter out all classes that are not present here
                eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
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
            pseudo_seg = pseudo_seg[:, selected_slice]

            # the line of death lol
            # this needs to be a separate variable because we could otherwise permanently overwrite
            # properties['class_locations']
            # selected_class_or_region is:
            # - None if we do not have an ignore label and force_fg is False OR if force_fg is True but there is no foreground in the image
            # - A tuple of all (non-ignore) labels if there is an ignore label and force_fg is False
            # - a class or region if force_fg is True
            class_locations = {
                selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
            } if (selected_class_or_region is not None) else None

            # print(properties)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else False,
                                               class_locations, overwrite_class=selected_class_or_region)


            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]
            pseudo_seg = pseudo_seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
            pseudo_seg_all[j] = np.pad(pseudo_seg, ((0, 0), *padding), 'constant', constant_values=-1)

        data_all, proper_size = self.shape_to_proper_size(data_all, return_size=True)
        seg_all = self.shape_to_proper_size(seg_all, proper_size)
        pseudo_seg_all = self.shape_to_proper_size(pseudo_seg_all, proper_size)
        ori_shape = np.array(ori_shape)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):

                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.uint8)
                    pseudo_seg_all = torch.from_numpy(pseudo_seg_all).to(torch.uint8)
                    images = []
                    segs = []
                    pseudo_segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{
                            'image': data_all[b],
                            'segmentation': seg_all[b],
                            'pseudo_segmentation': pseudo_seg_all[b],
                        })
                        # tmp['pseudo_segmentation'] = self.transforms._apply_to_segmentation(pseudo_seg_all[b])
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                        pseudo_segs.append(tmp['pseudo_segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                        pseudo_seg_all = [torch.stack([s[i] for s in pseudo_segs]) for i in range(len(pseudo_segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                        pseudo_seg_all = torch.stack(pseudo_segs)
                    del segs, images, pseudo_segs

            return {'data': data_all, 'target': seg_all, 'pseudo_target': pseudo_seg_all, 'keys': selected_keys, 'ori_shape': ori_shape}

        return {'data': data_all, 'target': seg_all, 'pseudo_target': pseudo_seg_all, 'keys': selected_keys, 'ori_shape': ori_shape}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset004_Hippocampus/2d'
    ds = nnUNetDataset(folder, None, 1000)  # this should not load the properties!
    dl = PseudoDataLoader2D(ds, 366, (65, 65), (56, 40), 0.33, None, None)
    a = next(dl)
