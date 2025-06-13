import torch
import random
import numpy as np
import os.path as osp
from loguru import logger
from collections import abc, defaultdict
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from src.utils.dataset import read_megadepth_color, get_resized_wh, get_divisible_wh

# --- Utility Functions ---

def collate_fn_skip_none(batch):
    """
    Collate function for DataLoader that filters out None items from a batch.
    This is useful when the Dataset's __getitem__ might return None for invalid samples.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # If the entire batch consists of None items, return None.
        # The training loop should be prepared to handle this.
        logger.warning("Entire batch filtered out, returning None.")
        return None
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        logger.error(f"Error during default_collate: {e}")
        # Depending on the error, you might want to return None or raise
        return None # Or raise e

# --- DataModule ---

class MultiSceneDataModule:
    """
    DataModule for loading preprocessed image pairs, masks, and bounding boxes.
    Uses PreprocessedDataset which reads data from .npz files and corresponding images.
    """

    def __init__(self, args, config, accelerator):
        super().__init__()
        self.config = config
        self.args = args
        self.seed = config.TRAINER.SEED
        self.accelerator = accelerator

        # --- Data Paths ---
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.test_data_root = config.DATASET.TEST_DATA_ROOT

        self.train_preprocessed_root = config.DATASET.TRAIN_NPZ_ROOT
        self.val_preprocessed_root = config.DATASET.VAL_NPZ_ROOT
        self.test_preprocessed_root = config.DATASET.TEST_NPZ_ROOT

        self.train_preprocessed_list_path = config.DATASET.TRAIN_LIST_PATH
        self.val_preprocessed_list_path = config.DATASET.VAL_LIST_PATH
        self.test_preprocessed_list_path = config.DATASET.TEST_LIST_PATH

        # --- Dataset & Image Loading Config ---
        self.img_resize = config.DATASET.MGDPT_IMG_RESIZE
        self.img_pad = config.DATASET.MGDPT_IMG_PAD
        self.df = config.DATASET.MGDPT_DF

        # --- Loader Parameters ---
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        # Use smaller batch size for validation/testing by default
        val_test_batch_size = getattr(args, 'val_batch_size', 2)
        self.val_loader_params = {
            'batch_size': val_test_batch_size,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.test_loader_params = {
            'batch_size': val_test_batch_size,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }

        logger.info("MultiSceneDataModule initialized for PreprocessedDataset.")
        logger.info(f"  Train: Img Root='{self.train_data_root}', NPZ Root='{self.train_preprocessed_root}', List='{self.train_preprocessed_list_path}'")
        logger.info(f"  Val:   Img Root='{self.val_data_root}', NPZ Root='{self.val_preprocessed_root}', List='{self.val_preprocessed_list_path}'")
        logger.info(f"  Test:  Img Root='{self.test_data_root}', NPZ Root='{self.test_preprocessed_root}', List='{self.test_preprocessed_list_path}'")
        
        # Call setup logic here
        self._setup_datasets()

    def _setup_datasets(self):
        """Setup train/val/test datasets."""
        # Handle debug mode where accelerator might be None
        if self.accelerator is not None:
            self.world_size = self.accelerator.num_processes
            self.rank = self.accelerator.process_index
        else:
            # Debug mode fallbacks
            self.world_size = 1
            self.rank = 0
        
        logger.info(f"[rank:{self.rank}/{self.world_size}] Setting up datasets...")

        self.train_dataset = self._create_dataset_instance(mode='train')
        self.val_dataset = self._create_dataset_instance(mode='val')
        self.test_dataset = self._create_dataset_instance(mode='test')
        
        if self.train_dataset:
            logger.info(f'[rank:{self.rank}] Train Dataset loaded.')
        if self.val_dataset:
            logger.info(f'[rank:{self.rank}] Val Dataset loaded.')
        if self.test_dataset:
            logger.info(f'[rank:{self.rank}] Test Dataset loaded.')

    def _create_dataset_instance(self, mode):
        """Helper to create PreprocessedDataset instances."""
        if mode == 'train':
            data_root = self.train_data_root
            preprocessed_dir = self.train_preprocessed_root
            list_path = self.train_preprocessed_list_path
            n_samples_per_subset = self.config.TRAINER.N_SAMPLES_PER_SUBSET
            seed_for_sampling = self.seed
        elif mode == 'val':
            data_root = self.val_data_root
            preprocessed_dir = self.val_preprocessed_root
            list_path = self.val_preprocessed_list_path
            n_samples_per_subset = None
            seed_for_sampling = None
        else: # mode == 'test'
            data_root = self.test_data_root
            preprocessed_dir = self.test_preprocessed_root
            list_path = self.test_preprocessed_list_path
            n_samples_per_subset = None
            seed_for_sampling = None

        if not preprocessed_dir or not list_path:
             logger.warning(f"Dataset paths not configured for mode '{mode}'. Dataset will be None.")
             return None

        common_kwargs = {
            'mode': mode,
            'img_resize': self.img_resize,
            'df': self.df,
            'img_padding': self.img_pad,
            'n_samples_per_subset': n_samples_per_subset if mode == 'train' else None,
            'seed': seed_for_sampling if mode == 'train' else None
        }

        # Handle multiple validation sets specified by a list of list_paths
        if mode == 'val' and isinstance(list_path, (list, tuple)):
            datasets = []
            if not isinstance(preprocessed_dir, (list, tuple)):
                preprocessed_dir = [preprocessed_dir] * len(list_path)
            if not isinstance(data_root, (list, tuple)):
                data_root = [data_root] * len(list_path)

            if not (len(data_root) == len(preprocessed_dir) == len(list_path)):
                raise ValueError("Mismatch in number of validation roots/dirs/lists.")

            for dr, pp_dir, pp_list in zip(data_root, preprocessed_dir, list_path):
                logger.info(f"[rank {self.rank}] Loading validation subset: List='{pp_list}', NPZ Dir='{pp_dir}', Img Root='{dr}'")
                datasets.append(PreprocessedDataset(
                    data_root=dr,
                    processed_data_dir=pp_dir,
                    list_file_path=pp_list,
                    **common_kwargs
                ))
            return datasets if datasets else None
        else:
            logger.info(f"[rank {self.rank}] Loading dataset: List='{list_path}', NPZ Dir='{preprocessed_dir}', Img Root='{data_root}'")
            if not osp.exists(list_path):
                logger.warning(f"List file not found for mode '{mode}': {list_path}. Dataset will be None.")
                return None
            return PreprocessedDataset(
                data_root=data_root,
                processed_data_dir=preprocessed_dir,
                list_file_path=list_path,
                **common_kwargs
            )

    def _create_dataloader(self, dataset, loader_params, shuffle=False, drop_last=False):
        """Helper to create DataLoader with distributed sampler and collate function."""
        if dataset is None:
             logger.warning("Dataset is None, cannot create DataLoader.")
             return None
        
        # Handle debug mode where accelerator might be None
        if self.accelerator is not None:
            sampler = DistributedSampler(
                dataset, 
                shuffle=shuffle, 
                seed=self.seed, 
                drop_last=drop_last, 
                rank=self.accelerator.process_index, 
                num_replicas=self.accelerator.num_processes
            )
            return DataLoader(dataset, sampler=sampler, **loader_params, drop_last=drop_last, collate_fn=collate_fn_skip_none)
        else:
            # Debug mode: use simple DataLoader without DistributedSampler
            return DataLoader(dataset, **loader_params, drop_last=drop_last, collate_fn=collate_fn_skip_none)

    def train_dataloader(self):
        logger.info(f'[rank:{self.rank}/{self.world_size}] Creating training DataLoader.')
        return self._create_dataloader(self.train_dataset, self.train_loader_params, shuffle=True, drop_last=True)

    def val_dataloader(self):
        logger.info(f'[rank:{self.rank}/{self.world_size}] Creating validation DataLoader(s).')
        if self.val_dataset is None:
            return None
        if isinstance(self.val_dataset, abc.Sequence):
            return [self._create_dataloader(ds, self.val_loader_params, shuffle=False) for ds in self.val_dataset]
        else:
            return self._create_dataloader(self.val_dataset, self.val_loader_params, shuffle=False)

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}] Creating test DataLoader.')
        if self.test_dataset is None:
            return None
        return self._create_dataloader(self.test_dataset, self.test_loader_params, shuffle=False)

# --- Dataset Class ---

class PreprocessedDataset(Dataset):
    """
    Dataset class that loads preprocessed data pairs from .npz files.
    Each .npz file contains paths to original images, original masks/bboxes,
    and potentially other metadata like overlap score.
    This class loads the images, resizes masks/bboxes to match processed image dimensions.
    """
    def __init__(self,
                 data_root,
                 processed_data_dir,
                 list_file_path,
                 mode='train',
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 n_samples_per_subset=None,
                 seed=None,
                 **kwargs):
        super().__init__()
        self.data_root = data_root
        self.processed_data_dir = processed_data_dir
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.mode = mode
        self.n_samples_per_subset = n_samples_per_subset
        self.seed = seed
        self.npz_rel_paths = []

        if not osp.exists(list_file_path):
            logger.error(f"List file not found: {list_file_path}")
            return

        try:
            with open(list_file_path, 'r') as f:
                all_raw_paths = [line.strip() for line in f if line.strip()]
            
            if not all_raw_paths:
                logger.warning(f"List file {list_file_path} is empty or contains only whitespace.")
                self.npz_rel_paths = []
            elif self.mode == 'train' and self.n_samples_per_subset is not None and self.n_samples_per_subset > 0:
                logger.info(f"Applying N_SAMPLES_PER_SUBSET ({self.n_samples_per_subset}) for training set from {list_file_path}.")
                if self.seed is not None:
                    random.seed(self.seed)

                scene_to_paths = defaultdict(list)
                for path in all_raw_paths:
                    scene_name = path.split(osp.sep)[0] if osp.sep in path else "_allsamples_"
                    scene_to_paths[scene_name].append(path)

                sampled_paths = []
                total_original_paths = len(all_raw_paths)
                for scene_name, paths_in_scene in scene_to_paths.items():
                    if len(paths_in_scene) > self.n_samples_per_subset:
                        sampled_from_scene = random.sample(paths_in_scene, self.n_samples_per_subset)
                        logger.debug(f"Scene '{scene_name}': Sampled {len(sampled_from_scene)} out of {len(paths_in_scene)} paths.")
                    else:
                        sampled_from_scene = paths_in_scene
                        logger.debug(f"Scene '{scene_name}': Took all {len(sampled_from_scene)} paths (<= N_SAMPLES_PER_SUBSET).")
                    sampled_paths.extend(sampled_from_scene)
                
                self.npz_rel_paths = sampled_paths
                logger.info(f"Loaded {len(self.npz_rel_paths)} entries from {list_file_path} after sampling "
                            f"(originally {total_original_paths} entries across {len(scene_to_paths)} scenes).")
            else:
                self.npz_rel_paths = all_raw_paths
                logger.info(f"Loaded {len(self.npz_rel_paths)} entries from {list_file_path} (no per-scene sampling).")

        except Exception as e:
            logger.error(f"Error reading or processing list file {list_file_path}: {e}")
            self.npz_rel_paths = []

        if kwargs:
             logger.warning(f"PreprocessedDataset received unused arguments: {kwargs.keys()}")

    def __len__(self):
        return len(self.npz_rel_paths)

    def __getitem__(self, idx):
        if idx >= len(self.npz_rel_paths):
             logger.error(f"Index {idx} out of bounds for dataset of length {len(self.npz_rel_paths)}.")
             return None

        npz_rel_path = self.npz_rel_paths[idx]
        npz_full_path = osp.join(self.processed_data_dir, f"{npz_rel_path}.npz")

        try:
            data_npz = np.load(npz_full_path)
        except FileNotFoundError:
            logger.warning(f"Preprocessed file not found, skipping: {npz_full_path}")
            return None
        except Exception as e:
            logger.warning(f"Error loading npz file {npz_full_path}, skipping: {e}")
            return None

        try:
            img_path0_rel = str(data_npz['image_path0'])
            img_path1_rel = str(data_npz['image_path1'])
            valid_mask0_orig = data_npz['valid_mask0']
            valid_mask1_orig = data_npz['valid_mask1']
            bbox0_orig = data_npz['bbox0']
            bbox1_orig = data_npz['bbox1']
            overlap_score = float(data_npz['overlap_score'])
        except KeyError as e:
            logger.warning(f"Missing key '{e}' in npz file {npz_full_path}, skipping.")
            return None
        except ValueError as e:
            logger.warning(f"ValueError converting data in npz file {npz_full_path}: {e}, skipping.")
            return None

        img_path0_full = osp.join(self.data_root, img_path0_rel)
        img_path1_full = osp.join(self.data_root, img_path1_rel)

        try:
            image0, pad_mask0, scale0 = read_megadepth_color(
                img_path0_full, self.img_resize, self.df, self.img_padding, None)
            image1, pad_mask1, scale1 = read_megadepth_color(
                img_path1_full, self.img_resize, self.df, self.img_padding, None)
        except FileNotFoundError as e:
             logger.warning(f"Image file not found for npz {npz_full_path}, skipping: {e}")
             return None
        except IOError as e:
             logger.warning(f"IOError reading image for npz {npz_full_path}, skipping: {e}")
             return None
        except Exception as e:
             logger.warning(f"Unexpected error processing image for npz {npz_full_path}, skipping: {e}")
             return None

        if not isinstance(image0, torch.Tensor) or not isinstance(image1, torch.Tensor):
            logger.warning(f"read_megadepth_color did not return tensors for {npz_full_path}, skipping.")
            return None

        H0_proc, W0_proc = image0.shape[1:]
        H1_proc, W1_proc = image1.shape[1:]

        try:
            H0_orig, W0_orig = valid_mask0_orig.shape
            if scale0[0].item() == 0 or scale0[1].item() == 0:
                logger.warning(f"Zero scale factor for mask0 processing npz {npz_full_path}, skipping.")
                return None
            W0_intermediate = int(round(W0_orig / scale0[0].item()))
            H0_intermediate = int(round(H0_orig / scale0[1].item()))

            valid_mask0_orig_t = torch.from_numpy(valid_mask0_orig.astype(np.float32))
            mask0_intermediate_t = F.interpolate(
                valid_mask0_orig_t[None, None, :, :], 
                size=(H0_intermediate, W0_intermediate), 
                mode='nearest'
            ).squeeze().float()

            if pad_mask0 is None:
                gt_mask0_t = mask0_intermediate_t
            else:
                gt_mask0_t = torch.zeros((H0_proc, W0_proc), dtype=torch.float32, device=image0.device)
                gt_mask0_t[0:H0_intermediate, 0:W0_intermediate] = mask0_intermediate_t
            
            H1_orig, W1_orig = valid_mask1_orig.shape
            if scale1[0].item() == 0 or scale1[1].item() == 0:
                logger.warning(f"Zero scale factor for mask1 processing npz {npz_full_path}, skipping.")
                return None
            W1_intermediate = int(round(W1_orig / scale1[0].item()))
            H1_intermediate = int(round(H1_orig / scale1[1].item()))

            valid_mask1_orig_t = torch.from_numpy(valid_mask1_orig.astype(np.float32))
            mask1_intermediate_t = F.interpolate(
                valid_mask1_orig_t[None, None, :, :],
                size=(H1_intermediate, W1_intermediate),
                mode='nearest'
            ).squeeze().float()

            if pad_mask1 is None:
                gt_mask1_t = mask1_intermediate_t
            else:
                gt_mask1_t = torch.zeros((H1_proc, W1_proc), dtype=torch.float32, device=image1.device)
                gt_mask1_t[0:H1_intermediate, 0:W1_intermediate] = mask1_intermediate_t

            gt_masks = torch.stack([gt_mask0_t, gt_mask1_t], dim=0)

        except Exception as e:
            logger.warning(f"Error processing masks for npz {npz_full_path}, skipping: {e}")
            return None
            
        try:
            bbox0_orig_t = torch.from_numpy(bbox0_orig).float()
            bbox1_orig_t = torch.from_numpy(bbox1_orig).float()
            
            if scale0[0].item() == 0 or scale0[1].item() == 0 or scale1[0].item() == 0 or scale1[1].item() == 0:
                logger.warning(f"Zero scale factor encountered for npz {npz_full_path}, skipping bbox processing.")
                return None

            inv_scale0_tensor = torch.tensor([1.0/scale0[0].item(), 1.0/scale0[1].item(), 1.0/scale0[0].item(), 1.0/scale0[1].item()], dtype=torch.float32)
            inv_scale1_tensor = torch.tensor([1.0/scale1[0].item(), 1.0/scale1[1].item(), 1.0/scale1[0].item(), 1.0/scale1[1].item()], dtype=torch.float32)
            
            bbox0_t = bbox0_orig_t * inv_scale0_tensor
            bbox1_t = bbox1_orig_t * inv_scale1_tensor

            bbox0_t[0::2].clamp_(min=0, max=W0_proc - 1) 
            bbox0_t[1::2].clamp_(min=0, max=H0_proc - 1) 
            bbox1_t[0::2].clamp_(min=0, max=W1_proc - 1) 
            bbox1_t[1::2].clamp_(min=0, max=H1_proc - 1) 

            bbox0 = bbox0_t.long()
            bbox1 = bbox1_t.long()

        except Exception as e:
            logger.warning(f"Error processing bounding boxes for npz {npz_full_path}, skipping: {e}")
            return None

        data = {
            'image0': image0,
            'image1': image1,
            'gt_masks': gt_masks,
            'bbox0': bbox0,
            'bbox1': bbox1,
            'overlap_score': torch.tensor(overlap_score, dtype=torch.float32)
        }
        return data