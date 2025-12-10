from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset


class SingleDatasetBatchSampler(BatchSampler):
    """
    A batch sampler that samples from a single dataset per batch and handles distribution across GPUs.

    Args:
        datasets (List[Dataset]): List of datasets to sample from
        batch_size (int): Global batch size (will be divided across GPUs)
        drop_last (bool): Whether to drop the last incomplete batch
        generator (Optional[torch.Generator]): Random number generator
    """

    def __init__(
        self,
        datasets: List[Dataset],
        global_batch_size: int,
        drop_last: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.datasets = datasets
        self.global_batch_size = global_batch_size
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()
        self.initial_seed = self.generator.initial_seed()

        # Calculate dataset sizes and create index mappings
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        #### get start of each dataset #####
        self.cumsum_sizes = np.cumsum([0] + self.dataset_sizes).tolist()
        self.total_size = sum(self.dataset_sizes)

        # Create shuffled indices for each dataset
        self.indices_per_dataset = [
            torch.randperm(size, generator=self.generator).tolist() for size in self.dataset_sizes
        ]
        self.current_positions = [0] * len(datasets)

        self.available_datasets = list(range(len(datasets)))
        self.max_positions = [(size // self.global_batch_size) * self.global_batch_size for size in self.dataset_sizes]

    def __iter__(self) -> Iterator[List[int]]:
        # Reset state
        self.current_positions = [0] * len(self.datasets)
        self.available_datasets = list(range(len(self.datasets)))
        self.current_data_lengths = [size for size in self.dataset_sizes]  # full length, never shrinks

        while self.available_datasets:
            # Build probabilities for available datasets only
            lengths = [self.current_data_lengths[i] for i in self.available_datasets]
            total_length = sum(lengths)
            if total_length <= 0:
                break  # nothing left to sample

            probs = torch.tensor(lengths, dtype=torch.float) / total_length

            # Pick dataset
            dataset_idx_in_available = torch.multinomial(probs, num_samples=1, generator=self.generator).item()
            dataset_idx = self.available_datasets[dataset_idx_in_available]

            # Fetch batch
            dataset_indices = self.indices_per_dataset[dataset_idx]
            current_pos = self.current_positions[dataset_idx]
            end_pos = current_pos + self.global_batch_size

            if end_pos <= self.max_positions[dataset_idx]:
                batch_indices = [idx + self.cumsum_sizes[dataset_idx] for idx in dataset_indices[current_pos:end_pos]]
                self.current_positions[dataset_idx] = end_pos
                self.current_data_lengths[dataset_idx] = self.dataset_sizes[dataset_idx] - end_pos

                # Remove if exhausted
                if end_pos >= self.max_positions[dataset_idx]:
                    self.available_datasets.remove(dataset_idx)

                yield batch_indices
            else:
                # Not enough for a full batch
                self.available_datasets.remove(dataset_idx)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler.

        Args:
            epoch (int): Epoch number
        """
        torch_gen = torch.Generator()

        # Set seed based on epoch to ensure different shuffling each epoch
        new_seed = self.initial_seed + epoch
        torch_gen.manual_seed(new_seed)
        self.generator.manual_seed(new_seed)

        # Reshuffle indices for each dataset
        self.indices_per_dataset = [torch.randperm(size, generator=torch_gen).tolist() for size in self.dataset_sizes]

    @property
    def batch_size(self) -> int:
        return self.global_batch_size

    def __len__(self) -> int:
        return sum(size // self.global_batch_size for size in self.dataset_sizes)


class GroupedBatchSampler(BatchSampler):
    """
    Batch Sampler der Batches pro Gruppe erstellt (z.B. source oder language).
    
    Perfekt für source-basierte Hard Negatives:
    - Jeder Batch enthält nur Samples aus EINER Gruppe (source/language)
    - In-Batch Negatives sind automatisch Hard Negatives (gleiche Source!)
    - Fair sampling über alle Gruppen
    
    Args:
        dataset: Das Dataset (muss indexierbar sein)
        batch_size: Batch size
        group_column: Name der Spalte für Gruppierung (z.B. 'source' oder 'language')
        drop_last: Drop incomplete batches
        shuffle: Shuffle samples innerhalb jeder Gruppe
        seed: Random seed für Shuffling
    
    Example:
        >>> sampler = GroupedBatchSampler(dataset, batch_size=32, group_column='source')
        >>> # Batch 1: alle aus source='tatdqa'
        >>> # Batch 2: alle aus source='pdf'
        >>> # Batch 3: alle aus source='tatdqa' (wieder)
        >>> # ...
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        group_column: str = 'source',
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        num_replicas: int = None,  # Für DDP
        rank: int = None,  # Für DDP
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_column = group_column
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # DDP Support
        if num_replicas is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.num_replicas = dist.get_world_size()
            else:
                self.num_replicas = 1
        else:
            self.num_replicas = num_replicas
            
        if rank is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.rank = dist.get_rank()
            else:
                self.rank = 0
        else:
            self.rank = rank
        
        print(f"   → DDP: num_replicas={self.num_replicas}, rank={self.rank}")
        
        # Gruppiere Dataset nach group_column
        print(f"📊 Gruppiere Dataset nach '{group_column}' (ohne Image Loading)...")
        self.groups = {}  # group_value -> [indices]
        
        # CRITICAL: Direkter Arrow Table Zugriff um Image Loading zu vermeiden!
        try:
            # Versuche Arrow Table direkt zu lesen (HuggingFace Dataset)
            group_col_data = dataset.data.column(group_column)
            print(f"   → Using fast Arrow Table access (no image decoding)")
            
            for idx in range(len(dataset)):
                group_value = group_col_data[idx].as_py()
                
                if group_value is None:
                    group_value = 'default'
                
                if group_value not in self.groups:
                    self.groups[group_value] = []
                self.groups[group_value].append(idx)
                
        except Exception as e:
            # Fallback: Standard access (langsam, aber funktioniert immer)
            print(f"   ⚠️ Arrow Table access failed: {e}")
            print(f"   → Falling back to standard access (slower)")
            
            for idx in range(len(dataset)):
                sample = dataset[idx]
                group_value = sample.get(group_column)
                
                if group_value is None:
                    group_value = 'default'
                
                if group_value not in self.groups:
                    self.groups[group_value] = []
                self.groups[group_value].append(idx)
        
        self.group_names = list(self.groups.keys())
        self.group_sizes = {name: len(indices) for name, indices in self.groups.items()}
        
        print(f"   ✅ Gefunden: {len(self.group_names)} Gruppen")
        for name, size in sorted(self.group_sizes.items(), key=lambda x: -x[1])[:10]:
            print(f"      - {name}: {size:,} samples")
        
        # Calculate total batches
        self.batches_per_group = {
            name: len(indices) // batch_size 
            for name, indices in self.groups.items()
        }
        self.total_batches = sum(self.batches_per_group.values())
        
        print(f"   → Total batches (global): {self.total_batches:,}")
        if self.num_replicas > 1:
            print(f"   → Batches pro GPU (DDP round-robin): {self.total_batches // self.num_replicas:,}")

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle groups
        rng = np.random.RandomState(self.seed + self.epoch)
        
        # Shuffle indices within each group
        shuffled_groups = {}
        for group_name, indices in self.groups.items():
            if self.shuffle:
                shuffled_indices = rng.permutation(indices).tolist()
            else:
                shuffled_indices = indices.copy()
            shuffled_groups[group_name] = shuffled_indices
        
        # Create batches per group
        # CRITICAL: Jeder Batch hat batch_size samples (per-device!)
        # Bei DDP: Verschiedene Ranks bekommen verschiedene Batches (round-robin)
        all_batches = []
        
        for group_name, indices in shuffled_groups.items():
            num_batches = len(indices) // self.batch_size
            for i in range(num_batches):
                batch = indices[i * self.batch_size : (i + 1) * self.batch_size]
                all_batches.append((group_name, batch))
        
        # Shuffle batches (aber jeder Batch bleibt homogen!)
        if self.shuffle:
            rng.shuffle(all_batches)
        
        # DDP: Jeder Rank bekommt jeden n-ten Batch (round-robin)
        # Rank 0: Batches 0, 4, 8, 12, ...
        # Rank 1: Batches 1, 5, 9, 13, ...
        # Rank 2: Batches 2, 6, 10, 14, ...
        # Rank 3: Batches 3, 7, 11, 15, ...
        for idx, (group_name, batch) in enumerate(all_batches):
            if idx % self.num_replicas == self.rank:
                yield batch

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.epoch = epoch

    def __len__(self) -> int:
        # Bei DDP: Jeder Rank hat total_batches / num_replicas Batches (round-robin)
        return self.total_batches // self.num_replicas
