import random
import torch
import json
import os
import time
import pandas as pd
from sentencepiece import SentencePieceProcessor
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from collections import defaultdict
from glob import glob

class StreamingDataLoader(IterableDataset):
    # def __init__(self, file_dirs: dict[int, str], rank: int, world_size: int, tokenizer_path: str, max_length: int, min_length: int, min_text_length: int) -> None:
    def __init__(self, files: list[tuple[str, int]], rank: int, world_size: int, tokenizer_path: str, max_length: int, min_length: int, min_text_length: int) -> None:
        # self.file_dirs = file_dirs
        self.rank = rank
        self.world_size = world_size
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.max_length = max_length
        self.min_length = min_length
        self.min_text_length = min_text_length

        # self.stats = {
        #     'samples_per_language': defaultdict(int),
        #     'tokens_per_language': defaultdict(int),
        #     'files_per_language': defaultdict(int),
        #     'total_files_found': 0,
        #     'skipped_samples': defaultdict(int),
        #     'processing_time': 0.0,
        #     'total_samples_processed': 0,
        #     'start_time': time.time()
        # }

        # all_files_globbed = self._find_and_count_files(pattern="*")
        
        # random.shuffle(all_files_globbed)
        # self.files = all_files_globbed[self.rank::self.world_size]
        
        self.files = files

        # self._print_initialization_info()

    # def _find_and_count_files(self, pattern: str) -> list[tuple[str, int]]:
    #     files = []
    #     total_files = 0
    #     for language_id, directory in self.file_dirs.items():
    #         if not os.path.exists(directory):
    #             print(f"Warning: Directory {directory} for language {language_id} does not exist")
    #             continue

    #         found_files = glob(os.path.join(directory, '**', pattern), recursive=True)
    #         language_files = [(f, language_id) for f in found_files if os.path.isfile(f) and f.endswith(('.json', '.parquet'))]
            
    #         files.extend(language_files)
    #         self.stats['files_per_language'][language_id] = len(language_files)
    #         total_files += len(language_files)

    #     self.stats['total_files_found'] = total_files
    #     return files

    # def _print_initialization_info(self):
    #     print(f'Streaming dataset initialization on rank: {self.rank}')
    #     print(f'  Total files found across all ranks: {self.stats["total_files_found"]}')
    #     print(f'  Files assigned to this rank ({self.rank}): {len(self.files)}')
    #     for lang_id, count in self.stats['files_per_language'].items():
    #         print(f'    Language {lang_id}: {count} files found')

    # def _print_stats(self):
    #     elapsed = time.time() - self.stats['start_time']
    #     samples_per_sec = self.stats['total_samples_processed'] / max(elapsed, 1)
        
    #     print(f"\nStats (Rank {self.rank}) - {elapsed:.1f}s elapsed:")
    #     print(f"  Total samples: {self.stats['total_samples_processed']} ({samples_per_sec:.1f}/sec)")
    #     for lang_id in self.stats['samples_per_language']:
    #         samples = self.stats['samples_per_language'][lang_id]
    #         avg_tokens = self.stats['tokens_per_language'][lang_id] / max(samples, 1)
    #         skipped = self.stats['skipped_samples'][lang_id]
    #         print(f"    Lang {lang_id}: {samples} samples, {avg_tokens:.1f} avg tokens, {skipped} skipped")

    def _process_file(self, file_path: str, language_id: int):
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if idx > 1:
                        st = line[5 : -3]
                        yield st.strip(), language_id
                    if idx > 100:
                        break
                        
                    
        elif file_path.endswith('.parquet'):
            if os.path.exists(file_path):
                try:
                    data = pd.read_parquet(file_path, engine="fastparquet")
                    
                    # for chunk in range(500, len(data), 500):
                    #     for text in data['text'][:chunk]:
                    #         yield text.strip(), language_id
                    for text in data['text']:
                        yield text.strip(), language_id
                except Exception as e:
                    print(f'failed: {e}')
            
            
    
    def __iter__(self):
        if self.tokenizer is None:
            self.tokenizer = SentencePieceProcessor(model_file=self.tokenizer_path)

        random.shuffle(self.files)
        for file_path, language_id in self.files:
            for text, lang_id in self._process_file(file_path=file_path, language_id=language_id):
                if not text or len(text) <= self.min_text_length:
                    # self.stats['skipped_samples'][lang_id] += 1
                    continue

                tokens = self.tokenizer.encode(text)
                
                if not tokens or len(tokens) > self.max_length or len(tokens) < self.min_length:
                    # self.stats['skipped_samples'][lang_id] += 1
                    continue
                
                # self.stats['samples_per_language'][lang_id] += 1
                # self.stats['tokens_per_language'][lang_id] += len(tokens)
                # self.stats['total_samples_processed'] += 1
                
                # if self.stats['total_samples_processed'] > 0 and self.stats['total_samples_processed'] % 1000 == 0:
                #     self._print_stats()
                    
                yield {
                    'input_ids': torch.tensor(tokens, dtype=torch.long),
                    'language_id': torch.tensor(lang_id, dtype=torch.long)
                }

# def get_data_loader(data_dirs: dict[int, str], rank: int, world_size: int, tokenizer_path: str, max_length: int, min_length: int, min_text_length: int, batch_size: int, num_workers: int):
def get_data_loader(files: list[tuple[str, int]], rank: int, world_size: int, tokenizer_path: str, max_length: int, min_length: int, min_text_length: int, batch_size: int, num_workers: int):
    # dataset = StreamingDataLoader(file_dirs=data_dirs, rank=rank, world_size=world_size, tokenizer_path=tokenizer_path, max_length=max_length, min_text_length=min_text_length, min_length=min_length)
    dataset = StreamingDataLoader(files=files, rank=rank, world_size=world_size, tokenizer_path=tokenizer_path, max_length=max_length, min_text_length=min_text_length, min_length=min_length)
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
    pad_token_id = getattr(tokenizer, 'pad_id', lambda: 0)()

    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        dataset = worker_info.dataset
        dataset.tokenizer = SentencePieceProcessor(model_file=dataset.tokenizer_path)

    def collate_fn(batch):
        if not batch: return None

        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids_padded = []
        language_ids = torch.stack([item["language_id"] for item in batch])

        for item in batch:
            seq, pad_len = item["input_ids"], max_len - len(item["input_ids"])
            padded_input_ids = torch.cat([seq, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            input_ids_padded.append(padded_input_ids)

        inputs = torch.stack(input_ids_padded)
        labels = inputs.clone()
        labels[labels == pad_token_id] = -100

        return {"input_ids": inputs[:, :-1], "labels": labels[:, 1:], "language_ids": language_ids}
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=False, 
        worker_init_fn=worker_init_fn, 
        persistent_workers=(num_workers > 0)    
    )

