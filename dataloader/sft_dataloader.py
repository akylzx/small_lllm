import random
import torch
import json
import os
from sentencepiece import SentencePieceProcessor
from new_model.model.new_model import LANGUAGE_MAP

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer_path: str, data_files: dict[str, list[str]]):
        self.tokenizer_path = tokenizer_path
        self.data_files = data_files
        self.files = self._preprocess_files()
        self.samples = []
        self._generate_samples()
    
    def _preprocess_files(self) -> list[tuple[str, int]]:
        files = []

        for language in LANGUAGE_MAP.keys():
            if  language == "unk":
                continue

            language_id = LANGUAGE_MAP[language]

            for file in self.data_files[language]:
                if os.path.exists(file):
                    files.append((file, language_id))
        
        return files
    
    def _preprocess_file(self, file: str) -> list[dict[str, str]]:
        samples = []

        if file.endswith(".jsonl"):
            with open(file, 'r', encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line.strip())
                    system = data.get("system", "").strip()
                    user = data.get("user", "").strip()
                    assistant = data.get("assistant", "").strip()

                    sample = {
                        "system": system,
                        "user": user,
                        "assistant": assistant,
                    }

                    samples.append(sample)
        return samples
    
    def _generate_samples(self):
        tokenizer = SentencePieceProcessor(model_file=self.tokenizer_path)

        bos_token_id = tokenizer.bos_id() if tokenizer.bos_id() != -1 else None
        eos_token_id = tokenizer.eos_id() if tokenizer.eos_id() != -1 else None

        random.shuffle(self.files)

        for filepath, language_id in self.files:
            for sample in self._preprocess_file(file=filepath):
                full_prompt = f"{sample['system']}\n{sample['user']}\nAssistant:"
                full_response = f"{sample['assistant']}"

                inputs = tokenizer.encode(full_prompt)
                labels = tokenizer.encode(full_response)

                input_ids = []
                input_labels = []
                if bos_token_id is not None:
                    input_ids.append(bos_token_id)
                    input_labels.append(-100)

                input_ids.extend(inputs)
                input_ids.extend(labels)
                input_labels.extend([-100] * len(inputs))
                input_labels.extend(labels)

                if eos_token_id is not None:
                    input_ids.append(eos_token_id)
                    input_labels.append(-100)
                
                attention_mask = [1] * len(input_ids)
                
                assert len(input_ids) == len(attention_mask)
                assert len(input_ids) == len(input_labels)

                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "language_id": torch.tensor(language_id, dtype=torch.long),
                    "labels": torch.tensor(input_labels, dtype=torch.long),
                })
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) :
        return self.samples[index]


def get_sft_dataloader(data_files: dict[str, list[str]], rank: int, world_size: int, tokenizer_path: str, batch_size: int):
    sft_dataset = SFTDataset(data_files=data_files, tokenizer_path=tokenizer_path)
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)

    def collate_fn(batch):
        if batch is None:
            return 

        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        language_ids = [item['language_id'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]

        pad_token_id = tokenizer.pad_id() if tokenizer.pad_id() != -1 else 0

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        batch_size, seq_len = input_ids_padded.shape

        assert attention_mask_padded.shape == (batch_size, seq_len)
        assert labels_padded.shape == (batch_size, seq_len)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "language_ids": torch.stack(language_ids),
        }
    
    sampler = torch.utils.data.DistributedSampler(sft_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    return torch.utils.data.DataLoader(
        dataset=sft_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        sampler=sampler
    )

