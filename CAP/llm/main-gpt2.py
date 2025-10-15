from utils import set_seed
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from torch.utils.data import Dataset, DataLoader
import os

from model import GPT
import matplotlib.pyplot as plt
from trainer import Trainer

import re
import numpy as np


class Tokenizer:
    def __init__(self, vocab, pad_token_id=0):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.pad_token_id = pad_token_id

    def get_pad_token_id(self):
        return self.pad_token_id

    def encode(self, text):
        tokens = re.split(r'(\s+|[,.:;?_!"()\'’]|--)', text)
        tokens = [tok for tok in tokens if tok != ""]
        return [self.str_to_int[s] for s in tokens]

    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids]
        return "".join(tokens)


class StoryDataset(Dataset):
    def __init__(self, df, tokenizer, max_len_prompt, max_len_response):
        self.df = df
        self.tokenizer = tokenizer

        self.max_len_prompt = max_len_prompt
        self.max_len_response = max_len_response

        self.pad_token_id = self.tokenizer.get_pad_token_id()

    def __len__(self):
        return len(self.df)

    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.max_len_prompt + self.max_len_response - 1

    def __getitem__(self, idx):
        prompt = self.df.loc[idx, 'prompt']  # prompt
        response = self.df.loc[idx, 'response']

        # tokenizer
        prompt = self.tokenizer.encode(prompt)
        response = self.tokenizer.encode(response)

        if len(prompt) > self.max_len_prompt:
            prompt = prompt[:self.max_len_prompt]
        else:
            prompt = prompt + [self.pad_token_id] * (self.max_len_prompt - len(prompt))

        if len(response) > self.max_len_response:
            response = response[:self.max_len_response]
        else:
            response = response + [self.pad_token_id] * (self.max_len_response - len(response))

        cat = prompt + response
        
        x = torch.tensor(cat[:-1], dtype=torch.long)
        y = torch.tensor(cat[1:], dtype=torch.long)
        y[:self.max_len_prompt-1] = 0

        return x, y

class StoryTestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.df.loc[idx, 'prompt']
        response = self.df.loc[idx, 'response']
        prompt = self.tokenizer.encode(prompt)
        prompt = torch.tensor(prompt, dtype=torch.long)

        return prompt, response

if __name__ == '__main__':
    seed = 42
    set_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = "4"

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    df = pd.read_csv('../data/tiny_stories_with_prompt.csv')

    lengths_prompt_chars = [(len(p)) for p in df["prompt"]]
    lengths_response_chars = [(len(p)) for p in df["response"]]

    max_chars_prompt = int(np.mean(lengths_prompt_chars) + np.std(lengths_prompt_chars))
    min_chars_prompt = int(np.mean(lengths_prompt_chars) - np.std(lengths_prompt_chars))

    max_chars_response = int(np.mean(lengths_response_chars) + np.std(lengths_response_chars))
    min_chars_response = int(np.mean(lengths_response_chars) - np.std(lengths_response_chars))

    print(f"Prompt (chars): min={min_chars_prompt}, max={max_chars_prompt}")
    print(f"Response (chars): min={min_chars_response}, max={max_chars_response}")

    condition = df["prompt"].str.len().between(min_chars_prompt, max_chars_prompt) & df["response"].str.len().between(
        min_chars_response, max_chars_response)

    df_filtered = df[condition]
    df_filtered = df_filtered.reset_index(drop=True)
    print(len(df_filtered))

    # Create the vocab
    PAD_TOKEN = "<PAD>"
    pad_token_id = 0

    txt = " ".join(df_filtered['prompt'].tolist() + df_filtered['response'].tolist())
    tokens = re.split(r'(\s+|[,.:;?_!"()\'’]|--)', txt)
    preprocessed = [tok for tok in tokens if tok != ""]
    all_words = sorted(set(preprocessed))

    vocab = {token: i + 1 for i, token in enumerate(all_words)}
    vocab[PAD_TOKEN] = pad_token_id
    vocab_size = len(vocab)
    print('vocab_size =', vocab_size)

    tokenizer = Tokenizer(vocab, pad_token_id)

    lengths_prompt_tokens = [len(tokenizer.encode(p)) for p in df_filtered["prompt"]]
    lengths_response_tokens = [len(tokenizer.encode(p)) for p in df_filtered["response"]]
    
    max_len_prompt = int(np.mean(lengths_prompt_tokens) + np.std(lengths_prompt_tokens))
    min_len_prompt = int(np.mean(lengths_prompt_tokens) - np.std(lengths_prompt_tokens))
    
    max_len_response = int(np.mean(lengths_response_tokens) + np.std(lengths_response_tokens))
    min_len_response = int(np.mean(lengths_response_tokens) - np.std(lengths_response_tokens))
    
    print(f"Prompt (tokens): min={min_len_prompt}, max={max_len_prompt}")
    print(f"Response (tokens): min={min_len_response}, max={max_len_response}")

    train_dataset = StoryDataset(df_filtered, tokenizer, max_len_prompt=max_len_prompt,
                                 max_len_response=max_len_response)
    test_dataset = StoryTestDataset(df_filtered, tokenizer)

    model_name = 'gpt2'

    PATH = f'{model_name}_16092025.pt'
    
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = vocab_size
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)
    # model.load_state_dict(torch.load(PATH))
    model = model.to(device)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # 6e-6 # 6e-4
    train_config.max_iters = 200000
    train_config.num_workers = 0
    train_config.batch_size = 16
    trainer = Trainer(train_config, model, train_dataset)

    log_txt = f'out_{model_name}_16092025.txt'
    open(log_txt, 'w').close()

    test_loader = DataLoader(
            test_dataset,
            sampler=torch.utils.data.RandomSampler(test_dataset, replacement=False),
            pin_memory=True,
            batch_size=1,
            num_workers=train_config.num_workers,
        )


    def batch_end_callback(trainer):
        
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

            with open(log_txt, 'a') as f:
                f.write(f"\niter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}\n")

        if trainer.iter_num % 100 == 0:
            # evaluate both the train and test scores        
            model.eval()
            with torch.no_grad():
                # sample from the model...
                data_iter = iter(test_loader)
                _data = next(data_iter)
                x, y = _data
                x = x.to(device)

                y_pred = model.generate(x, max_len_response, do_sample=False)[0]
                y_pred = y_pred[max_len_prompt:]
                
                y_ids = [int(i) for i in y_pred]
                y_txt = tokenizer.decode(y_ids)
                print('\n Context \n')
                x = tokenizer.decode([int(i) for i in x.flatten()])
                print(x)
                print('\n Ground truth \n')
                print(y)
                print('\n Prediction: \n')
                print(y_txt)

                with open(log_txt, 'a') as f:
                    f.write('\n Context \n')
                    f.write(x)
                    f.write('\n Ground truth \n')
                    f.write(y[0])
                    f.write('\n Prediction: \n')
                    f.write(y_txt)

            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run(PATH, log_txt)