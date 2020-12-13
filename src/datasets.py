import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def create_minibatch(samples):
    if len(samples[0]) == 2:
        return (
            {
                k: pad_sequence([torch.as_tensor(s[0][k]) for s in samples], batch_first=True)
                for k in samples[0][0]
            },
            torch.as_tensor([s[1] for s in samples], dtype=torch.float),
        )
    else:
        return {
            k: pad_sequence([torch.as_tensor(s[k]) for s in samples], batch_first=True)
            for k in samples[0]
        }


class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, has_label=True) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.has_label = has_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.has_label:
            text, label = self.data[idx]
            return self.tokenizer(text), label
        else:
            text = self.data[idx]
            return self.tokenizer(text)
