import torch
import torch.nn as nn
import numpy as np
from embedding_as_service.text.encode import Encoder

EMB_DIM = 300
MAX_LEN = 32  # Максимальное количество слов, из которого может состоять название должности (оставим запас)

en = Encoder(embedding='fasttext', model='wiki_news_300_sub', max_seq_length=MAX_LEN)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("model.pt")
model.eval()


class Scorer(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super(Scorer, self).__init__()
        self.score = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        return self.score(input)


def predict(x, scorer=model):
    x = en.encode(texts=x, pooling='reduce_mean')
    x = np.float32(x)
    x = torch.from_numpy(x).to(device)
    score = scorer(x).cpu().detach().numpy()
    return score
