import torch
import torch.nn as nn
from embedding_as_service.text.encode import Encoder

EMB_DIM = 300
SEQ_LEN = 10
MAX_LEN = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Scorer(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super(Scorer, self).__init__()
        self.score = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        return self.score(input)


model = Scorer(EMB_DIM).to(device)
model.load_state_dict(torch.load("model_glove.pt", map_location=device))
model.eval()

en = Encoder(embedding='glove', model='wiki_300', max_seq_length=MAX_LEN)


def predict(titles):
    x = en.encode(texts=titles, pooling='reduce_mean').astype('float32')
    x = torch.from_numpy(x).to(device)
    with torch.no_grad():
        res = model(x)
    res = torch.sigmoid(0.71 * res - 23.05)
    return res.cpu().detach().numpy()
