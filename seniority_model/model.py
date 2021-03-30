import torch
import torch.nn as nn
import spacy

EMB_DIM = 96
nlp = spacy.load("en_core_web_sm")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Scorer(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super(Scorer, self).__init__()
        self.score = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input):
        return self.score(input)


model = Scorer(EMB_DIM).to(device)
model.load_state_dict(torch.load("model_spacy.pt", map_location=device))
model.eval()


def predict(title):
    x = nlp(title).vector
    x = torch.from_numpy(x).to(device)
    with torch.no_grad():
        res = torch.sigmoid(2.05 * model(x)-19.15).cpu().detach().numpy()
    return float(res)
