import torch
import torch.nn as nn
import torchvision
from tree_embedding import get_embedding


class ResNetEncoder(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNetEncoder, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)
        self.embedding_node = torch.nn.Embedding(1000, 128)

        
    def forward(self, x):
        out = self.resnet(x)
        return out

    
class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        # resnet encoder
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.embedding_node = torch.nn.Embedding(1000, 256)
        self.embedding_relation = torch.nn.Embedding(1000, 256)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_1 = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru_2 = torch.nn.Gru(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.p_r_embedding = get_embedding() # torch.Size([286, 30, 512])
        self.W = nn.Linear(512, 512)
        self.U = nn.Linear(512, 512)
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru_1(x, h0)

        out = self.fc(out[:, -1, :])
        return out

class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.fc_query = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.fc_attention = nn.Linear(params['dim_attention'], 1)

    def forward(self, ctx_val, ctx_key, ctx_mask, ht_query):

        attention_past = torch.zeros(B, 1, H, W).cuda()
        self.fc_query = nn.Linear(params['n'], params['dim_attention'], bias=False)
        ht_query = self.fc_query(ht_query)

        attention_score = torch.tanh(ctx_key + ht_query[:, None, None, :])
        attention_score = self.fc_attention(attention_score).squeeze(3)

        attention_score = attention_score - attention_score.max()
        attention_score = torch.exp(attention_score) * ctx_mask
        attention_score = attention_score / (attention_score.sum(2).sum(1)[:, None, None] + 1e-10)

        ct = (ctx_val * attention_score[:, None, :, :]).sum(3).sum(2)

        return ct, attention_score
# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 1

model = GRUModel(input_size, hidden_size, num_layers, output_size)
print(model)
