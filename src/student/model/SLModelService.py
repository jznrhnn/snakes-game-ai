import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request

MODEL_PATH = 'model//model_checkpoint.pth'
DATA_DIM = 7
NUM_LAYERS = 6
DROPOUT_RATE = 0.8


class SLPolicyNetwork(nn.Module):
    def __init__(self, INPUT_DIM=DATA_DIM, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE):
        super(SLPolicyNetwork, self).__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(INPUT_DIM if i == 0 else 64, 64, kernel_size=3,
                       stride=1, padding=1) for i in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64 * 14 * 14, 4)

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


model = SLPolicyNetwork()
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_tensor = torch.tensor(data['input'], dtype=torch.float)
    model.eval()
    output = torch.sort(
        model(input_tensor.reshape(-1, DATA_DIM, 14, 14)), 1, descending=True)
    indices = output.indices.numpy().tolist()
    return jsonify({'indices': indices})


if __name__ == '__main__':
    app.run(debug=True)
