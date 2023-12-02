import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import subprocess
from threading import Thread
import torch.nn.functional as F


# Constants
SEED = 1234
BATCH_SIZE = 64
MODEL_PATH = 'model//model_checkpoint.pth'
LOG_PATH = 'CNN_logs'
DATA_PATH = 'data//NegaVSNega_7x14x14_501102.csv'
TRAIN_RATIO = 0.85

# Set up seeds and device
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(DATA_PATH)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_reshaped = X.reshape(-1, 7, 14, 14)
X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Dataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(TRAIN_RATIO * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Model Definition
class SLPolicyNetwork(nn.Module):
    def __init__(self, INPUT_DIM=7, num_layers=6, dropout_rate=0.8):
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


# Set INPUT_DIM based on data
INPUT_DIM = len(train_loader.dataset[0][0])

# Model instantiation
model = SLPolicyNetwork(INPUT_DIM)

# Utility Functions


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def categorical_accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        batch_x, batch_y = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        predictions = model(batch_x)

        loss = criterion(predictions, batch_y)

        acc = categorical_accuracy(predictions, batch_y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            batch_x, batch_y = batch[0].to(device), batch[1].to(device)

            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)

            acc = categorical_accuracy(predictions, batch_y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Training Setup
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

# Load Model Function


def load_model():
    if os.path.exists(MODEL_PATH) == False:
        return

    global epoch
    global best_valid_loss
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_valid_loss = checkpoint['valid_loss']


def play():
    command = "java -jar AI-snake.jar student.BasicBot student.ModelRunner"
    result = subprocess.run(
        command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print("process success")
        print(result.stdout)
    else:
        print("process failed")
        print(result.stderr)


def plays():
    # The number of threads to create
    number_of_threads = 1

    # Create the threads
    threads = []
    for i in range(number_of_threads):
        thread = Thread(target=play)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


best_valid_acc = 0
best_valid_loss = float('inf')
epoch = 0
writer = SummaryWriter(LOG_PATH)
# Existing training loop
load_model()
N_EPOCHS = epoch + 20
while epoch < N_EPOCHS:

    start_time = time.time()

    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, val_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)
    writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
    writer.add_scalar('valid_acc', valid_acc, global_step=epoch)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    if valid_loss <= best_valid_loss:
        best_valid_loss = valid_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
        }, MODEL_PATH)
        print(f"Model saved  {valid_loss:.3f}, {valid_acc*100:.2f}")

    epoch += 1

    if train_acc > 0.995:
        break
