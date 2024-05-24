import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import itertools

from dataset import Shakespeare
from model import CharRNN, CharLSTM

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0

    for input_seq, target_seq in trn_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(input_seq.size(0))
        if isinstance(hidden, tuple):  # For LSTM
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:  # For RNN
            hidden = hidden.to(device)
        output, hidden = model(input_seq, hidden)
        loss = criterion(output.view(-1, model.output_size), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(output, dim=2)
        total_correct += (predicted == target_seq).sum().item()
        total_count += target_seq.numel()

    trn_loss = total_loss / len(trn_loader)
    trn_accuracy = total_correct / total_count
    return trn_loss, trn_accuracy

def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            hidden = model.init_hidden(input_seq.size(0))
            if isinstance(hidden, tuple):  # For LSTM
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:  # For RNN
                hidden = hidden.to(device)
            output, hidden = model(input_seq, hidden)
            loss = criterion(output.view(-1, model.output_size), target_seq.view(-1))
            total_loss += loss.item()

            _, predicted = torch.max(output, dim=2)
            total_correct += (predicted == target_seq).sum().item()
            total_count += target_seq.numel()

    val_loss = total_loss / len(val_loader)
    val_accuracy = total_correct / total_count
    return val_loss, val_accuracy

def plot_metrics(train_metrics, val_metrics, model_type, param_set, metric_name):
    plt.figure()
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{model_type} {metric_name}\n{param_set}')
    plt.legend()
    file_path='C:/Users/Islab/Desktop/islab_code/Language_Modeling/Train_loss_png/'
    filename = f"{model_type}_{metric_name}_{param_set}.png".replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
    plt.savefig(file_path+filename)
    plt.close()

def main():
    dataset = Shakespeare('shakespeare_train.txt')

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(0.8 * dataset_size)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    #파라매터 설정
    param_grid = {
        'batch_size': [32, 64],
        'hidden_size': [128, 256],
        'n_layers': [2, 3],
        'n_epochs': [20, 30, 50],
        'dropout': [0.2, 0.5],
    }

    
    param_combinations = list(itertools.product(*param_grid.values()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    best_rnn_val_loss = float('inf')
    best_lstm_val_loss = float('inf')
    best_rnn_params = None
    best_lstm_params = None

    for params in param_combinations:
        batch_size, hidden_size, n_layers, n_epochs, dropout = params

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        for model_type in ['RNN', 'LSTM']:
            if model_type == 'RNN':
                model = CharRNN(input_size=len(dataset.chars), hidden_size=hidden_size, output_size=len(dataset.chars), n_layers=n_layers, dropout=dropout).to(device)
                model_path = 'best_rnn_model.pth'
                best_val_loss = best_rnn_val_loss
                best_params = best_rnn_params
            else:
                model = CharLSTM(input_size=len(dataset.chars), hidden_size=hidden_size, output_size=len(dataset.chars), n_layers=n_layers, dropout=dropout).to(device)
                model_path = 'best_lstm_model.pth'
                best_val_loss = best_lstm_val_loss
                best_params = best_lstm_params

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            print(f"Training {model_type} with batch_size={batch_size}, hidden_size={hidden_size}, n_layers={n_layers}, n_epochs={n_epochs}, dropout={dropout}")
            
            for epoch in range(n_epochs):
                trn_loss, trn_accuracy = train(model, train_loader, device, criterion, optimizer)
                val_loss, val_accuracy = validate(model, val_loader, device, criterion)
                scheduler.step()
                train_losses.append(trn_loss)
                val_losses.append(val_loss)
                train_accuracies.append(trn_accuracy)
                val_accuracies.append(val_accuracy)
                print(f'{model_type} Epoch {epoch+1}/{n_epochs}, Train Loss: {trn_loss}, Validation Loss: {val_loss}, Train Accuracy: {trn_accuracy}, Validation Accuracy: {val_accuracy}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                torch.save(model.state_dict(), model_path)
                print(f"New best model: {model_type} with batch_size={batch_size}, hidden_size={hidden_size}, n_layers={n_layers}, n_epochs={n_epochs}, dropout={dropout}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

            plot_metrics(train_losses, val_losses, model_type, params, 'Loss')
            plot_metrics(train_accuracies, val_accuracies, model_type, params, 'Accuracy')

            if model_type == 'RNN':
                best_rnn_val_loss = best_val_loss
                best_rnn_params = best_params
            else:
                best_lstm_val_loss = best_val_loss
                best_lstm_params = best_params

    print(f"Best RNN model with params: batch_size={best_rnn_params[0]}, hidden_size={best_rnn_params[1]}, n_layers={best_rnn_params[2]}, n_epochs={best_rnn_params[3]}, dropout={best_rnn_params[4]}, Validation Loss: {best_rnn_val_loss}")
    print(f"Best LSTM model with params: batch_size={best_lstm_params[0]}, hidden_size={best_lstm_params[1]}, n_layers={best_lstm_params[2]}, n_epochs={best_lstm_params[3]}, dropout={best_lstm_params[4]}, Validation Loss: {best_lstm_val_loss}")

if __name__ == '__main__':
    main()
