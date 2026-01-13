#training the model

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from model import Net
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def train_nn(model, train_loader, val_loader, epochs=5, lr=0.01, output_file='output.txt'):
    with open(output_file, 'w') as f:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        training_loss = []
        validation_loss = []
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.view(inputs.shape[0], -1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Calculate average training loss for the epoch
            avg_train_loss = running_loss / len(train_loader)
            training_loss.append(avg_train_loss)
            print(f'Epoch: {epoch + 1}, Training Loss: {avg_train_loss}')
            f.write(f'Epoch: {epoch + 1}, Training Loss: {avg_train_loss}\n')
            
            #validation
            with torch.no_grad():
                correct = 0
                total = 0
                val_running_loss = 0.0  # New variable for validation loss
                for data in val_loader:
                    inputs, labels = data
                    inputs = inputs.view(inputs.shape[0], -1)
                    outputs = model(inputs)
                    # Calculate validation loss
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_running_loss / len(val_loader)  # Calculate average validation loss
                print(f'Validation Accuracy: {100 * correct / total}%')
                f.write(f'Validation Accuracy: {100 * correct / total}%\n')
                print(f'Epoch: {epoch + 1}, Validation Loss: {avg_val_loss}')
                f.write(f'Epoch: {epoch + 1}, Validation Loss: {avg_val_loss}\n')
                validation_loss.append(avg_val_loss)  # Append the validation loss
        
        print('Finished Training')
        f.write('Finished Training\n')
    return model, training_loss, validation_loss
def get_data_loader(train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    train_batch_size = 32
    val_batch_size = 128
    return DataLoader(dataset, batch_size=train_batch_size, shuffle=True), DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

def plot_loss(training_loss, validation_loss):
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.clf()
    plt.close()

    #plot just the training loss
    plt.plot(training_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.clf()
    plt.close()

    #plot just the validation loss
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('validation_loss.png')
    plt.clf()
    plt.close()

def main(args):
    train_loader, val_loader = get_data_loader()
    final_model, training_loss, validation_loss = train_nn(Net(), train_loader, val_loader, epochs=args.epochs, lr=args.lr, output_file=args.output_file)
    save_model(final_model)
    plot_loss(training_loss, validation_loss)

if __name__ == '__main__':
    # arg parse for epochs and lr and run name
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--run_name', type=str, default='run')
    parser.add_argument('--output_file', type=str, default='output.txt')
    args = parser.parse_args()
    main(args)
