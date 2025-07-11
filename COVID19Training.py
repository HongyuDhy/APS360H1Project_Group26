import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import Network
import CustomNetwork
import LoadData
import os
import argparse
import matplotlib.pyplot as plt
import time
from pathlib import Path

torch.manual_seed(21) # for reproducible results

def main():

    start_time = time.time()

    # get devide: cuda > cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    parser = argparse.ArgumentParser(description='APS360 Project')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers') # change to 0 if cpu doesn't support
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--scheduler', type=float, default=0.8, help='Scheduler changing rate')
    parser.add_argument('--model', type=str, default='custom_vgg', help='Model name')
    # Transform (Data Augmentation) parameters
    parser.add_argument('--resize', type=int, default=224, help='Resize')
    parser.add_argument('--noise', action='store_true', help='Add Gaussian Noise')
    parser.add_argument('--norm', action='store_true', help='Normalization')
    parser.add_argument('--norm_mean', type=float, default=[0.485, 0.456, 0.406], nargs='+', help='Normalization mean')
    parser.add_argument('--norm_std', type=float, default=[0.229, 0.224, 0.225], nargs='+', help='Normalization std')
    parser.add_argument('--rotation_degree', type=int, default=10, help='Rotation degree')
    parser.add_argument('--brightness', type=float, default=0.1, help='Brightness')
    parser.add_argument('--contrast', type=float, default=0.15, help='Contrast')
    parser.add_argument('--noise_mean', type=float, default=0, help='Noise mean')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Noise std')
    args = parser.parse_args(['--noise', '--norm'])

    # Network dictionary
    net_dict = {
        #'pretrained_vgg16': Network.pretrained_vgg16,
        #'pretrained_resnet50': Network.pretrained_resnet50,
        #'pretrained_inceptionv3': Network.pretrained_inceptionv3,
        'custom_resnet': CustomNetwork.custom_resnet,
        'custom_vgg': CustomNetwork.custom_vgg,
        'baseline': CustomNetwork.baseline
    }

    if args.model == 'baseline':
        net = Network.baseline(size=args.resize).to(device)
    else:
        net_class = net_dict[args.model]
        net = net_class().to(device)

    # Data Augmentation include Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, Add Gaussian Noise, Normalize
    transform = LoadData.transform(resize=args.resize, rotation_degree=args.rotation_degree,
                                   brightness=args.brightness,
                                   contrast=args.contrast, noise=args.noise, noise_mean=args.noise_mean,
                                   noise_std=args.noise_std, norm=args.norm)

    models_path = rf'D:\PythonProject\COVID19\models\{args.model}'  # best model saving path
    best_model_path = f'Best_Model_{args.model}.pth'  # best model saving name
    best_model_saving_path = Path(models_path) / best_model_path  # this is for test to use the best model

    # create folders for saving
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    # get batch loader
    train_loader, val_loader, test_loader, weights = LoadData.get_loader(args.batch_size, transform, num_workers=args.num_workers) # choose appropriate num_workers for cpu, default num_workers=0

    # criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.scheduler) # change lr to 0.8x every 5 epoches


    # from here, carefully change anything
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0.0

    for epoch in range(args.epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # training
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += pred.eq(labels.data).cpu().sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4%}")

        # validation
        net.eval()
        sum_loss = 0
        sum_correct = 0
        total_val = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                _, pred = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct = pred.eq(labels.data).cpu().sum()
                sum_loss += loss.item()
                sum_correct += correct.item()

        val_loss = sum_loss / len(val_loader)
        val_accuracy = sum_correct / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4%}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(net.state_dict(), f'{models_path}/{best_model_path}')
            print(f'Best model saved with validation accuracy: {best_accuracy:.2%}')

    print(f'Best Validation Accuracy: {best_accuracy:.2%}')

    # test
    net.eval()
    sum_loss_test = 0
    sum_correct_test = 0
    total_test = 0
    net.load_state_dict(torch.load(best_model_saving_path, map_location=device))
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct = pred.eq(labels.data).cpu().sum()
            sum_loss_test += loss.item()
            sum_correct_test += correct.item()

        test_loss = sum_loss_test / len(test_loader)
        test_accuracy = sum_correct_test / total_test

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4%}')

        # plot the results
        epoch_list = [int(i+1) for i in range(args.epochs)]
        plt.figure(figsize=(12, 5))

        # loss curve
        plt.subplot(1, 2, 1)
        plt.plot(epoch_list, train_losses, label='Training Loss')
        plt.plot(epoch_list, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epoch')
        plt.legend()
        plt.grid(True)

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epoch_list, train_accuracies, label='Training Accuracy')
        plt.plot(epoch_list, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{models_path}/loss_accuracy_{args.model}.png')
        plt.show()

        end_time = time.time()
        print(f'Total time: {end_time - start_time:.2f}')

if __name__ == "__main__":
        main()


