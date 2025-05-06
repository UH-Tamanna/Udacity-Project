import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import models

# Argument parser
def get_input_args():
    parser = argparse.ArgumentParser(description='Train an Image Classifier')
    parser.add_argument('--data_dir', type=str, default='flowers', help='Path to dataset')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet50', 'alexnet'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

# Load data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    
    return train_loader, valid_loader, train_dataset

# Build model
def build_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        input_size = 2048
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    if arch == 'resnet50':
        model.fc = classifier
    else:
        model.classifier = classifier
    
    return model

# Train model
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {running_loss/len(train_loader):.3f}")

# Save checkpoint
def save_checkpoint(model, train_dataset, arch):
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx
    }
    torch.save(checkpoint, 'checkpoint.pth')
    print("Model checkpoint saved.")

# Main function
def main():
    args = get_input_args()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader, train_dataset = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    train_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)
    save_checkpoint(model, train_dataset, args.arch)

if __name__ == '__main__':
    main()
