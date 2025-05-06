import torch
import argparse
import numpy as np
import json
from torchvision import models, transforms
from PIL import Image

def get_input_args():
    parser = argparse.ArgumentParser(description='Make predictions with a trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)

def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()
    
    image = process_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    probs = torch.exp(output)
    top_probs, top_classes = probs.topk(topk, dim=1)
    return top_probs.squeeze().tolist(), top_classes.squeeze().tolist()

def main():
    args = get_input_args()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(args.image_path, model, args.top_k, device)
    labels = [cat_to_name[str(cls)] for cls in classes]
    
    print("Predicted Classes and Probabilities:")
    for i in range(len(labels)):
        print(f"{labels[i]}: {probs[i]:.4f}")

if __name__ == "__main__":
    main()
