import torch
from pathlib import Path
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from scripts.train_vit import get_dataloaders
import argparse
import os


def build_model(num_classes, model_name='vit_base_patch16_224', pretrained=True, emb_dim=512):
    # Cargar el modelo base
    model = timm.create_model(model_name, pretrained=pretrained)

    # Obtener el número de características del clasificador original
    # ViT tiene el clasificador en `model.head`
    in_feats = model.head.in_features 

    # Crear una nueva capa clasificadora que incluye toda tu lógica personalizada
    new_classifier = nn.Sequential(
        # La primera capa toma el token CLS
        nn.Linear(in_feats, emb_dim), 
        nn.ReLU(),
        nn.LayerNorm(emb_dim),
        # La capa final predice las clases
        nn.Linear(emb_dim, num_classes)
    )

    # Reemplazar la capa clasificadora original del modelo
    model.head = new_classifier

    return model

# Definir evaluate
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y = y.squeeze().long()
            #print("Forma de y:", y.shape, "Tipo de y:", y.dtype, "Valores de y:", y.min(), y.max())
            logits = model(x)
            #print("Forma de logits:", logits.shape)
            preds = logits.argmax(dim=1)
            #print("Forma de preds:", preds.shape)
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def EvaluateModel(args):

    # Crear el modelo
    num_classes = 120  # Ajusta según tu dataset
    model = build_model(args.num_classes, model_name=args.model_name, pretrained=args.pretrained, emb_dim=args.emb_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar los pesos
    checkpoint_path = Path(args.checkpoint_dir) / "best_model.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)  # Usa map_location si es necesario
    model.load_state_dict(state_dict)

    # Mover al dispositivo

    model = model.to(device)

    # Crear DataLoader
    train_loader, val_loader, num_classes = get_dataloaders(args.data_dir, args.img_size, args.batch_size, args.workers)

    # Evaluar
    accuracy = evaluate(model, val_loader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=False, default='./data/images/Images')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint-dir', default='models/checkpoints')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--emb-dim', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=120)
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    EvaluateModel(args)