import os
import argparse
from pathlib import Path
import torch
import json
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_dataloaders(data_dir, img_size=224, batch_size=32, workers=4, val_split=0.2, seed=42):
    """
    Carga un dataset de imágenes y crea dataloaders de train y validación
    incluso si el dataset no está separado en carpetas train/val.
    """
    # Transformaciones
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    # Dataset completo para sacar índices
    full_dataset = ImageFolder(data_dir, transform=None)

    # Split indices
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Crear datasets independientes con sus transformaciones
    train_dataset = ImageFolder(data_dir, transform=train_tf)
    val_dataset   = ImageFolder(data_dir, transform=val_tf)

    # Aplicar los índices del split
    train_ds = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_ds   = torch.utils.data.Subset(val_dataset,   val_indices.indices)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)

    num_classes = len(full_dataset.classes)

    # Guardar mapping de clases
    idx_to_class = {idx: class_name for class_name, idx in full_dataset.class_to_idx.items()}

    if not os.path.exists('class_mapping.json'):
        with open('class_mapping.json', 'w') as f:
            json.dump(idx_to_class, f, indent=4)
        print("Mapeo de clases guardado en 'class_mapping.json'")
    else:
        print("El archivo 'class_mapping.json' ya existe, no se sobrescribe.")


    return train_loader, val_loader, num_classes



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

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y = y.squeeze().long()
        # Use the same forward pass as in the training loop
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, num_classes = get_dataloaders(args.data_dir, args.img_size, args.batch_size, args.workers)
    print("Número de clases:", num_classes)  # Verificar num_classes
    model = build_model(num_classes, args.model_name, pretrained=args.pretrained, emb_dim=args.emb_dim)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb = yb.squeeze().long()
            optimizer.zero_grad()
            logits = model(xb)          # forward simplificado
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
    
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} - loss {running_loss/len(train_loader):.4f} - val_acc {val_acc:.4f}")

        # checkpoint
        ckpt_path = Path(args.checkpoint_dir) / f"vit_epoch{epoch+1}.pt"
        torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, ckpt_path)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Path(args.checkpoint_dir)/"best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=False, default='./data/images/Images')
    parser.add_argument('--img-size', type=int, default=224)
    #parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model-name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint-dir', default='models/checkpoints')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--emb-dim', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
