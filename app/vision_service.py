import torch 
import torch.nn as nn 
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

class VisionService:
    def __init__(self, model_path, model_name='vit_base_patch16_224', num_classes=120, emb_dim=512, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Crear el modelo con la arquitectura correcta (igual que en el entrenamiento)
        self.model = timm.create_model(model_name, pretrained=False)
        in_feats = self.model.head.in_features
        
        new_classifier = nn.Sequential(
            nn.Linear(in_feats, emb_dim), 
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )
        self.model.head = new_classifier
        
        # 2. Cargar state_dict (sin cambios)
        sd = torch.load(model_path, map_location=self.device)
        if isinstance(sd, dict) and 'model_state' in sd:
            self.model.load_state_dict(sd['model_state'])
        else:
            self.model.load_state_dict(sd)
        
        # 3. Mover al dispositivo y modo evaluación
        self.model.to(self.device)
        self.model.eval()
        
        # 4. Definir transformaciones (sin cambios)
        self.transform = transforms.Compose([
            transforms.Resize(246),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

    def predict(self, pil_image: Image.Image):
        x = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # El forward pass ya está integrado en el modelo.
            # No necesitas llamar a forward_features, embed_proj, etc.
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(logits.argmax(dim=1).cpu().numpy()[0])
            
        return pred_idx, probs

