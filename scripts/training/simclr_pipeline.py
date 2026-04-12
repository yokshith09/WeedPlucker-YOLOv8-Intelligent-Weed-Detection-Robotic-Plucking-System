import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# 1. Dataset & Augmentation for SimCLR
# ==========================================
class SimCLRDataTransform:
    def __init__(self, size=640):
        # Color distortion and other standard SimCLR augmentations
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

class UnlabelledFieldDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG"]:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            return self.transform(image)
        return image, image

# ==========================================
# 2. SimCLR Model Wrapper for YOLOv8
# ==========================================
class SimCLRYOLOBackbone(nn.Module):
    def __init__(self, yolo_pt='best.pt', out_dim=128):
        super().__init__()
        print(f"Loading YOLOv8 model from {yolo_pt}...")
        yolo_instance = YOLO(yolo_pt)
        yolo_model_base = yolo_instance.model # Get the underlying PyTorch model
        
        # The YOLOv8 backbone is layers 0 to 9.
        self.backbone = nn.ModuleList(yolo_model_base.model[:10])
        
        # Freeze the first 5 layers for speed and stability
        for i in range(5):
            for param in self.backbone[i].parameters():
                param.requires_grad = False
        print("Froze first 5 layers of the backbone.")
                
        # Dummy pass to find output channels of layer 9
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            x = dummy_input
            for layer in self.backbone:
                x = layer(x)
        features_dim = x.shape[1] 
        print(f"Backbone features dimension at layer 9: {features_dim}")
        
        # Adaptive pooling to convert spatial features to 1D vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP Projection head (SimCLR standard)
        self.projection_head = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.BatchNorm1d(features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, out_dim)
        )

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        z = self.projection_head(x)
        return z

# ==========================================
# 3. NT-Xent Loss
# ==========================================
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    z = torch.cat([z_i, z_j], dim=0) 
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(mask, -9e15)
    
    labels = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(batch_size)], dim=0).to(z.device)
    return F.cross_entropy(sim_matrix, labels)

# ==========================================
# 4. Training Loop
# ==========================================
def train_simclr(image_dir, yolo_pt='best.pt', batch_size=8, epochs=15, lr=1e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = SimCLRDataTransform(size=640)
    dataset = UnlabelledFieldDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    print(f"Found {len(dataset)} images for pretraining in {image_dir}.")
    if len(dataset) == 0:
        print("Warning: No images found. Please check image_dir.")
        return
        
    model = SimCLRYOLOBackbone(yolo_pt=yolo_pt).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x_i, x_j in pbar:
            x_i, x_j = x_i.to(device), x_j.to(device)
            
            optimizer.zero_grad()
            z_i = model(x_i)
            z_j = model(x_j)
            
            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1}/{epochs} Complete. Average Loss: {total_loss/len(dataloader):.4f}")
    
    torch.save(model.backbone.state_dict(), 'simclr_backbone_weights.pt')
    print("Pretraining complete. Backbone weights saved to 'simclr_backbone_weights.pt'.")

# ==========================================
# 5. Reintegration / Retraining Function
# ==========================================
def inject_weights_and_retrain(original_pt='best.pt', 
                               simclr_weights='simclr_backbone_weights.pt', 
                               data_yaml='dataset.yaml', 
                               epochs=150):
    
    print(f"Reintegrating {simclr_weights} back into {original_pt}...")
    yolo = YOLO(original_pt)
    
    pretrained_backbone_state = torch.load(simclr_weights)
    
    # Update the backbone layers in YOLOv8
    backbone_module = nn.ModuleList(yolo.model.model[:10])
    backbone_module.load_state_dict(pretrained_backbone_state)
    
    for i in range(10):
        yolo.model.model[i] = backbone_module[i]
        
    enhanced_pt = 'best-simclr.pt'
    yolo.save(enhanced_pt)
    print(f"Enhanced model saved to {enhanced_pt}. Commencing fine-tuning...")
    
    enhanced_model = YOLO(enhanced_pt)
    
    # Train using ultralytics API with specific fine-tuning parameters
    enhanced_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        lr0=1e-4, 
        lrf=0.01, 
        optimizer='Adam', 
        patience=20,
        mosaic=1.0,
        copy_paste=0.5, # Specific parameters mentioned in prompt
        flipud=0.3,
        fliplr=0.5,
        workers=4 # Adjust based on CPU setup alongside GPU 
    )
    
    # Automatically update the user's original best.pt and last.pt
    import shutil
    try:
        # Find the latest training run directory
        runs_dir = os.path.join('.', 'runs', 'segment')
        if not os.path.exists(runs_dir):
            runs_dir = os.path.join('.', 'runs', 'detect')
            
        all_runs = sorted(glob.glob(os.path.join(runs_dir, '*')), key=os.path.getmtime)
        if all_runs:
            latest_run = all_runs[-1]
            new_best = os.path.join(latest_run, 'weights', 'best.pt')
            new_last = os.path.join(latest_run, 'weights', 'last.pt')
            
            orig_dir = os.path.dirname(original_pt)
            if not orig_dir: orig_dir = '.'
            
            if os.path.exists(new_best):
                shutil.copy(new_best, os.path.join(orig_dir, 'best.pt'))
            if os.path.exists(new_last):
                shutil.copy(new_last, os.path.join(orig_dir, 'last.pt'))
                
            print(f"✅ Successfully updated {orig_dir}/best.pt and last.pt with the newly trained models from {latest_run}!")
    except Exception as e:
        print(f"Could not auto-copy weights: {e}")

    print("Fine-tuning complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 SimCLR Training Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['pretrain', 'retrain'], help="Run 'pretrain' for SimCLR or 'retrain' for fine-tuning")
    parser.add_argument('--image_dir', type=str, help="Directory of unlabelled raw field images (for pretraining)")
    parser.add_argument('--data_yaml', type=str, help="Path to your dataset.yaml (for retraining)")
    parser.add_argument('--base_weights', type=str, default='best.pt', help="Initial weights")
    parser.add_argument('--simclr_weights', type=str, default='simclr_backbone_weights.pt', help="Path to save/load simclr weights")
    
    args = parser.parse_args()
    
    if args.mode == 'pretrain':
        if not args.image_dir:
            raise ValueError("Must provide --image_dir for pretraining mode")
        train_simclr(image_dir=args.image_dir, yolo_pt=args.base_weights, batch_size=32, epochs=150)
    elif args.mode == 'retrain':
        if not args.data_yaml:
            raise ValueError("Must provide --data_yaml for retrain mode")
        inject_weights_and_retrain(original_pt=args.base_weights, simclr_weights=args.simclr_weights, data_yaml=args.data_yaml)
