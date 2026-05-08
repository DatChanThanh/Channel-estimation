import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.optim as optim
import scipy.io
import pathlib
import time
import os
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & DATASET
# ==========================================
class RealMIMODataset(Dataset):
    def __init__(self, data_path: pathlib.Path, label_path: pathlib.Path):
        self.data_files = sorted(list(data_path.glob('*.mat')))
        self.label_files = sorted(list(label_path.glob('*.mat')))
        
        if len(self.data_files) != len(self.label_files):
            print(f"⚠️ WARNING: Dataset Mismatch! Data: {len(self.data_files)} vs Label: {len(self.label_files)}")
        
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        try:
            d_mat = scipy.io.loadmat(str(self.data_files[idx]))
            l_mat = scipy.io.loadmat(str(self.label_files[idx]))
            
            d_val = list(d_mat.values())[-1] 
            l_val = list(l_mat.values())[-1]
            
            data = torch.from_numpy(d_val).float()
            label = torch.from_numpy(l_val).float()
            
            if data.ndim == 3:
                data = data.permute(2, 0, 1)   
                label = label.permute(2, 0, 1)
            
            return data, label
        except Exception as e:
            # Fallback to zero tensor in case of corrupted files
            return torch.zeros(8, 612, 14), torch.zeros(8, 612, 14)

# =========================================================================
# 2. DYNAMIC INTERACTION MIXER MODULES
# =========================================================================

def get_pilot_mask(x):
    """Generates a mask: 1 where Pilot exists, 0 otherwise"""
    return (torch.max(torch.abs(x), dim=1, keepdim=True)[0] > 1e-6).float()

class PilotComparativeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False), 
            nn.Sigmoid()
        )
    def forward(self, target, neighbor):
        correlation = target * neighbor 
        features = torch.cat([target, neighbor, correlation], dim=1)
        return self.net(features)

class AdaptiveQuantumWave(nn.Module):
    """
    Adaptive Physics: Generates physical parameters dynamically based on input signal.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Hyper-Controller for parameter generation
        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 6 * dim) 
        )
        
        self.compress = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.Tanh()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        params = self.controller(x).view(B, 6 * C, 1, 1)
        
        # Extract physical parameters
        p_bessel_w    = torch.sigmoid(params[:, 0:C]) * 2.0 + 0.1
        p_bessel_phi  = params[:, C:2*C] * 0.1
        p_chirp_w     = torch.sigmoid(params[:, 2*C:3*C]) * 2.0 + 0.1
        p_chirp_gamma = torch.tanh(params[:, 3*C:4*C]) * 0.2
        p_delta       = torch.sigmoid(params[:, 4*C:5*C])
        p_alpha       = torch.sigmoid(params[:, 5*C:6*C])
        
        # Physics-based kernel computations
        bessel_term = torch.special.bessel_j0(p_bessel_w * x + p_bessel_phi)
        chirp_term = torch.cos((p_chirp_w * x) + (p_chirp_gamma * (x ** 2)))
        x_norm = torch.tanh(x) 
        legendre_term = p_delta * 0.5 * (5 * (x_norm ** 3) - 3 * x_norm)
        
        combined = torch.cat([bessel_term, chirp_term, legendre_term], dim=1)
        return self.compress(combined) * p_alpha

class PAE(nn.Module):
    """Physics-Aware Encoder with dynamic grouping"""
    def __init__(self, dim, num_groups=4):
        super().__init__()
        if num_groups > 1:
            assert dim % num_groups == 0, "Dimension must be divisible by num_groups!"
        
        self.num_groups = num_groups
        self.group_dim = dim // num_groups
        self.comparator = PilotComparativeAttention(self.group_dim)
        self.quantum_transform = AdaptiveQuantumWave(self.group_dim)
        
        self.num_neighbors = num_groups - 1
        if self.num_neighbors > 0:
            self.mix_w = nn.Parameter(torch.randn(num_groups, self.num_neighbors, 1, self.group_dim, 1, 1) * 0.02)

    def forward(self, x):
        if self.num_groups <= 1: return x
        
        groups = x.chunk(self.num_groups, dim=1)
        final_outputs = []

        for i in range(self.num_groups):
            current = groups[i] 
            others = [groups[j] for j in range(self.num_groups) if j != i]
            mask = get_pilot_mask(current) 
            
            # Comparative Heatmap Extraction
            total_heatmap_N = torch.zeros_like(current)
            for neighbor in others:
                n_i = self.comparator(current, neighbor)
                total_heatmap_N = total_heatmap_N + n_i
            total_heatmap_N = total_heatmap_N * mask

            # Dynamic Feature Mixing
            mixed_signal = current
            for idx, neighbor in enumerate(others):
                w = self.mix_w[i, idx]
                mixed_signal = mixed_signal + (w * neighbor)
            
            # Quantum Physical Transformation
            wave_out = self.quantum_transform(mixed_signal)
            
            # Modulation & Residual Addition
            delta = wave_out * total_heatmap_N * mask
            out = current + delta
            final_outputs.append(out)

        return torch.cat(final_outputs, dim=1)

# =========================================================================
# 3. NANO-TRANSFORMER COMPONENTS (OPTIMIZED)
# =========================================================================

class NanoSpectralGating(nn.Module):
    """Spectral Truncation: Learns critical frequency modes for global features"""
    def __init__(self, dim, h=612, w=14, modes_h=32, modes_w=8):
        super().__init__()
        self.dim = dim
        self.h, self.w = h, w
        self.modes_h = min(h, modes_h) 
        self.modes_w = min(w // 2 + 1, modes_w)
        
        self.complex_weight = nn.Parameter(
            torch.randn(dim, self.modes_h, self.modes_w, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        weight = torch.view_as_complex(self.complex_weight)
        x_fft_low = x_fft[:, :, :self.modes_h, :self.modes_w]
        x_fft_low = x_fft_low * weight
        
        out_fft = torch.zeros_like(x_fft)
        out_fft[:, :, :self.modes_h, :self.modes_w] = x_fft_low
        
        x = torch.fft.irfft2(out_fft, s=(H, W), norm='ortho')
        return x

class NanoPhaseRotation(nn.Module):
    """Phase rotation estimation using depthwise convolution"""
    def __init__(self, dim):
        super().__init__()
        self.theta_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, groups=dim), 
            nn.Tanh()
        )

    def forward(self, x):
        theta = self.theta_predictor(x) * 3.14159 
        x_r, x_i = x.chunk(2, dim=1)
        theta_r, theta_i = theta.chunk(2, dim=1)
        cos_t, sin_t = torch.cos(theta_r), torch.sin(theta_r)
        out_r = x_r * cos_t - x_i * sin_t
        out_i = x_r * sin_t + x_i * cos_t
        return torch.cat([out_r, out_i], dim=1)

class DifferentialConv(nn.Module):
    """Local differential feature extraction"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.theta = 0.7

    def forward(self, x):
        out_normal = self.conv(x)
        w = self.conv.weight
        w_sum = w.sum(dim=[2, 3], keepdim=True) - w[:, :, 1:2, 1:2]
        w_sum = w_sum.view(1, self.dim, 1, 1) 
        out_diff = out_normal - x * w_sum
        return (1 - self.theta) * out_normal + self.theta * out_diff

class Nano_TPD_Context(nn.Module):
    """Temporal-Phase-Differential fused context module"""
    def __init__(self, dim):
        super().__init__()
        self.branch_spectral = NanoSpectralGating(dim, modes_h=32)
        self.branch_phase    = NanoPhaseRotation(dim)
        self.branch_diff     = DifferentialConv(dim)
        
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1, groups=4, bias=False)
        self.shuffle = nn.ChannelShuffle(4)

    def forward(self, x):
        x1 = self.branch_spectral(x)
        x2 = self.branch_phase(x)
        x3 = self.branch_diff(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        return self.shuffle(self.fusion(x_cat))

class HIPA(nn.Module):
    """High-Intensity Physics Attention"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.shared_encoder = Nano_TPD_Context(dim)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        N = H * W

        encoded_feat = self.shared_encoder(x)
        q = k = encoded_feat 
        v = self.v_proj(x)
        
        q = q.reshape(B, self.num_heads, self.head_dim, N)
        k = k.reshape(B, self.num_heads, self.head_dim, N)
        v = v.reshape(B, self.num_heads, self.head_dim, N)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (k @ v.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ q)
        
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        return out.permute(0, 2, 3, 1)

class PRF(nn.Module):
    """Physics Refinement Feed-forward network"""
    def __init__(self, dim, expansion_factor=2.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=5, padding=2, 
                                groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def channel_shuffle(self, x, groups=2):
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.project_in(x)
        x = self.channel_shuffle(x, groups=2)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x.permute(0, 2, 3, 1)

class FPTR(nn.Module):
    """Fused Physics Transformer Block"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HIPA(dim, num_heads=num_heads)
        self.gamma1  = nn.Parameter(torch.ones(dim) * 1e-6)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = PRF(dim, expansion_factor=2.0)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x):
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x

class PiPNet(nn.Module):
    """Physics-in-Physics Network for MIMO CSI Feedback/Estimation"""
    def __init__(self, in_chans=8, out_chans=8, num_groups=2, dim=24, depth=3, heads=4):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.GELU()
        )
        self.mixer = PAE(dim=dim, num_groups=num_groups)
        
        self.pos_embed_h = nn.Parameter(torch.zeros(1, 1, 612, dim))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, 14, 1, dim))
        nn.init.trunc_normal_(self.pos_embed_h, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_w, std=0.02)

        self.blocks = nn.ModuleList([FPTR(dim, num_heads=heads) for _ in range(depth)])
        self.tail = nn.Conv2d(dim, out_chans, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.adapter(x)
        x = self.mixer(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.pos_embed_h.transpose(1, 2) + self.pos_embed_w.permute(0, 2, 1, 3)
        for blk in self.blocks: x = blk(x)
        x = x.permute(0, 3, 1, 2)
        return self.tail(x)

# ==========================================
# 4. TRAINING & VALIDATION LOGIC
# ==========================================

def calculate_nmse_linear(pred, target):
    mse = torch.sum((target - pred) ** 2)
    power = torch.sum(target ** 2)
    if power == 0: power = 1e-9
    return 10 * torch.log10(mse / power)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_nmse = 0.0, 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", colour="cyan", ncols=120)
    
    for i, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        nmse = calculate_nmse_linear(outputs, labels).item()
        running_nmse += nmse

        pbar.set_postfix(MSE=f"{running_loss/(i+1):.6f}", NMSE=f"{running_nmse/(i+1):.3f}dB")

    return running_loss / len(dataloader), running_nmse / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_nmse = 0.0, 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation", colour="yellow", ncols=120)
    
    with torch.no_grad():
        for i, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            nmse = calculate_nmse_linear(outputs, labels).item()
            running_nmse += nmse
            pbar.set_postfix(MSE=f"{running_loss/(i+1):.6f}", NMSE=f"{running_nmse/(i+1):.3f}dB")

    return running_loss / len(dataloader), running_nmse / len(dataloader)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device detected: {device}")
    
    # --- Data Path Configuration ---
    train_data_dir  = pathlib.Path(r'D:\LAB_AI\dataset_mimo\2x2\data')
    train_label_dir = pathlib.Path(r'D:\LAB_AI\dataset_mimo\2x2\label')
    val_data_dir    = pathlib.Path(r'D:\LAB_AI\dataset_mimo\test_2x2\test_data_2x2')
    val_label_dir   = pathlib.Path(r'D:\LAB_AI\dataset_mimo\test_2x2\test_label_2x2')

    train_dataset = RealMIMODataset(train_data_dir, train_label_dir)
    val_dataset = RealMIMODataset(val_data_dir, val_label_dir)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"🔢 Total Training Samples: {len(train_dataset)}")

    # --- Model Initialization ---
    model = PiPNet(in_chans=8, out_chans=8, num_groups=2, dim=24, depth=3, heads=4).to(device)
    
