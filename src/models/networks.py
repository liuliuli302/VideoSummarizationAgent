import torch
import torch.nn as nn
import torchvision.models as models

class VisionEncoder(nn.Module):
    def __init__(self, config):
        """
        Input: [Batch, Time, Channel, Height, Width]
        Output: [Batch, Time, EmbedDim]
        """
        super().__init__()
        self.embed_dim = config.get('embed_dim', 512) # ResNet18 default
        self.model_name = config.get('model_name', 'resnet18')
        
        # Load Pretrained ResNet
        print(f"Loading VisionEncoder: {self.model_name}")
        # Note: In a real offline setting, weights might be local.
        # Here we use default weights=None to avoid download errors in some envs, 
        # or weights='DEFAULT' if permitted.
        # Using a small ResNet for efficiency.
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the classification head (fc)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # Determine output dim of backbone (512 for ResNet18)
        backbone_dim = base_model.fc.in_features
        
        # Projector if needed
        if backbone_dim != self.embed_dim:
            self.projector = nn.Linear(backbone_dim, self.embed_dim)
        else:
            self.projector = nn.Identity()

    def forward(self, x):
        """
        x: [Batch, Time, Channel, Height, Width]
        """
        b, t, c, h, w = x.shape
        
        # Reshape for 2D CNN: [B*T, C, H, W]
        x_flat = x.view(b * t, c, h, w)
        
        # Extract features
        features = self.feature_extractor(x_flat) # [B*T, 512, 1, 1]
        features = features.flatten(1)            # [B*T, 512]
        
        # Project
        features = self.projector(features)       # [B*T, EmbedDim]
        
        # Reshape back: [Batch, Time, EmbedDim]
        out = features.view(b, t, -1)
        
        return out

class AgentCore(nn.Module):
    def __init__(self, visual_dim=512, hidden_dim=256, action_space=1, num_layers=2):
        """
        Temporal Modeling Agent
        """
        super().__init__()
        
        # Temporal Encoder (Transformer) or Bi-LSTM
        self.temporal_encoder = nn.LSTM(
            input_size=visual_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Decoder / Policy Head
        # Bi-LSTM output is 2 * hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space) # Output raw logits (e.g. for sigmoid)
        )
        
        print("AgentCore (LSTM-based) initialized.")

    def forward(self, visual_features):
        """
        visual_features: [Batch, Time, Dim]
        return: [Batch, Time, ActionSpace]
        """
        # Temporal Processing
        # h_n, c_n ignored
        feat_seq, _ = self.temporal_encoder(visual_features)
        
        # Decision
        logits = self.head(feat_seq)
        
        return logits
