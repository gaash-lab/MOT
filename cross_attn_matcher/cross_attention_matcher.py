import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAssociationEngine(nn.Module):
    def __init__(self, feat_dim=2048, pose_dim=34, hidden_dim=512):
        super().__init__()
        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by num_heads"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True  # Important for correct shape handling
        )
        
        self.cost_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

        self.concat_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, tracks, detections, cost_vector=None):
        """
        Args:
            tracks: Dict with keys:
                'features': (N_tracks, feat_dim)
                'poses': (N_tracks, pose_dim)
            detections: Dict with same structure
            cost_vector: Optional (N_tracks, N_dets, 2)
        Returns:
            attn_weights: (N_tracks, N_dets)
        """
        device = self.device
        
        # Ensure proper input shapes
        tracks['features'] = tracks['features'].squeeze(1) if tracks['features'].dim() == 3 else tracks['features']
        detections['features'] = detections['features'].squeeze(1) if detections['features'].dim() == 3 else detections['features']
        
        # Move to device
        tracks['features'] = tracks['features'].to(device)
        tracks['poses'] = tracks['poses'].to(device)
        detections['features'] = detections['features'].to(device)
        detections['poses'] = detections['poses'].to(device)
        
        if cost_vector is not None:
            cost_vector = cost_vector.to(device)

        track_feats = self.feat_proj(tracks['features'])  # (N_tracks, hidden_dim)
        det_feats = self.feat_proj(detections['features'])  # (N_dets, hidden_dim)
        
        track_feats = torch.cat([track_feats, self.pose_proj(tracks['poses'])], dim=1)
        det_feats = torch.cat([det_feats, self.pose_proj(detections['poses'])], dim=1)
        track_feats = self.concat_proj(track_feats)  # [N, 256]
        det_feats = self.concat_proj(det_feats)      # [N, 256]
        
        query = track_feats.unsqueeze(1).transpose(0, 1)  # (1, N_tracks, hidden_dim)
        key = value = det_feats.unsqueeze(1).transpose(0, 1)  # (1, N_dets, hidden_dim)

        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=value,
            need_weights=True
        )
        
        attn_weights = attn_weights.mean(dim=1)  # (N_tracks, N_dets)
        
        # Apply cost bias if provided
        if cost_vector is not None:
            cost_bias = self.cost_mlp(cost_vector).squeeze(-1)  # (N_tracks, N_dets)
            attn_weights = attn_weights * (1.0 - torch.sigmoid(cost_bias))
        return attn_weights

    def compute_loss(self, attn_weights, gt_matches):
        """Modified to handle variable batch sizes"""
        device = attn_weights.device
        targets = torch.zeros_like(attn_weights)
        
        for t_idx, d_idx in gt_matches:
            if t_idx < targets.shape[0] and d_idx < targets.shape[1]:
                targets[t_idx, d_idx] = 1
        
        return self.focal_loss(attn_weights, targets)

    def focal_loss(self, preds, targets, alpha=0.25, gamma=2):
        BCE_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (alpha * (1-pt)**gamma * BCE_loss).mean()