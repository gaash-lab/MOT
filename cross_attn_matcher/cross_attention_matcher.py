import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCrossAssociationEngine(nn.Module):
    def __init__(self, feat_dim=2048, pose_dim=34, hidden_dim=512, num_cost_types=7):
        super().__init__()
        assert hidden_dim % 8 == 0, "hidden_dim must be divisible by num_heads"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_heads = 8
        self.head_dim = hidden_dim // self.num_heads
        self.num_cost_types = num_cost_types
        
        # Feature projections
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )

        # Multi-head attention with larger capacity
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced cost processing network
        self.cost_processor = nn.Sequential(
            nn.Linear(num_cost_types, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output weights for cost modulation
        )

        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # Final matching head
        self.matching_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, tracks, detections, cost_matrix=None):
        """
        Enhanced forward pass with multi-cost processing
        
        Args:
            tracks: Dict with keys 'features', 'poses', 'boxes', 'confidences'
            detections: Dict with same structure
            cost_matrix: (N_tracks, N_dets, num_cost_types)
        Returns:
            attn_weights: (N_tracks, N_dets)
        """
        device = self.device
        
        # Handle dimension mismatches
        tracks['features'] = tracks['features'].squeeze(1) if tracks['features'].dim() == 3 else tracks['features']
        detections['features'] = detections['features'].squeeze(1) if detections['features'].dim() == 3 else detections['features']
        
        # Move to device
        for key in ['features', 'poses', 'boxes']:
            if key in tracks:
                tracks[key] = tracks[key].to(device)
                detections[key] = detections[key].to(device)
        
        if cost_matrix is not None:
            cost_matrix = cost_matrix.to(device)

        # Project features
        track_feats = self.feat_proj(tracks['features'])  # (N_tracks, hidden_dim)
        det_feats = self.feat_proj(detections['features'])  # (N_dets, hidden_dim)
        
        # Project poses
        track_pose_feats = self.pose_proj(tracks['poses'])
        det_pose_feats = self.pose_proj(detections['poses'])
        
        # Fuse appearance and pose features
        track_combined = torch.cat([track_feats, track_pose_feats], dim=1)
        det_combined = torch.cat([det_feats, det_pose_feats], dim=1)
        
        track_fused = self.feature_fusion(track_combined)  # (N_tracks, hidden_dim)
        det_fused = self.feature_fusion(det_combined)      # (N_dets, hidden_dim)
        
        # Cross-attention between tracks and detections
        # Prepare for attention: (batch_size=1, seq_len, hidden_dim)
        query = track_fused.unsqueeze(0)  # (1, N_tracks, hidden_dim)
        key = value = det_fused.unsqueeze(0)  # (1, N_dets, hidden_dim)

        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=value,
            need_weights=True
        )
        
        # Get attention weights: (1, N_tracks, N_dets) -> (N_tracks, N_dets)
        attn_weights = attn_weights.squeeze(0)
        
        # Apply cost-based modulation
        if cost_matrix is not None:
            # Process costs to get modulation weights
            cost_weights = self.cost_processor(cost_matrix)  # (N_tracks, N_dets, 1)
            cost_weights = cost_weights.squeeze(-1)  # (N_tracks, N_dets)
            
            # Modulate attention weights (multiplicative)
            attn_weights = attn_weights * cost_weights
        
        # Apply final matching head for refinement
        # Compute pairwise features for final decision
        track_expanded = track_fused.unsqueeze(1).expand(-1, det_fused.shape[0], -1)  # (N_tracks, N_dets, hidden_dim)
        det_expanded = det_fused.unsqueeze(0).expand(track_fused.shape[0], -1, -1)    # (N_tracks, N_dets, hidden_dim)
        
        # Element-wise product for interaction
        interaction_feats = track_expanded * det_expanded  # (N_tracks, N_dets, hidden_dim)
        
        # Final matching scores
        matching_scores = self.matching_head(interaction_feats).squeeze(-1)  # (N_tracks, N_dets)
        
        # Combine attention weights with matching scores
        final_scores = attn_weights + matching_scores
        
        return final_scores

    def compute_loss(self, attn_weights, gt_matches):
        """Standard focal loss computation"""
        device = attn_weights.device
        targets = torch.zeros_like(attn_weights)
        
        for t_idx, d_idx in gt_matches:
            if t_idx < targets.shape[0] and d_idx < targets.shape[1]:
                targets[t_idx, d_idx] = 1
        
        return self.focal_loss(attn_weights, targets)
    
    def compute_weighted_loss(self, attn_weights, gt_matches, sample_weights):
        """Weighted focal loss with hard negative mining"""
        device = attn_weights.device
        targets = torch.zeros_like(attn_weights)
        
        for t_idx, d_idx in gt_matches:
            if t_idx < targets.shape[0] and d_idx < targets.shape[1]:
                targets[t_idx, d_idx] = 1
        
        return self.weighted_focal_loss(attn_weights, targets, sample_weights)

    def focal_loss(self, preds, targets, alpha=0.25, gamma=2):
        """Standard focal loss"""
        BCE_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (alpha * (1-pt)**gamma * BCE_loss).mean()
    
    def weighted_focal_loss(self, preds, targets, weights, alpha=0.25, gamma=2):
        """Weighted focal loss for hard negative mining"""
        BCE_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_weight = alpha * (1-pt)**gamma
        
        # Apply sample weights
        weighted_loss = focal_weight * BCE_loss * weights
        
        return weighted_loss.sum() / (weights.sum() + 1e-8)

class CrossAssociationEngine(EnhancedCrossAssociationEngine):
    """Backward compatibility wrapper"""
    def __init__(self, feat_dim=2048, pose_dim=34, hidden_dim=256):
        # Use enhanced version with backward compatible defaults
        super().__init__(feat_dim=feat_dim, pose_dim=pose_dim, hidden_dim=max(hidden_dim, 512))