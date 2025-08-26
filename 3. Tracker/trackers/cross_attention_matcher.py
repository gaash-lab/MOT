import torch
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trackers.enhanced_costs import compute_enhanced_costs


def linear_assignment(cost_matrix, match_thr=0.5):
    """Modified to handle 2D cost matrices only"""
    assert cost_matrix.ndim == 2, f"Cost matrix must be 2D, got {cost_matrix.ndim}D"
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    u_tracks = []
    u_dets = []
    for r, c in zip(row_ind, col_ind):
        # print(cost_matrix[r, c])
        if cost_matrix[r, c] <= match_thr:
            matches.append((r, c))
        else:
            u_tracks.append(r)
            u_dets.append(c)
    all_tracks = set(range(cost_matrix.shape[0]))
    all_dets = set(range(cost_matrix.shape[1]))
    matched_tracks = set(r for r, c in matches)
    matched_dets = set(c for r, c in matches)
    u_tracks += list(all_tracks - matched_tracks)
    u_dets += list(all_dets - matched_dets)
    return matches, u_tracks, u_dets



class EnhancedCrossAssociationEngine(nn.Module):
    def __init__(self, feat_dim=2048, pose_dim=34, hidden_dim=512, num_cost_types=2):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be divisible by num_heads"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_heads = 2
        self.head_dim = hidden_dim // self.num_heads
        self.num_cost_types = num_cost_types
        
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

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.cost_processor = nn.Sequential(
            nn.Linear(num_cost_types, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output weights for cost modulation
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
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
            detections: Dict with keys 'features', 'poses', 'boxes', 'confidences'
            cost_matrix: (N_tracks, N_dets, num_cost_types)
        Returns:
            attn_weights: (N_tracks, N_dets)
        """
        device = self.device
        
        tracks['features'] = tracks['features'].squeeze(1) if tracks['features'].dim() == 3 else tracks['features']
        detections['features'] = detections['features'].squeeze(1) if detections['features'].dim() == 3 else detections['features']
        
        for key in ['features', 'poses', 'boxes']:
            if key in tracks:
                tracks[key] = tracks[key].to(device)
                detections[key] = detections[key].to(device)
        
        if cost_matrix is not None:
            cost_matrix = cost_matrix.to(device)

        track_feats = self.feat_proj(tracks['features'])  # (N_tracks, hidden_dim)
        det_feats = self.feat_proj(detections['features'])  # (N_dets, hidden_dim)
        
        track_pose_feats = self.pose_proj(tracks['poses'])
        det_pose_feats = self.pose_proj(detections['poses'])
        
        track_combined = torch.cat([track_feats, track_pose_feats], dim=1)
        det_combined = torch.cat([det_feats, det_pose_feats], dim=1)
        
        track_fused = self.feature_fusion(track_combined)  # (N_tracks, hidden_dim)
        det_fused = self.feature_fusion(det_combined)      # (N_dets, hidden_dim)
        
        query = track_fused.unsqueeze(0)  # (1, N_tracks, hidden_dim)
        key = value = det_fused.unsqueeze(0)  # (1, N_dets, hidden_dim)

        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=value,
            need_weights=True
        )
        
        attn_weights = attn_weights.squeeze(0)
        
        if cost_matrix is not None:
            cost_weights = self.cost_processor(cost_matrix)  # (N_tracks, N_dets, 1)
            cost_weights = cost_weights.squeeze(-1)  # (N_tracks, N_dets)
            
            attn_weights = attn_weights * cost_weights
        
        track_expanded = track_fused.unsqueeze(1).expand(-1, det_fused.shape[0], -1)  # (N_tracks, N_dets, hidden_dim)
        det_expanded = det_fused.unsqueeze(0).expand(track_fused.shape[0], -1, -1)    # (N_tracks, N_dets, hidden_dim)
        
        interaction_feats = track_expanded * det_expanded  # (N_tracks, N_dets, hidden_dim)
        
        matching_scores = self.matching_head(interaction_feats).squeeze(-1)  # (N_tracks, N_dets)
        
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
        super().__init__(feat_dim=feat_dim, pose_dim=pose_dim, hidden_dim=hidden_dim)


def cross_attention_assignment(tracks, detections, matcher, match_thr=0.5):
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
        
    track_dict = {
        'features': torch.stack([t.feat if isinstance(t.feat, torch.Tensor) else 
                               torch.tensor(t.feat, dtype=torch.float32) for t in tracks]),
        'poses': torch.stack([t.pose if isinstance(t.pose, torch.Tensor) else 
                            torch.tensor(t.pose, dtype=torch.float32) for t in tracks]),
        'boxes': torch.stack([t.x1y1x2y2 if isinstance(t.x1y1x2y2, torch.Tensor) else 
                            torch.tensor(t.x1y1x2y2, dtype=torch.float32) for t in tracks]),
        'scores': torch.tensor([t.score if isinstance(t.score, (float, int)) else 0.0 for t in tracks], dtype=torch.float32)
    }
    
    det_dict = {
        'features': torch.stack([d.feat if isinstance(d.feat, torch.Tensor) else 
                               torch.tensor(d.feat, dtype=torch.float32) for d in detections]),
        'poses': torch.stack([d.pose if isinstance(d.pose, torch.Tensor) else 
                            torch.tensor(d.pose, dtype=torch.float32) for d in detections]),
        'boxes': torch.stack([d.x1y1x2y2 if isinstance(d.x1y1x2y2, torch.Tensor) else 
                            torch.tensor(d.x1y1x2y2, dtype=torch.float32) for d in detections]),
        'scores': torch.tensor([d.score if isinstance(d.score, (float, int)) else 0.0 for d in detections], dtype=torch.float32)
    }
    
    cost_vector = None
    if len(tracks) > 0 and len(detections) > 0:
        cost_vector = compute_enhanced_costs(
            track_dict,
            det_dict
        )
    
    with torch.no_grad():
        attn_weights = matcher(track_dict, det_dict, cost_vector)
    if attn_weights.numel() > 0:
        attn_probs = torch.sigmoid(attn_weights)  # or F.softmax(attn_weights, dim=1)
        cost_matrix = 1.0 - attn_probs.cpu().numpy()
        return linear_assignment(cost_matrix, match_thr)
    else:
        return [], list(range(len(tracks))), list(range(len(detections)))