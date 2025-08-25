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
    

def box_iou(boxes1, boxes2, buffer=5):
    """
    Computes Buffered IoU between two sets of boxes.
    boxes1, boxes2: (N, 4) and (M, 4) in [x1, y1, x2, y2] format
    buffer: pixels to expand each box (applied only during intersection computation)
    """
    # Strip extra dimensions if present
    boxes1 = boxes1[:, :4] if boxes1.shape[-1] > 4 else boxes1
    boxes2 = boxes2[:, :4] if boxes2.shape[-1] > 4 else boxes2

    # Compute areas (original, no buffer)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Expand boxes by buffer (for intersection only)
    boxes1_exp = boxes1.clone()
    boxes2_exp = boxes2.clone()
    boxes1_exp[:, 0] -= buffer
    boxes1_exp[:, 1] -= buffer
    boxes1_exp[:, 2] += buffer
    boxes1_exp[:, 3] += buffer
    boxes2_exp[:, 0] -= buffer
    boxes2_exp[:, 1] -= buffer
    boxes2_exp[:, 2] += buffer
    boxes2_exp[:, 3] += buffer

    # Compute intersection from expanded boxes
    lt = torch.max(boxes1_exp[:, None, :2], boxes2_exp[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1_exp[:, None, 2:], boxes2_exp[None, :, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]  # (N, M)

    # Union uses original area
    union = area1[:, None] + area2[None, :] - inter

    return (inter + 1e-7) / (union + 1e-7)

def compute_oks(poses1, poses2, kappa=0.05):
    """Robust OKS computation for (17,2) pose format"""
    kps1 = poses1.view(-1, 17, 2)  
    kps2 = poses2.view(-1, 17, 2) 
    
    valid1 = (kps1.abs().sum(dim=-1) > 1e-3)  
    valid2 = (kps2.abs().sum(dim=-1) > 1e-3)  
    
    def get_scale(kps, valid):
        scales = []
        for i in range(len(kps)):
            vis_kps = kps[i][valid[i]]  
            if len(vis_kps) == 0:
                scales.append(0.0)
                continue
            scale = (vis_kps.max(dim=0)[0] - vis_kps.min(dim=0)[0]).prod()
            scales.append(scale)
        return torch.tensor(scales, device=kps1.device) 
    
    scale1 = get_scale(kps1, valid1) 
    scale2 = get_scale(kps2, valid2)  
    scale = (scale1[:,None] + scale2[None,:]) / 2  
    
    diff = kps1[:,None,:,:] - kps2[None,:,:,:]  
    sq_dist = (diff ** 2).sum(dim=-1)  
    
    oks_kpts = torch.exp(-sq_dist / (2 * scale[...,None] * kappa**2 + 1e-7))  
    
    valid_mask = valid1[:,None,:] & valid2[None,:,:]  
    oks_kpts = oks_kpts * valid_mask.float()
    
    valid_count = valid_mask.sum(dim=-1) 
    oks = oks_kpts.sum(dim=-1) / (valid_count + 1e-7)
    
    oks[valid_count == 0] = 0.0
    
    return oks


def compute_costs(track_boxes, det_boxes, track_poses, det_poses):
    """Compute combined spatial and shape costs"""
    with torch.no_grad():
        ious = box_iou(track_boxes, det_boxes)
        spatial_costs = 1.0 - ious
        
        shape_costs = 1.0 - compute_oks(track_poses, det_poses)
        
        return torch.stack([spatial_costs, shape_costs], dim=-1)



def cross_attention_assignment(tracks, detections, matcher, match_thr=0.5):
    """
    Robust cross-attention assignment with empty tensor handling
    """
    # Handle empty cases
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
        
    # Prepare inputs with shape checking
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
        # print(f"cost vector: {cost_vector}")
    
    with torch.no_grad():
        attn_weights = matcher(track_dict, det_dict, cost_vector)
    # print(f"Attention weights  {attn_weights}")
    # Apply Hungarian algorithm
    if attn_weights.numel() > 0:
        # print(attn_weights.numel())
        attn_probs = torch.sigmoid(attn_weights)  # or F.softmax(attn_weights, dim=1)
        cost_matrix = 1.0 - attn_probs.cpu().numpy()
        # cost_matrix = 1.0 - attn_weights.cpu().numpy()
        return linear_assignment(cost_matrix, match_thr)
    else:
        return [], list(range(len(tracks))), list(range(len(detections)))