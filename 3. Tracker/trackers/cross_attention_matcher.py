import torch
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class CrossAssociationEngine(nn.Module):
    def __init__(self, feat_dim=2048, pose_dim=34, hidden_dim=256):
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
            batch_first=True  
        )
        
        self.cost_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim//2),
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
        
        tracks['features'] = tracks['features'].squeeze(1) if tracks['features'].dim() == 3 else tracks['features']
        detections['features'] = detections['features'].squeeze(1) if detections['features'].dim() == 3 else detections['features']
        
        tracks['features'] = tracks['features'].to(device)
        tracks['poses'] = tracks['poses'].to(device)
        detections['features'] = detections['features'].to(device)
        detections['poses'] = detections['poses'].to(device)
        
        if cost_vector is not None:
            cost_vector = cost_vector.to(device)

        track_feats = self.feat_proj(tracks['features']) 
        det_feats = self.feat_proj(detections['features'])  
        
        # track_feats = track_feats + self.pose_proj(tracks['poses'])
        track_feats = torch.cat([track_feats, self.pose_proj(tracks['poses'])], dim=1)
        # det_feats = det_feats + self.pose_proj(detections['poses'])
        det_feats = torch.cat([det_feats, self.pose_proj(detections['poses'])], dim=1)
        track_feats = self.concat_proj(track_feats)  # [N, 256]
        det_feats = self.concat_proj(det_feats)      # [N, 256]
        # print(f"Track feats shape: {track_feats.shape}, Det feats shape: {det_feats.shape}")
        query = track_feats.unsqueeze(1).transpose(0, 1)  
        key = value = det_feats.unsqueeze(1).transpose(0, 1)  
        
        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=value,
            need_weights=True
        )
        
        attn_weights = attn_weights.mean(dim=1)  
        
        if cost_vector is not None:
            cost_bias = self.cost_mlp(cost_vector).squeeze(-1) 
            # attn_weights = attn_weights - cost_bias
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
    

# def box_iou(boxes1, boxes2):
#     """Vectorized IoU computation with proper dimension handling"""
#     boxes1 = boxes1[:,:4] if boxes1.shape[-1] > 4 else boxes1
#     boxes2 = boxes2[:,:4] if boxes2.shape[-1] > 4 else boxes2
    
#     area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
#     area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
    
#     lt = torch.max(boxes1[:,None,:2], boxes2[None,:,:2])  
#     rb = torch.min(boxes1[:,None,2:4], boxes2[None,:,2:4])  
    
#     wh = (rb - lt).clamp(min=0)
#     inter = wh[...,0] * wh[...,1]
    
#     return (inter + 1e-7) / (area1[:,None] + area2[None,:] - inter + 1e-7)


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

def shape_similarity_v2(tracks: torch.Tensor, dets: torch.Tensor) -> torch.Tensor:
    n_tracks = tracks.shape[0]
    n_dets = dets.shape[0]

    if n_tracks == 0 or n_dets == 0:
        # Return neutral similarity (all zeros or ones depending on logic)
        return torch.zeros((n_dets, n_tracks), dtype=torch.float32, device=dets.device)

    dw = (dets[:, 2] - dets[:, 0]).view(-1, 1)  # [N_dets, 1]
    dh = (dets[:, 3] - dets[:, 1]).view(-1, 1)
    tw = (tracks[:, 2] - tracks[:, 0]).view(1, -1)  # [1, N_tracks]
    th = (tracks[:, 3] - tracks[:, 1]).view(1, -1)

    w_diff = torch.abs(dw - tw) / torch.maximum(dw, tw)
    h_diff = torch.abs(dh - th) / torch.maximum(dh, th)

    similarity = torch.exp(-(w_diff + h_diff))  # shape: [N_dets, N_tracks]
    return similarity


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
                            torch.tensor(t.x1y1x2y2, dtype=torch.float32) for t in tracks])
    }
    
    det_dict = {
        'features': torch.stack([d.feat if isinstance(d.feat, torch.Tensor) else 
                               torch.tensor(d.feat, dtype=torch.float32) for d in detections]),
        'poses': torch.stack([d.pose if isinstance(d.pose, torch.Tensor) else 
                            torch.tensor(d.pose, dtype=torch.float32) for d in detections]),
        'boxes': torch.stack([d.x1y1x2y2 if isinstance(d.x1y1x2y2, torch.Tensor) else 
                            torch.tensor(d.x1y1x2y2, dtype=torch.float32) for d in detections])
    }
    
    cost_vector = None
    if len(tracks) > 0 and len(detections) > 0:
        cost_vector = compute_costs(
            track_dict['boxes'], 
            det_dict['boxes'],
            track_dict['poses'],
            det_dict['poses']
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