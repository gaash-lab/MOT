import torch
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def linear_assignment(cost_matrix, match_thr=0.3):
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
    

# class CrossAssociationEngine(nn.Module):
#     def __init__(self, feat_dim=2048, pose_dim=34, hidden_dim=512, ablation_mode="features_only"):
#         super().__init__()
#         assert hidden_dim % 4 == 0, "hidden_dim must be divisible by num_heads"
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.num_heads = 4
#         self.head_dim = hidden_dim // self.num_heads
#         self.ablation_mode = ablation_mode  # 'iou_only', 'pose_only', 'features_only', or None for full
        
#         # For all modes except iou_only
#         if ablation_mode != 'iou_only':
#             self.feat_proj = nn.Sequential(
#                 nn.Linear(feat_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.LayerNorm(hidden_dim)
#             )
#             self.pose_proj = nn.Sequential(
#                 nn.Linear(pose_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.LayerNorm(hidden_dim)
#             )

#             self.attention = nn.MultiheadAttention(
#                 embed_dim=hidden_dim,
#                 num_heads=self.num_heads,
#                 dropout=0.1,
#                 batch_first=True  # Important for correct shape handling
#             )
            
#             self.concat_proj = nn.Sequential(
#                 nn.Linear(hidden_dim * 2, hidden_dim),
#                 nn.ReLU(),
#                 nn.LayerNorm(hidden_dim)
#             )

#         # For all modes
#         self.cost_mlp = nn.Sequential(
#             nn.Linear(1, hidden_dim//2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim//2, 1)
#         )

#     def forward(self, tracks, detections, cost_vector=None):
#         """
#         Args:
#             tracks: Dict with keys:
#                 'features': (N_tracks, feat_dim)
#                 'poses': (N_tracks, pose_dim)
#             detections: Dict with same structure
#             cost_vector: Optional (N_tracks, N_dets, 2)
#         Returns:
#             attn_weights: (N_tracks, N_dets)
#         """
#         device = self.device
        
#         # Ensure proper input shapes
#         tracks['features'] = tracks['features'].squeeze(1) if tracks['features'].dim() == 3 else tracks['features']
#         detections['features'] = detections['features'].squeeze(1) if detections['features'].dim() == 3 else detections['features']
        
#         # Move to device
#         tracks['features'] = tracks['features'].to(device)
#         tracks['poses'] = tracks['poses'].to(device)
#         detections['features'] = detections['features'].to(device)
#         detections['poses'] = detections['poses'].to(device)
        
#         if cost_vector is not None:
#             cost_vector = cost_vector.to(device)

#         # IOU-ONLY ABLATION
#         if self.ablation_mode == 'iou_only':
#             if cost_vector is None:
#                 raise ValueError("cost_vector must be provided for IoU-only ablation")
            
#             # Process only IoU through MLP to get attention weights
#             attn_weights = self.cost_mlp(cost_vector).squeeze(-1)  # (N_tracks, N_dets)
            
#             # Convert IoU to attention weights (higher IoU = higher attention)
#             attn_weights = torch.sigmoid(attn_weights)
            
#             return attn_weights
        
#         # POSE-ONLY ABLATION
#         elif self.ablation_mode == 'pose_only':
#             # Use only pose information
#             track_feats = self.pose_proj(tracks['poses'])  # (N_tracks, hidden_dim)
#             det_feats = self.pose_proj(detections['poses'])  # (N_dets, hidden_dim)
            
#             # No concatenation needed for pose-only
            
#         # FEATURES-ONLY ABLATION
#         elif self.ablation_mode == 'features_only':
#             # Use only feature information
#             track_feats = self.feat_proj(tracks['features'])  # (N_tracks, hidden_dim)
#             det_feats = self.feat_proj(detections['features'])  # (N_dets, hidden_dim)
            
#             # No concatenation needed for features-only
            
#         # FULL MODEL (default)
#         else:
#             # Use both features and poses (original implementation)
#             track_feats = self.feat_proj(tracks['features'])  # (N_tracks, hidden_dim)
#             det_feats = self.feat_proj(detections['features'])  # (N_dets, hidden_dim)
            
#             track_feats = torch.cat([track_feats, self.pose_proj(tracks['poses'])], dim=1)
#             det_feats = torch.cat([det_feats, self.pose_proj(detections['poses'])], dim=1)
#             track_feats = self.concat_proj(track_feats)  # [N, 256]
#             det_feats = self.concat_proj(det_feats)      # [N, 256]
        
#         # For all modes except iou_only, apply attention
#         query = track_feats.unsqueeze(1).transpose(0, 1)  # (1, N_tracks, hidden_dim)
#         key = value = det_feats.unsqueeze(1).transpose(0, 1)  # (1, N_dets, hidden_dim)

#         attn_output, attn_weights = self.attention(
#             query=query,
#             key=key,
#             value=value,
#             need_weights=True
#         )
        
#         attn_weights = attn_weights.mean(dim=1)  # (N_tracks, N_dets)
        
#         # Apply cost bias if provided (for all modes)
#         if cost_vector is not None:
#             cost_bias = self.cost_mlp(cost_vector).squeeze(-1)  # (N_tracks, N_dets)
#             attn_weights = attn_weights * (1.0 - torch.sigmoid(cost_bias))
            
#         return attn_weights

#     def compute_loss(self, attn_weights, gt_matches):
#         """Modified to handle variable batch sizes"""
#         device = attn_weights.device
#         targets = torch.zeros_like(attn_weights)
        
#         for t_idx, d_idx in gt_matches:
#             if t_idx < targets.shape[0] and d_idx < targets.shape[1]:
#                 targets[t_idx, d_idx] = 1
        
#         return self.focal_loss(attn_weights, targets)

#     def focal_loss(self, preds, targets, alpha=0.25, gamma=2):
#         BCE_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         return (alpha * (1-pt)**gamma * BCE_loss).mean()


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
    SCALE_FACTOR = 1000.0
    poses1 = poses1 * SCALE_FACTOR  
    poses2 = poses2 * SCALE_FACTOR
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


def cos_distance(track_feats, det_feats):
    """Cosine distance between track and detection features"""
    # Ensure inputs are numpy arrays
    track_feats = np.asarray(track_feats)
    det_feats   = np.asarray(det_feats)

    # Squeeze extra dimensions if present (e.g., (N,1,D) -> (N,D))
    if track_feats.ndim == 3 and track_feats.shape[1] == 1:
        track_feats = track_feats.squeeze(1)
    if det_feats.ndim == 3 and det_feats.shape[1] == 1:
        det_feats = det_feats.squeeze(1)

    if track_feats.shape[0] == 0 or det_feats.shape[0] == 0:
        return np.ones((track_feats.shape[0], det_feats.shape[0]), dtype=np.float64)

    # Normalize features
    track_feats = track_feats / (np.linalg.norm(track_feats, axis=1, keepdims=True) + 1e-8)
    det_feats   = det_feats   / (np.linalg.norm(det_feats,   axis=1, keepdims=True) + 1e-8)

    # Cosine distance
    cos_dist = 1.0 - np.dot(track_feats, det_feats.T)  # (T, D)
    cos_dist = np.clip(cos_dist, 0., 1.)
    return cos_dist

def size_consistency_cost(track_boxes, det_boxes):
    """
    Penalize large changes in bounding box size
    """
    track_areas = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
    det_areas = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
    
    # Compute area ratios
    area_ratios = track_areas[:, None] / (det_areas[None, :] + 1e-7)
    
    # Cost increases with deviation from ratio of 1.0
    size_costs = torch.abs(torch.log(area_ratios + 1e-7))
    
    return torch.clamp(size_costs, 0.0, 3.0)  # Cap at reasonable value


# def compute_costs(track_boxes, det_boxes, track_poses, det_poses, track_feats, det_feats):
#     """
#     Compute combined spatial, shape, and feature costs
#     Returns a (num_tracks, num_dets, 3) tensor
#     """
#     with torch.no_grad():
#         # Spatial cost (IoU)
#         ious = box_iou(track_boxes, det_boxes)
#         spatial_costs = 1.0 - ious  # (T, D)

#         # Shape cost (OKS)
#         shape_costs = 1.0 - compute_oks(track_poses, det_poses)  # (T, D)

#         # Feature cost (Cosine distance)
#         track_feats_np = track_feats.cpu().numpy() if isinstance(track_feats, torch.Tensor) else track_feats
#         det_feats_np   = det_feats.cpu().numpy() if isinstance(det_feats, torch.Tensor) else det_feats
#         feat_costs = cos_distance(track_feats_np, det_feats_np)  # (T, D)
#         feat_costs = torch.from_numpy(feat_costs).to(spatial_costs.device).float()

#         size_costs = size_consistency_cost(track_boxes, det_boxes)

#         # Stack into a 3D cost tensor
#         return torch.stack([spatial_costs, shape_costs, feat_costs, size_costs], dim=-1)  # (T, D, 4)

def compute_costs(track_boxes, det_boxes, track_poses, det_poses, track_feats, det_feats):
    """
    Compute combined costs with ablation study options
    Comment/uncomment sections to test different metrics
    Returns a (num_tracks, num_dets, X) tensor where X depends on enabled metrics
    """
    with torch.no_grad():
        cost_components = []
                
        ious = box_iou(track_boxes, det_boxes)
        spatial_costs = 1.0 - ious  # (T, D)
        cost_components.append(spatial_costs)
        
        shape_costs = 1.0 - compute_oks(track_poses, det_poses)  # (T, D)
        cost_components.append(shape_costs)
        
        track_feats_np = track_feats.cpu().numpy() if isinstance(track_feats, torch.Tensor) else track_feats
        det_feats_np   = det_feats.cpu().numpy() if isinstance(det_feats, torch.Tensor) else det_feats
        feat_costs = cos_distance(track_feats_np, det_feats_np)  # (T, D)
        feat_costs = torch.from_numpy(feat_costs).to(track_boxes.device).float()
        cost_components.append(feat_costs)
        
        return torch.stack(cost_components, dim=-1)

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
            det_dict['poses'],
            track_dict['features'],
            det_dict['features']
        )
        # print(f"cost vector: {cost_vector}")
    
    with torch.no_grad():
        attn_weights = matcher(track_dict, det_dict, cost_vector)

        if torch.any(torch.isnan(attn_weights)) or torch.any(torch.isinf(attn_weights)):
                print("Warning: Invalid attention weights detected. Using uniform costs.")
                attn_weights = torch.ones_like(attn_weights) * 0.5
    # print(f"Attention weights  {attn_weights}")
    # Apply Hungarian algorithm
    if attn_weights.numel() > 0:
        # print(attn_weights.numel())
        attn_probs = torch.sigmoid(attn_weights)  # or F.softmax(attn_weights, dim=1)
        cost_matrix = 1.0 - attn_probs.cpu().numpy()

        if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
                print("Warning: Final cost matrix contains invalid values. Using fallback.")
                cost_matrix = np.ones_like(cost_matrix) * 0.5
        # cost_matrix = 1.0 - attn_weights.cpu().numpy()
        return linear_assignment(cost_matrix, match_thr)
    else:
        return [], list(range(len(tracks))), list(range(len(detections)))