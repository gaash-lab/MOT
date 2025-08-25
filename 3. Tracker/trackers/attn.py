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

class RobustCrossAssociationEngine(nn.Module):
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
        
        # Enhanced cost MLP for multiple distance types
        self.cost_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim//2),  # spatial, shape, conf, mhd, angle, temporal
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
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
            cost_vector: Optional (N_tracks, N_dets, 6) - [spatial, shape, conf, mhd, angle, temporal]
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
        
        track_feats = torch.cat([track_feats, self.pose_proj(tracks['poses'])], dim=1)
        det_feats = torch.cat([det_feats, self.pose_proj(detections['poses'])], dim=1)
        track_feats = self.concat_proj(track_feats)
        det_feats = self.concat_proj(det_feats)
        
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

def box_iou(boxes1, boxes2, buffer=5):
    """Computes Buffered IoU between two sets of boxes."""
    boxes1 = boxes1[:, :4] if boxes1.shape[-1] > 4 else boxes1
    boxes2 = boxes2[:, :4] if boxes2.shape[-1] > 4 else boxes2

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

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

    lt = torch.max(boxes1_exp[:, None, :2], boxes2_exp[None, :, :2])
    rb = torch.min(boxes1_exp[:, None, 2:], boxes2_exp[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
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

def conf_distance(tracks, dets):
    """Confidence distance with trajectory prediction"""
    if len(tracks) == 0 or len(dets) == 0:
        return torch.ones(len(tracks), len(dets), dtype=torch.float32)
    
    # Get previous scores for trajectory prediction
    t_score_prev = []
    for t in tracks:
        if hasattr(t, 'history') and len(t.history) > 1:
            frame_ids = sorted(list(t.history.keys()), reverse=True)
            frame_id = frame_ids[min(1, len(frame_ids) - 1)]
            t_score_prev.append(t.history[frame_id][1])
        else:
            t_score_prev.append(getattr(t, 'score', 0.5))
    
    # Linear projection for confidence trajectory
    t_score_prev = torch.tensor(t_score_prev, dtype=torch.float32)
    t_score = torch.tensor([getattr(t, 'score', 0.5) for t in tracks], dtype=torch.float32)
    t_score += (t_score - t_score_prev)  # Predict next confidence
    
    # Calculate confidence similarity
    d_score = torch.tensor([getattr(d, 'score', 0.5) for d in dets], dtype=torch.float32)
    conf_dist = torch.abs(t_score[:, None] - d_score[None, :])
    
    return conf_dist

def get_prev_box(history, frame_id, delta_t):
    """Get previous box from history with fallback"""
    for i in range(delta_t):
        target_key = frame_id - (i + 1)
        if target_key in history.keys():
            return history[target_key][0]
    return history[max(history.keys())][0]

def get_vel_t_d(b_1, b_2):
    """Calculate velocity vectors between box corners"""
    b_1, b_2 = b_1[:, np.newaxis, :], b_2[np.newaxis, :, :]
    
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_lb = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 3:4]**2) + 1e-5
    norm_rt = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_rb = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 3:4]**2) + 1e-5
    
    vel_lt = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_lt
    vel_lb = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 3] - b_1[:, :, 1]], axis=-1) / norm_lb
    vel_rt = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_rt
    vel_rb = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 3] - b_1[:, :, 1]], axis=-1) / norm_rb
    
    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=2)

def calc_angle(vel_t, vel_t_d):
    """Calculate angle between velocity vectors"""
    angle_ = 0
    for vdx in range(vel_t.shape[2]):
        vel_t_x = np.repeat(vel_t[:, :, vdx, 0], vel_t_d.shape[1], axis=1)
        vel_t_y = np.repeat(vel_t[:, :, vdx, 1], vel_t_d.shape[1], axis=1)
        
        angle = vel_t_x * vel_t_d[:, :, vdx, 0] + vel_t_y * vel_t_d[:, :, vdx, 1]
        angle = np.abs(np.arccos(np.clip(angle, a_min=-1, a_max=1))) / np.pi
        angle_ += angle / 4
    return angle_

def angle_distance(tracks, dets, frame_id, d_t=3):
    """Calculate angle distance between track velocity and predicted movement"""
    if len(tracks) == 0 or len(dets) == 0:
        return torch.ones(len(tracks), len(dets), dtype=torch.float32)
    
    # Get previous boxes for tracks
    track_boxes = []
    for t in tracks:
        if hasattr(t, 'history') and t.history:
            prev_box = get_prev_box(t.history, frame_id, d_t)
            track_boxes.append(prev_box)
        else:
            track_boxes.append(getattr(t, 'x1y1x2y2', [0, 0, 1, 1]))
    
    track_boxes = np.stack(track_boxes, axis=0)
    det_boxes = np.stack([getattr(d, 'x1y1x2y2', [0, 0, 1, 1]) for d in dets], axis=0)
    
    # Get velocity between track and detections
    vel_t_d = get_vel_t_d(track_boxes, det_boxes)
    
    # Get track velocities
    track_vels = []
    for t in tracks:
        if hasattr(t, 'velocity'):
            track_vels.append(t.velocity)
        else:
            # Default velocity if not available
            track_vels.append(np.zeros((1, 4, 2)))
    
    vel_t = np.stack(track_vels, axis=0)[:, np.newaxis]
    
    # Calculate angle distance
    angle_dist = calc_angle(vel_t, vel_t_d)
    
    # Weight by detection scores
    scores = np.array([getattr(d, 'score', 0.5) for d in dets])[np.newaxis, :]
    angle_dist *= scores
    
    return torch.tensor(angle_dist, dtype=torch.float32)

def mahalanobis_distance(tracks, dets):
    """Mahalanobis distance using track covariance"""
    if len(tracks) == 0 or len(dets) == 0:
        return torch.ones(len(tracks), len(dets), dtype=torch.float32)
    
    mhd_matrix = []
    for t in tracks:
        row = []
        t_pos = torch.tensor(getattr(t, 'x1y1x2y2', [0, 0, 1, 1])[:2], dtype=torch.float32)
        
        # Get covariance matrix (use identity if not available)
        if hasattr(t, 'covariance'):
            cov = t.covariance[:2, :2]  # Position covariance
        else:
            cov = torch.eye(2) * 10.0  # Default covariance
        
        for d in dets:
            d_pos = torch.tensor(getattr(d, 'x1y1x2y2', [0, 0, 1, 1])[:2], dtype=torch.float32)
            diff = t_pos - d_pos
            
            try:
                cov_inv = torch.inverse(cov + torch.eye(2) * 1e-6)
                mhd = torch.sqrt(diff.T @ cov_inv @ diff).item()
            except:
                mhd = torch.norm(diff).item()  # Fallback to Euclidean
            
            row.append(mhd)
        mhd_matrix.append(row)
    
    return torch.tensor(mhd_matrix, dtype=torch.float32)

def temporal_distance(tracks, dets, frame_id):
    """Temporal consistency based on track age and detection timing"""
    if len(tracks) == 0 or len(dets) == 0:
        return torch.zeros(len(tracks), len(dets), dtype=torch.float32)
    
    temporal_matrix = []
    for t in tracks:
        row = []
        track_age = getattr(t, 'age', 1)
        last_seen = getattr(t, 'last_frame', frame_id - 1)
        gap = frame_id - last_seen
        
        for d in dets:
            # Penalize associations with large temporal gaps
            temp_cost = np.exp(-gap / track_age) if track_age > 0 else 0.5
            row.append(1.0 - temp_cost)
        temporal_matrix.append(row)
    
    return torch.tensor(temporal_matrix, dtype=torch.float32)

def compute_enhanced_costs(track_boxes, det_boxes, track_poses, det_poses, 
                          tracks, dets, frame_id):
    """Compute all distance metrics for robust association"""
    with torch.no_grad():
        # Basic costs
        ious = box_iou(track_boxes, det_boxes)
        spatial_costs = 1.0 - ious
        
        shape_costs = 1.0 - compute_oks(track_poses, det_poses)
        
        # Enhanced costs for identity preservation
        conf_costs = conf_distance(tracks, dets)
        mhd_costs = mahalanobis_distance(tracks, dets)
        angle_costs = angle_distance(tracks, dets, frame_id)
        temporal_costs = temporal_distance(tracks, dets, frame_id)
        
        # Stack all costs
        return torch.stack([
            spatial_costs, shape_costs, conf_costs, 
            mhd_costs, angle_costs, temporal_costs
        ], dim=-1)

def robust_cross_attention_assignment(tracks, detections, matcher, frame_id, match_thr=0.3):
    """
    Enhanced cross-attention assignment with multiple distance metrics
    Lower threshold for more conservative matching
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
        
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
    
    # Compute enhanced cost vector with all distance metrics
    cost_vector = compute_enhanced_costs(
        track_dict['boxes'], 
        det_dict['boxes'],
        track_dict['poses'],
        det_dict['poses'],
        tracks,
        detections,
        frame_id
    )
    
    with torch.no_grad():
        attn_weights = matcher(track_dict, det_dict, cost_vector)
    
    if attn_weights.numel() > 0:
        attn_probs = torch.sigmoid(attn_weights)
        cost_matrix = 1.0 - attn_probs.cpu().numpy()
        
        # Apply additional filtering for high-confidence tracks
        for i, track in enumerate(tracks):
            if hasattr(track, 'hit_streak') and track.hit_streak > 5:  # Stable track
                # Be more conservative with stable tracks
                for j in range(cost_matrix.shape[1]):
                    if cost_matrix[i, j] > 0.2:  # Stricter threshold for stable tracks
                        cost_matrix[i, j] = 1.0
        
        return linear_assignment(cost_matrix, match_thr)
    else:
        return [], list(range(len(tracks))), list(range(len(detections)))