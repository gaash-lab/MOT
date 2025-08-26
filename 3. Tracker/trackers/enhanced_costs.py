import torch
import torch.nn.functional as F
import numpy as np

def cosine_feature_distance(track_features, det_features):
    """
    Cosine distance between appearance features
    Lower values = more similar appearances
    """
    # Normalize features
    if track_features.dim() == 3:
        track_features = track_features.squeeze(1)
    if det_features.dim() == 3:
        det_features = det_features.squeeze(1)
    track_norm = F.normalize(track_features, p=2, dim=1)
    det_norm = F.normalize(det_features, p=2, dim=1)
    
    # Compute cosine similarity matrix

    similarity = torch.mm(track_norm, det_norm.t())
    
    # Convert to distance (1 - similarity)
    distance = 1.0 - similarity
    return torch.clamp(distance, 0.0, 2.0)

def motion_prediction_cost(track_boxes, det_boxes, track_velocities=None, temporal_gap=1):
    """
    Predict next position based on velocity and compute distance to actual detection
    """
    if track_velocities is None:
        # Simple linear extrapolation using box centers
        track_centers = (track_boxes[:, :2] + track_boxes[:, 2:]) / 2
        det_centers = (det_boxes[:, :2] + det_boxes[:, 2:]) / 2
        
        # Assume zero velocity if no history (fallback to center distance)
        predicted_centers = track_centers
    else:
        # Use provided velocities to predict next position
        track_centers = (track_boxes[:, :2] + track_boxes[:, 2:]) / 2
        predicted_centers = track_centers + track_velocities * temporal_gap
        det_centers = (det_boxes[:, :2] + det_boxes[:, 2:]) / 2
    
    # Compute distances between predicted and actual positions
    distances = torch.cdist(predicted_centers, det_centers, p=2)
    
    # Normalize by image size (assuming 1920x1080 typical resolution)
    normalized_distances = distances / (1920 + 1080) * 2
    
    return torch.clamp(normalized_distances, 0.0, 1.0)

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

def pose_angle_consistency(track_poses, det_poses):
    """
    Compute angle differences for pose consistency
    Focus on key body orientation angles
    """
    def compute_body_angle(poses):
        # poses: (N, 17, 2) - COCO format
        poses_2d = poses.view(-1, 17, 2)
        
        # Key points for body orientation: shoulders (5,6) and hips (11,12)
        left_shoulder = poses_2d[:, 5]   # Left shoulder
        right_shoulder = poses_2d[:, 6]  # Right shoulder
        left_hip = poses_2d[:, 11]       # Left hip  
        right_hip = poses_2d[:, 12]      # Right hip
        
        # Shoulder angle
        shoulder_vec = right_shoulder - left_shoulder
        shoulder_angle = torch.atan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
        
        # Hip angle
        hip_vec = right_hip - left_hip
        hip_angle = torch.atan2(hip_vec[:, 1], hip_vec[:, 0])
        
        # Combine angles (you can weight these differently)
        combined_angle = (shoulder_angle + hip_angle) / 2
        return combined_angle
    
    track_angles = compute_body_angle(track_poses)  # (N_tracks,)
    det_angles = compute_body_angle(det_poses)      # (N_dets,)
    
    # Compute angle differences
    angle_diff = track_angles[:, None] - det_angles[None, :]
    
    # Normalize to [0, π]
    angle_cost = torch.abs(torch.remainder(angle_diff + np.pi, 2*np.pi) - np.pi) / np.pi
    
    return angle_cost

def confidence_consistency_cost(track_confs, det_confs, alpha=0.5):
    """
    Penalize large confidence changes (but allow some variation)
    """
    conf_diff = torch.abs(track_confs[:, None] - det_confs[None, :])
    
    # Apply smooth penalty (not too harsh for small changes)
    conf_cost = alpha * torch.tanh(conf_diff * 3.0)  # Smooth saturation
    
    return conf_cost

def temporal_smoothness_cost(track_features, det_features, track_history_features=None):
    """
    Encourage smooth feature transitions over time
    """
    if track_history_features is None:
        # Fallback to simple cosine distance
        return cosine_feature_distance(track_features, det_features)
    
    # Predict next features based on temporal trend
    # track_history_features: (N_tracks, history_len, feat_dim)
    if track_history_features.shape[1] >= 2:
        # Linear extrapolation
        recent_feats = track_history_features[:, -1]  # Most recent
        prev_feats = track_history_features[:, -2]    # Previous
        
        predicted_feats = recent_feats + (recent_feats - prev_feats)
    else:
        predicted_feats = track_features
    
    # Compute distance from predicted to actual
    return cosine_feature_distance(predicted_feats, det_features)

def compute_enhanced_costs(track_data, det_data, temporal_gap=1):
    """
    Combine multiple cost metrics for robust tracking
    
    Args:
        track_data: Dict with 'features', 'poses', 'boxes', 'scores'
        det_data: Dict with 'features', 'poses', 'boxes', 'scores'
        temporal_gap: Number of frames between track and detection
    
    Returns:
        cost_matrix: (N_tracks, N_dets, num_costs)
    """
    costs = []
    
    # appearance_cost = cosine_feature_distance(track_data['features'], det_data['features'])
    # costs.append(appearance_cost)
    
    spatial_cost = 1.0 - box_iou(track_data['boxes'], det_data['boxes'])
    costs.append(spatial_cost)
    
    # motion_cost = motion_prediction_cost(
    #     track_data['boxes'], 
    #     det_data['boxes'],
    #     track_data.get('velocities'),
    #     temporal_gap
    # )
    # costs.append(motion_cost)
    
    pose_cost = 1.0 - compute_oks(track_data['poses'], det_data['poses'])
    costs.append(pose_cost)
    
    # size_cost = size_consistency_cost(track_data['boxes'], det_data['boxes'])
    # costs.append(size_cost)
    
    # angle_cost = pose_angle_consistency(track_data['poses'], det_data['poses'])
    # costs.append(angle_cost)
    
    # if 'scores' in track_data and 'scores' in det_data:
    #     conf_cost = confidence_consistency_cost(track_data['scores'], det_data['scores'])
    #     print(conf_cost)
    #     costs.append(conf_cost)
    
    # Stack all costs: (N_tracks, N_dets, num_costs)
    # costs = [c.unsqueeze(-1) if c.dim() == 2 else c for c in costs]
    cost_matrix = torch.stack(costs, dim=-1)
    
    return cost_matrix

def box_iou(boxes1, boxes2, buffer=5):
    """IoU computation with optional buffer"""
    boxes1 = boxes1[:, :4] if boxes1.shape[-1] > 4 else boxes1
    boxes2 = boxes2[:, :4] if boxes2.shape[-1] > 4 else boxes2

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    if buffer > 0:
        boxes1_exp = boxes1.clone()
        boxes2_exp = boxes2.clone()
        boxes1_exp[:, [0, 1]] -= buffer
        boxes1_exp[:, [2, 3]] += buffer
        boxes2_exp[:, [0, 1]] -= buffer
        boxes2_exp[:, [2, 3]] += buffer
    else:
        boxes1_exp, boxes2_exp = boxes1, boxes2

    lt = torch.max(boxes1_exp[:, None, :2], boxes2_exp[None, :, :2])
    rb = torch.min(boxes1_exp[:, None, 2:], boxes2_exp[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return (inter + 1e-7) / (union + 1e-7)

def compute_oks(poses1, poses2, kappa=0.05):
    """Object Keypoint Similarity computation"""
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