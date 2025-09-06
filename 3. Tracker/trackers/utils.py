import lap
import numpy as np
from cython_bbox import bbox_overlaps
import torch


def find_deleted_detections(dets, dets_95):
    # Get boxes
    a_x1y1x2y2 = np.ascontiguousarray(dets[:, :4], dtype=np.float64)
    b_x1y1x2y2 = np.ascontiguousarray(dets_95[:, :4], dtype=np.float64)

    # Calculate IoU
    ious = bbox_overlaps(a_x1y1x2y2, b_x1y1x2y2)

    # Find deleted detections
    dets_del = dets_95[np.max(ious, axis=0) < 0.97]

    return dets_del

def add_buffer(boxes, buffer=5):
    # boxes: [N, 4] in (x1, y1, x2, y2)
    boxes_buf = boxes.copy()
    boxes_buf[:, 0] += buffer  # x1
    boxes_buf[:, 1] += buffer  # y1
    boxes_buf[:, 2] += buffer  # x2
    boxes_buf[:, 3] += buffer  # y2
    return boxes_buf

def iou_distance(a_tracks, b_tracks):
    # Get boxes
    a_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in b_tracks], dtype=np.float64)

    # Calculate IoU distance
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        iou_sim = np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float64)
        iou_dist = 1 - iou_sim
    else:
        # Calculate HIoU
        h_iou = (np.minimum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.maximum(a_boxes[:, 1:2], b_boxes[:, 1:2].T))
        h_iou /= (np.maximum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.minimum(a_boxes[:, 1:2], b_boxes[:, 1:2].T))

        # Calculate HMIoU
        # a_boxes_buf = add_buffer(a_boxes, buffer=20)
        # b_boxes_buf = add_buffer(b_boxes, buffer=20)
        # iou_sim = bbox_overlaps(a_boxes_buf, b_boxes_buf)
        iou_sim = bbox_overlaps(a_boxes, b_boxes)
        iou_dist = 1 - h_iou * iou_sim

    return iou_sim, iou_dist


def cos_distance(tracks, dets):
    # Check
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Calculate cosine distance
    t_feat = np.concatenate([t.feat for t in tracks], axis=0)
    d_feat = np.concatenate([d.feat for d in dets], axis=0)
    cos_dist = np.clip(1 - np.dot(t_feat, d_feat.T), a_min=0., a_max=1.)

    return cos_dist

def cos_pose(tracks, dets):
    # Handle empty case
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Stack pose vectors
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    t_feat = np.stack([t.pose for t in tracks], axis=0)  # shape: [N_tracks, pose_dim]
    d_feat = np.stack([d.pose for d in dets], axis=0)    # shape: [N_dets, pose_dim]

    t_feat_norm = t_feat / np.linalg.norm(t_feat, axis=1, keepdims=True)
    d_feat_norm = d_feat / np.linalg.norm(d_feat, axis=1, keepdims=True)

    # Compute cosine distance
    cos_dist = np.clip(1 - np.dot(t_feat_norm, d_feat_norm.T), a_min=0., a_max=1.)
    return cos_dist

def conf_distance(tracks, dets):
    # Check
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Get previous scores
    t_score_prev = []
    for t in tracks:
        frame_ids = sorted(list(t.history.keys()), reverse=True)
        frame_id = frame_ids[min(1, len(frame_ids) - 1)]
        t_score_prev.append(t.history[frame_id][1])

    # Linear projection
    t_score_prev = np.array(t_score_prev)
    t_score = np.array([t.score for t in tracks])
    t_score += (t_score - t_score_prev)

    # Calculate confidence similarity
    d_score = np.array([d.score for d in dets])
    conf_dist = np.abs(t_score[:, None] - d_score[None, :])

    return conf_dist


def get_prev_box(history, frame_id, delta_t):
    # Try
    for i in range(delta_t):
        target_key = frame_id - delta_t
        if target_key in history.keys():
            return history[target_key][0]

    # If there are no recent observation
    return history[max(history.keys())][0]


def get_vel_t_d(b_1, b_2):
    # Expand boxes
    b_1, b_2 = b_1[:, np.newaxis, :], b_2[np.newaxis, :, :]

    # Get normalization factors
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_lb = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 3:4]**2) + 1e-5
    norm_rt = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_rb = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 3:4]**2) + 1e-5

    # Get velocities
    vel_lt = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_lt
    vel_lb = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 3] - b_1[:, :, 1]], axis=-1) / norm_lb
    vel_rt = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_rt
    vel_rb = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 3] - b_1[:, :, 1]], axis=-1) / norm_rb

    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=2)


def calc_angle(vel_t, vel_t_d):
    angle_ = 0
    for vdx in range(vel_t.shape[2]):
        # Divide & Repeat
        vel_t_x = np.repeat(vel_t[:, :, vdx, 0], vel_t_d.shape[1], axis=1)
        vel_t_y = np.repeat(vel_t[:, :, vdx, 1], vel_t_d.shape[1], axis=1)

        # Calculate angle, Normalize to range (0 ~ 1)
        angle = vel_t_x * vel_t_d[:, :, vdx, 0] + vel_t_y * vel_t_d[:, :, vdx, 1]
        angle = np.abs(np.arccos(np.clip(angle, a_min=-1, a_max=1))) / np.pi
        angle_ += angle / 4

    return angle_


def angle_distance(tracks, dets, frame_id, d_t=3):
    # Initialization
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Get velocity between track and detections
    track_boxes = np.stack([get_prev_box(t.history, frame_id, d_t) for t in tracks], axis=0)
    vel_t_d = get_vel_t_d(track_boxes, np.stack([d.x1y1x2y2 for d in dets], axis=0))

    # Get angle distance
    angle_dist = calc_angle(np.stack([t.velocity for t in tracks], axis=0)[:, np.newaxis], vel_t_d)

    # Fuse score
    scores = np.array([d.score for d in dets])[np.newaxis, :]
    angle_dist *= scores

    return angle_dist


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def associate(cost, match_thr):
    # Initialization
    matches = []

    # Run
    if cost.shape[0] > 0 and cost.shape[1] > 0:
        # Get index for minimum similarity
        min_ddx = np.argmin(cost, axis=1)
        min_tdx = np.argmin(cost, axis=0)

        # Match tracks with detections
        for tdx, ddx in enumerate(min_ddx):
            if min_tdx[ddx] == tdx and cost[tdx, ddx] < match_thr:
                matches.append([tdx, ddx])

    return matches


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


# def compute_oks(poses1, poses2, kappa=0.005):
#     kps1 = poses1.view(-1, 17, 2)  
#     kps2 = poses2.view(-1, 17, 2) 
    
#     valid1 = (kps1.abs().sum(dim=-1) > 1e-3)  
#     valid2 = (kps2.abs().sum(dim=-1) > 1e-3)  
    
#     def get_scale(kps, valid):
#         scales = []
#         for i in range(len(kps)):
#             vis_kps = kps[i][valid[i]]  
#             if len(vis_kps) == 0:
#                 scales.append(0.0)
#                 continue
#             scale = (vis_kps.max(dim=0)[0] - vis_kps.min(dim=0)[0]).prod()
#             scales.append(scale)
#         return torch.tensor(scales, device=kps1.device) 
    
#     scale1 = get_scale(kps1, valid1) 
#     scale2 = get_scale(kps2, valid2)  
#     scale = (scale1[:,None] + scale2[None,:]) / 2  
    
#     diff = kps1[:,None,:,:] - kps2[None,:,:,:]  
#     sq_dist = (diff ** 2).sum(dim=-1)  
    
#     oks_kpts = torch.exp(-sq_dist / (2 * scale[...,None] * kappa**2 + 1e-7))  
    
#     valid_mask = valid1[:,None,:] & valid2[None,:,:]  
#     oks_kpts = oks_kpts * valid_mask.float()
    
#     valid_count = valid_mask.sum(dim=-1) 
#     oks = oks_kpts.sum(dim=-1) / (valid_count + 1e-7)
    
#     oks[valid_count == 0] = 0.0
    
#     return oks

def compute_oks(poses1, poses2, kappa=0.05):
    """
    Compute Object Keypoint Similarity between two sets of poses.
    
    Args:
        poses1: Tensor of shape (N, 34) - first 17 keypoints as [x1,y1,x2,y2,...]
        poses2: Tensor of shape (M, 34) - same format as poses1
        kappa: Scaling factor (default 0.05 as in COCO)
    
    Returns:
        OKS matrix of shape (N, M) where higher values indicate better matches
    """
    # Reshape to (N, 17, 2) and (M, 17, 2)
    SCALE_FACTOR = 1000.0
    poses1 = poses1 * SCALE_FACTOR
    poses2 = poses2 * SCALE_FACTOR
    # print(f"poses1 shape: {poses1.shape}, poses2 shape: {poses2.shape}")
    kps1 = poses1.view(-1, 17, 2)
    kps2 = poses2.view(-1, 17, 2)
    
    # Create valid mask (keypoints with non-zero values)
    valid1 = (kps1.abs().sum(dim=-1) > 1e-3)
    valid2 = (kps2.abs().sum(dim=-1) > 1e-3)

    # print(f"Valid keypoints - tracks: {valid1.sum()}/{valid1.numel()}")
    # print(f"Valid keypoints - dets: {valid2.sum()}/{valid2.numel()}")

    
    def get_scale(kps, valid):
        """Calculate scale as area of bounding box around visible keypoints"""
        scales = []
        for i in range(len(kps)):
            vis_kps = kps[i][valid[i]]
            if len(vis_kps) < 2:  # Need at least 2 points for scale
                scales.append(1.0)  # Default scale
                continue
            bbox = torch.stack([vis_kps.min(dim=0)[0], vis_kps.max(dim=0)[0]])
            scale = (bbox[1] - bbox[0]).prod()  # Area of bounding box
            scales.append(max(scale.item(), 1e-3))  # Ensure scale > 0
        return torch.tensor(scales, device=kps.device)
    
    # Calculate scales for each instance
    scale1 = get_scale(kps1, valid1)
    scale2 = get_scale(kps2, valid2)
    scale = (scale1[:, None] + scale2[None, :]) / 2  # (N, M)

    # print(f"Scale values - tracks: {scale1}")
    # print(f"Scale values - dets: {scale2}")
    # print(f"Scale matrix range: {scale.min()} to {scale.max()}")

    
    # Calculate squared distances between all keypoints
    diff = kps1[:, None, :, :] - kps2[None, :, :, :]  # (N, M, 17, 2)
    sq_dist = (diff ** 2).sum(dim=-1)  # (N, M, 17)

    # print(f"Squared distance range: {sq_dist.min()} to {sq_dist.max()}")

    
    # Compute OKS for each keypoint
    oks_kpts = torch.exp(-sq_dist / (2 * scale[..., None] * kappa**2 + 1e-7))
    
    # Apply valid mask (both keypoints must be valid)
    valid_mask = valid1[:, None, :] & valid2[None, :, :]
    oks_kpts = oks_kpts * valid_mask.float()
    
    # Sum over keypoints and normalize by valid count
    valid_count = valid_mask.sum(dim=-1)
    oks = oks_kpts.sum(dim=-1) / (valid_count + 1e-7)
    
    # Set OKS to 0 if no valid keypoints
    oks[valid_count == 0] = 0.0

    # print(f"Final OKS range: {oks.min()} to {oks.max()}")
    # print("=== END DEBUG ===\n")
    
    return oks

def iterative_assignment(tracks, dets_high, dets_low, dets_del_high, match_thr, penalty_p, penalty_q,
                        reduce_step, frame_id, d_t=3):
    # Initialization
    matches = []
    
    
    poses_1 = torch.tensor([t.pose for t in tracks], dtype=torch.float32)
    poses_2 = torch.tensor([d.pose for d in (dets_high + dets_low)], dtype=torch.float32)
    dets = dets_high + dets_low

    cos_dist = cos_distance(tracks, dets)

    iou_sim, iou_dist = iou_distance(tracks, dets) 
    if len(poses_1) == 0 or len(poses_2) == 0:
        pose_dist = np.ones((len(tracks), len(dets)))
    else:
        oks_sim = compute_oks(poses_1, poses_2).numpy()
        
        pose_dist = 1.0 - oks_sim  # Convert to distance

    conf_dist = conf_distance(tracks, dets) 
    angle_dist = angle_distance(tracks, dets, frame_id, d_t) 

    # weights = [0.45, 0.45, 0.05, 0.05]  # iou, pose, conf, angle
    weights = [0.5,0.3, 0.2, 0.05]
    cost = (weights[0] * iou_dist + 
            weights[1] * pose_dist + 
            weights[2] * cos_dist + 
            weights[3] * conf_dist)
    
    # # Give penalty
    # cost[:, len(dets_high):len(dets_high + dets_low)] += penalty_p
    # cost[:, len(dets_high + dets_low):] += penalty_q
    
    cost[:, len(dets_high):len(dets_high + dets_low)] = np.minimum(
        cost[:, len(dets_high):len(dets_high + dets_low)] + penalty_p, 1.0)
    cost[:, len(dets_high + dets_low):] = np.minimum(
        cost[:, len(dets_high + dets_low):] + penalty_q, 1.0)
    
    # Constraint
    cost[iou_sim <= 0.20] = 1.0
    cost = np.clip(cost, 0, 1)

    # # Linear assignment
    # matches, u_tracks, u_dets = linear_assignment(cost, match_thr)

    # Match
    while True:
        # Match tracks with detections
        matches_ = associate(cost, match_thr)
        match_thr -= reduce_step

        # Check (if there are no more matchable pairs)
        if len(matches_) == 0:
            break

        # Append
        matches += matches_

        # Update cost matrix
        for t, d in matches:
            cost[t, :] = 1.
            cost[:, d] = 1.

    # Find indices of unmatched tracks and detections
    m_tracks = [t for t, _ in matches]
    u_tracks = [t for t in range(len(tracks)) if t not in m_tracks]
    m_dets = [d for _, d in matches]
    u_dets = [d for d in range(len(dets)) if d not in m_dets]

    return matches, u_tracks, u_dets

def track_aware_nms(pair_sims, scores, num_tracks, nms_thresh, score_thresh):
    # Initialization
    num_dets = len(pair_sims) - num_tracks
    allow_indices = np.ones(num_dets) * (scores > score_thresh)

    # Run
    for idx in range(num_dets):
        # Check 1
        if allow_indices[idx] == 0:
            continue

        # Check 2
        if num_tracks > 0:
            if np.max(pair_sims[num_tracks + idx, :num_tracks]) > nms_thresh:
                allow_indices[idx] = 0
                continue

        # Check 3
        for jdx in range(num_dets):
            if idx != jdx and allow_indices[jdx] == 1 and scores[idx] > scores[jdx]:
                if pair_sims[num_tracks + idx, num_tracks + jdx] > nms_thresh:
                    allow_indices[jdx] = 0

    return allow_indices == 1


