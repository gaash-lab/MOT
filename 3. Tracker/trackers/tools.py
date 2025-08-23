# import lap
# import numpy as np
# from cython_bbox import bbox_overlaps
# import torch
# import warnings
# from copy import deepcopy
# from typing import Optional

# # Import the functions from the second script
# def shape_similarity(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
#     # from default_settings import BoostTrackSettings
#     # if not BoostTrackSettings['s_sim_corr']:
#     #     return shape_similarity_v1(detects, tracks)
#     # else:
#     return shape_similarity_v2(detects, tracks)

# def shape_similarity_v1(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
#     if detects.size == 0 or tracks.size == 0:
#         return np.zeros((0, 0))

#     dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
#     dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
#     tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
#     th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
#     return np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dw, tw)))

# def shape_similarity_v2(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
#     if detects.size == 0 or tracks.size == 0:
#         return np.zeros((0, 0))

#     dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
#     dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
#     tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
#     th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
#     return np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dh, th)))

# def MhDist_similarity(mahalanobis_distance: np.ndarray, softmax_temp: float = 1.0) -> np.ndarray:
#     limit = 13.2767  # 99% conf interval
#     mahalanobis_distance = deepcopy(mahalanobis_distance)
#     mask = mahalanobis_distance > limit
#     mahalanobis_distance[mask] = limit
#     mahalanobis_distance = limit - mahalanobis_distance

#     mahalanobis_distance = np.exp(mahalanobis_distance/softmax_temp) / np.exp(mahalanobis_distance/softmax_temp).sum(0).reshape((1, -1))
#     mahalanobis_distance = np.where(mask, 0, mahalanobis_distance)
#     return mahalanobis_distance

# def iou_batch(bboxes1, bboxes2):
#     """
#     From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
#     """
#     bboxes2 = np.expand_dims(bboxes2, 0)
#     bboxes1 = np.expand_dims(bboxes1, 1)

#     xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
#     yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
#     xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
#     yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
#     w = np.maximum(0.0, xx2 - xx1)
#     h = np.maximum(0.0, yy2 - yy1)
#     wh = w * h
#     o = wh / (
#         (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
#         + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
#         - wh
#     )

#     return o

# def match(cost_matrix: np.ndarray, threshold: float) -> np.ndarray:
#     if cost_matrix.size > 0:
#         a = (cost_matrix > threshold).astype(np.int32)
#         if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#             matched_indices = np.stack(np.where(a), axis=1)
#         else:
#             _, x, y = lap.lapjv(-cost_matrix, extend_cost=True)
#             matched_indices = np.array([[y[i], i] for i in x if i >= 0])
#     else:
#         matched_indices = np.empty(shape=(0, 2))
#     return matched_indices

# def linear_assignment(detections: np.ndarray, trackers: np.ndarray,
#                       iou_matrix: np.ndarray, cost_matrix: np.ndarray,
#                       threshold: float, emb_cost: Optional[np.ndarray] = None):
#     if iou_matrix is None and cost_matrix is None:
#         raise Exception("Both iou_matrix and cost_matrix are None!")
#     if iou_matrix is None:
#         iou_matrix = deepcopy(cost_matrix)
#     if cost_matrix is None:
#         cost_matrix = deepcopy(iou_matrix)
#     matched_indices = match(cost_matrix, threshold)
#     unmatched_detections = []
#     for d, det in enumerate(detections):
#         if d not in matched_indices[:, 0]:
#             unmatched_detections.append(d)
#     unmatched_trackers = []
#     for t, trk in enumerate(trackers):
#         if t not in matched_indices[:, 1]:
#             unmatched_trackers.append(t)

#     # filter out matched with low IOU
#     matches = []
#     for m in matched_indices:
#         valid_match = iou_matrix[m[0], m[1]] >= threshold or (False if emb_cost is None else (iou_matrix[m[0], m[1]] >= threshold / 2 and emb_cost[m[0], m[1]] >= 0.75))
#         if valid_match:
#             matches.append(m.reshape(1, 2))
#         else:
#             unmatched_detections.append(m[0])
#             unmatched_trackers.append(m[1])

#     if len(matches) == 0:
#         matches = np.empty((0, 2), dtype=int)
#     else:
#         matches = np.concatenate(matches, axis=0)

#     return matches, np.array(unmatched_detections), np.array(unmatched_trackers), cost_matrix

# def associate(
#         detections,
#         trackers,
#         iou_threshold,
#         mahalanobis_distance: Optional[np.ndarray] = None,
#         track_confidence: Optional[np.ndarray] = None,
#         detection_confidence: Optional[np.ndarray] = None,
#         emb_cost: Optional[np.ndarray] = None,
#         lambda_iou: float = 0.5,
#         lambda_mhd: float = 0.25,
#         lambda_shape: float = 0.25
# ):
#     if len(trackers) == 0:
#         return (
#             np.empty((0, 2), dtype=int),
#             np.arange(len(detections)),
#             np.empty((0, 5), dtype=int),
#             np.empty((0, 0))
#         )
#     iou_matrix = iou_batch(detections, trackers)

#     cost_matrix = deepcopy(iou_matrix)

#     if detection_confidence is not None and track_confidence is not None:
#         conf = np.multiply(detection_confidence.reshape((-1, 1)), track_confidence.reshape((1, -1)))
#         conf[iou_matrix < iou_threshold] = 0

#         cost_matrix += lambda_iou * conf * iou_batch(detections, trackers)
#     else:
#         warnings.warn("Detections or tracklet confidence is None and detection-tracklet confidence cannot be computed!")
#         conf = None

#     if mahalanobis_distance is not None and mahalanobis_distance.size > 0:
#         mahalanobis_distance = MhDist_similarity(mahalanobis_distance)

#         cost_matrix += lambda_mhd * mahalanobis_distance
#         if conf is not None:
#             cost_matrix += lambda_shape * conf * shape_similarity(detections, trackers)

#     if emb_cost is not None:
#         lambda_emb = (1+lambda_iou+lambda_shape+lambda_mhd) * 1.5
#         cost_matrix += lambda_emb * emb_cost

#     return linear_assignment(detections, trackers, iou_matrix, cost_matrix, iou_threshold, emb_cost)

# # Keep your existing utility functions
# def find_deleted_detections(dets, dets_95):
#     # Get boxes
#     a_x1y1x2y2 = np.ascontiguousarray(dets[:, :4], dtype=np.float64)
#     b_x1y1x2y2 = np.ascontiguousarray(dets_95[:, :4], dtype=np.float64)

#     # Calculate IoU
#     ious = bbox_overlaps(a_x1y1x2y2, b_x1y1x2y2)

#     # Find deleted detections
#     dets_del = dets_95[np.max(ious, axis=0) < 0.97]

#     return dets_del

# def get_mh_dist_matrix(detections, trackers, n_dims=4):
#     """Calculate Mahalanobis distance between detections and trackers.
    
#     Args:
#         detections: np.array of detections in x1y1x2y2 format
#         trackers: List of track objects with kf (KalmanFilter) attribute
#         n_dims: Number of dimensions to use (default: 4 for bbox)
    
#     Returns:
#         np.array: Mahalanobis distance matrix (dets x tracks)
#     """
#     if len(trackers) == 0:
#         return np.zeros((0, 0))
    
#     def x1y1x2y2_to_cxcywh(bbox):
#         x1, y1, x2, y2 = bbox[:4]
#         width = x2 - x1
#         height = y2 - y1
#         cx = x1 + width/2
#         cy = y1 + height/2
#         return np.array([cx, cy, width, height])
    
#     z = np.array([x1y1x2y2_to_cxcywh(d) for d in detections])
#     x = np.zeros((len(trackers), n_dims))
#     sigma_inv = np.zeros_like(x)
    
#     for i, tracker in enumerate(trackers):
#         x[i] = tracker.mean[:n_dims]  # Kalman filter state vector
#         sigma_inv[i] = 1 / np.diag(tracker.covariance[:n_dims, :n_dims])  # Inverse of diagonal covariance
    
#     return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)

# def cos_distance(tracks, dets):
#     # Check
#     if len(tracks) == 0 or len(dets) == 0:
#         return np.ones((len(tracks), len(dets)), dtype=np.float64)

#     # Calculate cosine distance
#     t_feat = np.concatenate([t.feat for t in tracks], axis=0)
#     d_feat = np.concatenate([d.feat for d in dets], axis=0)
#     cos_dist = np.clip(1 - np.dot(t_feat, d_feat.T), a_min=0., a_max=1.)

#     return cos_dist

# # NEW: Modified iterative assignment function using the associate function
# def iterative_assignment_with_associate(tracks, dets_high, dets_low, dets_del_high, match_thr, penalty_p, penalty_q,
#                                        reduce_step, frame_id, d_t=3,
#                                        lambda_iou=0.5, lambda_mhd=0.25, lambda_shape=0.25):
#     """
#     Modified version using the associate function from the second script.
    
#     Args:
#         tracks: List of track objects with .x1y1x2y2, .score, .feat attributes and potentially .kf (Kalman filter)
#         dets_high: List of high confidence detection objects  
#         dets_low: List of low confidence detection objects
#         dets_del_high: List of deleted high confidence detections
#         match_thr: IoU threshold for matching
#         penalty_p: Penalty for low confidence detections
#         penalty_q: Penalty for deleted detections  
#         reduce_step: Step size for reducing threshold in iterative matching
#         frame_id: Current frame ID
#         d_t: Time delta for velocity calculations
#         lambda_iou, lambda_mhd, lambda_shape: Weighting factors for different similarity measures
#     """
    
#     # Combine all detections
#     dets = dets_high + dets_low  # + dets_del_high (commented out like in original)
    
#     if len(tracks) == 0 or len(dets) == 0:
#         return [], list(range(len(tracks))), list(range(len(dets)))
    
#     # Prepare detection and track arrays for the associate function
#     det_boxes = np.array([d.x1y1x2y2 for d in dets])  # Shape: (N_dets, 4)
#     track_boxes = np.array([t.x1y1x2y2 for t in tracks])  # Shape: (N_tracks, 4) 
    
#     # Prepare confidence scores
#     detection_confidence = np.array([d.score for d in dets])
#     track_confidence = np.array([t.score for t in tracks])
    
#     # Calculate Mahalanobis distance if tracks have Kalman filters
#     mahalanobis_distance = None
#     if hasattr(tracks[0], 'kalman_filter') and tracks[0].kalman_filter is not None:
#         try:
#             mahalanobis_distance = get_mh_dist_matrix(det_boxes, tracks)
#         except:
#             mahalanobis_distance = None
    
#     # Calculate embedding cost (cosine similarity) if available
#     emb_cost = None
#     if hasattr(tracks[0], 'feat') and hasattr(dets[0], 'feat'):
#         try:
#             cos_dist = cos_distance(tracks, dets)  # Shape: (N_tracks, N_dets)
#             emb_cost = 1 - cos_dist.T  # Convert to similarity and transpose to (N_dets, N_tracks)
#         except:
#             emb_cost = None
    
#     # Apply penalties to detection confidence based on detection type
#     penalized_detection_conf = detection_confidence.copy()
    
#     # Apply penalty to low confidence detections
#     penalized_detection_conf[len(dets_high):len(dets_high + dets_low)] -= penalty_p
    
#     # Apply penalty to deleted high confidence detections (if included)
#     # penalized_detection_conf[len(dets_high + dets_low):] -= penalty_q
    
#     # Use the associate function
#     matches, unmatched_dets, unmatched_tracks, cost_matrix = associate(
#         detections=det_boxes,
#         trackers=track_boxes, 
#         iou_threshold=match_thr,
#         mahalanobis_distance=mahalanobis_distance,
#         track_confidence=track_confidence,
#         detection_confidence=penalized_detection_conf,
#         emb_cost=emb_cost,
#         lambda_iou=lambda_iou,
#         lambda_mhd=lambda_mhd, 
#         lambda_shape=lambda_shape
#     )
    
#     # Convert matches format to match original function output
#     matches_list = [[t, d] for d, t in matches]  # Note: associate returns (det_idx, track_idx)
    
#     return matches_list, unmatched_tracks.tolist(), unmatched_dets.tolist()

# # Keep your existing track_aware_nms function unchanged
# def track_aware_nms(pair_sims, scores, num_tracks, nms_thresh, score_thresh):
#     # Initialization
#     num_dets = len(pair_sims) - num_tracks
#     allow_indices = np.ones(num_dets) * (scores > score_thresh)

#     # Run
#     for idx in range(num_dets):
#         # Check 1
#         if allow_indices[idx] == 0:
#             continue

#         # Check 2
#         if num_tracks > 0:
#             if np.max(pair_sims[num_tracks + idx, :num_tracks]) > nms_thresh:
#                 allow_indices[idx] = 0
#                 continue

#         # Check 3
#         for jdx in range(num_dets):
#             if idx != jdx and allow_indices[jdx] == 1 and scores[idx] > scores[jdx]:
#                 if pair_sims[num_tracks + idx, num_tracks + jdx] > nms_thresh:
#                     allow_indices[jdx] = 0

#     return allow_indices == 1



import lap
import numpy as np
from cython_bbox import bbox_overlaps
import torch

def get_mh_dist_matrix(detections, trackers, n_dims=4):
    """Calculate Mahalanobis distance between detections and trackers.
    
    Args:
        detections: np.array of detections in x1y1x2y2 format
        trackers: List of track objects with kf (KalmanFilter) attribute
        n_dims: Number of dimensions to use (default: 4 for bbox)
    
    Returns:
        np.array: Mahalanobis distance matrix (dets x tracks)
    """
    if len(trackers) == 0:
        return np.zeros((0, 0))
    
    def x1y1x2y2_to_cxcywh(bbox):
        x1, y1, x2, y2 = bbox[:4]
        width = x2 - x1
        height = y2 - y1
        cx = x1 + width/2
        cy = y1 + height/2
        return np.array([cx, cy, width, height])
    
    z = np.array([x1y1x2y2_to_cxcywh(d) for d in detections])
    x = np.zeros((len(trackers), n_dims))
    sigma_inv = np.zeros_like(x)
    
    for i, tracker in enumerate(trackers):
        x[i] = tracker.mean[:n_dims]  # Kalman filter state vector
        sigma_inv[i] = 1 / np.diag(tracker.covariance[:n_dims, :n_dims])  # Inverse of diagonal covariance
    
    return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)

def mahalanobis_distance(a_tracks, b_tracks):
    """Calculate Mahalanobis distance between tracks and detections"""
    if len(a_tracks) == 0 or len(b_tracks) == 0:
        mh_dist = np.zeros((len(a_tracks), len(b_tracks)), dtype=np.float64)
        return mh_dist
    
    det_boxes = np.array([track.x1y1x2y2 for track in b_tracks])
    
    mh_matrix = get_mh_dist_matrix(det_boxes, a_tracks)
    
    mh_dist = mh_matrix.T
    
    # Normalize Mahalanobis distance to [0, 1] range
    # Chi-square 99% confidence limit for 4 DOF
    chi2_limit = 13.2767
    mh_dist_normalized = np.clip(mh_dist / chi2_limit, 0, 1)
    
    return mh_dist_normalized

def add_buffer(boxes, buffer=5):
    # boxes: [N, 4] in (x1, y1, x2, y2)
    boxes_buf = boxes.copy()
    boxes_buf[:, 0] += buffer  # x1
    boxes_buf[:, 1] += buffer  # y1
    boxes_buf[:, 2] += buffer  # x2
    boxes_buf[:, 3] += buffer  # y2
    return boxes_buf

def iou_distance(a_tracks, b_tracks, buffer=15):
    """Calculate IoU distance between tracks - keeping for IoU similarity calculation"""
    # Get boxes
    a_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in b_tracks], dtype=np.float64)

    # Calculate IoU distance
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        iou_sim = np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float64)
        iou_dist = 1 - iou_sim
    else:
        # Calculate standard IoU
        a_boxes_buf = add_buffer(a_boxes, buffer=20)
        b_boxes_buf = add_buffer(b_boxes, buffer=20)
        iou_sim = bbox_overlaps(a_boxes_buf, b_boxes_buf)
        # iou_sim = bbox_overlaps(a_boxes, b_boxes)
        iou_dist = 1 - iou_sim

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

def iterative_assignment_mahalanobis(tracks, dets_high, dets_low, dets_del_high, match_thr, penalty_p, penalty_q,
                                   reduce_step, frame_id, d_t=3, use_iou_constraint=True, iou_threshold=0.10):
    """
    Modified iterative assignment using Mahalanobis distance instead of IoU distance.
    
    Args:
        tracks: List of track objects with .kf (KalmanFilter) attribute
        dets_high: List of high confidence detections
        dets_low: List of low confidence detections
        dets_del_high: List of deleted high confidence detections
        match_thr: Matching threshold
        penalty_p: Penalty for low confidence detections
        penalty_q: Penalty for deleted detections
        reduce_step: Step size for reducing threshold
        frame_id: Current frame ID
        d_t: Time delta
        use_iou_constraint: Whether to use IoU constraint (default: True)
        iou_threshold: IoU threshold for constraint (default: 0.10)
    """
    # Initialization
    matches = []
    dets = dets_high + dets_low  # + dets_del_high
    
    if len(tracks) == 0 or len(dets) == 0:
        return matches, list(range(len(tracks))), list(range(len(dets)))
    
    # Check if tracks have Kalman filters for Mahalanobis calculation
    has_kalman_filter = hasattr(tracks[0], 'kalman_filter') and tracks[0].kalman_filter is not None
    
    if not has_kalman_filter:
        raise ValueError("Tracks must have Kalman filter (.kf attribute) for Mahalanobis distance calculation")
    
    # Calculate Mahalanobis distance (this replaces iou_dist)
    mh_dist = mahalanobis_distance(tracks, dets)
    
    # Calculate IoU similarity for constraint (keeping this separate)
    iou_sim, _ = iou_distance(tracks, dets)
    
    # Calculate other distances
    cos_dist = cos_distance(tracks, dets)
    
    # Calculate cost using Mahalanobis distance instead of IoU distance
    cost = 0.50 * mh_dist + 0.50 * cos_dist  # Using mh_dist instead of iou_dist
    cost += 0.10 * conf_distance(tracks, dets) + 0.05 * angle_distance(tracks, dets, frame_id, d_t)

    # Give penalty
    cost[:, len(dets_high):len(dets_high + dets_low)] += penalty_p
    cost[:, len(dets_high + dets_low):] += penalty_q

    # Apply IoU constraint if enabled
    if use_iou_constraint:
        cost[iou_sim <= iou_threshold] = 1.
    
    # Clip cost
    cost = np.clip(cost, 0, 1)

    # Iterative matching
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
        for t, d in matches_:
            cost[t, :] = 1.
            cost[:, d] = 1.

    # Find indices of unmatched tracks and detections
    m_tracks = [t for t, _ in matches]
    u_tracks = [t for t in range(len(tracks)) if t not in m_tracks]
    m_dets = [d for _, d in matches]
    u_dets = [d for d in range(len(dets)) if d not in m_dets]

    return matches, u_tracks, u_dets

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


# Alternative version that combines both Mahalanobis and IoU distances
def iterative_assignment_combined(tracks, dets_high, dets_low, dets_del_high, match_thr, penalty_p, penalty_q,
                                reduce_step, frame_id, d_t=3, 
                                mh_weight=0.2, iou_weight=0.3, cos_weight=0.5):
    """
    Alternative version that combines Mahalanobis distance with IoU distance.
    """
    matches = []
    dets = dets_high + dets_low + dets_del_high
    
    if len(tracks) == 0 or len(dets) == 0:
        return matches, list(range(len(tracks))), list(range(len(dets)))
    
    has_kalman_filter = hasattr(tracks[0], 'kalman_filter') and tracks[0].kalman_filter is not None
    # if has_kalman_filter:
    #     mh_dist = mahalanobis_distance(tracks, dets)
    #     # print(f"mhd: {mh_dist}")
    #     iou_sim, iou_dist = iou_distance(tracks, dets)
        
    #     spatial_dist = mh_weight * mh_dist + iou_weight * iou_dist
    # else:
    #     print("Warning: No Kalman filter found, using IoU distance only")
    iou_sim, spatial_dist = iou_distance(tracks, dets)
    mh_weight, iou_weight = 0, 0.5
    cos_weight = 0.5
    
    cos_dist = cos_distance(tracks, dets)
    
    cost = spatial_dist + 0.5 * cos_dist 
    cost += 0.10 * conf_distance(tracks, dets) + 0.05 * angle_distance(tracks, dets, frame_id, d_t)

    cost[:, len(dets_high):len(dets_high + dets_low)] += penalty_p
    cost[:, len(dets_high + dets_low):] += penalty_q

    if has_kalman_filter:
        cost[iou_sim <= 0.10] = 1.
    
    cost = np.clip(cost, 0, 1)

    while True:
        matches_ = associate(cost, match_thr)
        match_thr -= reduce_step

        if len(matches_) == 0:
            break

        matches += matches_

        for t, d in matches_:
            cost[t, :] = 1.
            cost[:, d] = 1.

    m_tracks = [t for t, _ in matches]
    u_tracks = [t for t in range(len(tracks)) if t not in m_tracks]
    m_dets = [d for _, d in matches]
    u_dets = [d for d in range(len(dets)) if d not in m_dets]

    return matches, u_tracks, u_dets