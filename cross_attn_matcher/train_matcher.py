import argparse
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from cross_attention_matcher import CrossAssociationEngine

def prepare_batch(features_dict, device='cuda'):
    """Prepare training batch with guaranteed proper dimensions"""
    video = np.random.choice(list(features_dict.keys()))
    frames = list(features_dict[video].items())
    
    # Sample two frames with temporal gap
    frame1_idx = np.random.randint(0, len(frames)-3)
    frame2_idx = frame1_idx + np.random.randint(1, 3)
    
    frame1_id, frame1_dets = frames[frame1_idx]
    frame2_id, frame2_dets = frames[frame2_idx]
    
    # Convert to tensors with strict dimension checks
    def prepare_data(dets):
        if not dets:  # Handle empty detections
            return {
                'features': torch.zeros((0, 2048), dtype=torch.float32),
                'poses': torch.zeros((0, 34), dtype=torch.float32),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'ids': []
            }
            
        feats = torch.stack([torch.tensor(d['embedding'], dtype=torch.float32) for d in dets])
        poses = torch.stack([torch.tensor(d['pose'], dtype=torch.float32) for d in dets])
        boxes = torch.stack([torch.tensor(d['bbox'][:4], dtype=torch.float32) for d in dets])
        ids = [d['track_id'] for d in dets]
        return {
            'features': feats,
            'poses': poses,
            'boxes': boxes,
            'ids': ids
        }
    
    tracks = prepare_data(frame1_dets)
    detections = prepare_data(frame2_dets)
    
    # Compute costs only if we have detections in both frames
    if len(tracks['features']) > 0 and len(detections['features']) > 0:
        cost_vector = compute_costs(
            tracks['boxes'], 
            detections['boxes'],
            tracks['poses'],
            detections['poses'],
            tracks['features'],
            detections['features']
        )
    else:
        cost_vector = None
    
    # Prepare ground truth matches
    gt_matches = []
    track_ids = tracks['ids']
    det_ids = detections['ids']
    
    for t_idx, track_id in enumerate(track_ids):
        if track_id in det_ids:
            d_idx = det_ids.index(track_id)
            gt_matches.append((t_idx, d_idx))
    
    # Move to device with dimension preservation
    tracks = {
        'features': tracks['features'].to(device),
        'poses': tracks['poses'].to(device),
        'boxes': tracks['boxes'].to(device)
    }
    
    detections = {
        'features': detections['features'].to(device),
        'poses': detections['poses'].to(device),
        'boxes': detections['boxes'].to(device)
    }
    
    if cost_vector is not None:
        cost_vector = cost_vector.to(device)
    
    return tracks, detections, cost_vector, gt_matches

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
    kps1 = poses1.view(-1, 17, 2)  # (N,17,2)
    kps2 = poses2.view(-1, 17, 2)  # (M,17,2)
    
    valid1 = (kps1.abs().sum(dim=-1) > 1e-3)  # (N,17)
    valid2 = (kps2.abs().sum(dim=-1) > 1e-3)  # (M,17)
    
    def get_scale(kps, valid):
        scales = []
        for i in range(len(kps)):
            vis_kps = kps[i][valid[i]]  # (V,2)
            if len(vis_kps) == 0:
                scales.append(0.0)
                continue
            scale = (vis_kps.max(dim=0)[0] - vis_kps.min(dim=0)[0]).prod()
            scales.append(scale)
        return torch.tensor(scales, device=kps1.device)  # (N,)
    
    scale1 = get_scale(kps1, valid1)  # (N,)
    scale2 = get_scale(kps2, valid2)  # (M,)
    scale = (scale1[:,None] + scale2[None,:]) / 2  # (N,M)
    
    diff = kps1[:,None,:,:] - kps2[None,:,:,:]  # (N,M,17,2)
    sq_dist = (diff ** 2).sum(dim=-1)  # (N,M,17)
    
    oks_kpts = torch.exp(-sq_dist / (2 * scale[...,None] * kappa**2 + 1e-7))  # (N,M,17)
    
    valid_mask = valid1[:,None,:] & valid2[None,:,:]  # (N,M,17)
    oks_kpts = oks_kpts * valid_mask.float()
    
    valid_count = valid_mask.sum(dim=-1)  # (N,M)
    oks = oks_kpts.sum(dim=-1) / (valid_count + 1e-7)
    
    oks[valid_count == 0] = 0.0
    
    return oks

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

def compute_costs(track_boxes, det_boxes, track_poses, det_poses, track_feats, det_feats):
    """
    Compute combined spatial, shape, and feature costs
    Returns a (num_tracks, num_dets, 3) tensor
    """
    # with torch.no_grad():
        # Spatial cost (IoU)
    ious = box_iou(track_boxes, det_boxes)
    spatial_costs = 1.0 - ious  # (T, D)

    # Shape cost (OKS)
    shape_costs = 1.0 - compute_oks(track_poses, det_poses)  # (T, D)

    # Feature cost (Cosine distance)
    track_feats_np = track_feats.cpu().numpy() if isinstance(track_feats, torch.Tensor) else track_feats
    det_feats_np   = det_feats.cpu().numpy() if isinstance(det_feats, torch.Tensor) else det_feats
    feat_costs = cos_distance(track_feats_np, det_feats_np)  # (T, D)
    feat_costs = torch.from_numpy(feat_costs).to(spatial_costs.device).float()

    # size_costs = size_consistency_cost(track_boxes, det_boxes)

    # Stack into a 3D cost tensor
    return torch.stack([spatial_costs, shape_costs, feat_costs], dim=-1)  # (T, D, 4)

def save_checkpoint(model, optimizer, epoch, loss, output_dir, saved_checkpoints, best_loss, best_checkpoint_path):
    checkpoint_path = os.path.join(output_dir, f"model_{epoch}_loss_{loss:.4f}.pth")
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    saved_checkpoints.append(checkpoint_path)

    if loss < best_loss:
        best_loss = loss
        best_checkpoint_path = os.path.join(output_dir, f"best_model_loss_{best_loss:.4f}.pth")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': best_loss,
        }, best_checkpoint_path)

    if len(saved_checkpoints) > 10:
        to_delete = saved_checkpoints.pop(0)
        if os.path.abspath(to_delete) != os.path.abspath(best_checkpoint_path):
            try:
                os.remove(to_delete)
            except Exception as e:
                print(f"Warning: couldn't delete {to_delete} - {e}")

    return best_loss, best_checkpoint_path

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.train_pickle, 'rb') as f:
        features_dict = pickle.load(f)

    model = CrossAssociationEngine(feat_dim=2048, pose_dim=34, hidden_dim=512).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    saved_checkpoints = []
    best_loss = float('inf')
    best_checkpoint_path = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}")
        
        for _ in progress:
            tracks, dets, costs, gt_matches = prepare_batch(features_dict, device)

            if len(tracks['features']) == 0 or len(dets['features']) == 0:
                continue

            attn_weights = model(tracks, dets, costs)
            loss = model.compute_loss(attn_weights, gt_matches)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / max(1, args.steps_per_epoch)
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")

        best_loss, best_checkpoint_path = save_checkpoint(
            model, optimizer, epoch, avg_loss, args.output_dir,
            saved_checkpoints, best_loss, best_checkpoint_path
        )

def main():
    parser = argparse.ArgumentParser(description="Train CrossAssociationEngine on pose and spatial features")
    
    parser.add_argument('--train_pickle', type=str, default="/DATA/Tawheed/track_files/pickle_path/normalized/dance_train_with_pose.pickle", help='Path to training pickle file containing framewise detections.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--steps_per_epoch', type=int, default=500, help='Training steps per epoch.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per training step.')
    parser.add_argument('--output_dir', type=str, default='scaled_Pose', help='Directory to save model checkpoints.')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()