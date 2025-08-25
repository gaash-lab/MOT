import argparse
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from cross_attention_matcher import CrossAssociationEngine

# Import the enhanced cost functions
from enhanced_costs import compute_enhanced_costs, cosine_feature_distance

def prepare_enhanced_batch(features_dict, device='cuda', min_temporal_gap=1, max_temporal_gap=5):
    """Enhanced batch preparation with better temporal modeling and cost computation"""
    video = np.random.choice(list(features_dict.keys()))
    frames = list(features_dict[video].items())
    
    if len(frames) < max_temporal_gap + 1:
        max_temporal_gap = len(frames) - 1
    
    # Sample frames with variable temporal gaps (better for learning temporal consistency)
    frame1_idx = np.random.randint(0, len(frames) - max_temporal_gap)
    temporal_gap = np.random.randint(min_temporal_gap, max_temporal_gap + 1)
    frame2_idx = frame1_idx + temporal_gap
    
    frame1_id, frame1_dets = frames[frame1_idx]
    frame2_id, frame2_dets = frames[frame2_idx]
    
    def prepare_enhanced_data(dets):
        if not dets:
            return {
                'features': torch.zeros((0, 2048), dtype=torch.float32),
                'poses': torch.zeros((0, 34), dtype=torch.float32),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'confidences': torch.zeros((0,), dtype=torch.float32),
                'ids': []
            }
            
        feats = torch.stack([torch.tensor(d['embedding'], dtype=torch.float32) for d in dets])
        poses = torch.stack([torch.tensor(d['pose'], dtype=torch.float32) for d in dets])
        boxes = torch.stack([torch.tensor(d['bbox'][:4], dtype=torch.float32) for d in dets])
        
        # Extract confidences (assuming they exist in your data)
        print([d["bbox"] for d in dets])  # Debug print to check bbox format
        confs = torch.tensor([d["bbox"][4:] for d in dets], dtype=torch.float32)
        
        ids = [d['track_id'] for d in dets]
        
        return {
            'features': feats,
            'poses': poses,
            'boxes': boxes,
            'confidences': confs,
            'ids': ids
        }
    
    tracks = prepare_enhanced_data(frame1_dets)
    detections = prepare_enhanced_data(frame2_dets)
    
    # Compute enhanced costs with multiple metrics
    if len(tracks['features']) > 0 and len(detections['features']) > 0:
        # Prepare data for enhanced cost computation
        track_data = {
            'features': tracks['features'],
            'poses': tracks['poses'],
            'boxes': tracks['boxes'],
            'confidences': tracks['confidences']
        }
        
        det_data = {
            'features': detections['features'],
            'poses': detections['poses'], 
            'boxes': detections['boxes'],
            'confidences': detections['confidences']
        }
        
        cost_matrix = compute_enhanced_costs(track_data, det_data, temporal_gap)
    else:
        cost_matrix = None
    
    # Prepare ground truth matches
    gt_matches = []
    track_ids = tracks['ids']
    det_ids = detections['ids']
    
    for t_idx, track_id in enumerate(track_ids):
        if track_id in det_ids:
            d_idx = det_ids.index(track_id)
            gt_matches.append((t_idx, d_idx))
    
    # Move to device
    tracks = {k: v.to(device) if torch.is_tensor(v) else v for k, v in tracks.items()}
    detections = {k: v.to(device) if torch.is_tensor(v) else v for k, v in detections.items()}
    
    if cost_matrix is not None:
        cost_matrix = cost_matrix.to(device)
    
    return tracks, detections, cost_matrix, gt_matches, temporal_gap

def hard_negative_mining(attn_weights, gt_matches, neg_ratio=3):
    """
    Mine hard negatives to improve training
    Focus on high-confidence wrong matches
    """
    device = attn_weights.device
    targets = torch.zeros_like(attn_weights)
    
    # Set positive targets
    for t_idx, d_idx in gt_matches:
        if t_idx < targets.shape[0] and d_idx < targets.shape[1]:
            targets[t_idx, d_idx] = 1
    
    # Find hard negatives (high predicted score but wrong match)
    with torch.no_grad():
        probs = torch.sigmoid(attn_weights)
        
        # Get negative samples (where target = 0)
        neg_mask = (targets == 0)
        neg_probs = probs * neg_mask.float()
        
        # Select top-k hard negatives
        num_pos = len(gt_matches)
        num_neg = min(num_pos * neg_ratio, neg_mask.sum().item())
        
        if num_neg > 0:
            _, hard_neg_indices = neg_probs.view(-1).topk(num_neg)
            hard_neg_mask = torch.zeros_like(neg_mask.view(-1))
            hard_neg_mask[hard_neg_indices] = 1
            hard_neg_mask = hard_neg_mask.view(neg_mask.shape)
            
            # Create sample weights (give more weight to positives and hard negatives)
            sample_weights = torch.ones_like(targets) * 0.1  # Low weight for easy negatives
            sample_weights[targets == 1] = 1.0  # High weight for positives
            sample_weights[hard_neg_mask] = 0.5  # Medium weight for hard negatives
        else:
            sample_weights = torch.ones_like(targets)
    
    return sample_weights

def compute_identity_consistency_loss(attn_weights, gt_matches, temperature=0.1):
    """
    Additional loss to encourage identity consistency
    Uses contrastive learning principles
    """
    if len(gt_matches) == 0:
        return torch.tensor(0.0, device=attn_weights.device)
    
    device = attn_weights.device
    
    # Create positive and negative pairs
    pos_scores = []
    neg_scores = []
    
    for t_idx, d_idx in gt_matches:
        if t_idx < attn_weights.shape[0] and d_idx < attn_weights.shape[1]:
            # Positive score
            pos_scores.append(attn_weights[t_idx, d_idx])
            
            # Negative scores (all other detections for this track)
            neg_mask = torch.ones(attn_weights.shape[1], dtype=torch.bool)
            neg_mask[d_idx] = False
            if neg_mask.any():
                neg_scores.extend(attn_weights[t_idx, neg_mask].tolist())
    
    if not pos_scores or not neg_scores:
        return torch.tensor(0.0, device=device)
    
    pos_scores = torch.tensor(pos_scores, device=device)
    neg_scores = torch.tensor(neg_scores, device=device)
    
    # Contrastive loss
    pos_exp = torch.exp(pos_scores / temperature)
    neg_exp = torch.exp(neg_scores / temperature)
    
    # InfoNCE-style loss
    loss = -torch.log(pos_exp.mean() / (pos_exp.mean() + neg_exp.mean() + 1e-8))
    
    return loss

def train_with_enhanced_costs(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.train_pickle, 'rb') as f:
        features_dict = pickle.load(f)

    # Enhanced model with larger hidden dimension for handling more cost types
    model = CrossAssociationEngine(feat_dim=2048, pose_dim=34, hidden_dim=512).float().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    saved_checkpoints = []
    best_loss = float('inf')
    best_checkpoint_path = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_focal_loss = 0
        total_consistency_loss = 0
        
        progress = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}")
        
        for step in progress:
            tracks, dets, cost_matrix, gt_matches, temporal_gap = prepare_enhanced_batch(
                features_dict, device, min_temporal_gap=1, max_temporal_gap=5
            )

            if len(tracks['features']) == 0 or len(dets['features']) == 0 or len(gt_matches) == 0:
                continue

            # Forward pass
            attn_weights = model(tracks, dets, cost_matrix)
            
            # Compute main focal loss with hard negative mining
            sample_weights = hard_negative_mining(attn_weights, gt_matches)
            focal_loss = model.compute_weighted_loss(attn_weights, gt_matches, sample_weights)
            
            # Compute identity consistency loss
            consistency_loss = compute_identity_consistency_loss(attn_weights, gt_matches)
            
            # Combined loss
            total_step_loss = focal_loss + 0.1 * consistency_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_step_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Accumulate losses
            total_loss += total_step_loss.item()
            total_focal_loss += focal_loss.item()
            total_consistency_loss += consistency_loss.item()
            
            # Update progress bar
            progress.set_postfix({
                'loss': total_step_loss.item(),
                'focal': focal_loss.item(),
                'consist': consistency_loss.item()
            })
        
        # Step scheduler
        scheduler.step()
        
        avg_loss = total_loss / max(1, args.steps_per_epoch)
        avg_focal = total_focal_loss / max(1, args.steps_per_epoch)
        avg_consist = total_consistency_loss / max(1, args.steps_per_epoch)
        
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Focal: {avg_focal:.4f}, Consistency: {avg_consist:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        best_loss, best_checkpoint_path = save_checkpoint(
            model, optimizer, epoch, avg_loss, args.output_dir,
            saved_checkpoints, best_loss, best_checkpoint_path
        )

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

    # Keep only last 10 checkpoints
    if len(saved_checkpoints) > 10:
        to_delete = saved_checkpoints.pop(0)
        if os.path.abspath(to_delete) != os.path.abspath(best_checkpoint_path):
            try:
                os.remove(to_delete)
            except Exception as e:
                print(f"Warning: couldn't delete {to_delete} - {e}")

    return best_loss, best_checkpoint_path

def main():
    parser = argparse.ArgumentParser(description="Train Enhanced CrossAssociationEngine")
    
    parser.add_argument('--train_pickle', type=str, 
                       default="/DATA/Tawheed/track_files/dance_train_with_pose.pickle", 
                       help='Path to training pickle file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Training steps per epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per training step')
    parser.add_argument('--output_dir', type=str, default='checkpoints_enhanced', 
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_with_enhanced_costs(args)

if __name__ == "__main__":
    main()