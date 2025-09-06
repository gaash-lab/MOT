from trackers.cmc import *
from trackers.utils import *
from trackers.track import *
from trackers.tools import *
import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
import torch.nn as nn
import random
import pickle
import os
from tqdm import tqdm
from trackers.cross_attention_matcher import *



class FeatureProjector(nn.Module):
    def __init__(self, feat_dim=2048, pose_dim=34, fusion_dim=2048):
        super().__init__()
        
        # Appearance projections
        self.query_app = nn.Linear(feat_dim, fusion_dim)
        self.key_app = nn.Linear(feat_dim, fusion_dim)
        self.value_app = nn.Linear(feat_dim, fusion_dim)

        self.query_pose = nn.Linear(pose_dim, fusion_dim)
        self.key_pose = nn.Linear(pose_dim, fusion_dim)
        self.value_pose = nn.Linear(pose_dim, fusion_dim)

        self.ln_app = nn.LayerNorm(fusion_dim)
        self.ln_pose = nn.LayerNorm(fusion_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, feat, pose, mode):
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).float().to(self.device)
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).float().to(self.device)

        if mode == 'query':
            app_proj = self.ln_app(self.query_app(feat))
            pose_proj = self.ln_pose(self.query_pose(pose))
        elif mode == 'key':
            app_proj = self.ln_app(self.key_app(feat))
            pose_proj = self.ln_pose(self.key_pose(pose))
        elif mode == 'value':
            app_proj = self.ln_app(self.value_app(feat))
            pose_proj = self.ln_pose(self.value_pose(pose))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        app_proj = app_proj.squeeze()
        pose_proj = pose_proj.squeeze()
        fused_input = torch.cat([app_proj, pose_proj], dim=-1)
        fused_embedding = self.fusion_mlp(fused_input)

        return fused_embedding



class CrossAttentionMatcher(nn.Module):
    def __init__(self, feat_dim=2048, pose_dim=34):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projector = FeatureProjector(feat_dim, pose_dim, feat_dim).to(self.device)
        self.scale = np.sqrt(feat_dim)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0).to(self.device)  #https://www.v7labs.com/blog/triplet-loss

    def compute_loss(self, anchor, positive, negative):
        q_anchor = self.projector(anchor["feat"], anchor["pose"], 'query')
        k_pos = self.projector(positive["feat"], positive["pose"], 'key')
        k_neg = self.projector(negative["feat"], negative["pose"], 'key')
        return self.triplet_loss(q_anchor, k_pos, k_neg)

    def process(self, vector):
        if vector.dim() == 1: # Convert 1D to 2D
            vector = vector.unsqueeze(0) 
        elif vector.dim() > 2: # Convert higher dimensions to 2D
            vector = vector.squeeze()  
            if vector.dim() > 1: # Ensure it's 2D
                vector = vector[0]  
            vector = vector.unsqueeze(0)

        return vector # Ensure it's 2D


    def compute_attention(self, query_feat, query_pose, memory_feats, memory_poses):
        """
        Handles all cases of input dimensions:
        - query_feat: (1, 2048) or (2048,)
        - pose_vector: (34,)
        - memory_feats: list of (1, 2048), (2048,), or (N, 2048)
        """
        # with torch.no_grad():  # Disable gradient calculation
            # Convert query to proper 2D tensor (1, 2048)
        q_tensor = torch.from_numpy(query_feat).float().to(self.device)
        q_tensor = self.process(q_tensor)

        q_pose = torch.from_numpy(query_pose).float().to(self.device)
        q_pose = self.process(q_pose)

        # Handle empty memory bank
        if len(memory_feats) == 0:
            return np.zeros_like(q_tensor.squeeze(0).numpy()), np.zeros(0)
            
        mem_tensors = []
        mem_poses = []
        for f, p in zip(memory_feats, memory_poses):
            # Process appearance feature
            ft = torch.from_numpy(f).float().to(self.device) if isinstance(f, np.ndarray) else f.float().to(self.device)
            ft = self.process(ft)
            mem_tensors.append(ft)
            
            # Process pose vector
            pt = torch.from_numpy(p).float().to(self.device) if isinstance(p, np.ndarray) else p.float().to(self.device)
            pt = self.process(pt)
            mem_poses.append(pt)
        
        
        k_tensor = torch.cat(mem_tensors, dim=0) 
        k_pose = torch.cat(mem_poses, dim=0)

        q = self.projector(q_tensor, q_pose, 'query')  
        k = self.projector(k_tensor, k_pose, 'key')    
        v = self.projector(k_tensor, k_pose, 'value')  

        if q.dim() == 1:
            q = q.unsqueeze(0)
        if k.dim() == 1:
            k = k.unsqueeze(0)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        scores = torch.mm(q, k.t()) / self.scale  
        attn = torch.softmax(scores, dim=-1)
        attended = torch.mm(attn, v).squeeze(0)  
        
        return attended.detach().cpu().numpy(), attn.squeeze(0).detach().cpu().numpy()



def cosine_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    return dot(a, b) / (norm(a) * norm(b) + 1e-6)


class Tracker(object):
    def __init__(self, args, vid_name):
        self.args = args
        self.max_time_lost = args.max_time_lost
        self.memory_bank = []  # Stores dicts with id, feat, area, hw
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = CrossAttentionMatcher(args.feat_dim, args.pose_dim).to(self.device)
        self.association_engine = CrossAssociationEngine(args.feat_dim, args.pose_dim).to(self.device)
        self.load_checkpoint(args.association_checkpoint, self.association_engine)
        self.association_engine.eval()  # Set to evaluation mode
        if args.mode == "train_memory":
            self.optimizer = torch.optim.Adam(self.matcher.parameters(), lr=1e-4) 
        else:
            self.load_checkpoint(args.checkpoint_path, self.matcher)
            self.matcher.eval()  # Set to evaluation mode
        self.train_pickle = args.train_pickle
        self.epochs = args.num_epochs
        self.hard_mining = args.hard_mining

        self.tracks = []
        self.frame_id = 0
        self.counter = TrackCounter()

        if args.mode != "train_memory":
            self.cmc = CMC(vid_name)

    def load_checkpoint(self, checkpoint_path, matcher):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        matcher.load_state_dict(checkpoint["model_state"])


    def train_memory_bank(self):
        with open(self.train_pickle, "rb") as f:
            features_dict = pickle.load(f)

        best_loss = float('inf')
        batch_size = 32  

        for epoch in range(self.epochs):
            losses = []

            for _ in tqdm(range(100), desc=f"Epoch {epoch}"):  
                video = random.choice(list(features_dict.keys()))
                all_frames = features_dict[video]

                detections = []
                for frame_id, dets in all_frames.items():
                    for d in dets:
                        detections.append({
                            "video": video,
                            "frame": frame_id,
                            "track_id": d["track_id"],
                            "feat": d["embedding"],
                            "pose": d["pose"]
                        })


                batch_anchors = random.sample(detections, min(batch_size, len(detections)))

                for anchor in batch_anchors:
                    positives = [d for d in detections if d["track_id"] == anchor["track_id"] and d["frame"] != anchor["frame"]]
                    negatives = [d for d in detections if d["track_id"] != anchor["track_id"]]

                    if not positives or not negatives:
                        continue  # skip this anchor if we can't sample properly

                    positive = random.choice(positives)

                    anchor_emb = self.matcher.projector(
                        torch.from_numpy(np.array(anchor["feat"])).unsqueeze(0).to(self.device).float(),
                        torch.from_numpy(np.array(anchor["pose"])).unsqueeze(0).to(self.device).float(),
                        mode='query'
                    ).detach()

                    neg_feats = np.stack([neg["feat"] for neg in negatives])  
                    neg_poses = np.stack([neg["pose"] for neg in negatives]) 

                    neg_feats_tensor = torch.from_numpy(neg_feats).float().to(self.device)
                    neg_poses_tensor = torch.from_numpy(neg_poses).float().to(self.device)

                    neg_embs = self.matcher.projector(neg_feats_tensor, neg_poses_tensor, mode='key').detach()  

                    if anchor_emb.dim() == 1:
                        anchor_emb = anchor_emb.unsqueeze(0)

                    sims = torch.nn.functional.cosine_similarity(anchor_emb, neg_embs, dim=-1)

                    hardest_idx = torch.argmax(sims).item()
                    hardest_negative = negatives[hardest_idx]

                    loss = self.matcher.compute_loss(anchor, positive, hardest_negative)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())

            mean_loss = np.mean(losses)
            print(f"[Epoch {epoch}] Loss: {mean_loss:.4f}")

            self.save_checkpoint(epoch, mean_loss)
            if mean_loss < best_loss:
                best_loss = mean_loss
                self.save_checkpoint(epoch, mean_loss, best=True)

    def sample_triplet(self, features_dict):
        video = random.choice(list(features_dict.keys()))
        all_frames = features_dict[video]

        detections = []
        for frame_id, dets in all_frames.items():
            for d in dets:
                detections.append({
                    "video": video,
                    "frame": frame_id,
                    "track_id": d["track_id"],
                    "feat": d["embedding"],
                    "pose": d["pose"]
                })

        anchor = random.choice(detections)
        same_id = [d for d in detections if d["track_id"] == anchor["track_id"] and d["frame"] != anchor["frame"]]
        diff_id = [d for d in detections if d["track_id"] != anchor["track_id"]]

        if not same_id or not diff_id:
            raise ValueError("Insufficient positive/negative samples in video.")

        positive = random.choice(same_id)
        negative = random.choice(diff_id)

        return anchor, positive, negative

    def save_checkpoint(self, epoch, loss, best=False):
        filename = os.path.join(self.args.output_memory, f"epoch_{epoch}_loss_{loss}.pth" if not best else f"memory_bank_best_loss_{loss}.pth")
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "model_state": self.matcher.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, filename)


    def match_with_memory(self, det):
        feature = det.feat
        pose = det.pose

        if not self.memory_bank:
            return None

        mem_features = [mem['feat'] for mem in self.memory_bank]
        mem_pose  = [mem['pose'] for mem in self.memory_bank]

        with torch.no_grad():  # Disable gradient calculation
            attended_feat, attn_weights = self.matcher.compute_attention(feature, pose, mem_features, mem_pose)

        best_idx = np.argmax(attn_weights)
        best_score = attn_weights[best_idx]
        best_match = self.memory_bank[best_idx] if best_score > 0.70 else None
        if not best_match:
            return None
        # self.memory_bank.remove(best_match)
        # print(f"Reusing track ID {best_match['id']} with score {best_score:.2f}")
        # return best_match['id']
        
        # Also verify with cosine similarity for robustness
        if best_match:
            sim = cosine_sim(best_match['feat'], feature)
            if sim > 0.60:
                self.memory_bank.remove(best_match)
                print(f"Reusing track ID {best_match['id']} at frame {self.frame_id} with score {sim:.2f}")
                return best_match['id']
        return None
        
    def init_tracks(self, dets):
        tracks = [t for t in self.tracks if t.state in (TrackState.Tracked, TrackState.New)]
        iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
        scores = np.array([d.score for d in dets])

        allow_indices = track_aware_nms(iou_sim, scores, len(tracks), self.args.tai_thr, self.args.init_thr)

        for idx, flag in enumerate(allow_indices):
            if flag:
                det = dets[idx]
                reused_id = self.match_with_memory(det)
                if reused_id is not None:
                    det.initiate(self.frame_id, self.counter, reused_id)
                else:
                    det.initiate(self.frame_id, self.counter)
                self.tracks.append(det)

    def update(self, dets, dets_95):
        self.frame_id += 1

        dets_del = find_deleted_detections(dets, dets_95)
        dets = [Track(self.args, d) for d in dets]
        dets_del = [Track(self.args, d) for d in dets_del]

        dets_high = [d for d in dets if d.score > self.args.det_thr]
        dets_low = [d for d in dets if d.score <= self.args.det_thr]
        dets_del_high = [d for d in dets_del if d.score > self.args.det_thr]

        tracked_lost = [t for t in self.tracks if t.state in (TrackState.Tracked, TrackState.Lost)]
        new = [t for t in self.tracks if t.state == TrackState.New]

        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(tracked_lost, warp_matrix)
        apply_cmc(new, warp_matrix)

        [t.predict() for t in tracked_lost + new]

        dets_combined = dets_high + dets_low + dets_del_high
        matches, u_tracks, u_dets = iterative_assignment(tracked_lost, dets_high, dets_low, dets_del_high,
                                                         self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
                                                         self.args.reduce_step, self.frame_id)

        # matches, u_tracks, u_dets = cross_attention_assignment(tracked_lost, dets_high, self.association_engine, self.args.match_thr)
        
        # Match unmatched tracks (u_tracks) with low confidence detections (dets_low)
        # if len(u_tracks) > 0 and len(dets_low) > 0:
        #     tracks_unmatched = [tracked_lost[i] for i in u_tracks]
        #     matches_low, u_tracks_low, u_dets_low = cross_attention_assignment(
        #     tracks_unmatched, dets_low, self.association_engine, self.args.match_thr
        #     )
            
        #     # Update original matches with new matches from low confidence detections
        #     offset = len(dets_high)  # Offset for indexing into dets_combined
        #     for t, d in matches_low:
        #         matches.append((u_tracks[t], d + offset))  # Add offset to detection index
            
        #     # Update u_tracks to those still unmatched after low dets matching
        #     u_tracks = [u_tracks[i] for i in u_tracks_low]
            
        #     # Update dets_low to only include unmatched detections
        #     dets_low = [dets_low[i] for i in u_dets_low]
            
        for t, d in matches:
            tracked_lost[t].update(self.frame_id, dets_combined[d])

        for t in u_tracks:
            tracked_lost[t].mark_lost()

        dets_high_left = [dets_high[i] for i in u_dets if i < len(dets_high)]
        matches, u_tracks, u_dets = iterative_assignment(new, dets_high_left, [], [],
                                   self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
                                   self.args.reduce_step, self.frame_id)

        for t, d in matches:
            new[t].update(self.frame_id, dets_high_left[d])

        for t in u_tracks:
            new[t].mark_removed()

        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                h, w = track.x1y1wh[3], track.x1y1wh[2]
                area = h * w
                feat = track.feat
                self.memory_bank.append({
                    'hw': (h, w),
                    'area': area,
                    'id': track.track_id,
                    'feat': feat,
                    'pose': track.pose.copy()
                })
                track.mark_removed()

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        self.init_tracks([dets_high_left[i] for i in u_dets])

        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def update_without_detections(self):
        self.frame_id += 1
        self.tracks = [t for t in self.tracks if t.state != TrackState.New]

        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(self.tracks, warp_matrix)
        [t.predict() for t in self.tracks]

        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return [t for t in self.tracks if t.state == TrackState.Tracked]
