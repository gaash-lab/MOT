from trackers.cmc import *
from trackers.utils import *
from trackers.track import *
import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
import torch.nn as nn


class FeatureProjector(nn.Module):
    def __init__(self, feat_dim=2048, pose_dim=34):
        super().__init__()
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)

        self.pose_query = nn.Linear(pose_dim, feat_dim)  # Project pose to same dimension as appearance
        self.pose_key = nn.Linear(pose_dim, feat_dim)
        self.pose_value = nn.Linear(pose_dim, feat_dim)
        
        
    def forward(self, feat, pose, mode):
        if mode == 'query':
            app_query = self.query(feat)
            pose_query = self.pose_query(pose)
            return app_query + pose_query  
        elif mode == 'key':
            app_key = self.key(feat)
            pose_key = self.pose_key(pose)
            return app_key + pose_key  
        elif mode == 'value':
            app_value = self.value(feat)
            pose_value = self.pose_value(pose)
            return app_value + pose_value  
        return feat



class CrossAttentionMatcher:
    def __init__(self, feat_dim=2048):
        self.projector = FeatureProjector(feat_dim)
        self.scale = np.sqrt(feat_dim)
        
    def compute_attention(self, query_feat, query_pose, memory_feats, memory_poses):
        """
        Handles all cases of input dimensions:
        - query_feat: (1, 2048) or (2048,)
        - pose_vector: (34,)
        - memory_feats: list of (1, 2048), (2048,), or (N, 2048)
        """
        # Convert query to proper 2D tensor (1, 2048)
        q_tensor = torch.from_numpy(query_feat).float()
        if q_tensor.dim() == 1:
            q_tensor = q_tensor.unsqueeze(0)  # (2048,) -> (1, 2048)
        elif q_tensor.dim() > 2:
            q_tensor = q_tensor.squeeze()  # (1,1,2048) -> (2048,)
            if q_tensor.dim() > 1:
                q_tensor = q_tensor[0]  # Take first if still multi-dimensional
            q_tensor = q_tensor.unsqueeze(0)

        q_pose = torch.from_numpy(query_pose).float()
        if q_pose.dim() == 1:
            q_pose = q_pose.unsqueeze(0)  
        elif q_pose.dim() > 2:
            q_pose = q_pose.squeeze()  
            if q_pose.dim() > 1:
                q_pose = q_pose[0]  
            q_pose = q_pose.unsqueeze(0)

        # Handle empty memory bank
        if len(memory_feats) == 0:
            return np.zeros_like(q_tensor.squeeze(0).numpy()), np.zeros(0)
            
        mem_tensors = []
        mem_poses = []
        for f, p in zip(memory_feats, memory_poses):
            # Process appearance feature
            ft = torch.from_numpy(f).float() if isinstance(f, np.ndarray) else f.float()
            if ft.dim() == 1:
                ft = ft.unsqueeze(0)
            elif ft.dim() > 2:
                ft = ft.squeeze()
                if ft.dim() > 1:
                    ft = ft[0]
                ft = ft.unsqueeze(0)
            mem_tensors.append(ft)
            
            # Process pose vector
            pt = torch.from_numpy(p).float() if isinstance(p, np.ndarray) else p.float()
            if pt.dim() == 1:
                pt = pt.unsqueeze(0)
            elif pt.dim() > 2:
                pt = pt.squeeze()
                if pt.dim() > 1:
                    pt = pt[0]
                pt = pt.unsqueeze(0)
            mem_poses.append(pt)
        
        
        k_tensor = torch.cat(mem_tensors, dim=0) 
        k_pose = torch.cat(mem_poses, dim=0)

        q = self.projector(q_tensor, q_pose, 'query')  
        k = self.projector(k_tensor, k_pose, 'key')    
        v = self.projector(k_tensor, k_pose, 'value')  

        # Compute attention scores
        scores = torch.mm(q, k.t()) / self.scale  
        attn = torch.softmax(scores, dim=-1)
        attended = torch.mm(attn, v).squeeze(0)  
        
        return attended.detach().numpy(), attn.squeeze(0).detach().numpy()


def cosine_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    return dot(a, b) / (norm(a) * norm(b) + 1e-6)


class Tracker(object):
    def __init__(self, args, vid_name):
        self.args = args
        self.max_time_lost = args.max_time_lost
        self.memory_bank = []  # Stores dicts with id, feat, area, hw
        self.matcher = CrossAttentionMatcher(args.feat_dim, args.pose_dim)

        self.tracks = []
        self.frame_id = 0
        self.counter = TrackCounter()

        self.cmc = CMC(vid_name)

    def match_with_memory(self, det):
        feature = det.feat
        pose = det.pose

        if not self.memory_bank:
            return None

        mem_features = [mem['feat'] for mem in self.memory_bank]
        mem_pose  = [mem['pose'] for mem in self.memory_bank]

        attended_feat, attn_weights = self.matcher.compute_attention(feature, pose, mem_features, mem_pose)

        best_idx = np.argmax(attn_weights)
        best_score = attn_weights[best_idx]
        best_match = self.memory_bank[best_idx] if best_score > 0.50 else None
        if not best_match:
            return None
        self.memory_bank.remove(best_match)
        print(f"Reusing track ID {best_match['id']} with score {best_score:.2f}")
        return best_match['id']
        
        # Also verify with cosine similarity for robustness
        # if best_match:
        #     sim = cosine_sim(best_match['feat'], feature)
        #     if sim > 0.70:
        #         self.memory_bank.remove(best_match)
        #         return best_match['id']
        
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
                # if not any(mem['id'] == track.track_id for mem in self.memory_bank):
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
