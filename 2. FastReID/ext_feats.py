import os
import cv2
import pickle
import random
import argparse
import numpy as np
from fastreid.emb_computer import EmbeddingComputer


def make_parser():
    # Initialization
    parser = argparse.ArgumentParser("Track")

    # Data args
    parser.add_argument("--dataset", type=str, default="mot17_half")
    parser.add_argument("--data_path", type=str, default="/DATA/Tawheed/MOTDatasets/mot17_half/train/")
    parser.add_argument("--pickle_path", type=str, default="/home/tawheed/MOT/mot17_half_train.pickle")
    parser.add_argument("--output_path", type=str, default="/DATA/Tawheed/track_files/pickle_path/features/mot17_half_train.pickle")
    parser.add_argument("--config_path", type=str, default="configs/MOT17_half/sbs_S50.yml")
    parser.add_argument("--weight_path", type=str, default="weights/mot17_half_sbs_S50.pth")

    # Else
    parser.add_argument("--seed", type=float, default=10000)

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    embedder = EmbeddingComputer(config_path=args.config_path, weight_path=args.weight_path)

    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    
    for vid_name in detections.keys():
        for frame_id in detections[vid_name].keys():
            # If there is no detection
            if detections[vid_name][frame_id] is None:
                continue
            img_path = os.path.join(args.data_path + vid_name + '/img1/%06d.jpg' % frame_id)
            
            if not os.path.exists(img_path):
                img_path = os.path.join(args.data_path + vid_name + '/img1/%08d.jpg' % frame_id)

            img = cv2.imread(img_path)
            
            # Get detection
            detection = detections[vid_name][frame_id]

            # Get features
            if detection is not None:
                detection = np.array(detection)
                embedding = embedder.compute_embedding(img, detection[:, :4])
                detections[vid_name][frame_id] = np.concatenate([detection, embedding], axis=1)

            # Logging
            print(vid_name, frame_id, flush=True)

    # Save
    with open(args.output_path, 'wb') as handle:
        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
