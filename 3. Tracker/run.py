import os
import torch
import pickle
import argparse
from utils.etc import *
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from trackers.tracker import Tracker
from utils.gbi import gb_interpolation


def make_parser():
    parser = argparse.ArgumentParser("Tracker")

    # Basic
    parser.add_argument("--pickle_dir", type=str, default="/DATA/Tawheed/track_files/pickle_path/normalized/")
    parser.add_argument("--output_dir", type=str, default="../outputs/3. track/")
    parser.add_argument("--train_pickle", type=str, default="/DATA/Tawheed/track_files/dance_train_with_pose1.pickle")
    parser.add_argument("--data_dir", type=str, default="/DATA/Tawheed/MOTDatasets/")
    parser.add_argument("--checkpoint_path", type=str, default="/home/tawheed/MOT/TrackTrack/outputs/memory/epoch_17_loss_0.24095295429229735.pth")
    parser.add_argument("--association_checkpoint", type=str, default="/home/tawheed/MOT/TrackTrack/cross_attn_matcher/checkpoints/best_model_loss_0.0419.pth")
    # /home/tawheed/MOT/TrackTrack/outputs/memory_1/memory_bank_best_loss_0.13229473876953124.pth
    parser.add_argument("--output_memory", type=str, default="../outputs/memory/")
    parser.add_argument("--dataset", type=str, default="DanceTrack")
    parser.add_argument("--mode", type=str, default="train_memory")
    parser.add_argument("--seed", type=float, default=10000)

    # For trackers
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--max_time_lost", type=float, default=25)
    parser.add_argument("--penalty_p", type=float, default=0.20)
    parser.add_argument("--penalty_q", type=float, default=0.40)
    parser.add_argument("--reduce_step", type=float, default=0.05)
    parser.add_argument("--tai_thr", type=float, default=0.55)
    parser.add_argument("--feat_dim", type=int, default=2048)
    parser.add_argument("--pose_dim", type=int, default=34)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--hard_mining", type=bool, default=True)

    return parser


def track(detections, detections_95, data_path, result_folder, mode):
    # For each video
    total_time, total_count = 0, 0
    for vid_name in detections.keys():
        print(f"\n\n\nTracking {vid_name}...")
        # Set proper parameters
        set_parameters(args, vid_name, mode)

        # Set max time lost
        seq_info = open(data_path + vid_name + '/seqinfo.ini', mode='r')
        for s_i in seq_info.readlines():
            if 'frameRate' in s_i:
                args.max_time_lost = int(s_i.split('=')[-1])
            if 'imWidth' in s_i:
                args.img_w = int(s_i.split('=')[-1])
            if 'imHeight' in s_i:
                args.img_h = int(s_i.split('=')[-1])

        # Set tracker
        tracker = Tracker(args, vid_name)

        # For each frame
        results = []
        for frame_id in detections[vid_name].keys():
            # Run tracking
            start = time.time()
            if detections[vid_name][frame_id] is not None:
                track_results = tracker.update(detections[vid_name][frame_id], detections_95[vid_name][frame_id])
            else:
                track_results = tracker.update_without_detections()
            total_time += time.time() - start
            total_count += 1

            # Filter out the results
            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                # Check aspect ratio
                if 'MOT' in data_path.split('/')[-2] and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
                    print(f"Skipping track {t.track_id} in frame {frame_id} due to aspect ratio.")
                    continue

                # Check track id, minimum box area
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)

            # Merge
            results.append([frame_id, track_ids, x1y1whs, scores])

        # Logging & Write results
        result_filename = os.path.join(result_folder, '{}.txt'.format(vid_name))
        write_results(result_filename, results)

    return total_time, total_count


def run():
    if args.mode == "train_memory":
        if args.mode == "train_memory":
            tracker = Tracker(args, "")
            tracker.train_memory_bank()
    else:
        model = PostLinker()
        model.load_state_dict(torch.load('./AFLink/AFLink_epoch20.pth'))
        aflink_dataset = LinkData('', '')

        # Logging & Set proper parameters
        print('Running %s %s...' % (args.dataset, args.mode))
        set_parameters(args, args.dataset, args.mode)

        # Make result folder
        trackers_to_eval = args.pickle_path.split('/')[-1].split('.pickle')[0]
        result_folder = os.path.join(args.output_dir, trackers_to_eval)
        os.makedirs(result_folder, exist_ok=True)
        os.makedirs(result_folder + '_post/', exist_ok=True)

        # Read detection result
        with open(args.pickle_path, 'rb') as f:
            detections = pickle.load(f)
        with open(args.pickle_path_95, 'rb') as f:
            detections_95 = pickle.load(f)

        # Track
        total_time, total_count = track(detections, detections_95, args.data_path, result_folder, args.mode)

        # Post-processing
        print('Running post-processing...')
        for result_file in os.listdir(result_folder):
            # Set Path
            path_in = result_folder + '/' + str(result_file)
            path_out = result_folder + '_post/' + str(result_file)

            # Link
            if 'Dance' in args.dataset:
                linker = AFLink(path_in=path_in, path_out=path_out, model=model, dataset=aflink_dataset,
                                thrT=(0, 20), thrS=100, thrP=0.05)
                linker.link()

            # Gaussian Interpolation
            if 'MOT' in args.dataset:
                gb_interpolation(path_in, path_out, interval=30, tau=12)

        # Evaluation
        if args.mode == 'val':
            print('Evaluating...')
            evaluate(args, trackers_to_eval + '_post', args.dataset)

        # Logging
        print(total_count / total_time, flush=True)
        print('', flush=True)


if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Run
    run()
