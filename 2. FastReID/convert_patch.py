import os
import cv2
import pandas as pd
from tqdm import tqdm

def extract_mot_patches(gt_path, img_dir, output_dir, cam_id='001'):
    df = pd.read_csv(gt_path, header=None)
    df.columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'class', 'visibility', 'ignore']

    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(gt_path)):
        if int(row['class']) != 1 or float(row['visibility']) < 0.5:
            continue

        frame_id = int(row['frame'])
        track_id = int(row['track_id'])

        img_file = os.path.join(img_dir, f"{frame_id:08d}.jpg") # for dancetrack 8 and other 6
        if not os.path.exists(img_file):
            continue

        img = cv2.imread(img_file)
        if img is None:
            continue

        height, width = img.shape[:2]

        x1, y1 = max(0, int(row['x'])), max(0, int(row['y']))
        x2 = min(width, x1 + int(row['w']))
        y2 = min(height, y1 + int(row['h']))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Output format: {track_id}_c{cam_id}_{frame_id}.jpg
        save_filename = f"{track_id}_c{cam_id}_{frame_id:06d}.jpg"
        save_path = os.path.join(output_dir, save_filename)
        cv2.imwrite(save_path, crop)


def extract_all_sequences(dataset_root):
    train_dir = os.path.join(dataset_root, 'train')
    patch_dir = os.path.join(dataset_root, 'patch')
    os.makedirs(patch_dir, exist_ok=True)

    for seq_name in os.listdir(train_dir):
        seq_path = os.path.join(train_dir, seq_name)
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        img_dir = os.path.join(seq_path, 'img1')

        if not os.path.exists(gt_path) or not os.path.exists(img_dir):
            continue

        print(f"Processing {seq_name}...")

        # Extract cam_id from sequence name if available (e.g., MOT17-02-FRCNN → cam_id=002)
        cam_id = ''.join(filter(str.isdigit, seq_name))[-3:].zfill(3)

        extract_mot_patches(gt_path, img_dir, patch_dir, cam_id=cam_id)


if __name__ == "__main__":
    extract_all_sequences("/DATA/Tawheed/MOTDatasets/DanceTrack/")
