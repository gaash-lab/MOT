import os
import cv2
import pandas as pd
from tqdm import tqdm

def extract_mot_patches(gt_path, img_dir, output_dir, cam_id='001'):
    try:
        df = pd.read_csv(gt_path, header=None)
    except Exception as e:
        print(f"\nERROR reading {gt_path}: {str(e)}")
        return

    if df.empty:
        print(f"\nEmpty ground truth file: {gt_path}")
        return
    
    df.columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'class', 'visibility', 'ignore']
    saved_count = 0
    debug_count = 0  # Track first few saves for debugging

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(img_dir)}"):
        # Skip non-person or low visibility objects
        if int(row['class']) != 1 or float(row['visibility']) < 0.25:  # Lowered visibility threshold
            continue
        
        frame_id = int(row['frame'])
        track_id = int(row['track_id'])
        img_file = os.path.join(img_dir, f"{frame_id:08d}.jpg")
        if not os.path.exists(img_file):
            print(f"Does not exist: {img_file}")
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
        print(f"\n\n\n{track_id}_c{cam_id}_{frame_id:06d}.jpg")
        print(output_dir)
        # Create output filename directly in patch folder
        save_filename = f"{track_id}_c{cam_id}_{frame_id:06d}.jpg"
        save_path = os.path.join(output_dir, save_filename)
        
        # Debug output for first 5 saves
        if debug_count < 5:
            print(f"\nDebug save {debug_count + 1}:")
            print(f"Source: {img_file}")
            print(f"Crop size: {crop.shape}")
            print(f"Saving to: {save_path}")
            debug_count += 1

        if cv2.imwrite(save_path, crop):
            saved_count += 1
        else:
            print(f"\nFAILED to save: {save_path}")

    print(f"\nTotal saved patches: {saved_count} to {output_dir}")

def extract_all_sequences(dataset_root):
    patch_dir = os.path.join(dataset_root, 'patch')
    os.makedirs(patch_dir, exist_ok=True)
    print(f"\nAll patches will be saved directly to: {patch_dir}")

    for split in ['train']:
        split_dir = os.path.join(dataset_root, split)
        
        if not os.path.exists(split_dir):
            print(f"\nSplit directory not found: {split_dir}")
            continue
            
        print(f"\nProcessing split: {split}")
        
        for seq_name in sorted(os.listdir(split_dir)):
            seq_path = os.path.join(split_dir, seq_name)
            
            if not os.path.isdir(seq_path):
                continue
                
            gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
            img_dir = os.path.join(seq_path, 'img1')
            
            if not os.path.exists(gt_path):
                print(f"\nGT file missing: {gt_path}")
                continue
                
            if not os.path.exists(img_dir):
                print(f"\nImage directory missing: {img_dir}")
                continue

            print(f"\nProcessing sequence: {seq_name}")
            
            cam_id = seq_name[-3:].zfill(3)
            
            extract_mot_patches(gt_path, img_dir, patch_dir, cam_id=cam_id)

if __name__ == "__main__":
    dataset_path = "/DATA/Tawheed/MOTDatasets/DanceTrack/"
    print(f"Starting extraction from: {dataset_path}")
    
    test_file = os.path.join(dataset_path, 'patch', 'test_write.txt')
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("Write test successful")
    except Exception as e:
        print(f"\nERROR: Cannot write to output directory: {str(e)}")
        exit(1)
    
    extract_all_sequences(dataset_path)
    print("\nExtraction complete!")