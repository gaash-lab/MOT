# import pickle

# file_path = '/DATA/Tawheed/2. det_feat/dance_val_0.80.pickle'

# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

# print("Type of data:", type(data))

# if isinstance(data, dict):
#     print("Keys:", list(data.keys()))
#     for vid in list(data.keys())[:2]:
#         for frame_id in list(data[vid].keys())[:2]:
#             print(f"Video: {vid}, Frame ID: {frame_id}, Data: {data[vid][frame_id]}")
# elif isinstance(data, list):
#     print("First 3 items:")
#     for item in data[:3]:
#         print(item)
# else:
#     print("Data content preview:", data)


import pickle
import numpy as np

# Load the file
with open('/DATA/Tawheed/2. det_feat/dance_val_0.80.pickle', 'rb') as f:
    data = pickle.load(f)

# Choose one video/frame
video = list(data.keys())[0]
frame_data = data[video]
frame_id = list(frame_data.keys())[0]
arr = frame_data[frame_id]

# Analyze a row
row = arr[0]
print("Total length:", len(row))

# Print first 10 values
print("First 10 values:", row[:10])

# Try slicing for detection vs feature assumption
for i in range(4, 10):
    print(f"First {i} values as detection:")
    print("Detections:", row[:i])
    print("Features:", row[i:i+5], "...")  # Print next 5 as sample
    print("------")
