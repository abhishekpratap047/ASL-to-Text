import numpy as np
import os
import random
from tqdm import tqdm
from scipy.interpolate import interp1d

# -------------------------------
# Paths
# -------------------------------
INPUT_DIR = "clips_npy"       # Original npy features
OUTPUT_DIR = "clips_npy_aug"  # Augmented features
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Augmentation Functions
# -------------------------------
def add_noise(seq, noise_level=0.01):
    """Add small Gaussian noise to keypoints"""
    return seq + np.random.normal(0, noise_level, seq.shape)

def scale_translate(seq, scale_range=(0.95, 1.05), translate_range=(-0.05, 0.05)):
    """Scale and translate entire sequence"""
    scale = random.uniform(*scale_range)
    tx = random.uniform(*translate_range)
    ty = random.uniform(*translate_range)
    seq_aug = seq.copy()
    # Pose x/y
    seq_aug[:, ::4] = seq_aug[:, ::4] * scale + tx
    seq_aug[:, 1::4] = seq_aug[:, 1::4] * scale + ty
    # Left hand x/y
    lh_start = 33*4
    seq_aug[:, lh_start::3] = seq_aug[:, lh_start::3] * scale + tx
    seq_aug[:, lh_start+1::3] = seq_aug[:, lh_start+1::3] * scale + ty
    # Right hand x/y
    rh_start = lh_start + 21*3
    seq_aug[:, rh_start::3] = seq_aug[:, rh_start::3] * scale + tx
    seq_aug[:, rh_start+1::3] = seq_aug[:, rh_start+1::3] * scale + ty
    return seq_aug

def time_shift(seq, shift_range=(-3, 3)):
    """Shift sequence forward/backward by a few frames"""
    shift = random.randint(*shift_range)
    if shift == 0:
        return seq
    elif shift > 0:
        return np.vstack([seq[shift:], np.tile(seq[-1], (shift,1))])
    else:
        return np.vstack([np.tile(seq[0], (-shift,1)), seq[:shift]])

def speed_augment(seq, speed_range=(0.9, 1.1)):
    """Change speed by resampling frames"""
    speed = random.uniform(*speed_range)
    num_frames = seq.shape[0]
    new_length = int(num_frames / speed)
    # Interpolate each feature dimension over frames
    x = np.arange(num_frames)
    x_new = np.linspace(0, num_frames-1, new_length)
    f = interp1d(x, seq, axis=0, kind='linear')
    seq_resampled = f(x_new)
    # If resampled length != original, pad or trim
    if seq_resampled.shape[0] > num_frames:
        seq_resampled = seq_resampled[:num_frames]
    elif seq_resampled.shape[0] < num_frames:
        pad_len = num_frames - seq_resampled.shape[0]
        seq_resampled = np.vstack([seq_resampled, np.tile(seq_resampled[-1], (pad_len,1))])
    return seq_resampled

def augment_sequence(seq):
    """Apply all augmentations"""
    seq_aug = seq.copy()
    seq_aug = add_noise(seq_aug, noise_level=0.01)
    seq_aug = scale_translate(seq_aug)
    seq_aug = time_shift(seq_aug)
    seq_aug = speed_augment(seq_aug, speed_range=(0.9, 1.1))
    return seq_aug

# -------------------------------
# Process all .npy files
# -------------------------------
for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    save_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    for npy_file in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        file_path = os.path.join(class_path, npy_file)
        seq = np.load(file_path)

        # Save original
        np.save(os.path.join(save_class_dir, npy_file), seq)

        # Generate augmented sequences (e.g., 2 per original)
        for i in range(2):
            seq_aug = augment_sequence(seq)
            base_name = os.path.splitext(npy_file)[0]
            save_name = f"{base_name}_aug{i+1}.npy"
            np.save(os.path.join(save_class_dir, save_name), seq_aug)

print("Augmentation with speed changes completed!")
