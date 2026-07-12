import numpy as np

pos = np.load("_tmp_initial_pos.npy")
diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
dist = np.linalg.norm(diff, axis=-1)
# ignore self
np.fill_diagonal(dist, 999.0)

min_dist = np.min(dist)
min_idx = np.unravel_index(np.argmin(dist), dist.shape)

print(f"Global Minimum distance: {min_dist:.6f} between {min_idx[0]} and {min_idx[1]}")

# Also check distances < 0.4
close_pairs = np.argwhere(dist < 0.4)
count = 0
for i, j in close_pairs:
    if i < j:
        print(f"Close pair {i}-{j}: {dist[i, j]:.6f} nm")
        count += 1
        if count > 20:
            print("... and more.")
            break
