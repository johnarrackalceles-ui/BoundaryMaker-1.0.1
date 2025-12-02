import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# 1) SET YOUR PATHS HERE
# =========================
img_blue_path = r"C:\Users\Celes\PycharmProjects\lab_unet_formac\data\images\images_blue\image1.tif"
img_red_path  = r"C:\Users\Celes\PycharmProjects\lab_unet_formac\data\images\images_red\image1.tif"

out_dir = Path(r"C:\Users\Celes\PycharmProjects\lab_unet_formac\results_doublepos")
out_dir.mkdir(parents=True, exist_ok=True)

print("BLUE image path:", img_blue_path)
print("RED  image path:", img_red_path)
print("Output dir:", out_dir)

# =========================
# 2) LOAD IMAGES
# =========================
img_blue = cv2.imread(img_blue_path)   # BGR
img_red  = cv2.imread(img_red_path)    # BGR

print("img_blue loaded:", img_blue is not None)
print("img_red  loaded:", img_red is not None)
assert img_blue is not None and img_red is not None, "Check image paths!"

assert img_blue.shape == img_red.shape, f"Shape mismatch: {img_blue.shape} vs {img_red.shape}"
H, W, _ = img_blue.shape
print(f"Image size: {W} x {H}")

# Save debug inputs so you can visually confirm
cv2.imwrite(str(out_dir / "debug_blue_input.png"), img_blue)
cv2.imwrite(str(out_dir / "debug_red_input.png"), img_red)
print("Saved debug_blue_input.png and debug_red_input.png")

# =========================
# 3) BLENDED IMAGE FOR DISPLAY
# =========================
blend = cv2.addWeighted(img_blue, 0.5, img_red, 0.5, 0)

# =========================
# 4) SPLIT & NORMALIZE CHANNELS
# =========================
# Blue image → use BLUE channel
b_blue, g_blue, r_blue = cv2.split(img_blue)
# Red image → use RED channel
b_red, g_red, r_red = cv2.split(img_red)

# Normalize to [0,1]
b_blue_norm = b_blue.astype(np.float32) / 255.0
r_red_norm  = r_red.astype(np.float32)  / 255.0

print("Red channel stats BEFORE boost:")
print("  r_red_norm min:", float(r_red_norm.min()))
print("  r_red_norm max:", float(r_red_norm.max()))

# =========================
# 5) BOOST RED IF EXTREMELY DIM
# =========================
if r_red_norm.max() < 0.05:
    print("Boosting red contrast (very dim signal)...")
    r_red_norm = r_red_norm / (r_red_norm.max() + 1e-8)
    r_red_norm = np.clip(r_red_norm, 0.0, 1.0)

print("Red channel stats AFTER boost (for thresholding):")
print("  r_red_norm min:", float(r_red_norm.min()))
print("  r_red_norm max:", float(r_red_norm.max()))

# =========================
# 6) CHOOSE THRESHOLDS
# =========================
print("Blue channel stats:")
print("  min:", b_blue_norm.min())
print("  max:", b_blue_norm.max())

# auto threshold if blue is dim
if b_blue_norm.max() < 0.2:
    blue_thr = b_blue_norm.max() * 0.5
    print(f"Auto-adjusting blue_thr to {blue_thr:.4f}")
else:
    blue_thr = 0.3

if r_red_norm.max() < 0.2:
    red_thr = float(r_red_norm.max() * 0.5)  # half of max if weak
    print(f"Auto-adjusting red_thr to {red_thr:.4f} (weak signal)")
else:
    red_thr = 0.3

print(f"Using thresholds → blue_thr={blue_thr:.3f}, red_thr={red_thr:.4f}")

# =========================
# 7) BUILD MASKS
# =========================
mask_blue   = (b_blue_norm > blue_thr).astype(np.uint8) * 255   # all cells
mask_red    = (r_red_norm  > red_thr ).astype(np.uint8) * 255   # T-cell channel
mask_double = cv2.bitwise_and(mask_blue, mask_red)              # intersection

# Clean double-positive with morphology
kernel = np.ones((3, 3), np.uint8)
mask_double_clean = cv2.morphologyEx(mask_double, cv2.MORPH_OPEN, kernel, iterations=1)

# =========================
# 8) COUNT DOUBLE-POSITIVE BLOBS
# =========================
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mask_double_clean, connectivity=8
)
print(f"connectedComponents found {num_labels - 1} labeled regions (excluding background)")

double_pos_count = 0
min_area = 20   # px^2, tweak for your cell size
max_area = 500
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if min_area < area < max_area:
        double_pos_count += 1

print(f"Double-positive cells detected (area-filtered): {double_pos_count}")

# =========================
# 9) DRAW BOUNDARIES ON DOUBLE-POSITIVES
# =========================
contours, _ = cv2.findContours(mask_double_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contours found in double-positive mask: {len(contours)}")

overlay = blend.copy()
cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

cv2.putText(
    overlay,
    f"T-cells (double+): {double_pos_count}",
    (15, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)

# =========================
# 10) SAVE OUTPUTS
# =========================
blue_path_out   = out_dir / "mask_blue_all_cells.png"
red_path_out    = out_dir / "mask_red_tcells.png"
double_path_out = out_dir / "mask_double_positive.png"
overlay_out     = out_dir / "double_positive_overlay.png"

cv2.imwrite(str(blue_path_out),   mask_blue)
cv2.imwrite(str(red_path_out),    mask_red)
cv2.imwrite(str(double_path_out), mask_double_clean)
cv2.imwrite(str(overlay_out),     overlay)

print("Saved:")
print("  ", blue_path_out)
print("  ", red_path_out)
print("  ", double_path_out)
print("  ", overlay_out)

# =========================
# 11) VISUAL PREVIEW
# =========================
plt.figure(figsize=(12, 8))
plt.subplot(2,2,1); plt.title("Blended RGB"); plt.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(2,2,2); plt.title("Blue mask (all cells)"); plt.imshow(mask_blue, cmap="gray"); plt.axis("off")
plt.subplot(2,2,3); plt.title("Red mask (T-cells)"); plt.imshow(mask_red, cmap="gray"); plt.axis("off")
plt.subplot(2,2,4); plt.title(f"Double-positive mask (count={double_pos_count})"); plt.imshow(mask_double_clean, cmap="gray"); plt.axis("off")
plt.tight_layout()
plt.show()