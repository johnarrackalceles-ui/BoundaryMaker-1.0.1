import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

# ----- GUI + logging -----
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd

# Use the Windows-style embedded editor you saved earlier
from contour_editor_gui import ContourEditorGUI

# =========================
# 1) Paths & Setup
# =========================
MODEL_PATH      = r"C:\Users\Celes\PycharmProjects\unet_small_gpu\unet_finetuned.pth"
TEST_IMAGES_DIR = "data/images/test"
RESULTS_DIR     = "results_tiled_humanlog"
LOG_XLSX        = os.path.join(RESULTS_DIR, "results_log.xlsx")
LOG_CSV         = os.path.join(RESULTS_DIR, "results_log.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# =========================
# 2) Model
# =========================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# =========================
# 3) Transforms
# =========================
to_tensor = transforms.Compose([transforms.ToTensor()])

# =========================
# 4) Tiling helpers
# =========================
def tile_image(img, tile_size=512, overlap=32):
    """
    Split image into overlapping tiles.
    Returns:
      tiles: list of ((x, y), tile_np)
      H, W: original height/width
    """
    h, w = img.shape[:2]
    tiles = []
    step = tile_size - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            cut = img[y:y_end, x:x_end]
            tiles.append(((x, y), cut))
    return tiles, h, w

def stitch_tiles(preds, coords, full_shape):
    """
    Simple averaging stitch to reduce seams.
    preds: list of 2D float arrays in [0,1]
    coords: list of (x, y)
    full_shape: (H, W)
    """
    H, W = full_shape
    acc = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)
    for (x, y), p in zip(coords, preds):
        ph, pw = p.shape
        acc[y:y+ph, x:x+pw] += p
        wgt[y:y+ph, x:x+pw] += 1.0
    wgt[wgt == 0] = 1.0
    return acc / wgt

# =========================
# 5) Results log helpers
# =========================
def ensure_log():
    """Create results_log.xlsx (and csv mirror) if not present."""
    if not os.path.exists(LOG_XLSX):
        df = pd.DataFrame(columns=["Image Name", "Manual Count", "Comments"])
        df.to_excel(LOG_XLSX, index=False)
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=["Image Name", "Manual Count", "Comments"]).to_csv(LOG_CSV, index=False)

def append_log_row(image_name, count, comment):
    """Append one row to Excel + CSV (CSV as safe mirror)."""
    # Excel
    try:
        df = pd.read_excel(LOG_XLSX)
        df.loc[len(df)] = [image_name, count, comment]
        df.to_excel(LOG_XLSX, index=False)
    except Exception as e:
        print(f"⚠️ Excel write failed: {e}")

    # CSV mirror
    try:
        if os.path.exists(LOG_CSV):
            dfc = pd.read_csv(LOG_CSV)
            dfc.loc[len(dfc)] = [image_name, count, comment]
            dfc.to_csv(LOG_CSV, index=False)
        else:
            pd.DataFrame([[image_name, count, comment]],
                         columns=["Image Name", "Manual Count", "Comments"]).to_csv(LOG_CSV, index=False)
    except Exception as e:
        print(f"⚠️ CSV write failed: {e}")

def prompt_manual_entry(img_name):
    """
    Pop-up (Tk) to collect manual T-cell count + comments from the user.
    Returns (count, comment) or (None, None) if skipped/cancelled.
    """
    root = tk.Tk()
    root.withdraw()

    count = simpledialog.askinteger("Manual Count",
                                    f"Enter T-cell count for {img_name} (or Cancel to skip):",
                                    minvalue=0)
    if count is None:
        messagebox.showinfo("Skipped", f"Skipped logging for {img_name}.")
        root.destroy()
        return None, None

    comment = simpledialog.askstring("Comments", "Any notes about this image?")
    root.destroy()
    return count, (comment or "")

# =========================
# 6) Inference loop (tiled + human log)
# =========================
ensure_log()
test_images = [f for f in os.listdir(TEST_IMAGES_DIR)
               if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

for img_name in test_images:
    print(f"\n==== Processing {img_name} ====")
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    pil = Image.open(img_path).convert("RGB")
    img_rgb = np.array(pil)  # H,W,3 (RGB)

    # --- Tile
    tiles, H, W = tile_image(img_rgb, tile_size=512, overlap=32)
    preds, coords = [], []

    for (x, y), tile in tiles:
        tile_tensor = to_tensor(Image.fromarray(tile)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tile_tensor)
            pred = torch.sigmoid(out).squeeze().cpu().numpy()  # Ht, Wt
        preds.append(pred)
        coords.append((x, y))

    # --- Stitch back
    prob_full = stitch_tiles(preds, coords, (H, W))
    mask = (prob_full > 0.5).astype(np.uint8) * 255

    # --- Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Filter out small contours ---
    filtered_contours = []
    min_area = 1500  # adjust this threshold depending on your scale
    max_area = 999999  # optional, can cap to ignore huge blobs if needed
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            filtered_contours.append(cnt)

    print(f"Kept {len(filtered_contours)} out of {len(contours)} contours after filtering.")
    contours = filtered_contours

    # --- Open editor (Windows-style embedded TK GUI)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    editor = ContourEditorGUI(img_bgr.copy(), contours)
    edited_contours = editor.start()  # list of cv2-style contours or None

    if edited_contours is not None and len(edited_contours) > 0:
        contours = edited_contours
        print(f"✅ User edited {len(contours)} contours.")
    else:
        print("ℹ️ No edits applied (kept model contours).")

    # --- Draw final contours & save visuals
    overlay_bgr = img_bgr.copy()
    cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2)

    save_mask    = os.path.join(RESULTS_DIR, f"{os.path.splitext(img_name)[0]}_mask.png")
    save_overlay = os.path.join(RESULTS_DIR, f"{os.path.splitext(img_name)[0]}_overlay.png")
    cv2.imwrite(save_mask, mask)
    cv2.imwrite(save_overlay, overlay_bgr)
    print(f"💾 Saved: {save_mask}")
    print(f"💾 Saved: {save_overlay}")

    # --- Preview (optional)
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.title("Original"); plt.imshow(pil); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Predicted Mask (Tiled)"); plt.imshow(mask, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Final Overlay"); plt.imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- Ask *you* (human) for the manual count + comment
    count, comment = prompt_manual_entry(img_name)
    if count is not None:
        append_log_row(img_name, count, comment)
        print(f"🧾 Logged: {img_name} | Count={count} | Notes='{comment}'")
    else:
        print("📝 Logging skipped.")

print(f"\n📊 Results log saved to:\n  - {LOG_XLSX}\n  - {LOG_CSV} (mirror)")
print("✅ Done.")