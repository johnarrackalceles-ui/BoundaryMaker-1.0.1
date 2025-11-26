import torch
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


# --- Import contour editor ---
from contour_editor_gui import ContourEditorGUI

# --- 0️⃣ Physical scale settings (EDIT THESE) ---
# Micrometers per pixel (example value – change to your actual calibration)
PIXEL_SIZE_UM = 0.5          # µm / pixel
# How long the bar should be in real units
SCALE_BAR_UM = 20            # µm
SCALE_BAR_UNIT = "µm"        # label text

def add_scale_bar(ax, pixel_size_um, bar_um, unit="µm", loc="lower right",
                  color="white", fontsize=8, bar_thickness=2):
    """
    Add a Google-Maps-like scale bar to an image axis.
    Assumes ax.imshow(..) has been called, so data units are pixels.
    """
    if pixel_size_um <= 0:
        return  # avoid division by zero if not configured

    # Length of the bar in pixels
    bar_length_px = bar_um / pixel_size_um

    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(
        ax.transData,
        bar_length_px,                                # length in data units (pixels)
        f"{bar_um:g} {unit}",                         # label
        loc,                                          # e.g. 'lower right'
        pad=0.3,
        color=color,
        frameon=True,
        size_vertical=bar_thickness,
        fontproperties=fontprops
    )
    ax.add_artist(scalebar)
def draw_scale_bar_opencv(
    img,
    pixel_size_um,
    bar_um,
    unit="µm",
    color=(255, 255, 255),
    thickness=3,
    margin_frac=0.05,
):
    """
    Draw a simple scale bar on the bottom-right of an image (in-place).

    img            : BGR image (OpenCV)
    pixel_size_um  : micrometers per pixel
    bar_um         : length of bar in micrometers
    unit           : text label unit (e.g., 'µm')
    """
    if pixel_size_um <= 0 or bar_um <= 0:
        return

    h, w = img.shape[:2]

    # length of bar in pixels
    bar_px = int(round(bar_um / pixel_size_um))

    # clamp to at most 1/3 of the image width
    max_bar_px = int(w * 0.33)
    if bar_px > max_bar_px:
        bar_px = max_bar_px
        bar_um = bar_px * pixel_size_um  # update label to match

    # margins from borders
    margin_x = int(round(w * margin_frac))
    margin_y = int(round(h * margin_frac))

    # bar endpoints (bottom-right)
    x1 = w - margin_x
    x0 = x1 - bar_px
    y = h - margin_y

    # draw the bar
    cv2.line(img, (x0, y), (x1, y), color, thickness)

    # label text (centered above bar)
    label = f"{bar_um:g} {unit}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_x = x0 + (bar_px - text_w) // 2
    text_y = y - 5  # a few pixels above bar

    # draw background box for readability (optional)
    cv2.rectangle(
        img,
        (text_x - 2, text_y - text_h - 2),
        (text_x + text_w + 2, text_y + 2),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(img, label, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)


# --- 1️⃣ Paths ---
MODEL_PATH = r"C:\Users\Celes\PycharmProjects\unet_small_gpu\unet_finetuned.pth"  # your trained weights
TEST_IMAGES_DIR = "data/images/test"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 2️⃣ Device Selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple Metal (MPS GPU)")
else:
    device = torch.device("cpu")
    print("🧠 Using CPU")

# --- 3️⃣ Load Model (ResNet34 UNet) ---
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

# --- 4️⃣ Load trained weights ---
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded successfully from {MODEL_PATH}")

# --- 5️⃣ Image Transform ---
transform = transforms.Compose([transforms.ToTensor()])

# --- 6️⃣ Inference Loop ---
test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for img_name in test_images:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output_sigmoid = torch.sigmoid(output).squeeze().cpu().numpy()

    print(f"{img_name} → output min: {output_sigmoid.min():.4f}, max: {output_sigmoid.max():.4f}")

    # --- Threshold + Contours ---
    mask_clean = (output_sigmoid > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    cv2.drawContours(img_cv, contours, -1, (0, 0, 255), 2)

    # --- 🪟 Open Windows-style interactive contour editor ---
    print("🖋 Launching interactive contour editor window...")
    editor = ContourEditorGUI(img_cv.copy(), contours)
    edited_contours = editor.start()  # blocks until user saves or cancels

    # Use edited contours if available
    if edited_contours and len(edited_contours) > 0:
        contours = edited_contours
        print(f"✅ User edited {len(contours)} contours successfully.")
    else:
        print("⚠️ Contours unchanged or edit canceled.")

    # --- Re-draw final contours ---
    final_img = np.array(image)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    cv2.drawContours(final_img, contours, -1, (0, 0, 255), 2)

    # --- Draw scale bar directly on the saved image ---
    draw_scale_bar_opencv(
        final_img,
        pixel_size_um=PIXEL_SIZE_UM,
        bar_um=SCALE_BAR_UM,
        unit=SCALE_BAR_UNIT,
    )

    # --- Save Results ---
    result_path = os.path.join(RESULTS_DIR, img_name)
    cv2.imwrite(result_path, final_img)
    print(f"💾 Saved final result to {result_path}")

    # --- Preview ---
    plt.figure(figsize=(15, 5))

    # Original
    ax1 = plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.axis("off")

    # Predicted mask
    ax2 = plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask_clean, cmap="gray")
    plt.axis("off")

    # Final overlay + SCALE BAR
    ax3 = plt.subplot(1, 3, 3)
    plt.title("Final Overlay (After Edit)")
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # NEW: add scale bar to the final overlay
    def add_scale_bar(ax, pixel_size_um, bar_um, unit="µm",
                      color="white", linewidth=4, fontsize=10,
                      frac_of_width=0.2, margin_frac=0.05):
        """
        Draw a simple Google-Maps style scale bar on an image axis.

        - pixel_size_um: micrometers per pixel
        - bar_um: length of the bar in micrometers
        - frac_of_width: optional, if bar_um is None we can use fraction of width
        """
        if pixel_size_um <= 0:
            return

        # Get image size from the first image in this axis
        if not ax.images:
            return
        img = ax.images[0].get_array()
        h, w = img.shape[:2]

        # Length of bar in pixels
        bar_px = bar_um / pixel_size_um

        # If requested bar is too long, clamp it to some fraction of width
        max_bar_px = w * frac_of_width
        if bar_px > max_bar_px:
            bar_px = max_bar_px
            bar_um = bar_px * pixel_size_um  # update label value

        # Position near bottom-right with a small margin
        margin_x = w * margin_frac
        margin_y = h * margin_frac

        x0 = w - margin_x - bar_px
        x1 = w - margin_x
        y0 = h - margin_y

        # Draw the bar (line)
        ax.plot([x0, x1], [y0, y0],
                color=color, linewidth=linewidth, solid_capstyle="butt")

        # Draw the label centered above the bar
        ax.text((x0 + x1) / 2, y0 - h * 0.02,
                f"{bar_um:g} {unit}",
                color=color, fontsize=fontsize,
                ha="center", va="bottom")
    add_scale_bar(
        ax3,
        pixel_size_um=PIXEL_SIZE_UM,
        bar_um=SCALE_BAR_UM,
        unit=SCALE_BAR_UNIT,
        color="white",
        fontsize=8,
    )

    plt.tight_layout()
    plt.show()
