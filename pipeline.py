import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ۱. بارگذاری تصویر
img_bgr = cv.imread('input.jpg')
if img_bgr is None:
    raise FileNotFoundError("فایل input.jpg در مسیر پروژه پیدا نشد!")

img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
print(f"Image loaded. Shape: {img_bgr.shape}")

# ۲. شبیه‌سازی انکود و دکود در پایپ‌لاین
_, encoded_bytes = cv.imencode('.jpg', img_bgr, [cv.IMWRITE_JPEG_QUALITY, 85])
decoded_img = cv.imdecode(encoded_bytes, cv.IMREAD_COLOR)

# ۳. کشش کنتراست (Contrast Stretching)
r_min, r_max = np.min(img_gray), np.max(img_gray)
stretched_img = ((img_gray - r_min) / (r_max - r_min) * 255).astype(np.uint8)

# ۴. آستانه‌گذاری (Thresholding)
_, thresh_img = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)

# ۵. تصویر نگاتیو (Negative Transformation)
negative_img = 255 - img_gray

# ۶. متعادل‌سازی هیستوگرام (Histogram Equalization)
equalized_img = cv.equalizeHist(img_gray)

# ۷. تصحیح گاما (Gamma Correction)
gamma = 0.5
table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
gamma_img = cv.LUT(img_gray, table)

# ۸. برش سطوح روشنایی (Intensity-Level Slicing)
slicing_img = img_gray.copy()
mask = (slicing_img >= 100) & (slicing_img <= 180)
slicing_img[mask] = 255

# ۹. محاسبه هیستوگرام‌ها
hist_orig, _ = np.histogram(img_gray.ravel(), bins=256, range=[0, 256])
hist_equ, _ = np.histogram(equalized_img.ravel(), bins=256, range=[0, 256])

# ۱۰. مصورسازی و ذخیره خروجی
fig, axes = plt.subplots(3, 3, figsize=(14, 14))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("1. Original RGB")
axes[0, 0].axis('off')

axes[0, 1].imshow(img_gray, cmap='gray')
axes[0, 1].set_title("2. Grayscale Input")
axes[0, 1].axis('off')

axes[0, 2].imshow(stretched_img, cmap='gray')
axes[0, 2].set_title("3. Contrast Stretched")
axes[0, 2].axis('off')

axes[1, 0].imshow(equalized_img, cmap='gray')
axes[1, 0].set_title("4. Equalized (equalizeHist)")
axes[1, 0].axis('off')

axes[1, 1].imshow(negative_img, cmap='gray')
axes[1, 1].set_title("5. Negative Transformation")
axes[1, 1].axis('off')

axes[1, 2].imshow(gamma_img, cmap='gray')
axes[1, 2].set_title(f"6. Gamma Corrected (γ={gamma})")
axes[1, 2].axis('off')

axes[2, 0].imshow(slicing_img, cmap='gray')
axes[2, 0].set_title("7. Intensity Slicing")
axes[2, 0].axis('off')

axes[2, 1].bar(range(256), hist_orig, color='gray', width=1.0)
axes[2, 1].set_title("Hist: Original Gray")
axes[2, 1].set_xlim([0, 255])

axes[2, 2].bar(range(256), hist_equ, color='navy', width=1.0)
axes[2, 2].set_title("Hist: Equalized")
axes[2, 2].set_xlim([0, 255])

plt.tight_layout()
plt.savefig("pipeline_results.png", dpi=300)
print("Pipeline executed successfully. Output saved to pipeline_results.png")
plt.show()
