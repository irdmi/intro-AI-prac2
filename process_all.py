import os
import glob
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision
from torchvision.transforms import functional as F

# 1. Настройка путей и папок
FRAMES_DIR = "frames"
DETECTION_DIR = "annotated_detection"
SEGMENTATION_DIR = "annotated_segmentation"

os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_DIR, exist_ok=True)

# 2. Загрузка предобученной модели Mask R-CNN
print("Загрузка модели Mask R-CNN...")
weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
model.eval()

categories = weights.meta["categories"]

# Цветовая палитра для классов сегментации
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(categories), 3), dtype=np.uint8)

# 3. Получение списка кадров
frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
print(f"Найдено кадров: {len(frame_files)}")

det_images = []
seg_images = []

for frame_path in frame_files:
    img_orig = Image.open(frame_path).convert("RGB")
    img_tensor = F.to_tensor(img_orig)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # Фильтрация по уверенности (confidence > 0.5)
    scores = prediction["scores"].cpu().numpy()
    high_conf_mask = scores > 0.5

    boxes = prediction["boxes"][high_conf_mask].cpu().numpy()
    labels = prediction["labels"][high_conf_mask].cpu().numpy()
    masks = prediction["masks"][high_conf_mask].cpu().numpy()
    scores = scores[high_conf_mask]

    # --- ЗАДАНИЕ 5: Визуализация детекции (Bounding Box + Label + Confidence) ---
    img_det = img_orig.copy()
    draw = ImageDraw.Draw(img_det)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        class_name = categories[label]
        caption = f"{class_name}: {score:.2f}"
        
        # Отрисовка рамки и текста
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), caption, fill="red")

    det_save_path = os.path.join(DETECTION_DIR, os.path.basename(frame_path))
    img_det.save(det_save_path)
    det_images.append(img_det)

    # --- ЗАДАНИЕ 6: Визуализация семантической сегментации (Маски по классам) ---
    np_img = np.array(img_orig, dtype=np.uint8)
    mask_overlay = np.zeros_like(np_img, dtype=np.uint8)

    for mask, label in zip(masks, labels):
        binary_mask = mask[0] > 0.5
        color = COLORS[label]
        mask_overlay[binary_mask] = color

    # Наложение цветной маски на исходный кадр (прозрачность 50%)
    alpha = 0.5
    blended = np.where(mask_overlay > 0, (alpha * mask_overlay + (1 - alpha) * np_img).astype(np.uint8), np_img)
    img_seg = Image.fromarray(blended)

    seg_save_path = os.path.join(SEGMENTATION_DIR, os.path.basename(frame_path))
    img_seg.save(seg_save_path)
    seg_images.append(img_seg)

print("Инференс и отрисовка кадров завершены.")

# --- Сборка анимированных GIF ---
print("Сборка GIF № 1 (Детекция)...")
if det_images:
    det_images[0].save(
        "output_detection.gif",
        save_all=True,
        append_images=det_images[1:],
        duration=500,  # 2 кадра в секунду (fps=2)
        loop=0
    )

print("Сборка GIF № 2 (Сегментация)...")
if seg_images:
    seg_images[0].save(
        "output_segmentation.gif",
        save_all=True,
        append_images=seg_images[1:],
        duration=500,
        loop=0
    )

print("Готово! Результаты сохранены в файлы output_detection.gif и output_segmentation.gif")