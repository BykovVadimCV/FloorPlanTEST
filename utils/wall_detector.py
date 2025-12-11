"""
Быков Вадим Олегович - 12.12.2025

Модуль морфологического распознавания стен на изображениях - v1.0

ПАЙПЛАЙН:

1. Загрузка изображения & преобразования к инверсивной бинарной маске;
2. Удаление текста при помощи OCR + фильтрация мелких и вытянутых компонентов;
3. Вычисление локальной полу-толщины для классификации по толщине и извлечене стен согласно толщине;
4. Закрытие стен горизонтальными и вертикальными структурными элементами;
5. Ректификация маски стен: скелетизация, устранение мелких компонент, закрытие разрывов, очистка шумов;
6. Генерация heatmap толщин стен (опционально);
7. Сохранение результатов;

Последнее изменение - 12.12.25 - Initial Release
"""

import os
import cv2
import numpy as np
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Optional
from ocr import FloorPlanOCR, mask_image


@dataclass
class WallIsolationConfig:
    min_wall_thickness: int = 3
    max_wall_thickness: int = 12
    text_removal_min_area: int = 50
    text_removal_max_area: int = 500
    small_component_threshold: int = 100
    noise_removal_kernel: int = 2
    gap_closing_threshold: int = 5
    straighten_walls: bool = True


# ------------------------------------------------------------------ #
# LOADING & PREPROCESSING
# ------------------------------------------------------------------ #


def load_image(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {img_path}")

    if len(img.shape) > 2 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        img_rgb = img[:, :, :3]
        white_bg = np.ones_like(img_rgb) * 255
        alpha_factor = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
        img_composite = (img_rgb * alpha_factor + white_bg * (1 - alpha_factor)).astype(np.uint8)
    else:
        img_composite = img

    gray = cv2.cvtColor(img_composite, cv2.COLOR_BGR2GRAY)

    return img_composite, gray


def create_binary_mask(gray: np.ndarray, threshold: int = 200) -> np.ndarray:
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


# ------------------------------------------------------------------ #
# OCR
# ------------------------------------------------------------------ #


def remove_text_by_analysis(binary: np.ndarray, config: WallIsolationConfig) -> np.ndarray:
    cleaned = binary.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if area < config.text_removal_min_area:
            mask = (labels == i)
            cleaned[mask] = 0
            continue

        if area < config.text_removal_max_area:
            aspect_ratio = max(width, height) / (min(width, height) + 1)
            if aspect_ratio > 1.5 and aspect_ratio < 8 and max(width, height) < 100:
                mask = (labels == i)
                cleaned[mask] = 0
                continue
            extent = area / (width * height)
            if extent < 0.3:
                mask = (labels == i)
                cleaned[mask] = 0
                continue

    return cleaned


# ------------------------------------------------------------------ #
# MORPHOLOGICAL FILTERING
# ------------------------------------------------------------------ #

def compute_distance_transform(binary: np.ndarray) -> np.ndarray:
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return dist_transform


def straighten_walls_morphological(binary: np.ndarray) -> np.ndarray:
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v, iterations=1)

    straightened = cv2.bitwise_or(horizontal, vertical)

    straightened = cv2.bitwise_and(straightened, binary)

    kernel_smooth = np.ones((3, 3), np.uint8)
    straightened = cv2.morphologyEx(straightened, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)

    return straightened


# ------------------------------------------------------------------ #
# EXTRACTION & REFINEMENT
# ------------------------------------------------------------------ #

def extract_walls_by_thickness(binary: np.ndarray,
                               dist_transform: np.ndarray,
                               config: WallIsolationConfig) -> np.ndarray:
    min_thickness = config.min_wall_thickness / 2.0
    max_thickness = config.max_wall_thickness / 2.0

    wall_mask = np.zeros_like(binary)
    wall_mask[(dist_transform >= min_thickness) & (dist_transform <= max_thickness)] = 255
    wall_mask = cv2.bitwise_and(wall_mask, binary)

    kernel = np.ones((3, 3), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if config.straighten_walls:
        wall_mask = straighten_walls_morphological(wall_mask)

    return wall_mask


def refine_wall_mask(wall_mask: np.ndarray,
                     original_binary: np.ndarray,
                     config: WallIsolationConfig) -> np.ndarray:
    refined = wall_mask.copy()

    try:
        skeleton = cv2.ximgproc.thinning(wall_mask)
    except AttributeError:
        skeleton = cv2.erode(wall_mask, np.ones((3, 3), np.uint8))

    kernel_dilate = np.ones((3, 3), np.uint8)
    expanded = cv2.dilate(skeleton, kernel_dilate, iterations=2)
    refined = cv2.bitwise_and(expanded, original_binary)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < config.small_component_threshold:
            mask = (labels == i)
            refined[mask] = 0

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,
                                             (config.gap_closing_threshold,
                                              config.gap_closing_threshold))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)

    kernel_clean = np.ones((config.noise_removal_kernel, config.noise_removal_kernel), np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_clean)

    if config.straighten_walls:
        refined = straighten_walls_morphological(refined)

    return refined

# ------------------------------------------------------------------ #
# HEATMAP
# ------------------------------------------------------------------ #

def create_thickness_heatmap(dist_transform: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    heatmap = np.zeros_like(wall_mask, dtype=np.uint8)
    masked_dist = dist_transform.copy()
    masked_dist[wall_mask == 0] = 0
    if masked_dist.max() > 0:
        normalized = (masked_dist / masked_dist.max() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        heatmap[wall_mask == 0] = 0
    return heatmap


# ------------------------------------------------------------------ #
# PROCESSING
# ------------------------------------------------------------------ #

def process_floor_plan(img_path: str,
                       output_dir: str,
                       ocr_engine: FloorPlanOCR,
                       db_path: str,
                       heatmap_dir: Optional[str] = None,
                       enable_heatmap: bool = False,
                       config: Optional[WallIsolationConfig] = None):
    if config is None:
        config = WallIsolationConfig()

    print(f"Обработка: {img_path}")

    img, _ = load_image(img_path)
    print(f"[+] Изображение загружено: {img.shape[1]}x{img.shape[0]}")

    print("[*] Запуск OCR анализа...")
    detections = ocr_engine.detect(img)
    print(f"[+] OCR: Найдено {len(detections)} текстовых элементов")

    db_entry = {
        "image": os.path.basename(img_path),
        "detections": [asdict(d) for d in detections]
    }
    ocr_engine.save_database(db_path, db_entry)

    img_cleaned = mask_image(img, detections)
    gray_cleaned = cv2.cvtColor(img_cleaned, cv2.COLOR_BGR2GRAY)

    binary = create_binary_mask(gray_cleaned)

    binary_cleaned = remove_text_by_analysis(binary, config)

    dist_transform = compute_distance_transform(binary_cleaned)
    wall_mask = extract_walls_by_thickness(binary_cleaned, dist_transform, config)
    wall_mask_refined = refine_wall_mask(wall_mask, binary_cleaned, config)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(output_dir, f"{base_name}_walls_mask.png")
    cv2.imwrite(mask_path, wall_mask_refined)
    print(f"[SAVE] Маска стен сохранена: {mask_path}")

    overlay = img.copy()
    overlay[wall_mask_refined > 0] = [0, 255, 0]
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    overlay_path = os.path.join(output_dir, f"{base_name}_walls_overlay.png")
    cv2.imwrite(overlay_path, blended)

    if enable_heatmap and heatmap_dir:
        heatmap = create_thickness_heatmap(dist_transform, wall_mask_refined)
        heatmap_path = os.path.join(heatmap_dir, f"{base_name}_thickness_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)

    return wall_mask_refined


def get_image_files(input_path: str):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        image_files = []
        for filename in sorted(os.listdir(input_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(input_path, filename))
        return image_files
    raise ValueError(f"Путь не существует: {input_path}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Модуль морфологического распознавания стен на изображениях v1.0")
    parser.add_argument('--input', type=str, required=True, help='Входной каталог или файл')
    parser.add_argument('--output', type=str, default='output', help='Каталог для результатов')
    parser.add_argument('--db-file', type=str, default='ocr_database.json', help='Файл базы данных OCR')
    parser.add_argument('--heatmap', action='store_true', help='Генерация heatmap')
    parser.add_argument('--min-thickness', type=int, default=2, help='Мин толщина стены')
    parser.add_argument('--max-thickness', type=int, default=20, help='Макс толщина стены')
    parser.add_argument('--no-straighten', action='store_true', help='Отключить выпрямление стен')

    args = parser.parse_args()

    config = WallIsolationConfig(
        min_wall_thickness=args.min_thickness,
        max_wall_thickness=args.max_thickness,
        straighten_walls=not args.no_straighten
    )

    os.makedirs(args.output, exist_ok=True)
    heatmap_dir = os.path.join(args.output, 'heatmaps') if args.heatmap else None
    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)

    print("Инициализация OCR движка (это может занять время)...")
    ocr = FloorPlanOCR()

    image_files = get_image_files(args.input)
    db_path = os.path.join(args.output, args.db_file)

    print(f"Найдено изображений: {len(image_files)}")

    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {os.path.basename(img_path)}")
        try:
            process_floor_plan(
                img_path,
                args.output,
                ocr,
                db_path,
                heatmap_dir,
                args.heatmap,
                config
            )
        except Exception as e:
            print(f"[ERROR] Ошибка обработки: {e}")


if __name__ == "__main__":
    main()