"""
Быков Вадим Олегович - 12.12.2025 - ReFloorTEST

Система опознавания стен, дверей и окон - v1.0

ПАЙПЛАЙН:
1. Считывание и маскировка текста при помощи OCR;
2. Считывание дверей и окон при помощи двух моделей YOLOv9;
3. Обнаружение и морфологическое выпрямление стен;
4. Экспорт в JSON и визуализация обнаруженных данных.

Последнее изменение - 12.12.25 - Initial Release
"""

import os
import cv2
import json
import numpy as np
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
from ocr import FloorPlanOCR
from wall_detector import WallIsolationConfig, load_image, create_binary_mask
from wall_detector import remove_text_by_analysis, compute_distance_transform
from wall_detector import extract_walls_by_thickness, refine_wall_mask


@dataclass
class Wall:
    id: str
    points: List[List[int]]
    thickness: float
    length: float


@dataclass
class Door:
    id: str
    bbox: List[List[int]]
    confidence: float
    center: List[int]
    source: str


@dataclass
class Window:
    id: str
    bbox: List[List[int]]
    confidence: float
    center: List[int]
    source: str


@dataclass
class FloorPlanData:
    meta: Dict[str, str]
    walls: List[Dict]
    doors: List[Dict]
    windows: List[Dict]


# ------------------------------------------------------------------ #
# YOLO
# ------------------------------------------------------------------ #

def run_yolo_detection(input_path: str, yolo_weights: str, output_dir: str,
                       imgsz: int = 1024, conf_thres: float = 0.6,
                       device: str = "0", classes: str = "3 8",
                       model_name: str = "model1"):
    yolo_output = os.path.join(output_dir, f"yolo_detections_{model_name}")
    os.makedirs(yolo_output, exist_ok=True)

    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".venv", "Scripts", "python.exe")

    if not os.path.exists(venv_python):
        venv_python = "python"

    cmd = [
        venv_python, "detect.py",
        "--source", input_path,
        "--device", device,
        "--imgsz", str(imgsz),
        "--weights", yolo_weights,
        "--project", yolo_output,
        "--save-txt",
        "--save-conf",
        "--conf-thres", str(conf_thres),
        "--nosave"
    ]
    cmd += ["--classes"] + classes.split()

    print(f"[YOLO-{model_name}] Запуск детекции (weights: {Path(yolo_weights).name})...")
    try:
        subprocess.run(cmd, check=True, cwd="./utils/yolov9")
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] YOLO-{model_name} завершился с ошибкой: {e}")
    except FileNotFoundError:
        print(f"[WARNING] Не найден путь к YOLO или python. Пропуск {model_name}.")

    return yolo_output


def parse_yolo_results(yolo_dir: str, img_name: str, img_shape: Tuple[int, int],
                       door_class: int = 3, window_class: int = 8,
                       source_label: str = "model1") -> Tuple[List[Door], List[Window]]:
    h, w = img_shape
    base_name = Path(img_name).stem

    label_paths = list(Path(yolo_dir).rglob(f"{base_name}.txt"))

    if not label_paths:
        print(f"[WARNING] YOLO результаты не найдены для {img_name} в {yolo_dir}")
        return [], []

    label_path = label_paths[0]
    doors: List[Door] = []
    windows: List[Window] = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else 1.0

            cx, cy = int(cx_norm * w), int(cy_norm * h)
            bw, bh = int(w_norm * w), int(h_norm * h)

            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2

            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            center = [cx, cy]

            if cls_id == door_class:
                doors.append(Door(
                    id=f"door_{source_label}_{len(doors)}",
                    bbox=bbox,
                    confidence=conf,
                    center=center,
                    source=source_label
                ))
            elif cls_id == window_class:
                windows.append(Window(
                    id=f"win_{source_label}_{len(windows)}",
                    bbox=bbox,
                    confidence=conf,
                    center=center,
                    source=source_label
                ))

    return doors, windows


def merge_detections(doors1: List[Door], windows1: List[Window],
                     doors2: List[Door], windows2: List[Window],
                     iou_threshold: float = 0.5) -> Tuple[List[Door], List[Window]]:
    def bbox_iou(bbox1, bbox2):
        x1_min = min(p[0] for p in bbox1)
        y1_min = min(p[1] for p in bbox1)
        x1_max = max(p[0] for p in bbox1)
        y1_max = max(p[1] for p in bbox1)

        x2_min = min(p[0] for p in bbox2)
        y2_min = min(p[1] for p in bbox2)
        x2_max = max(p[0] for p in bbox2)
        y2_max = max(p[1] for p in bbox2)

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    all_doors = doors1 + doors2
    all_windows = windows1 + windows2

    def nms(detections, iou_thresh):
        if not detections:
            return []

        detections_sorted = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        for det in detections_sorted:
            should_keep = True
            for kept in keep:
                if bbox_iou(det.bbox, kept.bbox) > iou_thresh:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)

        return keep

    merged_doors = nms(all_doors, iou_threshold)
    merged_windows = nms(all_windows, iou_threshold)

    for idx, door in enumerate(merged_doors):
        door.id = f"door_{idx}"

    for idx, window in enumerate(merged_windows):
        window.id = f"win_{idx}"

    print(f"[MERGE] Двери: {len(doors1)}+{len(doors2)} → {len(merged_doors)} (после NMS)")
    print(f"[MERGE] Окна: {len(windows1)}+{len(windows2)} → {len(merged_windows)} (после NMS)")

    return merged_doors, merged_windows


def create_element_mask(img_shape: Tuple[int, int],
                        doors: List[Door],
                        windows: List[Window],
                        ocr_detections: List,
                        expand_pixels: int = 5) -> np.ndarray:
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for door in doors:
        pts = np.array(door.bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    for window in windows:
        pts = np.array(window.bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    for detection in ocr_detections:
        pts = np.array(detection.bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    if expand_pixels > 0:
        kernel = np.ones((expand_pixels, expand_pixels), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


def process_walls_with_masking(img: np.ndarray,
                               element_mask: np.ndarray,
                               config: WallIsolationConfig) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_masked = gray.copy()
    gray_masked[element_mask > 0] = 255

    binary = create_binary_mask(gray_masked)

    binary_cleaned = remove_text_by_analysis(binary, config)

    dist_transform = compute_distance_transform(binary_cleaned)
    wall_mask = extract_walls_by_thickness(binary_cleaned, dist_transform, config)
    wall_mask_refined = refine_wall_mask(wall_mask, binary_cleaned, config)

    wall_mask_refined[element_mask > 0] = 0

    return wall_mask_refined


# ------------------------------------------------------------------ #
# WALLS
# ------------------------------------------------------------------ #

def trace_wall_polylines(wall_mask: np.ndarray, min_area: int = 50) -> List[Wall]:
    contours, hierarchy = cv2.findContours(wall_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    walls = []
    wall_id = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        points = []
        for pt in approx:
            x, y = int(pt[0][0]), int(pt[0][1])
            points.append([x, y])

        thickness = area / (perimeter + 1e-6)

        walls.append(Wall(
            id=f"w{wall_id}",
            points=points,
            thickness=float(thickness),
            length=float(perimeter)
        ))
        wall_id += 1

    return walls


# ------------------------------------------------------------------ #
# VISUALISATION
# ------------------------------------------------------------------ #

def visualize_results(img: np.ndarray,
                      wall_mask: np.ndarray,
                      walls: List[Wall],
                      doors: List[Door],
                      windows: List[Window]) -> np.ndarray:
    vis = img.copy()
    overlay = np.zeros_like(vis)

    overlay[wall_mask > 0] = (0, 255, 0)

    for door in doors:
        pts = np.array(door.bbox, dtype=np.int32)

        if door.source == "model1":
            color = (255, 0, 0)
            border_color = (255, 100, 100)
        else:
            color = (200, 0, 100)
            border_color = (220, 100, 150)

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, border_color, 2)

        center = door.center
        cv2.circle(overlay, tuple(center), 5, (255, 255, 255), -1)

        label = f"{door.id} [{door.source}]"
        cv2.putText(overlay, label, (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    for window in windows:
        pts = np.array(window.bbox, dtype=np.int32)

        if window.source == "model1":
            color = (0, 165, 255)
            border_color = (0, 200, 255)
        else:
            color = (0, 255, 165)
            border_color = (0, 255, 200)

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, border_color, 2)

        center = window.center
        cv2.circle(overlay, tuple(center), 5, (255, 255, 255), -1)

        label = f"{window.id} [{window.source}]"
        cv2.putText(overlay, label, (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    legend_y = 30
    cv2.rectangle(result, (10, legend_y - 20), (300, legend_y + 140), (255, 255, 255), -1)
    cv2.rectangle(result, (10, legend_y - 20), (300, legend_y + 140), (0, 0, 0), 2)

    cv2.rectangle(result, (20, legend_y), (40, legend_y + 15), (0, 255, 0), -1)
    cv2.putText(result, f"Walls ({len(walls)})", (50, legend_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.rectangle(result, (20, legend_y + 25), (40, legend_y + 40), (255, 0, 0), -1)
    cv2.putText(result, f"Doors Model1", (50, legend_y + 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.rectangle(result, (20, legend_y + 50), (40, legend_y + 65), (200, 0, 100), -1)
    cv2.putText(result, f"Doors Model2", (50, legend_y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.rectangle(result, (20, legend_y + 75), (40, legend_y + 90), (0, 165, 255), -1)
    cv2.putText(result, f"Windows Model1", (50, legend_y + 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.rectangle(result, (20, legend_y + 100), (40, legend_y + 115), (0, 255, 165), -1)
    cv2.putText(result, f"Windows Model2", (50, legend_y + 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    doors_m1 = len([d for d in doors if d.source == "model1"])
    doors_m2 = len([d for d in doors if d.source == "model2"])
    windows_m1 = len([w for w in windows if w.source == "model1"])
    windows_m2 = len([w for w in windows if w.source == "model2"])

    cv2.putText(result, f"Total: D={len(doors)} ({doors_m1}+{doors_m2}), W={len(windows)} ({windows_m1}+{windows_m2})",
                (15, legend_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    return result


# ------------------------------------------------------------------ #
# PROCESSING
# ------------------------------------------------------------------ #

def process_single_image(img_path: str,
                         output_dir: str,
                         doors: List[Door],
                         windows: List[Window],
                         wall_mask: np.ndarray,
                         visualize: bool = True) -> FloorPlanData:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {img_path}")

    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        img_rgb = img[:, :, :3]
        white_bg = np.ones_like(img_rgb) * 255
        alpha_factor = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
        img = (img_rgb * alpha_factor + white_bg * (1 - alpha_factor)).astype(np.uint8)

    walls = trace_wall_polylines(wall_mask)

    print(f"[RESULTS] Стен: {len(walls)} | Дверей: {len(doors)} | Окон: {len(windows)}")

    base_name = Path(img_path).stem

    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        else:
            return obj

    doors_dict = [
        {
            'id': str(d.id),
            'bbox': convert_to_native(d.bbox),
            'confidence': float(d.confidence),
            'center': convert_to_native(d.center),
            'source': str(d.source)
        }
        for d in doors
    ]

    windows_dict = [
        {
            'id': str(w.id),
            'bbox': convert_to_native(w.bbox),
            'confidence': float(w.confidence),
            'center': convert_to_native(w.center),
            'source': str(w.source)
        }
        for w in windows
    ]

    walls_dict = [
        {
            'id': str(w.id),
            'points': convert_to_native(w.points),
            'thickness': float(w.thickness),
            'length': float(w.length)
        }
        for w in walls
    ]

    floor_plan = FloorPlanData(
        meta={
            "source": str(Path(img_path).name),
            "image_size": f"{img.shape[1]}x{img.shape[0]}",
            "detection_method": "Dual_YOLO + Wall_Morphology"
        },
        walls=walls_dict,
        doors=doors_dict,
        windows=windows_dict
    )

    json_path = os.path.join(output_dir, f"{base_name}_floorplan.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(floor_plan), f, ensure_ascii=False, indent=2)
    print(f"[SAVE] JSON: {json_path}")

    mask_path = os.path.join(output_dir, f"{base_name}_walls_mask.png")
    cv2.imwrite(mask_path, wall_mask)

    if visualize:
        print("[VIS] Генерация визуализации...")
        vis_result = visualize_results(img, wall_mask, walls, doors, windows)
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        cv2.imwrite(vis_path, vis_result)
        print(f"[SAVE] Визуализация: {vis_path}")

    return floor_plan


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Система опознавания стен, дверей и окон - v1.0"
    )

    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output')

    parser.add_argument('--yolo1-weights', type=str, required=True,
                        help='Первая YOLO модель (классы 3)')
    parser.add_argument('--yolo2-weights', type=str, default='best.pt',
                        help='Вторая YOLO модель (классы 0, 1)')

    parser.add_argument('--yolo-device', type=str, default='0')
    parser.add_argument('--yolo-conf', type=float, default=0.5)
    parser.add_argument('--yolo-imgsz', type=int, default=1024)

    parser.add_argument('--min-wall-thickness', type=int, default=2)
    parser.add_argument('--max-wall-thickness', type=int, default=30)

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--skip-yolo1', action='store_true')
    parser.add_argument('--skip-yolo2', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("СИСТЕМА ДЕТЕКЦИИ С ДВУМЯ YOLO МОДЕЛЯМИ")
    print("Model 1: классы 3, 8 | Model 2: классы 0, 1 | Wall: морфология")
    print("=" * 70)

    print("\n[INIT] Инициализация OCR...")
    ocr = FloorPlanOCR()
    db_path = os.path.join(args.output, "ocr_database.json")

    wall_config = WallIsolationConfig(
        min_wall_thickness=args.min_wall_thickness,
        max_wall_thickness=args.max_wall_thickness
    )

    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [str(input_path)]
    else:
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend([str(f) for f in input_path.glob(ext)])

    print(f"\n[BATCH] Найдено изображений: {len(image_files)}\n")

    for idx, img_path in enumerate(sorted(image_files), 1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(image_files)}] Обработка: {Path(img_path).name}")
        print(f"{'=' * 70}")

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Не удалось прочитать изображение")
            img_shape = (img.shape[0], img.shape[1])

            print("\n[STEP 1/5] OCR детекция текста...")
            ocr_detections = ocr.detect(img)
            print(f"[OCR] Найдено {len(ocr_detections)} текстовых элементов")

            db_entry = {
                "image": os.path.basename(img_path),
                "detections": [asdict(d) for d in ocr_detections]
            }
            ocr.save_database(db_path, db_entry)

            doors1, windows1 = [], []
            if not args.skip_yolo1:
                print("\n[STEP 2/5] YOLO Model 1 детекция...")
                yolo1_output = run_yolo_detection(
                    img_path, args.yolo1_weights, args.output,
                    args.yolo_imgsz, args.yolo_conf, args.yolo_device,
                    "3 8", "model1"
                )
                doors1, windows1 = parse_yolo_results(
                    yolo1_output, Path(img_path).name, img_shape,
                    door_class=3, window_class=8, source_label="model1"
                )
            else:
                print("\n[STEP 2/5] YOLO Model 1 пропущена")

            doors2, windows2 = [], []
            if not args.skip_yolo2:
                print("\n[STEP 3/5] YOLO Model 2 детекция...")
                yolo2_output = run_yolo_detection(
                    img_path, args.yolo2_weights, args.output,
                    args.yolo_imgsz, args.yolo_conf, args.yolo_device,
                    "0 1", "model2"
                )
                doors2, windows2 = parse_yolo_results(
                    yolo2_output, Path(img_path).name, img_shape,
                    door_class=0, window_class=1, source_label="model2"
                )
            else:
                print("\n[STEP 3/5] YOLO Model 2 пропущена")

            print("\n[STEP 4/5] Объединение результатов...")
            doors, windows = merge_detections(doors1, windows1, doors2, windows2)

            print("\n[STEP 5/5] Wall Detection с маскированием...")
            element_mask = create_element_mask(
                img.shape, doors, windows, ocr_detections, expand_pixels=5
            )

            wall_mask = process_walls_with_masking(img, element_mask, wall_config)

            process_single_image(
                img_path, args.output, doors, windows, wall_mask, args.visualize
            )

            print(f"\n✓ Обработка завершена успешно")

        except Exception as e:
            print(f"\n✗ ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == "__main__":
    main()
