"""
Быков Вадим Олегович - 12.12.25

Модуль OCR для детекции текста на планах, очистки изображения от текста, и ведении JSON-базы считанного текста v. 1.0

ПАЙПЛАЙН:

1. Два прохода детекции (оригинальная ориентация + поворот на 90°;
2. Маскирование текста на изображении путем окрашивания в белый;
3. Трансформация координат после поворота;
4. Экспорт результатов в JSON, в т.ч для дальнейшего считывания детектором стен wall_detector.py.

Последнее изменение - 12.12.25 - Initial Release
"""

import cv2
import json
import numpy as np
import os
import torch
from dataclasses import dataclass, asdict
from doctr.models import ocr_predictor
from PIL import Image
from typing import List, Tuple, Dict, Any


@dataclass
class DetectedItem:
    text: str
    confidence: float
    bbox: List[List[int]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    rotation_state: str


# ------------------------------------------------------------------ #
# UTILS
# ------------------------------------------------------------------ #

def mask_image(image: np.ndarray, detections: List[DetectedItem]) -> np.ndarray:
    masked = image.copy()
    for item in detections:
        pts = np.array(item.bbox, dtype=np.int32)
        cv2.fillPoly(masked, [pts], (255, 255, 255))
    return masked


def _transform_bbox_after_rotation(bbox_pts: List[List[int]], orig_h: int, orig_w: int) -> List[List[int]]:
    new_pts = []
    for (x, y) in bbox_pts:
        new_x = y
        new_y = orig_h - 1 - x
        new_pts.append([new_x, new_y])

    # Re-order to maintain standard winding if necessary, but coordinate transform is sufficient
    # Find min/max for bounding box logic if needed, but here we just return transformed points
    return new_pts

# ------------------------------------------------------------------ #
# OCR PIPELINE
# ------------------------------------------------------------------ #


class FloorPlanOCR:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
            assume_straight_pages=True,
        ).to(self.device)

    def _extract_words(self, image: np.ndarray, rotation_state: str, w_scale: int, h_scale: int) -> List[DetectedItem]:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        W, H = pil_img.size
        result = self.detector([image])
        items = []

        if result.pages:
            for block in result.pages[0].blocks:
                for line in block.lines:
                    for word in line.words:
                        if word.confidence < 0.50:
                            continue

                        geom = word.geometry
                        if len(geom) == 2:
                            (x0, y0), (x1, y1) = geom
                            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                        else:
                            pts = list(geom)

                        abs_pts = [[int(p[0] * W), int(p[1] * H)] for p in pts]
                        items.append(DetectedItem(
                            text=word.value,
                            confidence=word.confidence,
                            bbox=abs_pts,
                            rotation_state=rotation_state
                        ))
        return items

    def detect(self, image: np.ndarray) -> List[DetectedItem]:
        h, w = image.shape[:2]

        items_orig = self._extract_words(image, 'original', w, h)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for item in items_orig:
            pts = np.array(item.bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        cleaned_for_rot = image.copy()
        cleaned_for_rot[mask == 255] = (255, 255, 255)  # Assuming white background

        rotated_img = cv2.rotate(cleaned_for_rot, cv2.ROTATE_90_CLOCKWISE)
        items_rot_raw = self._extract_words(rotated_img, 'rotated', h, w)  # H and W swap

        items_rot = []
        for item in items_rot_raw:
            new_bbox = _transform_bbox_after_rotation(item.bbox, h, w)
            items_rot.append(DetectedItem(
                text=item.text,
                confidence=item.confidence,
                bbox=new_bbox,
                rotation_state='rotated'
            ))

        return items_orig + items_rot

    @staticmethod
    def save_database(db_path: str, new_entry: Dict[str, Any]):
        data = []
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                pass

        data = [d for d in data if d['image'] != new_entry['image']]
        data.append(new_entry)

        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)