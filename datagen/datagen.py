"""
Быков Вадим Олегович - 12.12.25

Модуль генерации синтетических датасетов для расопзнавания дверей и окон на чертежах БТИ - v1.0

ПАЙПЛАЙН:

1. Сэмплирование стиля сцен;
2. Построение плана;
3. Рисование стен;
4. Построение элементов внутри комнат;
5. Добавление окон;
6. Добавление дверей;
7. Добавление текста;
8. Постобработка и аугментации;
9. Экспорт датасета;

Выходной формат:
  out_dir/
    images/
      00000.png
      00001.png
    labels/
      00000.txt  (YOLO format: <class_id> <cx> <cy> <w> <h>)

Классы:
  0 - door
  1 - window

Последнее изменение - 12.12.25 - Initial Release
"""

import os
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class Wall:
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: int
    border_thick: int
    is_external: bool


@dataclass
class Room:
    x1: int
    y1: int
    x2: int
    y2: int
    room_type: str


class BTIFloorPlanGeneratorV3:
    def __init__(self, img_size: int = 1024, seed: int = 0):
        self.size = img_size
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # DATASET
    # ------------------------------------------------------------------ #
    def generate_dataset(self, n_images: int, out_dir: str):
        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(n_images):
            img, labels = self.generate_single_plan()

            if self.rng.random() < 0.15:
                img, labels = self._apply_rotation(img, labels)

            dirty_threshold = int(n_images * 0.7)
            if i >= dirty_threshold:
                img = self._apply_print_defects(img)

            fname = f"{i:05d}"
            cv2.imwrite(os.path.join(img_dir, fname + ".png"), img)

            with open(os.path.join(lbl_dir, fname + ".txt"), "w", encoding="utf-8") as f:
                for cls_id, cx, cy, w, h in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # ------------------------------------------------------------------ #
    # SINGLE PLAN
    # ------------------------------------------------------------------ #
    def generate_single_plan(self) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        img = np.full((self.size, self.size), 255, dtype=np.uint8)

        ext_thick, int_thick, border_thick = self._sample_wall_style()

        margin_ratio_min, margin_ratio_max = 0.20, 0.27
        margin = int(self.size * self.rng.uniform(margin_ratio_min, margin_ratio_max))
        outer_left = margin
        outer_top = margin
        outer_right = self.size - margin
        outer_bottom = self.size - margin

        walls, rooms = self._create_1room_layout(
            outer_left, outer_top, outer_right, outer_bottom,
            ext_thick, int_thick, border_thick
        )

        self._draw_walls_with_corners(img, walls)

        self._add_double_rectangles(img, rooms, border_thick)

        labels: List[Tuple[int, float, float, float, float]] = []

        for w in walls:
            if not w.is_external:
                continue
            length = max(abs(w.x2 - w.x1), abs(w.y2 - w.y1))
            n_slots = 1
            if length > (outer_right - outer_left) * 0.55:
                n_slots = 2
            for _ in range(n_slots):
                if self.rng.random() < 0.75:
                    box = self._place_window(img, w)
                    if box:
                        labels.append((1, *box))

        internal_walls = [w for w in walls if not w.is_external]
        for w in internal_walls:
            length = max(abs(w.x2 - w.x1), abs(w.y2 - w.y1))
            n_slots = 1
            if length > (outer_bottom - outer_top) * 0.45:
                n_slots = 2
            for _ in range(n_slots):
                if self.rng.random() < 0.7:
                    box = self._place_door(img, w)
                    if box:
                        labels.append((0, *box))

        if not any(lbl[0] == 1 for lbl in labels):
            ext_walls = [w for w in walls if w.is_external]
            if ext_walls:
                w = self.rng.choice(ext_walls)
                box = self._place_window(img, w)
                if box:
                    labels.append((1, *box))

        if not any(lbl[0] == 0 for lbl in labels):
            if internal_walls:
                w = self.rng.choice(internal_walls)
                box = self._place_door(img, w)
                if box:
                    labels.append((0, *box))

        self._add_random_text(img, outer_left, outer_top, outer_right, outer_bottom)

        img = self._postprocess_base_plan(img)

        return img, labels

    # ------------------------------------------------------------------ #
    # LAYOUT / WALL STYLE
    # ------------------------------------------------------------------ #
    def _sample_wall_style(self) -> Tuple[int, int, int]:
        """
        Подбирает согласованный набор толщин для данного изображения.
        """
        scale = self.size / 1000.0

        border_thick = max(2, int(round(self.rng.uniform(2.2, 3.1) * scale)))
        ext_thick = max(
            border_thick * 3 + 2,
            int(round(self.rng.uniform(5, 30) * scale))
        )
        int_thick = max(
            border_thick * 2 + 1,
            int(round(ext_thick * self.rng.uniform(0.45, 0.65)))
        )
        int_thick = min(int_thick, ext_thick - 2)

        return ext_thick, int_thick, border_thick

    def _create_1room_layout(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        ext_thick: int,
        int_thick: int,
        border_thick: int,
    ) -> Tuple[List[Wall], List[Room]]:
        walls: List[Wall] = []
        rooms: List[Room] = []

        walls.append(Wall(left, top, right, top, ext_thick, border_thick, True))
        walls.append(Wall(left, bottom, right, bottom, ext_thick, border_thick, True))
        walls.append(Wall(left, top, left, bottom, ext_thick, border_thick, True))
        walls.append(Wall(right, top, right, bottom, ext_thick, border_thick, True))

        width = right - left
        height = bottom - top

        corridor_width = self.rng.randint(int(width * 0.14), int(width * 0.22))
        bathroom_size = self.rng.randint(int(min(width, height) * 0.14), int(min(width, height) * 0.20))
        kitchen_width = self.rng.randint(int(width * 0.22), int(width * 0.32))

        density_prob = 0.9
        extra_free_prob = 0.6

        if self.rng.random() < 0.5:
            corridor_x = left + corridor_width
            walls.append(Wall(corridor_x, top, corridor_x, bottom, int_thick, border_thick, False))

            bathroom_y = top + bathroom_size
            walls.append(Wall(left, bathroom_y, corridor_x, bathroom_y, int_thick, border_thick, False))

            kitchen_y = bottom - self.rng.randint(int(height * 0.25), int(height * 0.35))
            walls.append(Wall(corridor_x, kitchen_y, right, kitchen_y, int_thick, border_thick, False))

            kitchen_x = corridor_x + kitchen_width
            walls.append(Wall(kitchen_x, kitchen_y, kitchen_x, bottom, int_thick, border_thick, False))

            if self.rng.random() < density_prob:
                extra_y = top + self.rng.randint(int(height * 0.30), int(height * 0.55))
                walls.append(Wall(corridor_x, extra_y, kitchen_x, extra_y, int_thick, border_thick, False))

            if self.rng.random() < density_prob * 0.7:
                stub_y = top + self.rng.randint(int(height * 0.25), int(height * 0.70))
                stub_len = self.rng.randint(int(width * 0.10), int(width * 0.18))
                direction = self.rng.choice([-1, 1])
                if direction < 0:
                    x2 = max(left, corridor_x - stub_len)
                else:
                    x2 = min(right, corridor_x + stub_len)
                walls.append(Wall(corridor_x, stub_y, x2, stub_y, int_thick, border_thick, False))

            if self.rng.random() < density_prob * extra_free_prob:
                lv_y = top + self.rng.randint(int(height * 0.35), int(height * 0.80))
                lv_len = self.rng.randint(int(width * 0.10), int(width * 0.25))
                x1 = corridor_x + self.rng.randint(int(width * 0.05), int(width * 0.20))
                x2 = min(right - int(width * 0.05), x1 + lv_len)
                walls.append(Wall(x1, lv_y, x2, lv_y, int_thick, border_thick, False))

            rooms.append(Room(left, top, corridor_x, bathroom_y, "bathroom"))
            rooms.append(Room(left, bathroom_y, corridor_x, bottom, "corridor"))
            rooms.append(Room(corridor_x, kitchen_y, kitchen_x, bottom, "kitchen"))
            rooms.append(Room(corridor_x, top, right, kitchen_y, "living"))
        else:
            corridor_y = top + self.rng.randint(int(height * 0.14), int(height * 0.22))
            walls.append(Wall(left, corridor_y, right, corridor_y, int_thick, border_thick, False))

            bathroom_x = left + bathroom_size
            walls.append(Wall(bathroom_x, top, bathroom_x, corridor_y, int_thick, border_thick, False))

            kitchen_x = left + self.rng.randint(int(width * 0.25), int(width * 0.35))
            walls.append(Wall(kitchen_x, corridor_y, kitchen_x, bottom, int_thick, border_thick, False))

            if self.rng.random() < density_prob:
                extra_x = left + self.rng.randint(int(width * 0.40), int(width * 0.60))
                walls.append(Wall(extra_x, corridor_y, extra_x, bottom, int_thick, border_thick, False))

            if self.rng.random() < density_prob * 0.7:
                stub_x = left + self.rng.randint(int(width * 0.25), int(width * 0.75))
                stub_len = self.rng.randint(int(height * 0.08), int(height * 0.18))
                direction = self.rng.choice([-1, 1])
                if direction < 0:
                    y2 = max(top, corridor_y - stub_len)
                else:
                    y2 = min(bottom, corridor_y + stub_len)
                walls.append(Wall(stub_x, corridor_y, stub_x, y2, int_thick, border_thick, False))

            if self.rng.random() < density_prob * extra_free_prob:
                lv_x = left + self.rng.randint(int(width * 0.35), int(width * 0.80))
                lv_len = self.rng.randint(int(height * 0.10), int(height * 0.25))
                y1 = corridor_y + self.rng.randint(int(height * 0.05), int(height * 0.20))
                y2 = min(bottom - int(height * 0.05), y1 + lv_len)
                walls.append(Wall(lv_x, y1, lv_x, y2, int_thick, border_thick, False))

            rooms.append(Room(left, top, bathroom_x, corridor_y, "bathroom"))
            rooms.append(Room(bathroom_x, top, right, corridor_y, "corridor"))
            rooms.append(Room(left, corridor_y, kitchen_x, bottom, "kitchen"))
            rooms.append(Room(kitchen_x, corridor_y, right, bottom, "living"))

        if self.rng.random() < 0.45:
            balcony_wall_candidates = [w for w in walls if w.is_external and (w.x1 == w.x2 or w.y1 == w.y2)]
            if balcony_wall_candidates:
                balcony_wall = self.rng.choice(balcony_wall_candidates)
                if balcony_wall.x1 == balcony_wall.x2:
                    # вертикальная стена
                    mid_y = (balcony_wall.y1 + balcony_wall.y2) // 2
                    balcony_len = self.rng.randint(int(height * 0.12), int(height * 0.22))
                    balcony_depth = self.rng.randint(self.size // 40, self.size // 22)
                    by1 = mid_y - balcony_len // 2
                    by2 = mid_y + balcony_len // 2
                    if balcony_wall.x1 == right:
                        bx1 = right
                        bx2 = right + balcony_depth
                    else:
                        bx1 = left - balcony_depth
                        bx2 = left
                    walls.append(Wall(bx1, by1, bx2, by1, int_thick, border_thick, True))
                    walls.append(Wall(bx1, by2, bx2, by2, int_thick, border_thick, True))
                    walls.append(Wall(bx2, by1, bx2, by2, int_thick, border_thick, True))
                else:
                    mid_x = (balcony_wall.x1 + balcony_wall.x2) // 2
                    balcony_len = self.rng.randint(int(width * 0.14), int(width * 0.24))
                    balcony_depth = self.rng.randint(self.size // 40, self.size // 22)
                    bx1 = mid_x - balcony_len // 2
                    bx2 = mid_x + balcony_len // 2
                    if balcony_wall.y1 == bottom:
                        by1 = bottom
                        by2 = bottom + balcony_depth
                    else:
                        by1 = top - balcony_depth
                        by2 = top
                    walls.append(Wall(bx1, by1, bx2, by1, int_thick, border_thick, True))
                    walls.append(Wall(bx1, by2, bx2, by2, int_thick, border_thick, True))
                    walls.append(Wall(bx1, by1, bx1, by2, int_thick, border_thick, True))

        return walls, rooms

    # ------------------------------------------------------------------ #
    # DRAW WALLS
    # ------------------------------------------------------------------ #
    def _draw_walls_with_corners(self, img: np.ndarray, walls: List[Wall]):
        for w in walls:
            half = w.thickness // 2
            x1, x2 = min(w.x1, w.x2), max(w.x1, w.x2)
            y1, y2 = min(w.y1, w.y2), max(w.y1, w.y2)

            if w.x1 == w.x2:  # вертикальная
                pt1 = (x1 - half, y1 - half)
                pt2 = (x1 + half, y2 + half)
            else:             # горизонтальная
                pt1 = (x1 - half, y1 - half)
                pt2 = (x2 + half, y1 + half)

            cv2.rectangle(img, pt1, pt2, 0, -1)

        for w in walls:
            inner_thick = w.thickness - (2 * w.border_thick)
            if inner_thick <= 0:
                continue

            half_inner = inner_thick // 2
            x1, x2 = min(w.x1, w.x2), max(w.x1, w.x2)
            y1, y2 = min(w.y1, w.y2), max(w.y1, w.y2)

            if w.x1 == w.x2:
                pt1 = (x1 - half_inner, y1 - half_inner)
                pt2 = (x1 + half_inner, y2 + half_inner)
            else:
                pt1 = (x1 - half_inner, y1 - half_inner)
                pt2 = (x2 + half_inner, y1 + half_inner)

            cv2.rectangle(img, pt1, pt2, 255, -1)

    # ------------------------------------------------------------------ #
    # DOUBLE RECTANGLES IN ROOMS
    # ------------------------------------------------------------------ #
    def _add_double_rectangles(self, img: np.ndarray, rooms: List[Room], border_thick: int):
        for room in rooms:
            rw = room.x2 - room.x1
            rh = room.y2 - room.y1
            if rw < 50 or rh < 50:
                continue

            if room.room_type == "bathroom":
                n_rects = self.rng.randint(1, 3)
            elif room.room_type == "kitchen":
                n_rects = self.rng.randint(0, 2)
            else:
                n_rects = self.rng.randint(0, 2)

            margin = border_thick * 3
            if rw <= 2 * margin + 30 or rh <= 2 * margin + 30:
                continue

            inner_w_avail = rw - 2 * margin
            inner_h_avail = rh - 2 * margin

            for _ in range(n_rects):
                ow_min = int(inner_w_avail * 0.20)
                ow_max = int(inner_w_avail * 0.70)
                oh_min = int(inner_h_avail * 0.20)
                oh_max = int(inner_h_avail * 0.60)
                if ow_max <= ow_min or oh_max <= oh_min:
                    continue

                outer_w = self.rng.randint(ow_min, ow_max)
                outer_h = self.rng.randint(oh_min, oh_max)

                x0_min = room.x1 + margin
                x0_max = room.x1 + margin + inner_w_avail - outer_w
                y0_min = room.y1 + margin
                y0_max = room.y1 + margin + inner_h_avail - outer_h
                if x0_max <= x0_min or y0_max <= y0_min:
                    continue

                x0 = self.rng.randint(x0_min, x0_max)
                y0 = self.rng.randint(y0_min, y0_max)

                max_gap_by_size = int(min(outer_w, outer_h) * 0.25)
                if max_gap_by_size <= border_thick * 1.2:
                    continue

                gap_min = max(border_thick, int(border_thick * 1.0))
                gap_max = min(max_gap_by_size, border_thick * 5)
                if gap_max <= gap_min:
                    gap = gap_min
                else:
                    gap = self.rng.randint(gap_min, gap_max)

                ix1 = x0 + gap
                iy1 = y0 + gap
                ix2 = x0 + outer_w - gap
                iy2 = y0 + outer_h - gap

                if ix2 - ix1 < 3 * border_thick or iy2 - iy1 < 3 * border_thick:
                    continue

                cv2.rectangle(img, (x0, y0), (x0 + outer_w, y0 + outer_h), 0, border_thick)
                cv2.rectangle(img, (ix1, iy1), (ix2, iy2), 0, border_thick)

    # ------------------------------------------------------------------ #
    # WINDOWS / DOORS
    # ------------------------------------------------------------------ #
    def _place_window(self, img: np.ndarray, w: Wall) -> Optional[Tuple[float, float, float, float]]:
        half = w.thickness // 2

        t_outer = max(1, w.border_thick // 2)
        t_inner = max(1, w.border_thick // 2)

        min_gap_ratio = 0.25
        max_gap_ratio = 0.65
        gap_ratio = self.rng.uniform(min_gap_ratio, max_gap_ratio)
        inner_gap = int(w.thickness * gap_ratio)

        pad = max(30, w.thickness * 2)

        if abs(w.x2 - w.x1) > abs(w.y2 - w.y1):
            axis_start, axis_end = min(w.x1, w.x2), max(w.x1, w.x2)
            y_center = (w.y1 + w.y2) // 2
            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None

            min_len = int(w.thickness * 2.5)
            max_len = int(min(avail, w.thickness * 5.0))
            if max_len < min_len:
                return None
            win_len = self.rng.randint(min_len, max_len)
            start = axis_start + pad + self.rng.randint(0, max(1, avail - win_len))
            end = start + win_len

            x0, x1 = int(start), int(end)
            y_top = y_center - half
            y_bottom = y_center + half

            p1 = (x0, y_top)
            p2 = (x0, y_bottom)
            p3 = (x1, y_top)
            p4 = (x1, y_bottom)
            cv2.line(img, p1, p2, 0, t_outer)
            cv2.line(img, p3, p4, 0, t_outer)

            offset = inner_gap // 2
            ya = y_center - offset
            yb = y_center + offset
            q1 = (x0, ya)
            q2 = (x1, ya)
            q3 = (x0, yb)
            q4 = (x1, yb)
            cv2.line(img, q1, q2, 0, t_inner)
            cv2.line(img, q3, q4, 0, t_inner)

            return self._points_to_yolo([p1, p2, p3, p4, q1, q2, q3, q4])
        else:
            axis_start, axis_end = min(w.y1, w.y2), max(w.y1, w.y2)
            x_center = (w.x1 + w.x2) // 2
            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None

            min_len = int(w.thickness * 2.5)
            max_len = int(min(avail, w.thickness * 5.0))
            if max_len < min_len:
                return None
            win_len = self.rng.randint(min_len, max_len)
            start = axis_start + pad + self.rng.randint(0, max(1, avail - win_len))
            end = start + win_len

            y0, y1 = int(start), int(end)
            x_left = x_center - half
            x_right = x_center + half

            p1 = (x_left, y0)
            p2 = (x_right, y0)
            p3 = (x_left, y1)
            p4 = (x_right, y1)
            cv2.line(img, p1, p2, 0, t_outer)
            cv2.line(img, p3, p4, 0, t_outer)

            offset = inner_gap // 2
            xa = x_center - offset
            xb = x_center + offset
            q1 = (xa, y0)
            q2 = (xa, y1)
            q3 = (xb, y0)
            q4 = (xb, y1)
            cv2.line(img, q1, q2, 0, t_inner)
            cv2.line(img, q3, q4, 0, t_inner)

            return self._points_to_yolo([p1, p2, p3, p4, q1, q2, q3, q4])

    def _place_door(self, img: np.ndarray, w: Wall) -> Optional[Tuple[float, float, float, float]]:
        half = w.thickness // 2

        t_notch = max(1, w.border_thick)
        t_edge = max(1, w.border_thick)

        pad = max(35, int(w.thickness * 2.5))  # отступ от углов/стыков

        horizontal = abs(w.x2 - w.x1) > abs(w.y2 - w.y1)

        if horizontal:
            axis_start, axis_end = min(w.x1, w.x2), max(w.x1, w.x2)
            center_line = (w.y1 + w.y2) // 2
        else:
            axis_start, axis_end = min(w.y1, w.y2), max(w.y1, w.y2)
            center_line = (w.x1 + w.x2) // 2

        avail = axis_end - axis_start - 2 * pad
        if avail <= w.thickness * 1.5:
            return None

        if self.rng.random() < 0.35:
            min_ratio, max_ratio = 3.0, 6.0
        else:
            min_ratio, max_ratio = 1.6, 2.8

        min_len = int(w.thickness * min_ratio)
        max_len = int(w.thickness * max_ratio)
        max_len = min(max_len, int(avail * 0.95))
        if max_len <= 0:
            return None
        min_len = min(min_len, max_len)

        door_len = self.rng.randint(max(1, min_len), max_len)

        start = axis_start + pad + self.rng.randint(0, max(1, int(avail - door_len)))
        end = start + door_len

        if horizontal:
            y_center = center_line
            y_top = y_center - half
            y_bottom = y_center + half

            x_start = int(start)
            x_end = int(end)
            x_mid = (x_start + x_end) // 2

            edge1_top = (x_start, y_top)
            edge1_bot = (x_start, y_bottom)
            edge2_top = (x_end, y_top)
            edge2_bot = (x_end, y_bottom)
            cv2.line(img, edge1_top, edge1_bot, 0, t_edge)
            cv2.line(img, edge2_top, edge2_bot, 0, t_edge)

            notch_len = int(door_len * self.rng.uniform(0.8, 1.1))
            notch_half = notch_len // 2
            p1 = (x_mid, y_center - notch_half)
            p2 = (x_mid, y_center + notch_half)
            cv2.line(img, p1, p2, 0, t_notch)

            all_pts = [p1, p2, edge1_top, edge1_bot, edge2_top, edge2_bot]
            return self._points_to_yolo(all_pts)

        else:
            x_center = center_line
            x_left = x_center - half
            x_right = x_center + half

            y_start = int(start)
            y_end = int(end)
            y_mid = (y_start + y_end) // 2

            edge1_left = (x_left, y_start)
            edge1_right = (x_right, y_start)
            edge2_left = (x_left, y_end)
            edge2_right = (x_right, y_end)
            cv2.line(img, edge1_left, edge1_right, 0, t_edge)
            cv2.line(img, edge2_left, edge2_right, 0, t_edge)

            notch_len = int(door_len * self.rng.uniform(0.8, 1.1))
            notch_half = notch_len // 2
            p1 = (x_center - notch_half, y_mid)
            p2 = (x_center + notch_half, y_mid)
            cv2.line(img, p1, p2, 0, t_notch)

            all_pts = [p1, p2, edge1_left, edge1_right, edge2_left, edge2_right]
            return self._points_to_yolo(all_pts)

    # ------------------------------------------------------------------ #
    # LABEL UTILITIES
    # ------------------------------------------------------------------ #
    def _points_to_yolo(self, pts: List[Tuple[int, int]]) -> Optional[Tuple[float, float, float, float]]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 3
        x_min = max(0, min(xs) - pad)
        x_max = min(self.size - 1, max(xs) + pad)
        y_min = max(0, min(ys) - pad)
        y_max = min(self.size - 1, max(ys) + pad)

        if x_max <= x_min or y_max <= y_min:
            return None

        cx = (x_min + x_max) / 2.0 / self.size
        cy = (y_min + y_max) / 2.0 / self.size
        w = (x_max - x_min) / self.size
        h = (y_max - y_min) / self.size

        return (
            min(max(cx, 0.0), 1.0),
            min(max(cy, 0.0), 1.0),
            min(max(w, 0.0), 1.0),
            min(max(h, 0.0), 1.0),
        )

    # ------------------------------------------------------------------ #
    # TEXT AND BASE POSTPROCESSING
    # ------------------------------------------------------------------ #
    def _add_random_text(self, img: np.ndarray, left: int, top: int, right: int, bottom: int):
        n_txt = self.rng.randint(16, 30)
        scale_factor = self.size / 1024.0
        for _ in range(n_txt):
            x = self.rng.randint(left + 20, right - 20)
            y = self.rng.randint(top + 20, bottom - 20)
            val = f"{self.rng.randint(1, 30)}.{self.rng.randint(0, 9)}"
            font_scale = self.rng.uniform(0.40, 0.80) * scale_factor
            thickness = self.rng.choice([1, 1, 2])
            cv2.putText(
                img,
                val,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                0,
                thickness,
                cv2.LINE_AA,
            )

    def _postprocess_base_plan(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.8:
            ksize = self.rng.choice([3, 3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), self.rng.uniform(0.4, 0.9))

        thr = self.rng.randint(210, 235)
        _, img_bin = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

        return img_bin

    # ------------------------------------------------------------------ #
    # ROTATION
    # ------------------------------------------------------------------ #
    def _apply_rotation(
        self, img: np.ndarray, labels: List[Tuple[int, float, float, float, float]]
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        angle = self.rng.uniform(-45, 45)
        center = (self.size // 2, self.size // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (self.size, self.size), borderValue=255)

        rotated_labels: List[Tuple[int, float, float, float, float]] = []
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for cls_id, cx, cy, w, h in labels:
            cx_px = cx * self.size
            cy_px = cy * self.size
            w_px = w * self.size
            h_px = h * self.size

            corners = [
                (cx_px - w_px / 2, cy_px - h_px / 2),
                (cx_px + w_px / 2, cy_px - h_px / 2),
                (cx_px + w_px / 2, cy_px + h_px / 2),
                (cx_px - w_px / 2, cy_px + h_px / 2),
            ]

            rotated_corners = []
            for x, y in corners:
                x_rel = x - center[0]
                y_rel = y - center[1]
                x_rot = x_rel * cos_a - y_rel * sin_a
                y_rot = x_rel * sin_a + y_rel * cos_a
                rotated_corners.append((x_rot + center[0], y_rot + center[1]))

            xs = [c[0] for c in rotated_corners]
            ys = [c[1] for c in rotated_corners]
            x_min = max(0, min(xs))
            x_max = min(self.size, max(xs))
            y_min = max(0, min(ys))
            y_max = min(self.size, max(ys))

            new_cx = (x_min + x_max) / 2 / self.size
            new_cy = (y_min + y_max) / 2 / self.size
            new_w = (x_max - x_min) / self.size
            new_h = (y_max - y_min) / self.size

            if new_w > 0.01 and new_h > 0.01:
                rotated_labels.append(
                    (
                        cls_id,
                        min(max(new_cx, 0.0), 1.0),
                        min(max(new_cy, 0.0), 1.0),
                        min(max(new_w, 0.0), 1.0),
                        min(max(new_h, 0.0), 1.0),
                    )
                )

        return rotated_img, rotated_labels

    # ------------------------------------------------------------------ #
    # LINE-LEVEL DEFECTS
    # ------------------------------------------------------------------ #
    def _apply_line_defects(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        work = img.copy()

        thr = self.rng.randint(180, 220)
        _, bin_inv = cv2.threshold(work, thr, 255, cv2.THRESH_BINARY_INV)

        n_ops = self.rng.randint(1, 3)
        for _ in range(n_ops):
            op = self.rng.choice(["erode", "dilate", "open", "close"])
            ksize = self.rng.choice([(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
            if op == "erode":
                bin_inv = cv2.erode(bin_inv, kernel, iterations=1)
            elif op == "dilate":
                bin_inv = cv2.dilate(bin_inv, kernel, iterations=1)
            elif op == "open":
                bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel, iterations=1)
            else:
                bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel, iterations=1)

        work = cv2.bitwise_not(bin_inv)

        n_cuts = self.rng.randint(10, 25)
        for _ in range(n_cuts):
            orientation = self.rng.choice(["h", "v"])
            if orientation == "h":
                y = self.rng.randint(int(h * 0.1), int(h * 0.9))
                x1 = self.rng.randint(int(w * 0.1), int(w * 0.8))
                length = self.rng.randint(int(w * 0.02), int(w * 0.08))
                x2 = min(w - 1, x1 + length)
                thickness = self.rng.randint(1, 2)
                cv2.line(work, (x1, y), (x2, y), 255, thickness)
            else:
                x = self.rng.randint(int(w * 0.1), int(w * 0.9))
                y1 = self.rng.randint(int(h * 0.1), int(h * 0.8))
                length = self.rng.randint(int(h * 0.02), int(h * 0.08))
                y2 = min(h - 1, y1 + length)
                thickness = self.rng.randint(1, 2)
                cv2.line(work, (x, y1), (x, y2), 255, thickness)

        edges = cv2.Canny(work, 50, 150)
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, dil_kernel, iterations=1)

        noise = (np.random.randn(h, w) * self.rng.uniform(5, 15)).astype(np.int16)
        noise = np.clip(noise, -40, 40)

        work16 = work.astype(np.int16)
        work16[edges > 0] += noise[edges > 0]
        work = np.clip(work16, 0, 255).astype(np.uint8)

        return work

    # ------------------------------------------------------------------ #
    # GLOBAL PRINT / SCAN DEFECTS
    # ------------------------------------------------------------------ #
    def _apply_print_defects(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)

        defect_type = self.rng.choice(
            [
                "gaussian_noise",
                "salt_pepper",
                "blur",
                "scan_lines",
                "jpeg",
                "line_defects",
                "combined",
            ]
        )

        if defect_type in ["line_defects", "combined"]:
            img_ld = self._apply_line_defects(img_float.astype(np.uint8))
            img_float = img_ld.astype(np.float32)

        if defect_type in ["gaussian_noise", "combined"]:
            noise_std = self.rng.uniform(3, 12)
            noise = np.random.normal(0, noise_std, img_float.shape)
            img_float = img_float + noise

        if defect_type in ["salt_pepper", "combined"]:
            prob = self.rng.uniform(0.002, 0.01)
            mask = np.random.rand(*img_float.shape)
            img_float[mask < prob / 2] = 0
            img_float[mask > 1 - prob / 2] = 255

        if defect_type in ["blur", "combined"]:
            blur_type = self.rng.choice(["gaussian", "motion"])
            if blur_type == "gaussian":
                ksize = self.rng.choice([3, 5])
                img_float = cv2.GaussianBlur(img_float, (ksize, ksize), 0)
            else:
                kernel_size = self.rng.randint(3, 7)
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size // 2, :] = 1.0 / kernel_size
                angle = self.rng.uniform(-15, 15)
                M = cv2.getRotationMatrix2D(
                    (kernel_size // 2, kernel_size // 2), angle, 1.0
                )
                kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
                img_float = cv2.filter2D(img_float, -1, kernel)

        if defect_type == "scan_lines":
            for _ in range(self.rng.randint(2, 6)):
                y = self.rng.randint(0, img_float.shape[0] - 1)
                thickness = self.rng.randint(1, 3)
                brightness_change = self.rng.uniform(-30, 30)
                img_float[
                    max(0, y - thickness) : min(img_float.shape[0], y + thickness), :
                ] += brightness_change

        if defect_type == "jpeg":
            quality = self.rng.randint(40, 75)
            _, encoded = cv2.imencode(
                ".jpg",
                img_float.astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            img_float = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if self.rng.random() < 0.3:
            brightness_map = gaussian_filter(
                np.random.randn(
                    img_float.shape[0] // 10, img_float.shape[1] // 10
                )
                * 15,
                sigma=2,
            )
            brightness_map = cv2.resize(
                brightness_map,
                (img_float.shape[1], img_float.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            img_float += brightness_map

        img_float = np.clip(img_float, 0, 255)
        return img_float.astype(np.uint8)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Модуль генерации синтетических датасетов для расопзнавания дверей и "
                                                 "окон на чертежах БТИ - v1.0")
    parser.add_argument("--output", type=str, default="bti_synth_dataset", help="Каталог для датасета")
    parser.add_argument("--count", type=int, default=200, help="Количество генерируемых изображений")
    parser.add_argument("--seed", type=int, default=11, help="Сид случайных чисел")
    parser.add_argument("--size", type=int, default=1024, help="Размер изображений в пикселях")

    args = parser.parse_args()

    gen = BTIFloorPlanGeneratorV3(img_size=args.size, seed=args.seed)
    gen.generate_dataset(args.count, args.output)
    print(f"Сгенерировано {args.count} изображений в папку {args.output}/")
    print(f"  - ~{int(args.count * 0.3)} чистых изображений")
    print(f"  - ~{int(args.count * 0.7)} изображений с имитацией дефектов скана & печати")
    print(f"  - ~{int(args.count * 0.15)} изображений с поворотом плана")