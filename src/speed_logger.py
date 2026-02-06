# speed_logger.py
# Логирование скорости с отметкой нарушений

import os
import json
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class SpeedEvent:
    """Данные о скорости транспорта"""
    track_id: int = 0
    speed_kmh: float = 0.0
    timestamp: str = ""
    frame_idx: int = 0
    camera_id: str = ""

    # Лимит и нарушение
    speed_limit: int = 70
    is_violation: bool = False

    # Номер (если распознан)
    plate_text: str = ""
    plate_conf: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Конвертируем numpy типы
        for key, value in d.items():
            if isinstance(value, (np.bool_, bool)):
                d[key] = bool(value)
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                d[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                d[key] = float(value)
        return d


class SpeedLogger:
    """
    Логирование скорости транспорта.

    Структура:
    output_dir/
    └── speeds/
        ├── all_speeds.json     # все измерения
        └── violations.json     # только нарушители (> speed_limit)
    """

    def __init__(
        self,
        output_dir: str,
        camera_id: str = "camera_01",
        speed_limit: int = 70,
    ):
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.speed_limit = speed_limit

        # Папка для скоростей
        self.speeds_dir = os.path.join(output_dir, "speeds")
        os.makedirs(self.speeds_dir, exist_ok=True)

        # Лучшая скорость для каждого track_id (максимальная)
        self.speeds: Dict[int, SpeedEvent] = {}

        # Статистика
        self.stats = {
            "total_measurements": 0,
            "unique_vehicles": 0,
            "violations": 0,
            "max_speed": 0.0,
            "avg_speed": 0.0,
        }

        print(f"🚗 Лимит скорости: {speed_limit} км/ч")

    def update(
        self,
        track_id: int,
        speed_kmh: float,
        frame_idx: int = 0,
        plate_text: str = "",
        plate_conf: float = 0.0,
    ):
        """Обновляет скорость для track_id (сохраняет максимальную)"""
        if speed_kmh <= 0:
            return

        self.stats["total_measurements"] += 1

        current = self.speeds.get(track_id)

        # Сохраняем максимальную скорость
        if current is None or speed_kmh > current.speed_kmh:
            event = SpeedEvent(
                track_id=track_id,
                speed_kmh=round(speed_kmh, 1),
                timestamp=datetime.now().isoformat(),
                frame_idx=frame_idx,
                camera_id=self.camera_id,
                speed_limit=self.speed_limit,
                is_violation=speed_kmh > self.speed_limit,
                plate_text=plate_text,
                plate_conf=round(plate_conf, 2) if plate_conf else 0.0,
            )
            self.speeds[track_id] = event

            # Логируем нарушения
            if event.is_violation and (current is None or not current.is_violation):
                print(f"🚨 НАРУШЕНИЕ! ID:{track_id} → {speed_kmh:.0f} км/ч (лимит {self.speed_limit})")

    def update_plate(self, track_id: int, plate_text: str, plate_conf: float):
        """Обновляет номер для существующей записи скорости"""
        if track_id in self.speeds:
            self.speeds[track_id].plate_text = plate_text
            self.speeds[track_id].plate_conf = round(plate_conf, 2)

    def finalize(self):
        """Сохраняет все результаты на диск"""
        all_speeds = []
        violations = []

        speeds_list = list(self.speeds.values())

        for event in speeds_list:
            data = event.to_dict()
            all_speeds.append(data)

            if event.is_violation:
                violations.append(data)

        # Статистика
        if speeds_list:
            self.stats["unique_vehicles"] = len(speeds_list)
            self.stats["violations"] = len(violations)
            self.stats["max_speed"] = max(e.speed_kmh for e in speeds_list)
            self.stats["avg_speed"] = round(sum(e.speed_kmh for e in speeds_list) / len(speeds_list), 1)

        # Сохраняем все скорости
        all_path = os.path.join(self.speeds_dir, "all_speeds.json")
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump({
                "camera_id": self.camera_id,
                "speed_limit": self.speed_limit,
                "stats": self.stats,
                "vehicles": all_speeds,
            }, f, ensure_ascii=False, indent=2)

        # Сохраняем нарушителей
        viol_path = os.path.join(self.speeds_dir, "violations.json")
        with open(viol_path, "w", encoding="utf-8") as f:
            json.dump({
                "camera_id": self.camera_id,
                "speed_limit": self.speed_limit,
                "total_violations": len(violations),
                "vehicles": violations,
            }, f, ensure_ascii=False, indent=2)

        print(f"\n🚗 Статистика скорости:")
        print(f"   Всего измерений: {self.stats['total_measurements']}")
        print(f"   Уникальных ТС: {self.stats['unique_vehicles']}")
        print(f"   Средняя скорость: {self.stats['avg_speed']} км/ч")
        print(f"   Макс. скорость: {self.stats['max_speed']} км/ч")
        print(f"   🚨 Нарушителей (>{self.speed_limit} км/ч): {self.stats['violations']}")
        print(f"\n📁 Скорости: {self.speeds_dir}")
