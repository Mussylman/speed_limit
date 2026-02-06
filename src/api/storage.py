# storage.py
# Работа с outputs/ - чтение, кэш, tail

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from threading import Lock
from datetime import datetime
import time


class OutputStorage:
    """
    Менеджер для работы с outputs/.

    Функции:
    - Чтение JSON файлов (violations, vehicles, speeds)
    - Кэширование с TTL
    - Tail для jsonl файлов
    - Поиск последнего run_dir для камеры
    """

    def __init__(self, outputs_dir: str, cache_ttl: float = 5.0):
        self.outputs_dir = Path(outputs_dir)
        self.cache_ttl = cache_ttl

        self._cache: Dict[str, tuple] = {}  # {path: (data, timestamp)}
        self._cache_lock = Lock()

    def _get_cached(self, path: Path) -> Optional[Any]:
        """Возвращает данные из кэша если не устарели"""
        key = str(path)
        with self._cache_lock:
            if key in self._cache:
                data, ts = self._cache[key]
                if time.time() - ts < self.cache_ttl:
                    return data
        return None

    def _set_cache(self, path: Path, data: Any):
        """Сохраняет в кэш"""
        key = str(path)
        with self._cache_lock:
            self._cache[key] = (data, time.time())

    def _load_json(self, path: Path) -> List[Dict]:
        """Загружает JSON файл"""
        if not path.exists():
            return []

        # Проверяем кэш
        cached = self._get_cached(path)
        if cached is not None:
            return cached

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._set_cache(path, data)
                    return data
                return [data] if data else []
        except (json.JSONDecodeError, IOError):
            return []

    def get_latest_run_dir(self, camera_id: str) -> Optional[Path]:
        """Находит последнюю папку run для камеры"""
        if not self.outputs_dir.exists():
            return None

        pattern = f"{camera_id}_run_*"
        dirs = sorted(self.outputs_dir.glob(pattern), reverse=True)
        return dirs[0] if dirs else None

    def get_all_run_dirs(self) -> List[tuple]:
        """Возвращает все run_dirs с camera_id"""
        if not self.outputs_dir.exists():
            return []

        result = []
        for d in self.outputs_dir.iterdir():
            if d.is_dir() and "_run_" in d.name:
                cam_id = d.name.split("_run_")[0]
                result.append((cam_id, d))
        return result

    # =========================================================
    # Нарушения
    # =========================================================

    def get_violations(
        self,
        camera_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple:
        """
        Возвращает нарушения.

        Returns:
            (items, total)
        """
        all_violations = []

        if camera_id:
            run_dir = self.get_latest_run_dir(camera_id)
            if run_dir:
                path = run_dir / "speeds" / "violations.json"
                violations = self._load_json(path)
                for v in violations:
                    v["camera_id"] = camera_id
                all_violations.extend(violations)
        else:
            for cam_id, run_dir in self.get_all_run_dirs():
                path = run_dir / "speeds" / "violations.json"
                violations = self._load_json(path)
                for v in violations:
                    v["camera_id"] = cam_id
                all_violations.extend(violations)

        # Сортировка по времени
        all_violations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        total = len(all_violations)
        items = all_violations[offset:offset + limit]

        return items, total

    # =========================================================
    # Транспорт / Номера
    # =========================================================

    def get_vehicles(
        self,
        camera_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple:
        """
        Возвращает распознанные номера.

        Returns:
            (items, total)
        """
        all_vehicles = []

        if camera_id:
            run_dir = self.get_latest_run_dir(camera_id)
            if run_dir:
                path = run_dir / "passed" / "results.json"
                vehicles = self._load_json(path)
                for v in vehicles:
                    v["camera_id"] = camera_id
                all_vehicles.extend(vehicles)
        else:
            for cam_id, run_dir in self.get_all_run_dirs():
                path = run_dir / "passed" / "results.json"
                vehicles = self._load_json(path)
                for v in vehicles:
                    v["camera_id"] = cam_id
                all_vehicles.extend(vehicles)

        all_vehicles.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        total = len(all_vehicles)
        items = all_vehicles[offset:offset + limit]

        return items, total

    # =========================================================
    # Скорости
    # =========================================================

    def get_speeds(
        self,
        camera_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Возвращает измерения скорости"""
        all_speeds = []

        if camera_id:
            run_dir = self.get_latest_run_dir(camera_id)
            if run_dir:
                path = run_dir / "speeds" / "all_speeds.json"
                speeds = self._load_json(path)
                for s in speeds:
                    s["camera_id"] = camera_id
                all_speeds.extend(speeds)
        else:
            for cam_id, run_dir in self.get_all_run_dirs():
                path = run_dir / "speeds" / "all_speeds.json"
                speeds = self._load_json(path)
                for s in speeds:
                    s["camera_id"] = cam_id
                all_speeds.extend(speeds)

        all_speeds.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_speeds[:limit]

    # =========================================================
    # Tail (для jsonl)
    # =========================================================

    def tail_jsonl(self, camera_id: str, filename: str, n: int = 100) -> List[Dict]:
        """
        Читает последние N строк из jsonl файла.

        Args:
            camera_id: ID камеры
            filename: имя файла (например "metrics.jsonl")
            n: количество строк

        Returns:
            Список записей
        """
        run_dir = self.get_latest_run_dir(camera_id)
        if not run_dir:
            return []

        path = run_dir / filename
        if not path.exists():
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            result = []
            for line in lines[-n:]:
                line = line.strip()
                if line:
                    try:
                        result.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return result
        except IOError:
            return []

    # =========================================================
    # Утилиты
    # =========================================================

    def get_image_path(self, camera_id: str, relative_path: str) -> Optional[Path]:
        """Возвращает полный путь к изображению"""
        run_dir = self.get_latest_run_dir(camera_id)
        if not run_dir:
            return None

        full_path = run_dir / relative_path
        if full_path.exists():
            return full_path
        return None

    def clear_cache(self):
        """Очищает кэш"""
        with self._cache_lock:
            self._cache.clear()

    def get_summary(self, camera_id: str) -> Optional[Dict]:
        """Возвращает summary метрик для камеры"""
        run_dir = self.get_latest_run_dir(camera_id)
        if not run_dir:
            return None

        path = run_dir / "metrics_summary.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None
