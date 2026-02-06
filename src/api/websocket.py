# websocket.py
# WebSocket manager для real-time событий

import asyncio
import json
from typing import Dict, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from fastapi import WebSocket


@dataclass
class WSMessage:
    """WebSocket сообщение"""
    channel: str
    event: str
    data: Any
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


class WebSocketManager:
    """
    Менеджер WebSocket подключений.

    Каналы:
        speed:{camera_id}      - обновления скорости
        violations:{camera_id} - нарушения
        plates:{camera_id}     - распознанные номера
        stats:{camera_id}      - статистика
        system                 - системные события
    """

    def __init__(self):
        self._subscriptions: Dict[str, Set[WebSocket]] = {}
        self._ws_channels: Dict[WebSocket, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Принимает подключение"""
        await websocket.accept()
        async with self._lock:
            self._ws_channels[websocket] = set()

    async def disconnect(self, websocket: WebSocket):
        """Отключает клиента"""
        async with self._lock:
            channels = self._ws_channels.pop(websocket, set())
            for channel in channels:
                if channel in self._subscriptions:
                    self._subscriptions[channel].discard(websocket)
                    if not self._subscriptions[channel]:
                        del self._subscriptions[channel]

    async def subscribe(self, websocket: WebSocket, channel: str):
        """Подписывает на канал"""
        async with self._lock:
            if channel not in self._subscriptions:
                self._subscriptions[channel] = set()
            self._subscriptions[channel].add(websocket)

            if websocket in self._ws_channels:
                self._ws_channels[websocket].add(channel)

        await self._send(websocket, WSMessage(
            channel=channel,
            event="subscribed",
            data={"channel": channel}
        ))

    async def unsubscribe(self, websocket: WebSocket, channel: str):
        """Отписывает от канала"""
        async with self._lock:
            if channel in self._subscriptions:
                self._subscriptions[channel].discard(websocket)
                if not self._subscriptions[channel]:
                    del self._subscriptions[channel]

            if websocket in self._ws_channels:
                self._ws_channels[websocket].discard(channel)

        await self._send(websocket, WSMessage(
            channel=channel,
            event="unsubscribed",
            data={"channel": channel}
        ))

    async def broadcast(self, channel: str, event: str, data: Any):
        """Отправляет всем подписчикам канала"""
        message = WSMessage(channel=channel, event=event, data=data)

        async with self._lock:
            subscribers = self._subscriptions.get(channel, set()).copy()

        if not subscribers:
            return

        disconnected = []
        for ws in subscribers:
            try:
                await ws.send_text(message.to_json())
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            await self.disconnect(ws)

    async def broadcast_pattern(self, pattern: str, event: str, data: Any):
        """Отправляет по паттерну (speed:* → все speed:cam1, speed:cam2...)"""
        async with self._lock:
            matching = [
                ch for ch in self._subscriptions.keys()
                if self._match(ch, pattern)
            ]

        for channel in matching:
            await self.broadcast(channel, event, data)

    def _match(self, channel: str, pattern: str) -> bool:
        if pattern.endswith("*"):
            return channel.startswith(pattern[:-1])
        return channel == pattern

    async def _send(self, websocket: WebSocket, message: WSMessage):
        try:
            await websocket.send_text(message.to_json())
        except Exception:
            await self.disconnect(websocket)

    async def handle_message(self, websocket: WebSocket, data: str):
        """
        Обрабатывает сообщение от клиента.

        Формат:
            {"action": "subscribe", "channel": "speed:cam1"}
            {"action": "unsubscribe", "channel": "speed:cam1"}
            {"action": "ping"}
        """
        try:
            msg = json.loads(data)
            action = msg.get("action")
            channel = msg.get("channel")

            if action == "subscribe" and channel:
                await self.subscribe(websocket, channel)
            elif action == "unsubscribe" and channel:
                await self.unsubscribe(websocket, channel)
            elif action == "ping":
                await self._send(websocket, WSMessage(
                    channel="system",
                    event="pong",
                    data={}
                ))
        except json.JSONDecodeError:
            pass

    def get_stats(self) -> dict:
        return {
            "total_connections": len(self._ws_channels),
            "channels": {ch: len(subs) for ch, subs in self._subscriptions.items()}
        }


# Глобальный экземпляр
ws_manager = WebSocketManager()
