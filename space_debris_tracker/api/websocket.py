"""
WebSocket Connection Manager
Manages real-time connections for satellite tracking
"""

import json
from typing import Dict, List

from fastapi import WebSocket


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, List[int]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept and store new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_json(self, message: Dict, websocket: WebSocket):
        """Send JSON message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending to client: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def subscribe(self, websocket: WebSocket, satellite_ids: List[int]):
        """Subscribe client to specific satellites"""
        self.subscriptions[websocket] = satellite_ids

    async def broadcast_to_subscribers(self, satellite_id: int, message: Dict):
        """Broadcast to clients subscribed to specific satellite"""
        for websocket, subs in self.subscriptions.items():
            if satellite_id in subs:
                await self.send_json(message, websocket)


if __name__ == "__main__":
    # Test connection manager
    manager = ConnectionManager()
    print("Connection manager created")
    print(f"Active connections: {len(manager.active_connections)}")
