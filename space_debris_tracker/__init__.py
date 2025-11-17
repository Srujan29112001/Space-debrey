"""
Space Debris Tracking & Autonomous Collision Prediction System
===============================================================

An AI-powered system for detecting, tracking, and predicting space debris
collisions using state-of-the-art computer vision, physics-informed ML,
and autonomous monitoring agents.

Modules:
--------
- computer_vision: YOLOv7, DINO v2, DeepSORT detection and tracking
- trajectory_prediction: PINN, Transformer, Mamba2 orbit prediction
- knowledge_graph: Neo4j-based GraphRAG for space situational awareness
- monitoring_agents: Multi-agent MCP system for autonomous monitoring
- api: FastAPI REST/GraphQL/WebSocket endpoints
- dashboard: Streamlit 3D visualization with Cesium.js

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Space Debris Tracking Team"
__license__ = "MIT"

from .computer_vision import SpaceDebrisDetector
from .knowledge_graph import SpaceKnowledgeGraph
from .monitoring_agents import SpaceMonitoringAgent
from .trajectory_prediction import OrbitPredictionEngine

__all__ = [
    "SpaceDebrisDetector",
    "OrbitPredictionEngine",
    "SpaceKnowledgeGraph",
    "SpaceMonitoringAgent",
]
