"""
Main FastAPI Server
Provides REST, GraphQL, and WebSocket endpoints
"""

import asyncio
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import main components
from space_debris_tracker import (
    OrbitPredictionEngine,
    SpaceDebrisDetector,
    SpaceKnowledgeGraph,
)

from .graphql_api import graphql_app
from .rest import router as rest_router
from .websocket import ConnectionManager


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Space Debris Tracking API",
        description="AI-Powered Space Debris Tracking & Collision Prediction System",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production: specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(rest_router, prefix="/api/v1", tags=["REST API"])

    # Mount GraphQL
    app.mount("/graphql", graphql_app)

    # WebSocket connection manager
    manager = ConnectionManager()

    # Store components
    app.state.detector = None
    app.state.predictor = None
    app.state.knowledge_graph = None
    app.state.manager = manager

    @app.on_event("startup")
    async def startup_event():
        """Initialize components on startup"""
        print("Initializing Space Debris Tracking System...")

        # Initialize detector
        try:
            app.state.detector = SpaceDebrisDetector()
            print("✓ Detector initialized")
        except Exception as e:
            print(f"✗ Detector initialization failed: {e}")

        # Initialize predictor
        try:
            app.state.predictor = OrbitPredictionEngine()
            print("✓ Predictor initialized")
        except Exception as e:
            print(f"✗ Predictor initialization failed: {e}")

        # Initialize knowledge graph
        try:
            app.state.knowledge_graph = SpaceKnowledgeGraph()
            print("✓ Knowledge graph connected")
        except Exception as e:
            print(f"✗ Knowledge graph connection failed: {e}")

        print("System ready!")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        if app.state.knowledge_graph:
            app.state.knowledge_graph.close()
        print("System shutdown complete")

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "Space Debris Tracking API",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "docs": "/api/docs",
                "rest": "/api/v1",
                "graphql": "/graphql",
                "websocket": "/ws/tracking",
            },
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "detector": app.state.detector is not None,
                "predictor": app.state.predictor is not None,
                "knowledge_graph": app.state.knowledge_graph is not None,
            },
        }

    @app.websocket("/ws/tracking")
    async def websocket_tracking(websocket: WebSocket):
        """
        Real-time tracking WebSocket endpoint

        Clients can subscribe to satellite tracking updates
        """
        await manager.connect(websocket)

        try:
            # Receive subscription info
            data = await websocket.receive_json()
            satellite_ids = data.get("satellites", [])
            update_rate = data.get("update_rate", 1.0)  # Hz

            # Send updates
            while True:
                updates = []

                # Get updates for subscribed satellites
                for sat_id in satellite_ids:
                    try:
                        # Get satellite state (placeholder)
                        state = {
                            "satellite_id": sat_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "position": [6778.0, 0.0, 0.0],
                            "velocity": [0.0, 7.66, 0.0],
                            "risk_level": "LOW",
                        }
                        updates.append(state)
                    except Exception as e:
                        print(f"Error getting state for {sat_id}: {e}")

                # Send updates
                await manager.send_json({"type": "position_update", "data": updates}, websocket)

                # Wait based on update rate
                await asyncio.sleep(1.0 / update_rate)

        except WebSocketDisconnect:
            manager.disconnect(websocket)
            print("Client disconnected")
        except Exception as e:
            print(f"WebSocket error: {e}")
            manager.disconnect(websocket)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "space_debris_tracker.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
