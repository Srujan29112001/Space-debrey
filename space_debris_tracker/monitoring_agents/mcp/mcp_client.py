"""
MCP Client for Inter-Agent Communication
Uses ZeroMQ for message passing between monitoring agents
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List

try:
    import zmq
    import zmq.asyncio

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("Warning: pyzmq not available. Install with: pip install pyzmq")


class MCPClient:
    """
    MCP (Model Context Protocol) Client
    Enables communication between monitoring agents
    """

    def __init__(self, agent_id: str, broker_url: str = "tcp://localhost:5555"):
        """
        Initialize MCP client

        Args:
            agent_id: Unique agent identifier
            broker_url: ZeroMQ broker URL
        """
        self.agent_id = agent_id
        self.broker_url = broker_url

        if ZMQ_AVAILABLE:
            self.context = zmq.asyncio.Context()

            # Publisher socket (for broadcasting)
            self.publisher = self.context.socket(zmq.PUB)

            # Subscriber socket (for receiving)
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all

            # Connect sockets
            try:
                self.publisher.connect(broker_url)
                self.subscriber.connect(broker_url)
                print(f"MCP Client '{agent_id}' connected to {broker_url}")
            except Exception as e:
                print(f"Warning: Could not connect to MCP broker: {e}")
        else:
            self.context = None
            self.publisher = None
            self.subscriber = None

        self.message_queue = []

    async def broadcast(self, message: Dict):
        """
        Broadcast message to all agents

        Args:
            message: Message dictionary
        """
        if not self.publisher:
            return

        # Add metadata
        message["sender"] = self.agent_id
        message["timestamp"] = datetime.utcnow().isoformat()

        # Serialize and send
        try:
            msg_bytes = json.dumps(message).encode("utf-8")
            await self.publisher.send(msg_bytes)
        except Exception as e:
            print(f"Error broadcasting message: {e}")

    async def receive_messages(self, timeout: float = 0.1) -> List[Dict]:
        """
        Receive messages from other agents

        Args:
            timeout: Timeout in seconds

        Returns:
            List of received messages
        """
        if not self.subscriber:
            return []

        messages = []

        try:
            # Non-blocking receive with timeout
            while True:
                try:
                    msg_bytes = await asyncio.wait_for(self.subscriber.recv(), timeout=timeout)

                    message = json.loads(msg_bytes.decode("utf-8"))

                    # Filter out own messages
                    if message.get("sender") != self.agent_id:
                        messages.append(message)

                except asyncio.TimeoutError:
                    break

        except Exception as e:
            print(f"Error receiving messages: {e}")

        return messages

    async def send_to_agent(self, target_agent: str, message: Dict):
        """
        Send message to specific agent

        Args:
            target_agent: Target agent ID
            message: Message dictionary
        """
        message["target"] = target_agent
        await self.broadcast(message)

    def close(self):
        """Close connections"""
        if self.publisher:
            self.publisher.close()
        if self.subscriber:
            self.subscriber.close()
        if self.context:
            self.context.term()


if __name__ == "__main__":
    # Test MCP client
    async def test():
        client = MCPClient("test_agent_1")

        # Broadcast test message
        await client.broadcast({"type": "test", "data": "Hello from agent 1"})

        # Receive messages
        messages = await client.receive_messages()
        print(f"Received {len(messages)} messages")

        client.close()

    asyncio.run(test())
