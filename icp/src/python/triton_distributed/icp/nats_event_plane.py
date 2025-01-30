import uuid
from datetime import datetime
from typing import Optional

import nats

from triton_distributed.icp.event_plane import Event, Topic
from triton_distributed.icp.protos import event_pb2


class NatsEventPlane:
    """EventPlane implementation using NATS."""

    def __init__(self, server_uri: str, component_id: uuid.UUID):
        self._server_uri = server_uri
        self._component_id = component_id
        self._nc = nats.NATS()

    async def connect(self):
        await self._nc.connect(self._server_uri)

    async def create_event(self, event_type: str, topic: Topic, payload: bytes):
        event = Event(
            event_id=uuid.uuid4(),
            topic=topic,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            component_id=self._component_id,
            payload=payload,
        )
        return event

    async def publish(self, event: Event):
        event_pb = event.to_protobuf()
        message = event_pb.SerializeToString()
        subject = f"ep.{event.event_type}.{event.component_id}.{event.topic}.trunk"
        await self._nc.publish(subject, message)

    async def subscribe(
        self,
        callback,
        topic: Optional[Topic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ):
        async def message_handler(msg):
            event_pb = event_pb2.Event()
            event_pb.ParseFromString(msg.data)
            event = Event.from_protobuf(event_pb)
            await callback(event)

        subject = f"ep.{event_type or '*'}.{component_id or '*'}.{str(topic) + '.' if topic else ''}>"
        await self._nc.subscribe(subject, cb=message_handler)

    async def disconnect(self):
        await self._nc.close()
