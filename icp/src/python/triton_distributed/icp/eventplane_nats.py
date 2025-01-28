import uuid
from datetime import datetime
from typing import Optional

import nats

from triton_distributed.icp.eventplane import Channel, Event
from triton_distributed.icp.protos import event_pb2


class EventPlaneNats:
    """EventPlane implementation using NATS."""

    def __init__(self, server_url: str, component_id: uuid.UUID):
        self.server_url = server_url
        self.component_id = component_id
        self.nc = nats.NATS()

    async def connect(self):
        await self.nc.connect(self.server_url)

    async def create_event(self, event_type: str, channel: Channel, payload: bytes):
        event = Event(
            event_id=uuid.uuid4(),
            channel=channel,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            component_id=self.component_id,
            payload=payload,
        )
        return event

    async def publish(self, event: Event):
        event_pb = event.to_protobuf()
        message = event_pb.SerializeToString()
        subject = f"ep.{event.event_type}.{event.component_id}.{event.channel}.trunk"
        await self.nc.publish(subject, message)

    async def subscribe(
        self,
        callback,
        channel: Optional[Channel] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ):
        async def message_handler(msg):
            event_pb = event_pb2.Event()
            event_pb.ParseFromString(msg.data)
            event = Event.from_protobuf(event_pb)
            await callback(event)

        subject = f"ep.{event_type or '*'}.{component_id or '*'}.{str(channel) + '.' if channel else ''}>"
        await self.nc.subscribe(subject, cb=message_handler)

    async def disconnect(self):
        await self.nc.close()
