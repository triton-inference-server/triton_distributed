import uuid
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

from triton_distributed.icp.protos import event_pb2


class EventTopic:
    """Event topic class for identifying event streams."""

    def __init__(self, channel: Union[List[str], str]):
        if isinstance(channel, str):
            self.chunks = channel.split(".")
        else:
            self.chunks = channel

    def __str__(self):
        return ".".join(self.chunks)

    def to_string(self):
        """Convert Topic to string."""
        return str(self)

    @staticmethod
    def from_string(channel_str: str):
        """Create Topic from string."""
        return EventTopic(channel_str.split("."))


@dataclass
class Event:
    """Event class for representing events."""

    event_id: uuid.UUID
    channel: EventTopic
    event_type: str
    timestamp: datetime
    component_id: uuid.UUID
    payload: bytes

    def to_protobuf(self):
        """Convert Event to Protobuf message."""
        event_pb = event_pb2.Event()
        event_pb.event_id = str(self.event_id)
        event_pb.channel = self.channel.to_string()
        event_pb.event_type = self.event_type
        event_pb.timestamp.FromDatetime(self.timestamp)
        event_pb.component_id = str(self.component_id)
        event_pb.payload = self.payload
        return event_pb

    @staticmethod
    def from_protobuf(event_pb):
        """Create Event from Protobuf message."""

        return Event(
            event_id=uuid.UUID(event_pb.event_id),
            channel=EventTopic.from_string(event_pb.channel),
            event_type=event_pb.event_type,
            timestamp=event_pb.timestamp.ToDatetime(),
            component_id=uuid.UUID(event_pb.component_id),
            payload=event_pb.payload,
        )


class EventPlane:
    """EventPlane interface for publishing and subscribing to events."""

    def __init__(self, server_url: str):
        pass

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def create_event(self, event_type: str, channel: EventTopic, payload: bytes):
        pass

    @abstractmethod
    async def publish(self, event: Event):
        pass

    @abstractmethod
    async def subscribe(
        self,
        callback,
        channel: Optional[EventTopic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ):
        pass

    @abstractmethod
    async def disconnect(self):
        pass
