import uuid
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

from triton_distributed.icp.protos import event_pb2


class EventTopic:
    """Event topic class for identifying event streams."""

    def __init__(self, event_topic: Union[List[str], str]):
        if isinstance(event_topic, str):
            self._event_topic = event_topic
        else:
            self._event_topic = ".".join(event_topic)

    def __str__(self):
        return self._event_topic

    def to_string(self):
        """Convert Topic to string."""
        return str(self)

    @staticmethod
    def from_string(event_topic_str: str):
        """Create Topic from string."""
        return EventTopic(event_topic_str)


@dataclass
class Event:
    """Event class for representing events."""

    event_id: uuid.UUID
    event_topic: EventTopic
    event_type: str
    timestamp: datetime
    component_id: uuid.UUID
    payload: bytes

    def to_protobuf(self):
        """Convert Event to Protobuf message."""
        event_pb = event_pb2.Event()
        event_pb.event_id = str(self.event_id)
        event_pb.event_topic = self.event_topic.to_string()
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
            event_topic=EventTopic.from_string(event_pb.event_topic),
            event_type=event_pb.event_type,
            timestamp=event_pb.timestamp.ToDatetime(),
            component_id=uuid.UUID(event_pb.component_id),
            payload=event_pb.payload,
        )


class EventPlane:
    """EventPlane interface for publishing and subscribing to events."""

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def create_event(
        self, event_type: str, event_topic: EventTopic, payload: bytes
    ):
        pass

    @abstractmethod
    async def publish(self, event: Event):
        pass

    @abstractmethod
    async def subscribe(
        self,
        callback,
        event_topic: Optional[EventTopic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ):
        pass

    @abstractmethod
    async def disconnect(self):
        pass
