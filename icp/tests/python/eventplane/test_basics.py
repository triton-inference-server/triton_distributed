import uuid
from datetime import datetime

import pytest

from triton_distributed.icp.eventplane import Event, EventTopic
from triton_distributed.icp.eventplane_nats import EventPlaneNats


class TestEventTopic:
    def test_from_string(self):
        event_topic_str = "level1.level2"
        event_topic = EventTopic.from_string(event_topic_str)
        assert event_topic.chunks == ["level1", "level2"]

    def test_to_string(self):
        event_topic = EventTopic(["level1", "level2"])
        assert event_topic.to_string() == "level1.level2"


class TestEvent:
    @pytest.fixture
    def sample_event(self):
        event_topic = EventTopic("test.event_topic")
        return Event(
            event_id=uuid.uuid4(),
            event_topic=event_topic,
            event_type="test_event",
            timestamp=datetime.utcnow(),
            component_id=uuid.uuid4(),
            payload=b"test_payload",
        )

    def test_to_protobuf(self, sample_event):
        proto = sample_event.to_protobuf()
        assert proto.event_id == str(sample_event.event_id)
        assert proto.event_topic == str(sample_event.event_topic)
        assert proto.event_type == sample_event.event_type

    def test_from_protobuf(self, sample_event):
        proto = sample_event.to_protobuf()
        event = Event.from_protobuf(proto)
        assert event.event_id == sample_event.event_id
        assert event.event_topic.chunks == sample_event.event_topic.chunks


class TestEventPlaneNats:
    @pytest.fixture
    def event_plane_instance(self):
        server_url = "nats://localhost:4222"
        component_id = uuid.uuid4()
        return EventPlaneNats(server_url, component_id)

    @pytest.mark.asyncio
    async def test_create_event(self, event_plane_instance):
        event_type = "test_event"
        event_topic = EventTopic("test.event_topic")
        payload = b"test_payload"
        event = await event_plane_instance.create_event(
            event_type, event_topic, payload
        )
        assert event.event_type == event_type
        assert event.event_topic.to_string() == event_topic.to_string()
