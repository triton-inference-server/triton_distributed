import uuid
from datetime import datetime

import pytest

from triton_distributed.icp.event_plane import Event, Topic
from triton_distributed.icp.event_plane_nats import EventPlaneNats


class TestEventTopic:
    def test_from_string(self):
        topic_str = "level1.level2"
        topic = Topic.from_string(topic_str)
        assert topic.chunks == ["level1", "level2"]

    def test_to_string(self):
        topic = Topic(["level1", "level2"])
        assert topic.to_string() == "level1.level2"


class TestEvent:
    @pytest.fixture
    def sample_event(self):
        topic = Topic("test.topic")
        return Event(
            event_id=uuid.uuid4(),
            topic=topic,
            event_type="test_event",
            timestamp=datetime.utcnow(),
            component_id=uuid.uuid4(),
            payload=b"test_payload",
        )

    def test_to_protobuf(self, sample_event):
        proto = sample_event.to_protobuf()
        assert proto.event_id == str(sample_event.event_id)
        assert proto.topic == str(sample_event.topic)
        assert proto.event_type == sample_event.event_type

    def test_from_protobuf(self, sample_event):
        proto = sample_event.to_protobuf()
        event = Event.from_protobuf(proto)
        assert event.event_id == sample_event.event_id
        assert event.topic.chunks == sample_event.topic.chunks


class TestEventPlaneNats:
    @pytest.fixture
    def event_plane_instance(self):
        server_url = "nats://localhost:4222"
        component_id = uuid.uuid4()
        return EventPlaneNats(server_url, component_id)

    @pytest.mark.asyncio
    async def test_create_event(self, event_plane_instance):
        event_type = "test_event"
        topic = Topic("test.topic")
        payload = b"test_payload"
        event = await event_plane_instance.create_event(event_type, topic, payload)
        assert event.event_type == event_type
        assert event.topic.to_string() == topic.to_string()
