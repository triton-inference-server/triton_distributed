import uuid
from datetime import datetime

import pytest

from triton_distributed.eventplane.eventplane import Event, Channel
from triton_distributed.eventplane.eventplane_nats import EventPlaneNats


class TestChannel:
    def test_from_string(self):
        channel_str = "level1.level2"
        channel = Channel.from_string(channel_str)
        assert channel.chunks == ["level1", "level2"]

    def test_to_string(self):
        channel = Channel(["level1", "level2"])
        assert channel.to_string() == "level1.level2"


class TestEvent:
    @pytest.fixture
    def sample_event(self):
        channel = Channel("test.channel")
        return Event(
            event_id=uuid.uuid4(),
            channel=channel,
            event_type="test_event",
            timestamp=datetime.utcnow(),
            component_id=uuid.uuid4(),
            payload=b"test_payload",
        )

    def test_to_protobuf(self, sample_event):
        proto = sample_event.to_protobuf()
        assert proto.event_id == str(sample_event.event_id)
        assert proto.channel == str(sample_event.channel)
        assert proto.event_type == sample_event.event_type

    def test_from_protobuf(self, sample_event):
        proto = sample_event.to_protobuf()
        event = Event.from_protobuf(proto)
        assert event.event_id == sample_event.event_id
        assert event.channel.chunks == sample_event.channel.chunks


class TestEventPlaneNats:
    @pytest.fixture
    def event_plane_instance(self):
        server_url = "nats://localhost:4222"
        component_id = uuid.uuid4()
        return EventPlaneNats(server_url, component_id)

    @pytest.mark.asyncio
    async def test_create_event(self, event_plane_instance):
        event_type = "test_event"
        channel = Channel("test.channel")
        payload = b"test_payload"
        event = await event_plane_instance.create_event(event_type, channel, payload)
        assert event.event_type == event_type
        assert event.channel.to_string() == channel.to_string()
