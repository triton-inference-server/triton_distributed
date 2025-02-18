# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import uuid
from datetime import datetime

import pytest

from triton_distributed.icp.nats_event_plane import (
    EventMetadata,
    EventTopic,
    NatsEventPlane,
)

pytestmark = pytest.mark.pre_merge


class TestEventTopic:
    def test_from_string(self):
        topic_str = "level1"
        event_topic = EventTopic(topic_str)
        assert event_topic.event_topic == topic_str

    def test_to_string(self):
        event_topic = EventTopic(["level1", "level2"])
        assert str(event_topic) == "level1.level2"


class TestEvent:
    @pytest.fixture
    def sample_event_metadata(self):
        event_topic = EventTopic("test.event_topic")
        return EventMetadata(
            event_id=uuid.uuid4(),
            event_topic=event_topic,
            event_type="test_event",
            timestamp=datetime.utcnow(),
            component_id=uuid.uuid4(),
        )


class TestEventPlaneNats:
    @pytest.fixture
    def event_plane_instance(self):
        server_url = "tls://localhost:4222"
        component_id = uuid.uuid4()
        return NatsEventPlane(server_url, component_id)
