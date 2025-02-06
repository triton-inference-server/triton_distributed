<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Event Plane example

A basic example that demonstrates how to use the Event Plane API to create an event plane, register an event, and trigger the event.

## Code overview

### 1) Initialize NATS server and create an event plane
```python
    server_url = "nats://localhost:4222"
    component_id = uuid.uuid4()
    plane = NatsEventPlane(server_url, component_id)
    await plane.connect()
```

### 2) Define the callback function for receiving events
```python
    received_events = []
    async def callback(event, metadata):
        print(metadata)
        received_events.append(metadata)
```

### 3) Prepare the event event_topic, event type, and event payload
```python
    event_topic = EventTopic(["test", "event_topic"])
    event_type = "test_event"
    event = b"my_payload"
```

### 4) Subscribe to the event event_topic and type and register the callback function
```python
    await plane.subscribe(callback, event_topic=event_topic, event_type=event_type)
```

### 5) Publish the event
```python
    await plane.publish(event, event_type, event_topic)
```
