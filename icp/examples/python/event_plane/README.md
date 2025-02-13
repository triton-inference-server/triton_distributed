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

### Using context managers (recommended)

```python
async def example_with_context_managers():
    server_url = "tls://localhost:4222"
    component_id = uuid.uuid4()

    async with NatsEventPlane(server_url, component_id) as plane:
        received_events = []

        async def callback(event, error):
            if error:
                print(f"Error occurred: {error}")
                return
            print(event)
            received_events.append(event)

        event_topic = EventTopic(["test", "event_topic"])
        event_type = "test_event"
        event = b"my_payload"

        # Subscribe using context manager
        async with await plane.subscribe(callback, event_topic=event_topic, event_type=event_type):
            # Publish event
            await plane.publish(event, event_type, event_topic)
            # Allow time for message to propagate
            await asyncio.sleep(2)
```

### Manual resource management

#### 1) Initialize NATS server and create an event plane
```python
    server_url = "tls://localhost:4222"
    component_id = uuid.uuid4()
    plane = NatsEventPlane(server_url, component_id)
    await plane.connect()
```

#### 2) Define the callback function for receiving events
```python
    received_events = []
    async def callback(event, error):
        if error:
            print(f"Error occurred: {error}")
            return
        print(event)
        received_events.append(event)
```

#### 3) Prepare the event event_topic, event type, and event payload
```python
    event_topic = EventTopic(["test", "event_topic"])
    event_type = "test_event"
    event = b"my_payload"
```

#### 4) Subscribe to the event event_topic and type and register the callback function
```python
    subscription = await plane.subscribe(callback, event_topic=event_topic, event_type=event_type)
```

#### 5) Publish the event
```python
    await plane.publish(event, event_type, event_topic)
```

#### 6) Clean up resources
```python
    # Unsubscribe when done
    await subscription.unsubscribe()

    # Disconnect from NATS server
    await plane.disconnect()
```
