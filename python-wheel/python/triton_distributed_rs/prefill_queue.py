import asyncio

from nats.aio.client import Client as NATS
from nats.js.errors import NotFoundError


class PrefillQueue:
    def __init__(self, nats_host="localhost", nats_port=4222, dequeue_timeout=1):
        self.nats_url = f"nats://{nats_host}:{nats_port}"
        self.nc = None
        self.js = None
        self.stream_name = "prefill_queue"
        self.subject = f"{self.stream_name}.*"
        self.dequeue_timeout = dequeue_timeout
        self._subscriber = None

    async def connect(self):
        """Establish connection and create stream if needed"""
        if self.nc is None:
            self.nc = NATS()
            await self.nc.connect(self.nats_url)
            self.js = self.nc.jetstream()
            # Check if stream exists, if not create it
            try:
                await self.js.stream_info(self.stream_name)
            except NotFoundError:
                await self.js.add_stream(name=self.stream_name, subjects=[self.subject])
                print(f"Stream '{self.stream_name}' created")
            # Create persistent subscriber
            self._subscriber = await self.js.pull_subscribe(
                f"{self.stream_name}.queue", durable="worker-group"
            )

    async def ensure_connection(self):
        """Ensure we have an active connection"""
        if self.nc is None or self.nc.is_closed:
            await self.connect()

    async def close(self):
        """Close the connection when done"""
        if self.nc:
            await self.nc.close()
            self.nc = None
            self.js = None
            self._subscriber = None

    async def enqueue_task(self, task_data: str):
        """Enqueue a task using JSON string from Pydantic model_dump_json()"""
        await self.ensure_connection()
        await self.js.publish(f"{self.stream_name}.queue", task_data.encode("utf-8"))

    async def dequeue_task(self):
        """Dequeue and return a task as JSON string, or None if queue is empty"""
        await self.ensure_connection()
        try:
            msgs = await self._subscriber.fetch(1, timeout=self.dequeue_timeout)
            if msgs:
                msg = msgs[0]
                await msg.ack()
                return msg.data.decode("utf-8")
        except asyncio.TimeoutError:
            return None
