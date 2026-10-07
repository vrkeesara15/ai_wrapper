"""Work queues with retries and a dead-letter queue.

Used by the ingestion pipeline (one message per document), the embedding service (separate live and bulk
queues) and the agent runtime (run queue). ``memory`` for a single process, ``redis`` (lists) for workers
spread across pods. Other brokers (Kafka, SQS, RabbitMQ) plug in via ``class: module:Class``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Protocol

from pydantic import BaseModel, Field

from aiready.core.plugins import create, register


class Message(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    topic: str
    payload: dict[str, Any]
    attempts: int = 0
    enqueued_at: float = Field(default_factory=time.time)
    last_error: str | None = None


class Queue(Protocol):
    async def put(self, topic: str, payload: dict[str, Any]) -> Message: ...
    async def get(self, topic: str, timeout: float | None = None) -> Message | None: ...
    async def ack(self, msg: Message) -> None: ...
    async def nack(self, msg: Message, error: str, max_attempts: int = 3) -> bool: ...
    async def depth(self, topic: str) -> int: ...
    async def dead_letters(self, topic: str) -> list[Message]: ...
    async def replay_dead_letters(self, topic: str) -> int: ...


@register("queue", "memory")
class MemoryQueue:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Message]] = {}
        self._dlq: dict[str, list[Message]] = {}

    def _q(self, topic: str) -> asyncio.Queue[Message]:
        if topic not in self._queues:
            self._queues[topic] = asyncio.Queue()
        return self._queues[topic]

    async def put(self, topic: str, payload: dict[str, Any]) -> Message:
        msg = Message(topic=topic, payload=payload)
        await self._q(topic).put(msg)
        return msg

    async def get(self, topic: str, timeout: float | None = None) -> Message | None:
        try:
            if timeout is None:
                return self._q(topic).get_nowait()
            return await asyncio.wait_for(self._q(topic).get(), timeout)
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            return None

    async def ack(self, msg: Message) -> None:
        return None

    async def nack(self, msg: Message, error: str, max_attempts: int = 3) -> bool:
        """Requeue with backoff or move to the dead-letter queue. Returns True if dead-lettered."""
        msg.attempts += 1
        msg.last_error = error
        if msg.attempts >= max_attempts:
            self._dlq.setdefault(msg.topic, []).append(msg)
            return True
        await asyncio.sleep(min(0.05 * 2 ** msg.attempts, 2.0))
        await self._q(msg.topic).put(msg)
        return False

    async def depth(self, topic: str) -> int:
        return self._q(topic).qsize()

    async def dead_letters(self, topic: str) -> list[Message]:
        return list(self._dlq.get(topic, []))

    async def replay_dead_letters(self, topic: str) -> int:
        items = self._dlq.pop(topic, [])
        for m in items:
            m.attempts = 0
            await self._q(topic).put(m)
        return len(items)


@register("queue", "redis")
class RedisQueue:
    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "aiready:q:"):
        import redis.asyncio as aioredis

        self._r = aioredis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix

    async def put(self, topic: str, payload: dict[str, Any]) -> Message:
        msg = Message(topic=topic, payload=payload)
        await self._r.lpush(self.prefix + topic, msg.model_dump_json())
        return msg

    async def get(self, topic: str, timeout: float | None = None) -> Message | None:
        if timeout:
            item = await self._r.brpoplpush(self.prefix + topic, self.prefix + topic + ":processing", int(timeout) or 1)
        else:
            item = await self._r.rpoplpush(self.prefix + topic, self.prefix + topic + ":processing")
        return Message.model_validate_json(item) if item else None

    async def ack(self, msg: Message) -> None:
        await self._r.lrem(self.prefix + msg.topic + ":processing", 1, self._find_raw(msg))

    def _find_raw(self, msg: Message) -> str:
        return msg.model_dump_json()

    async def nack(self, msg: Message, error: str, max_attempts: int = 3) -> bool:
        await self.ack(msg)
        msg.attempts += 1
        msg.last_error = error
        if msg.attempts >= max_attempts:
            await self._r.lpush(self.prefix + msg.topic + ":dlq", msg.model_dump_json())
            return True
        await self._r.lpush(self.prefix + msg.topic, msg.model_dump_json())
        return False

    async def depth(self, topic: str) -> int:
        return int(await self._r.llen(self.prefix + topic))

    async def dead_letters(self, topic: str) -> list[Message]:
        raw = await self._r.lrange(self.prefix + topic + ":dlq", 0, -1)
        return [Message.model_validate_json(r) for r in raw]

    async def replay_dead_letters(self, topic: str) -> int:
        n = 0
        while True:
            item = await self._r.rpop(self.prefix + topic + ":dlq")
            if not item:
                return n
            data = json.loads(item)
            data["attempts"] = 0
            await self._r.lpush(self.prefix + topic, json.dumps(data))
            n += 1


def make_queue(spec: dict[str, Any] | str | None) -> Queue:
    return create("queue", spec or {"type": "memory"})
