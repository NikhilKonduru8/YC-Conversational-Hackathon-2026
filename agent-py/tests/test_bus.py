"""Unit tests for the ROS-style in-process message bus."""

import pytest

from bus import Bus


async def test_publish_fans_out_to_all_subscribers():
    bus = Bus()
    seen: list[str] = []
    bus.subscribe("/t", lambda m: seen.append(f"a:{m}"))

    async def async_sub(m):
        seen.append(f"b:{m}")

    bus.subscribe("/t", async_sub)
    await bus.publish("/t", "hi")
    assert seen == ["a:hi", "b:hi"]


async def test_publish_to_topic_with_no_subscribers_is_noop():
    await Bus().publish("/nobody", "x")  # must not raise


async def test_subscriber_exception_does_not_break_others():
    bus = Bus()
    seen: list[str] = []

    def boom(_m):
        raise RuntimeError("bad subscriber")

    bus.subscribe("/t", boom)
    bus.subscribe("/t", lambda m: seen.append(m))
    await bus.publish("/t", "ok")
    assert seen == ["ok"]


async def test_service_call_returns_handler_result():
    bus = Bus()

    async def handler(req):
        return req * 2

    bus.register_service("double", handler)
    assert await bus.call("double", 21) == 42


async def test_sync_service_handler_supported():
    bus = Bus()
    bus.register_service("inc", lambda n: n + 1)
    assert await bus.call("inc", 1) == 2


async def test_calling_unknown_service_raises():
    with pytest.raises(KeyError):
        await Bus().call("missing")
