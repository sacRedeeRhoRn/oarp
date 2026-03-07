from __future__ import annotations

from oarp.plugins.base import TopicPlugin
from oarp.plugins.topic_plugins.materials_ni_silicide import (
    build_plugin as build_ni_silicide_plugin,
)


def load_topic_plugin(plugin_id: str) -> TopicPlugin | None:
    clean = str(plugin_id or "").strip().lower()
    if not clean:
        return None
    if clean in {"materials_ni_silicide", "ni_silicide", "ni-silicide"}:
        return build_ni_silicide_plugin()
    return None


__all__ = ["load_topic_plugin"]
