from __future__ import annotations

from oarp.plugins.base import TopicPlugin
from oarp.topic_spec import TopicSpec


class NiSilicideTopicPlugin:
    plugin_id = "materials_ni_silicide"

    _entity_terms = ("ni2si", "nisi", "nisi2", "nickel silicide", "ni silicide")
    _negative_terms = (
        "hfo2",
        "hafnia",
        "ferroelectric",
        "zro2",
        "zr-doped",
        "undoped-hfo2",
    )

    def expand_query(self, spec: TopicSpec, query: str) -> list[str]:
        base = query.strip()
        extras = [
            "nickel silicide phase formation thin film annealing",
            "Ni2Si NiSi NiSi2 thickness temperature",
            "nickel film silicon reaction temperature phase sequence",
            "Ni silicide transformation temperature thickness dependence",
        ]
        if spec.keywords:
            extras.append(" ".join(spec.keywords))
        out: list[str] = []
        for item in [base] + extras:
            clean = " ".join(item.split())
            if clean and clean not in out:
                out.append(clean)
        return out

    def candidate_boost(self, article_row: dict, spec: TopicSpec) -> float:
        title = str(article_row.get("title") or "").lower()
        abstract = str(article_row.get("abstract") or "").lower()
        text = f"{title} {abstract}"
        boost = 0.0

        entity_hits = sum(1 for token in self._entity_terms if token in text)
        boost += min(entity_hits * 0.12, 0.45)

        if "thickness" in text:
            boost += 0.08
        if "anneal" in text or "annealing" in text or "temperature" in text:
            boost += 0.08

        has_negative = any(term in text for term in self._negative_terms)
        has_positive = any(term in text for term in self._entity_terms)
        if has_negative and not has_positive:
            boost -= 0.28

        return boost

    def entity_alias_overrides(self, spec: TopicSpec) -> dict[str, str]:  # noqa: ARG002
        return {
            "ni₂si": "Ni2Si",
            "ni si": "NiSi",
            "ni-si": "NiSi",
            "ni si2": "NiSi2",
            "ni-si2": "NiSi2",
            "nickel monosilicide": "NiSi",
            "nickel disilicide": "NiSi2",
        }


def build_plugin() -> TopicPlugin:
    return NiSilicideTopicPlugin()
