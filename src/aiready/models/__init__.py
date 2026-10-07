"""Model serving layer: backends (OpenAI-compatible, Anthropic, mock), pools with an inference router, and the
logical model catalog. Only the gateway calls this package; everything else goes through ``connect("gateway")``.
"""

from aiready.models.catalog import (
    Category,
    LiveCatalog,
    ModelCatalog,
    ModelsConfig,
    ModelSpec,
    PoolSpec,
    Tier,
)

__all__ = ["Category", "LiveCatalog", "ModelCatalog", "ModelSpec", "ModelsConfig", "PoolSpec", "Tier"]
