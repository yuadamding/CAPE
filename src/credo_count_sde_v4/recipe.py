"""Discovery metadata for the future stable bridge.

Development `4.0.dev29` intentionally does not register this object through an
entry point. The standalone v4 CLI is the complete executor and loader.
"""

from dataclasses import dataclass

from .version import RECIPE_ID, RECIPE_VERSION


@dataclass(frozen=True)
class RecipeDescriptor:
    recipe_id: str = RECIPE_ID
    recipe_version: str = RECIPE_VERSION
    executor: str = "credo-v4"
    loader: str = "credo-v4 open-run"
    discovery_only: bool = True


recipe = RecipeDescriptor()
