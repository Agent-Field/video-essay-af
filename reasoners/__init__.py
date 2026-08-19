"""Routers for video-essay-af — one per phase, plus the research tools."""

from .editorial import router as editorial_router
from .production import router as production_router
from .research import router as research_router
from .script import router as script_router
from .tools import router as tools_router

__all__ = [
    "tools_router",
    "research_router",
    "script_router",
    "editorial_router",
    "production_router",
]
