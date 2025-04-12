"""
HMAS (Hierarchical Multi-Agent System) with DREAMv1 AGI
Dynamic Reasoning and Evolving Autonomous Mind

A sophisticated artificial general intelligence framework implementing
a hierarchical multi-agent system for complex problem-solving.
"""

from shared.branding import (
    PROJECT_NAME,
    PROJECT_FULL_NAME,
    AGI_NAME,
    AGI_FULL_NAME,
    VERSION,
    BUILD,
    COPYRIGHT
)

__version__ = VERSION
__build__ = BUILD
__author__ = "HMAS Team"
__copyright__ = COPYRIGHT

# Package exports
from hmas.core import Agent, Environment
from hmas.config import Settings

__all__ = [
    "Agent",
    "Environment",
    "Settings",
    "__version__",
    "__build__",
    "__author__",
    "__copyright__"
] 