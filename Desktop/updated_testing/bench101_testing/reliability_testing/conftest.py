"""
Pytest configuration for reliability tests.
"""

import pytest
import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "bp-whatsappbot-version-mansi"))

# Import orchestrator
try:
    from agents.multi_agent_v2.orchestrator import MultiAgentOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    ORCHESTRATOR_AVAILABLE = False
    logger.warning(f"Could not import MultiAgentOrchestrator: {e}")


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "reliability: marks tests as reliability tests"
    )
    config.addinivalue_line(
        "markers", "long_running: marks tests as long-running tests"
    )
    config.addinivalue_line(
        "markers", "chaos: marks tests as chaos engineering tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async tests"
    )

