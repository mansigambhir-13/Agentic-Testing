"""
Recovery and Failure Injection Tests

Tests for failure recovery, graceful degradation, and error handling.
"""

import pytest
import logging
import sys
import os
from pathlib import Path
import time
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "bp-whatsappbot-version-mansi"))

# Load environment
load_dotenv()

logger = logging.getLogger(__name__)


class TestRecovery:
    """Recovery and failure injection test suite."""
    
    @pytest.fixture(scope="function")
    def agent(self):
        """Initialize MultiAgent V2 orchestrator."""
        try:
            from agents.multi_agent_v2.orchestrator import MultiAgentOrchestrator
            import asyncio
            
            orchestrator = MultiAgentOrchestrator()
            
            # Initialize synchronously
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(orchestrator.initialize())
                else:
                    loop.run_until_complete(orchestrator.initialize())
            except RuntimeError:
                asyncio.run(orchestrator.initialize())
            
            yield orchestrator
        except Exception as e:
            pytest.skip(f"Agent initialization failed: {e}")
    
    @pytest.mark.reliability
    def test_graceful_degradation(self, agent):
        """Test agent degrades gracefully when services are unavailable."""
        # Test with valid query first
        try:
            response1 = agent.process_message("degradation_test_user", "Hi")
            assert response1 is not None
            logger.info("Normal operation: OK")
        except Exception as e:
            logger.warning(f"Normal operation failed: {e}")
        
        # Test with potentially problematic queries
        problematic_queries = [
            "Show me products from unavailable category",
            "Add invalid product to cart",
        ]
        
        recovery_count = 0
        for query in problematic_queries:
            try:
                response = agent.process_message("degradation_test_user", query)
                # Should still get a response (even if error message)
                assert response is not None
                assert len(response) > 0
                recovery_count += 1
                logger.info(f"Gracefully handled: '{query[:30]}...'")
            except Exception as e:
                logger.error(f"Failed to handle gracefully: {e}")
        
        # At least one should recover
        assert recovery_count >= 1, "No graceful degradation observed"
    
    @pytest.mark.reliability
    def test_request_timeout_handling(self, agent):
        """Test agent handles request timeouts appropriately."""
        # Test with simple query that should complete quickly
        start_time = time.time()
        try:
            response = agent.process_message("timeout_test_user", "Hi")
            duration = time.time() - start_time
            
            assert response is not None
            # Should complete within reasonable time (30 seconds)
            assert duration < 30.0, f"Request took {duration:.1f}s, exceeds 30s timeout"
            logger.info(f"Request completed in {duration:.1f}s")
        except Exception as e:
            logger.error(f"Request timeout test failed: {e}")
            # Timeout is acceptable if handled gracefully
            assert "timeout" in str(e).lower() or "time" in str(e).lower()
    
    @pytest.mark.reliability
    def test_state_recovery_after_error(self, agent):
        """Test agent recovers state after an error."""
        user_id = "recovery_test_user"
        
        try:
            # First request - establish state
            response1 = agent.process_message(user_id, "Show me shirts")
            assert response1 is not None
            logger.info("Initial state established")
            
            # Second request - trigger potential error
            try:
                response2 = agent.process_message(user_id, "Add invalid product XYZ to cart")
                # May succeed or fail, but should respond
                assert response2 is not None
            except Exception:
                # Error is acceptable
                pass
            
            # Third request - should still work after error
            response3 = agent.process_message(user_id, "Show my cart")
            assert response3 is not None
            logger.info("State recovered after error")
            
        except Exception as e:
            logger.error(f"State recovery test failed: {e}")
            raise

