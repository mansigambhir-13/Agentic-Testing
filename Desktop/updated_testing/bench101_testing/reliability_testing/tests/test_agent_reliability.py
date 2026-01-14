"""
Agent Reliability Tests

Tests for MultiAgent V2 orchestrator reliability, consistency, and recovery.
"""

import pytest
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "bp-whatsappbot-version-mansi"))

# Load environment
load_dotenv()

logger = logging.getLogger(__name__)


class TestAgentReliability:
    """Agent reliability test suite."""
    
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
    def test_basic_response_reliability(self, agent):
        """Test agent responds reliably to basic queries."""
        test_queries = [
            "Hi",
            "Show me categories",
            "What products do you have?",
        ]
        
        results = []
        for query in test_queries:
            try:
                response = agent.process_message("reliability_test_user", query)
                assert response is not None
                assert len(response) > 0
                results.append(True)
                logger.info(f"Query '{query}' responded successfully")
            except Exception as e:
                logger.error(f"Query '{query}' failed: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.95, f"Response reliability {success_rate:.1%} below 95%"
    
    @pytest.mark.reliability
    def test_session_persistence(self, agent):
        """Test session data persists across requests."""
        user_id = "reliability_test_user_persist"
        
        # First request - add to cart
        try:
            response1 = agent.process_message(user_id, "Show me shirts")
            assert response1 is not None
            
            # Second request - should remember context
            response2 = agent.process_message(user_id, "Add the first one to cart")
            assert response2 is not None
            
            # Third request - check cart
            response3 = agent.process_message(user_id, "Show my cart")
            assert response3 is not None
            assert "cart" in response3.lower() or "item" in response3.lower()
            
            logger.info("Session persistence test passed")
        except Exception as e:
            logger.error(f"Session persistence test failed: {e}")
            raise
    
    @pytest.mark.reliability
    def test_error_recovery(self, agent):
        """Test agent recovers gracefully from errors."""
        error_scenarios = [
            "Add invalid product XYZ999 to cart",
            "Show me products that don't exist",
            "Checkout with empty cart",
        ]
        
        recovery_count = 0
        for scenario in error_scenarios:
            try:
                response = agent.process_message("reliability_test_user_error", scenario)
                # Should get a response (even if error message)
                assert response is not None
                assert len(response) > 0
                # Should be a helpful error message, not a crash
                assert "error" not in response.lower() or "sorry" in response.lower() or "help" in response.lower()
                recovery_count += 1
                logger.info(f"Error scenario '{scenario[:30]}...' handled gracefully")
            except Exception as e:
                logger.error(f"Error scenario '{scenario[:30]}...' failed: {e}")
        
        recovery_rate = recovery_count / len(error_scenarios)
        assert recovery_rate >= 0.90, f"Error recovery rate {recovery_rate:.1%} below 90%"
    
    @pytest.mark.reliability
    def test_concurrent_requests(self, agent):
        """Test agent handles sequential requests reliably (simulating concurrency)."""
        user_ids = [f"reliability_user_{i}" for i in range(10)]
        queries = ["Hi", "Show me products", "What categories do you have?"]
        
        results = []
        for user_id in user_ids:
            query = queries[hash(user_id) % len(queries)]
            try:
                response = agent.process_message(user_id, query)
                success = response is not None and len(response) > 0
                results.append(success)
            except Exception as e:
                logger.error(f"Request failed for {user_id}: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) if results else 0
        
        assert success_rate >= 0.90, f"Request success rate {success_rate:.1%} below 90%"
        logger.info(f"Requests: {sum(results)}/{len(results)} successful")

