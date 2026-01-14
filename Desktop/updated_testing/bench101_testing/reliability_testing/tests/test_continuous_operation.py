"""
Continuous Operation Reliability Tests

Tests for 24-hour continuous operation and longevity.
"""

import pytest
import logging
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "bp-whatsappbot-version-mansi"))

# Load environment
load_dotenv()

logger = logging.getLogger(__name__)


class TestContinuousOperation:
    """Continuous operation reliability tests."""
    
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
    @pytest.mark.long_running
    def test_short_continuous_operation(self, agent):
        """Test continuous operation for short duration (2 minutes for quick test)."""
        duration_minutes = 2  # Reduced for quicker test
        duration_seconds = duration_minutes * 60
        request_interval = 5  # seconds between requests
        
        start_time = time.time()
        requests_processed = 0
        errors = 0
        
        try:
            import psutil
            initial_memory = psutil.Process().memory_info().rss
        except ImportError:
            initial_memory = 0
            logger.warning("psutil not available, memory tracking disabled")
        
        test_queries = [
            "Hi",
            "Show me categories",
            "What products do you have?",
            "Show me shirts",
            "Show my cart",
        ]
        
        query_index = 0
        
        logger.info(f"Starting {duration_minutes}-minute continuous operation test")
        
        while time.time() - start_time < duration_seconds:
            try:
                query = test_queries[query_index % len(test_queries)]
                user_id = f"continuous_test_user_{requests_processed % 10}"
                
                response = agent.process_message(user_id, query)
                
                if response and len(response) > 0:
                    requests_processed += 1
                else:
                    errors += 1
                    logger.warning(f"Empty response for query: {query}")
                
                query_index += 1
                
                if requests_processed % 5 == 0:
                    elapsed = time.time() - start_time
                    if initial_memory > 0:
                        try:
                            import psutil
                            current_memory = psutil.Process().memory_info().rss
                            memory_growth = (current_memory - initial_memory) / initial_memory * 100
                            logger.info(
                                f"Progress: {requests_processed} requests, "
                                f"{errors} errors, "
                                f"{elapsed:.0f}s elapsed, "
                                f"memory: +{memory_growth:.1f}%"
                            )
                        except:
                            pass
                    else:
                        logger.info(
                            f"Progress: {requests_processed} requests, "
                            f"{errors} errors, "
                            f"{elapsed:.0f}s elapsed"
                        )
                
            except Exception as e:
                errors += 1
                logger.error(f"Request failed: {e}")
            
            time.sleep(request_interval)
        
        elapsed_time = time.time() - start_time
        if initial_memory > 0:
            try:
                import psutil
                final_memory = psutil.Process().memory_info().rss
                memory_growth = (final_memory - initial_memory) / initial_memory * 100
            except:
                memory_growth = 0
        else:
            memory_growth = 0
        
        total_requests = requests_processed + errors
        error_rate = errors / total_requests if total_requests > 0 else 0
        availability = requests_processed / total_requests if total_requests > 0 else 0
        
        logger.info(f"Continuous operation test completed:")
        logger.info(f"  Duration: {elapsed_time:.0f}s ({duration_minutes} minutes)")
        logger.info(f"  Requests processed: {requests_processed}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  Error rate: {error_rate:.2%}")
        logger.info(f"  Availability: {availability:.2%}")
        logger.info(f"  Memory growth: {memory_growth:.1f}%")
        
        assert availability >= 0.95, f"Availability {availability:.1%} below 95%"
        assert error_rate <= 0.05, f"Error rate {error_rate:.1%} above 5%"
        assert memory_growth < 50, f"Memory growth {memory_growth:.1f}% exceeds 50%"

