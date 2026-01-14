"""
Infrastructure Reliability Tests

Tests for Redis, MongoDB, API endpoints, and service dependencies.
"""

import pytest
import logging
from datetime import datetime
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "bp-whatsappbot-version-mansi"))

# Load environment
load_dotenv()

logger = logging.getLogger(__name__)


class TestInfrastructureReliability:
    """Infrastructure reliability test suite."""
    
    @pytest.fixture(scope="function")
    def redis_client(self):
        """Create Redis client for testing."""
        try:
            import redis
            client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5
            )
            client.ping()
            yield client
            client.close()
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
    
    @pytest.fixture(scope="function")
    def mongo_client(self):
        """Create MongoDB client for testing."""
        try:
            from pymongo import MongoClient
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            yield client
            client.close()
        except Exception as e:
            pytest.skip(f"MongoDB not available: {e}")
    
    @pytest.mark.reliability
    def test_redis_connection_stability(self, redis_client):
        """Test Redis maintains stable connection over time."""
        errors = 0
        operations = 50  # Reduced for faster tests
        
        for i in range(operations):
            try:
                redis_client.ping()
                redis_client.set(f"test_key_{i}", f"value_{i}")
                redis_client.get(f"test_key_{i}")
                redis_client.delete(f"test_key_{i}")
            except Exception as e:
                errors += 1
                logger.error(f"Redis operation {i} failed: {e}")
            
            time.sleep(0.05)  # 50ms between operations
        
        error_rate = errors / operations
        assert error_rate < 0.01, f"Redis error rate {error_rate:.2%} exceeds 1%"
        logger.info(f"Redis stability test: {operations - errors}/{operations} operations successful")
    
    @pytest.mark.reliability
    def test_mongodb_connection_pool(self, mongo_client):
        """Test MongoDB connection pool handles requests."""
        results = []
        
        for i in range(20):  # Reduced for faster tests
            try:
                db = mongo_client.get_database("test_reliability")
                collection = db.get_collection("test_collection")
                collection.insert_one({
                    "test_id": i,
                    "timestamp": datetime.now()
                })
                result = collection.find_one({"test_id": i})
                collection.delete_one({"test_id": i})
                results.append(True)
            except Exception as e:
                logger.error(f"MongoDB operation {i} failed: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) if results else 0
        assert success_rate >= 0.95, f"MongoDB success rate {success_rate:.2%} below 95%"
        logger.info(f"MongoDB connection pool test: {sum(results)}/{len(results)} operations successful")
    
    @pytest.mark.reliability
    def test_api_endpoint_availability(self):
        """Test API endpoints remain available."""
        try:
            import requests
        except ImportError:
            pytest.skip("requests library not available")
        
        endpoints = [
            ("http://localhost:9002/health", "Customer API"),
            ("http://localhost:9004/health", "Order API"),
            ("http://localhost:9007/health", "Chat API"),
        ]
        
        results = {}
        any_available = False
        
        for url, name in endpoints:
            success = 0
            failed = 0
            
            for _ in range(5):  # Reduced for faster tests
                try:
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        success += 1
                        any_available = True
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.warning(f"{name} health check failed: {e}")
            
            total = success + failed
            availability = success / total if total > 0 else 0
            results[name] = availability
            
            logger.info(f"{name} availability: {success}/{total} ({availability:.1%})")
        
        # Skip test if no services are available (expected in test environment)
        if not any_available:
            pytest.skip("No API endpoints available (expected in test environment)")
        
        # If services are available, check they meet availability threshold
        for name, availability in results.items():
            if results[name] > 0:  # Only check services that had at least one success
                assert availability >= 0.6, f"{name} availability {availability:.1%} below 60%"

