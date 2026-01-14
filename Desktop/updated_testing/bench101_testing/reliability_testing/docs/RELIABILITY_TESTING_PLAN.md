# Reliability Testing Plan
## MultiAgent V2 - E-Commerce Agent

**Version**: 1.0  
**Date**: December 2025  
**System**: MultiAgent V2 (Strands Framework)  
**Testing Framework**: Empty Bench 101 + Custom Reliability Suite

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Objectives & Success Criteria](#2-objectives--success-criteria)
3. [Reliability Metrics](#3-reliability-metrics)
4. [Test Categories](#4-test-categories)
5. [Test Scenarios](#5-test-scenarios)
6. [Infrastructure Reliability](#6-infrastructure-reliability)
7. [Agent Behavior Reliability](#7-agent-behavior-reliability)
8. [Failure Injection Testing](#8-failure-injection-testing)
9. [Long-Running Tests](#9-long-running-tests)
10. [Monitoring & Alerting](#10-monitoring--alerting)
11. [Test Schedule](#11-test-schedule)
12. [Tools & Resources](#12-tools--resources)
13. [Reporting](#13-reporting)

---

## 1. Executive Summary

### Purpose
This reliability testing plan ensures MultiAgent V2 can operate consistently, recover gracefully from failures, and maintain performance over extended periods in production environments.

### Scope
| Component | Covered |
|-----------|---------|
| Chat Agent (Emma) | ✅ |
| Customer API | ✅ |
| Order API | ✅ |
| Redis Session Management | ✅ |
| MongoDB Cart Storage | ✅ |
| PostgreSQL/Supabase | ✅ |
| WhatsApp Integration | ✅ |
| Tool Execution Pipeline | ✅ |

### Current Baseline (From Previous Testing)
- **Overall Success Rate**: 99.00%
- **Average Response Time**: 2.90s
- **Error Rate**: 0.99%
- **Multi-Turn Handling**: 6.48 avg turns

---

## 2. Objectives & Success Criteria

### Primary Objectives

| Objective | Target | Measurement |
|-----------|--------|-------------|
| System Availability | ≥ 99.5% | Uptime over 7-day period |
| MTBF (Mean Time Between Failures) | ≥ 24 hours | Continuous operation |
| MTTR (Mean Time To Recovery) | ≤ 30 seconds | Auto-recovery time |
| Response Consistency | ≥ 95% | Same query → similar response |
| Error Recovery Rate | 100% | Graceful handling of all errors |
| Data Integrity | 100% | No cart/session data loss |

### Success Criteria

```
✅ PASS if:
  - Availability ≥ 99.5% over 7 days
  - No critical failures during 24-hour continuous run
  - All failure scenarios recover within 30 seconds
  - Zero data loss in cart/session during failures
  - Response quality maintained after recovery

❌ FAIL if:
  - Any unrecoverable crash
  - Data corruption or loss
  - Availability drops below 99%
  - Recovery time exceeds 2 minutes
```

---

## 3. Reliability Metrics

### 3.1 Core Metrics

| Metric | Formula | Target | Priority |
|--------|---------|--------|----------|
| **Availability** | (Uptime / Total Time) × 100 | ≥ 99.5% | Critical |
| **MTBF** | Total Uptime / Number of Failures | ≥ 24 hrs | Critical |
| **MTTR** | Total Downtime / Number of Failures | ≤ 30 sec | Critical |
| **Failure Rate** | Failures / Total Requests | ≤ 1% | High |
| **Error Recovery Rate** | Recovered / Total Errors | 100% | High |

### 3.2 Agent-Specific Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Response Consistency Score** | Similarity of responses to identical queries | ≥ 0.85 |
| **Context Retention Rate** | Accurate context after N turns | ≥ 95% |
| **Tool Success Rate** | Successful tool executions | ≥ 98% |
| **Session Persistence Rate** | Sessions maintained across restarts | 100% |
| **Cart Data Integrity** | Cart data preserved during failures | 100% |

### 3.3 Performance Under Load

| Metric | Normal Load | Peak Load | Stress |
|--------|-------------|-----------|--------|
| **Response Time P50** | ≤ 1.5s | ≤ 2.5s | ≤ 4s |
| **Response Time P95** | ≤ 3s | ≤ 5s | ≤ 8s |
| **Throughput** | 100 req/min | 200 req/min | 300 req/min |
| **Error Rate** | ≤ 1% | ≤ 2% | ≤ 5% |

---

## 4. Test Categories

### 4.1 Category Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RELIABILITY TEST CATEGORIES                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  AVAILABILITY    │  │  RECOVERABILITY  │  │  CONSISTENCY │  │
│  │                  │  │                  │  │              │  │
│  │  • Uptime tests  │  │  • Failure       │  │  • Response  │  │
│  │  • Continuous    │  │    injection     │  │    stability │  │
│  │    operation     │  │  • Auto-recovery │  │  • Behavior  │  │
│  │  • Load endurance│  │  • Failover      │  │    patterns  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  DATA INTEGRITY  │  │  DEGRADATION     │  │  LONGEVITY   │  │
│  │                  │  │                  │  │              │  │
│  │  • Cart persist  │  │  • Graceful      │  │  • 24-hour   │  │
│  │  • Session state │  │    degradation   │  │    soak test │  │
│  │  • Transaction   │  │  • Partial       │  │  • Memory    │  │
│  │    safety        │  │    failure       │  │    leaks     │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Priority Matrix

| Category | Priority | Duration | Frequency |
|----------|----------|----------|-----------|
| Availability Testing | P0 - Critical | 24-168 hrs | Weekly |
| Failure Recovery | P0 - Critical | 4-8 hrs | Daily |
| Data Integrity | P0 - Critical | 2-4 hrs | Daily |
| Response Consistency | P1 - High | 2-4 hrs | Daily |
| Graceful Degradation | P1 - High | 4-8 hrs | Weekly |
| Longevity/Soak Testing | P2 - Medium | 24-72 hrs | Weekly |

---

## 5. Test Scenarios

### 5.1 Scenario Matrix

| ID | Scenario | Category | Duration | Automation |
|----|----------|----------|----------|------------|
| R-001 | 24-hour continuous operation | Availability | 24 hrs | ✅ |
| R-002 | Redis connection failure/recovery | Recovery | 30 min | ✅ |
| R-003 | MongoDB failover simulation | Recovery | 30 min | ✅ |
| R-004 | API endpoint timeout handling | Recovery | 1 hr | ✅ |
| R-005 | High load sustained operation | Degradation | 4 hrs | ✅ |
| R-006 | Response consistency validation | Consistency | 2 hrs | ✅ |
| R-007 | Cart data persistence during restart | Integrity | 1 hr | ✅ |
| R-008 | Session recovery after crash | Recovery | 1 hr | ✅ |
| R-009 | Concurrent user session handling | Availability | 2 hrs | ✅ |
| R-010 | Memory leak detection (72-hr soak) | Longevity | 72 hrs | ✅ |
| R-011 | Network partition simulation | Recovery | 1 hr | ✅ |
| R-012 | OpenAI API rate limit handling | Degradation | 1 hr | ✅ |
| R-013 | Graceful shutdown and restart | Availability | 30 min | ✅ |
| R-014 | Database connection pool exhaustion | Recovery | 1 hr | ✅ |
| R-015 | Tool execution timeout recovery | Recovery | 1 hr | ✅ |

### 5.2 Detailed Scenario Specifications

#### R-001: 24-Hour Continuous Operation

```yaml
scenario_id: R-001
name: "24-Hour Continuous Operation"
category: Availability
priority: P0

description: |
  Run the agent continuously for 24 hours with simulated 
  user traffic to verify stable operation.

preconditions:
  - All services healthy
  - Monitoring enabled
  - Logging active

test_parameters:
  duration: 24 hours
  request_rate: 50 requests/minute
  user_patterns:
    - new_users: 30%
    - returning_users: 50%
    - power_users: 20%
  conversation_types:
    - simple_queries: 40%
    - multi_turn: 35%
    - complex_workflows: 25%

success_criteria:
  - uptime: ">= 99.5%"
  - error_rate: "<= 1%"
  - response_time_p95: "<= 4s"
  - memory_growth: "<= 10%"
  - no_crashes: true

measurements:
  - requests_processed
  - errors_count
  - response_times
  - memory_usage
  - cpu_usage
  - active_sessions
```

#### R-002: Redis Connection Failure/Recovery

```yaml
scenario_id: R-002
name: "Redis Connection Failure/Recovery"
category: Recovery
priority: P0

description: |
  Simulate Redis connection failures and verify the agent 
  falls back to MongoDB and recovers when Redis is restored.

test_steps:
  1. Establish baseline with Redis active
  2. Process 10 conversations successfully
  3. Simulate Redis failure (stop container)
  4. Verify MongoDB fallback activates
  5. Continue processing (degraded mode)
  6. Restore Redis connection
  7. Verify automatic reconnection
  8. Confirm session data integrity

success_criteria:
  - fallback_time: "<= 5s"
  - recovery_time: "<= 30s"
  - data_loss: "0"
  - error_messages: "user_friendly"

measurements:
  - time_to_detect_failure
  - time_to_fallback
  - time_to_recovery
  - requests_during_failure
  - data_consistency_check
```

#### R-007: Cart Data Persistence During Restart

```yaml
scenario_id: R-007
name: "Cart Data Persistence During Restart"
category: Data Integrity
priority: P0

description: |
  Verify cart data is preserved during planned and 
  unplanned agent restarts.

test_steps:
  1. Create 10 user sessions with cart items
  2. Verify cart data in Redis
  3. Perform graceful shutdown
  4. Restart agent
  5. Query all user carts
  6. Verify data matches pre-restart state
  7. Repeat with forced termination (kill -9)
  8. Verify MongoDB sync preserved data

success_criteria:
  - cart_items_preserved: "100%"
  - cart_totals_accurate: "100%"
  - session_continuity: "100%"

test_data:
  users:
    - id: "test_user_001"
      cart_items: 3
      total: 2999.00
    - id: "test_user_002"
      cart_items: 5
      total: 4599.00
    # ... more test users
```

---

## 6. Infrastructure Reliability

### 6.1 Component Health Checks

```python
# tests/reliability/test_infrastructure.py

import pytest
import asyncio
from datetime import datetime, timedelta

class TestInfrastructureReliability:
    """Infrastructure reliability test suite."""
    
    @pytest.mark.reliability
    async def test_redis_connection_stability(self, redis_client):
        """Test Redis maintains stable connection over time."""
        errors = 0
        operations = 1000
        
        for i in range(operations):
            try:
                await redis_client.ping()
                await redis_client.set(f"test_key_{i}", f"value_{i}")
                await redis_client.get(f"test_key_{i}")
                await redis_client.delete(f"test_key_{i}")
            except Exception as e:
                errors += 1
            
            await asyncio.sleep(0.1)  # 100ms between operations
        
        error_rate = errors / operations
        assert error_rate < 0.01, f"Redis error rate {error_rate:.2%} exceeds 1%"
    
    @pytest.mark.reliability
    async def test_mongodb_connection_pool(self, mongo_client):
        """Test MongoDB connection pool handles concurrent requests."""
        async def mongo_operation(i):
            try:
                await mongo_client.test_db.test_collection.insert_one(
                    {"test_id": i, "timestamp": datetime.now()}
                )
                await mongo_client.test_db.test_collection.find_one(
                    {"test_id": i}
                )
                return True
            except Exception:
                return False
        
        # Run 100 concurrent operations
        results = await asyncio.gather(*[
            mongo_operation(i) for i in range(100)
        ])
        
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.99, f"MongoDB success rate {success_rate:.2%}"
    
    @pytest.mark.reliability
    async def test_api_endpoint_availability(self, api_client):
        """Test API endpoints remain available under load."""
        endpoints = [
            "/api/health",
            "/api/products",
            "/api/cart",
        ]
        
        results = {endpoint: {"success": 0, "failed": 0} for endpoint in endpoints}
        
        for _ in range(100):
            for endpoint in endpoints:
                try:
                    response = await api_client.get(endpoint)
                    if response.status_code == 200:
                        results[endpoint]["success"] += 1
                    else:
                        results[endpoint]["failed"] += 1
                except Exception:
                    results[endpoint]["failed"] += 1
        
        for endpoint, stats in results.items():
            total = stats["success"] + stats["failed"]
            availability = stats["success"] / total
            assert availability >= 0.99, \
                f"{endpoint} availability {availability:.2%} below 99%"
```

### 6.2 Service Dependency Map

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVICE DEPENDENCIES                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────────────┐                      │
│                    │   MultiAgent    │                      │
│                    │       V2        │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│         ┌───────────────────┼───────────────────┐          │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │    Redis    │    │   OpenAI    │    │  WhatsApp   │    │
│  │  (Primary)  │    │     API     │    │    API      │    │
│  └──────┬──────┘    └─────────────┘    └─────────────┘    │
│         │                                                   │
│         ▼ (Fallback)                                       │
│  ┌─────────────┐                                           │
│  │   MongoDB   │                                           │
│  │  (Backup)   │                                           │
│  └─────────────┘                                           │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ Customer API│    │  Order API  │    │  Chat API   │    │
│  │   :9002     │    │    :9004    │    │    :9007    │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Dependency Criticality:
  🔴 Critical (no fallback): OpenAI API, Customer API
  🟡 Important (has fallback): Redis → MongoDB
  🟢 Optional: WhatsApp (can queue messages)
```

### 6.3 Health Check Endpoints

| Service | Endpoint | Expected Response | Timeout |
|---------|----------|-------------------|---------|
| Agent | `/health` | `{"status": "healthy"}` | 5s |
| Customer API | `:9002/health` | `200 OK` | 3s |
| Order API | `:9004/health` | `200 OK` | 3s |
| Chat API | `:9007/health` | `200 OK` | 3s |
| Redis | `PING` | `PONG` | 1s |
| MongoDB | `db.adminCommand('ping')` | `{"ok": 1}` | 2s |

---

## 7. Agent Behavior Reliability

### 7.1 Response Consistency Testing

```python
# tests/reliability/test_consistency.py

import pytest
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class TestResponseConsistency:
    """Test agent response consistency."""
    
    @pytest.fixture
    def embedding_model(self):
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    @pytest.mark.reliability
    async def test_identical_query_consistency(
        self, agent, embedding_model
    ):
        """Same query should produce semantically similar responses."""
        query = "Show me red shirts under 1000 rupees"
        responses = []
        
        # Run same query 10 times
        for _ in range(10):
            response = await agent.chat(query)
            responses.append(response)
        
        # Calculate pairwise similarity
        embeddings = embedding_model.encode(responses)
        similarities = cosine_similarity(embeddings)
        
        # Average similarity (excluding diagonal)
        n = len(responses)
        avg_similarity = (similarities.sum() - n) / (n * (n - 1))
        
        assert avg_similarity >= 0.85, \
            f"Response consistency {avg_similarity:.2f} below 0.85 threshold"
    
    @pytest.mark.reliability
    async def test_context_retention_reliability(self, agent):
        """Context should be retained reliably across turns."""
        test_flows = [
            [
                ("Show me blue dresses", "dress"),
                ("What sizes are available?", "size"),
                ("Add medium to cart", "cart"),
            ],
            [
                ("I want running shoes", "shoe"),
                ("Show me Nike options", "nike"),
                ("What's the price of the first one?", "price"),
            ],
        ]
        
        success_count = 0
        total_checks = 0
        
        for flow in test_flows:
            for query, expected_keyword in flow:
                response = await agent.chat(query)
                total_checks += 1
                if expected_keyword.lower() in response.lower():
                    success_count += 1
        
        retention_rate = success_count / total_checks
        assert retention_rate >= 0.95, \
            f"Context retention {retention_rate:.2%} below 95%"
    
    @pytest.mark.reliability
    async def test_error_message_consistency(self, agent):
        """Error scenarios should produce consistent, helpful messages."""
        error_scenarios = [
            "Add product xyz123 to cart",  # Invalid product
            "What's the weather today?",    # Off-topic
            "Checkout now",                  # Empty cart
        ]
        
        for scenario in error_scenarios:
            responses = []
            for _ in range(5):
                response = await agent.chat(scenario)
                responses.append(response)
            
            # All responses should be helpful (not crash messages)
            for response in responses:
                assert "error" not in response.lower() or \
                       "sorry" in response.lower() or \
                       "help" in response.lower(), \
                       f"Unhelpful error response: {response[:100]}"
```

### 7.2 Tool Execution Reliability

```python
# tests/reliability/test_tools.py

import pytest
import asyncio
from collections import defaultdict

class TestToolReliability:
    """Test tool execution reliability."""
    
    TOOLS = [
        "search_products",
        "add_to_cart",
        "view_cart",
        "remove_from_cart",
        "get_recommendations",
        "check_order_status",
        "virtual_tryon",
    ]
    
    @pytest.mark.reliability
    async def test_tool_success_rate(self, agent):
        """Each tool should have >= 98% success rate."""
        tool_stats = defaultdict(lambda: {"success": 0, "failed": 0})
        
        test_queries = {
            "search_products": [
                "Show me shirts",
                "Find blue jeans",
                "Search for dresses",
            ],
            "add_to_cart": [
                "Add the first product to cart",
                "Put item 2 in my cart",
            ],
            "view_cart": [
                "Show my cart",
                "What's in my cart?",
            ],
            # ... more test queries per tool
        }
        
        for tool, queries in test_queries.items():
            for query in queries:
                for _ in range(10):  # Run each 10 times
                    try:
                        response = await agent.chat(query)
                        if "error" not in response.lower():
                            tool_stats[tool]["success"] += 1
                        else:
                            tool_stats[tool]["failed"] += 1
                    except Exception:
                        tool_stats[tool]["failed"] += 1
        
        for tool, stats in tool_stats.items():
            total = stats["success"] + stats["failed"]
            if total > 0:
                success_rate = stats["success"] / total
                assert success_rate >= 0.98, \
                    f"Tool {tool} success rate {success_rate:.2%} below 98%"
    
    @pytest.mark.reliability
    async def test_tool_timeout_handling(self, agent):
        """Tools should handle timeouts gracefully."""
        # Simulate slow responses
        slow_queries = [
            "Search for products with very specific criteria that might take long",
            "Get recommendations based on my entire purchase history",
        ]
        
        for query in slow_queries:
            try:
                response = await asyncio.wait_for(
                    agent.chat(query),
                    timeout=30.0
                )
                # Should get a response (even if timeout message)
                assert response is not None
                assert len(response) > 0
            except asyncio.TimeoutError:
                pytest.fail(f"Query timed out: {query[:50]}...")
```

---

## 8. Failure Injection Testing

### 8.1 Chaos Engineering Scenarios

```python
# tests/reliability/test_chaos.py

import pytest
import asyncio
import docker

class TestChaosEngineering:
    """Chaos engineering tests for failure injection."""
    
    @pytest.fixture
    def docker_client(self):
        return docker.from_env()
    
    @pytest.mark.chaos
    async def test_redis_failure_recovery(
        self, agent, docker_client, redis_container
    ):
        """Agent should recover from Redis failure."""
        # Establish baseline
        response1 = await agent.chat("Show me shirts")
        assert "shirt" in response1.lower()
        
        # Kill Redis
        redis_container.stop()
        await asyncio.sleep(2)
        
        # Agent should use MongoDB fallback
        response2 = await agent.chat("Add first to cart")
        assert "cart" in response2.lower() or "added" in response2.lower()
        
        # Restore Redis
        redis_container.start()
        await asyncio.sleep(5)
        
        # Agent should reconnect
        response3 = await agent.chat("Show my cart")
        assert "cart" in response3.lower()
    
    @pytest.mark.chaos
    async def test_network_partition(self, agent):
        """Agent should handle network partitions gracefully."""
        import subprocess
        
        # Simulate network latency
        subprocess.run([
            "tc", "qdisc", "add", "dev", "eth0", 
            "root", "netem", "delay", "500ms"
        ])
        
        try:
            start = asyncio.get_event_loop().time()
            response = await asyncio.wait_for(
                agent.chat("Show me products"),
                timeout=30.0
            )
            duration = asyncio.get_event_loop().time() - start
            
            assert response is not None
            # Response time should account for latency
            assert duration < 30.0
        finally:
            # Remove latency
            subprocess.run([
                "tc", "qdisc", "del", "dev", "eth0", "root"
            ])
    
    @pytest.mark.chaos
    async def test_memory_pressure(self, agent):
        """Agent should handle memory pressure gracefully."""
        import resource
        
        # Limit memory
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, hard))
        
        try:
            # Run multiple queries
            for i in range(20):
                response = await agent.chat(f"Search for product {i}")
                assert response is not None
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
```

### 8.2 Failure Injection Matrix

| Failure Type | Injection Method | Expected Behavior | Recovery Time |
|--------------|------------------|-------------------|---------------|
| Redis Down | Stop container | MongoDB fallback | < 5s |
| MongoDB Down | Stop container | Error message, no data loss | < 10s |
| API Timeout | Network delay | Timeout message | < 30s |
| High CPU | Stress test | Degraded performance | Immediate |
| Memory Pressure | Limit allocation | Graceful degradation | Immediate |
| Network Partition | iptables rules | Queued retries | < 60s |
| OpenAI Rate Limit | Mock 429 response | Backoff & retry | < 30s |
| Disk Full | Fill volume | Error logging, no crash | Immediate |

---

## 9. Long-Running Tests

### 9.1 Soak Testing Configuration

```yaml
# config/soak_test.yaml
soak_test:
  name: "MultiAgent V2 - 72 Hour Soak Test"
  duration: 72h
  
  traffic_pattern:
    base_rate: 30  # requests per minute
    peak_hours: [10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
    peak_multiplier: 2.5
    night_multiplier: 0.3
  
  user_simulation:
    new_users_ratio: 0.3
    returning_users_ratio: 0.5
    power_users_ratio: 0.2
    
    conversation_patterns:
      simple_query: 0.4
      multi_turn: 0.35
      complex_workflow: 0.25
  
  monitoring:
    interval: 60s  # Check every minute
    metrics:
      - response_time
      - error_rate
      - memory_usage
      - cpu_usage
      - active_connections
      - cache_hit_rate
    
    alerts:
      response_time_p95:
        threshold: 5000ms
        severity: warning
      error_rate:
        threshold: 2%
        severity: critical
      memory_usage:
        threshold: 85%
        severity: warning
  
  success_criteria:
    availability: ">= 99.5%"
    error_rate: "<= 1%"
    memory_leak: "< 5% growth per 24h"
    response_time_degradation: "< 10%"
```

### 9.2 Soak Test Runner

```python
# tests/reliability/soak_test.py

import asyncio
import time
from datetime import datetime, timedelta
import random
import psutil
import logging

class SoakTestRunner:
    """Run extended soak tests for reliability validation."""
    
    def __init__(self, agent, duration_hours: int = 72):
        self.agent = agent
        self.duration = timedelta(hours=duration_hours)
        self.start_time = None
        self.metrics = []
        self.errors = []
        self.logger = logging.getLogger("soak_test")
    
    async def run(self):
        """Execute the soak test."""
        self.start_time = datetime.now()
        end_time = self.start_time + self.duration
        
        self.logger.info(f"Starting {self.duration.total_seconds()/3600}h soak test")
        
        initial_memory = psutil.Process().memory_info().rss
        
        while datetime.now() < end_time:
            try:
                # Simulate realistic traffic
                await self._simulate_traffic_burst()
                
                # Collect metrics
                metrics = await self._collect_metrics()
                self.metrics.append(metrics)
                
                # Check for anomalies
                await self._check_health(metrics, initial_memory)
                
                # Wait before next burst
                await asyncio.sleep(60)
                
            except Exception as e:
                self.errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                })
                self.logger.error(f"Soak test error: {e}")
        
        return self._generate_report(initial_memory)
    
    async def _simulate_traffic_burst(self):
        """Simulate a burst of user traffic."""
        hour = datetime.now().hour
        
        # Adjust rate based on time of day
        if 10 <= hour <= 20:
            requests = random.randint(50, 100)
        else:
            requests = random.randint(10, 30)
        
        queries = [
            "Show me shirts",
            "Find blue jeans",
            "Add to cart",
            "Show my cart",
            "What do you recommend?",
            "Track my order",
        ]
        
        tasks = []
        for _ in range(requests):
            query = random.choice(queries)
            tasks.append(self._timed_query(query))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        errors = sum(1 for r in results if isinstance(r, Exception))
        if errors > 0:
            self.logger.warning(f"Burst had {errors}/{requests} errors")
    
    async def _timed_query(self, query: str):
        """Execute a query and measure response time."""
        start = time.time()
        response = await self.agent.chat(query)
        duration = time.time() - start
        return {"query": query, "duration": duration, "response": response}
    
    async def _collect_metrics(self):
        """Collect current system metrics."""
        process = psutil.Process()
        return {
            "timestamp": datetime.now().isoformat(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "connections": len(process.connections()),
            "threads": process.num_threads(),
        }
    
    async def _check_health(self, metrics, initial_memory):
        """Check for health anomalies."""
        current_memory = metrics["memory_mb"] * 1024 * 1024
        memory_growth = (current_memory - initial_memory) / initial_memory
        
        if memory_growth > 0.2:  # 20% growth
            self.logger.warning(f"Memory growth: {memory_growth:.1%}")
        
        if metrics["cpu_percent"] > 90:
            self.logger.warning(f"High CPU: {metrics['cpu_percent']}%")
    
    def _generate_report(self, initial_memory):
        """Generate soak test report."""
        elapsed = datetime.now() - self.start_time
        final_memory = psutil.Process().memory_info().rss
        
        return {
            "duration_hours": elapsed.total_seconds() / 3600,
            "total_errors": len(self.errors),
            "error_rate": len(self.errors) / len(self.metrics) if self.metrics else 0,
            "memory_growth_percent": (final_memory - initial_memory) / initial_memory * 100,
            "metrics_collected": len(self.metrics),
            "status": "PASSED" if len(self.errors) / max(len(self.metrics), 1) < 0.01 else "FAILED",
        }
```

---

## 10. Monitoring & Alerting

### 10.1 Monitoring Dashboard Metrics

```yaml
# Grafana Dashboard Configuration
dashboard:
  title: "MultiAgent V2 - Reliability Monitoring"
  
  panels:
    - title: "System Availability"
      type: stat
      query: "avg(up{job='multiagent-v2'})"
      thresholds:
        - value: 0.995
          color: green
        - value: 0.99
          color: yellow
        - value: 0
          color: red
    
    - title: "Response Time P95"
      type: graph
      query: "histogram_quantile(0.95, rate(response_duration_seconds_bucket[5m]))"
      alert:
        condition: "> 5s for 5m"
        severity: warning
    
    - title: "Error Rate"
      type: graph
      query: "rate(requests_errors_total[5m]) / rate(requests_total[5m])"
      alert:
        condition: "> 0.02 for 5m"
        severity: critical
    
    - title: "Memory Usage"
      type: graph
      query: "process_resident_memory_bytes / 1024 / 1024"
      alert:
        condition: "> 2048 for 10m"
        severity: warning
    
    - title: "Active Sessions"
      type: stat
      query: "redis_connected_clients"
    
    - title: "Tool Success Rate"
      type: bargauge
      query: "sum(tool_calls_success) / sum(tool_calls_total) by (tool_name)"
```

### 10.2 Alert Rules

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| HighErrorRate | error_rate > 2% for 5min | 🔴 Critical | Page on-call |
| HighLatency | p95 > 5s for 5min | 🟡 Warning | Slack notification |
| MemoryLeak | memory growth > 20% in 1hr | 🟡 Warning | Create ticket |
| ServiceDown | uptime = 0 for 1min | 🔴 Critical | Page on-call |
| RedisUnavailable | redis_up = 0 for 30s | 🟡 Warning | Check fallback |
| HighCPU | cpu > 90% for 10min | 🟡 Warning | Scale check |

---

## 11. Test Schedule

### 11.1 Daily Tests

| Time | Test | Duration | Automated |
|------|------|----------|-----------|
| 00:00 | Environment Reset | 10 min | ✅ |
| 00:15 | Health Check Suite | 15 min | ✅ |
| 02:00 | Failure Recovery Tests | 2 hrs | ✅ |
| 06:00 | Consistency Validation | 1 hr | ✅ |
| 12:00 | Mid-day Health Check | 15 min | ✅ |
| 18:00 | Load Test (Peak Simulation) | 2 hrs | ✅ |

### 11.2 Weekly Tests

| Day | Test | Duration |
|-----|------|----------|
| Monday | 24-Hour Continuous Operation | 24 hrs |
| Wednesday | Full Chaos Engineering Suite | 8 hrs |
| Friday | Graceful Degradation Tests | 4 hrs |
| Saturday | 72-Hour Soak Test (Start) | 72 hrs |

### 11.3 Monthly Tests

| Week | Test | Duration |
|------|------|----------|
| Week 1 | Full Disaster Recovery Drill | 8 hrs |
| Week 2 | Capacity Planning Tests | 4 hrs |
| Week 3 | Security & Reliability Audit | 8 hrs |
| Week 4 | Performance Baseline Update | 4 hrs |

---

## 12. Tools & Resources

### 12.1 Testing Tools

| Tool | Purpose | Version |
|------|---------|---------|
| pytest | Test framework | 7.x |
| locust | Load testing | 2.x |
| chaos-toolkit | Chaos engineering | 1.x |
| docker | Container management | 24.x |
| prometheus | Metrics collection | 2.x |
| grafana | Visualization | 10.x |

### 12.2 Required Infrastructure

```yaml
# Reliability Test Environment
infrastructure:
  compute:
    test_runner:
      cpu: 4 cores
      memory: 8 GB
      storage: 50 GB
    
    agent_instance:
      cpu: 2 cores
      memory: 4 GB
      storage: 20 GB
  
  services:
    redis:
      version: "7-alpine"
      memory: 1 GB
    
    mongodb:
      version: "6"
      storage: 10 GB
    
    postgres:
      version: "15-alpine"
      storage: 10 GB
  
  monitoring:
    prometheus:
      retention: 15d
    
    grafana:
      dashboards: reliability
```

---

## 13. Reporting

### 13.1 Report Template

```markdown
# Reliability Test Report - MultiAgent V2

## Test Run Summary
- **Date**: [DATE]
- **Duration**: [DURATION]
- **Environment**: [ENV]
- **Tester**: [NAME]

## Results Overview

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Availability | ≥ 99.5% | XX.XX% | ✅/❌ |
| MTBF | ≥ 24 hrs | XX hrs | ✅/❌ |
| MTTR | ≤ 30 sec | XX sec | ✅/❌ |
| Error Rate | ≤ 1% | X.XX% | ✅/❌ |

## Detailed Results

### Availability Testing
[Details]

### Recovery Testing
[Details]

### Consistency Testing
[Details]

## Issues Found
1. [Issue description]
2. [Issue description]

## Recommendations
1. [Recommendation]
2. [Recommendation]

## Sign-off
- [ ] QA Lead Approval
- [ ] Engineering Lead Approval
- [ ] Production Ready
```

### 13.2 Automated Report Generation

```python
# scripts/generate_reliability_report.py

def generate_report(test_results: dict) -> str:
    """Generate reliability test report."""
    
    template = f"""
# Reliability Test Report - MultiAgent V2

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Duration**: {test_results['duration_hours']:.1f} hours
**Status**: {'✅ PASSED' if test_results['passed'] else '❌ FAILED'}

## Summary

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Availability | ≥ 99.5% | {test_results['availability']:.2f}% | {'✅' if test_results['availability'] >= 99.5 else '❌'} |
| MTBF | ≥ 24 hrs | {test_results['mtbf_hours']:.1f} hrs | {'✅' if test_results['mtbf_hours'] >= 24 else '❌'} |
| MTTR | ≤ 30 sec | {test_results['mttr_seconds']:.1f} sec | {'✅' if test_results['mttr_seconds'] <= 30 else '❌'} |
| Error Rate | ≤ 1% | {test_results['error_rate']:.2f}% | {'✅' if test_results['error_rate'] <= 1 else '❌'} |

## Test Categories

### Availability Tests
- Continuous Operation: {'PASSED' if test_results['availability_passed'] else 'FAILED'}
- Uptime: {test_results['uptime_percent']:.2f}%

### Recovery Tests
- Redis Recovery: {'PASSED' if test_results['redis_recovery'] else 'FAILED'}
- MongoDB Fallback: {'PASSED' if test_results['mongo_fallback'] else 'FAILED'}
- API Timeout Handling: {'PASSED' if test_results['api_timeout'] else 'FAILED'}

### Consistency Tests
- Response Consistency: {test_results['consistency_score']:.2f}
- Context Retention: {test_results['context_retention']:.2f}%

## Recommendations

{chr(10).join(f'- {r}' for r in test_results.get('recommendations', []))}
"""
    return template
```

---

## Appendix A: Checklist

### Pre-Test Checklist
- [ ] All services running and healthy
- [ ] Test data seeded
- [ ] Monitoring enabled
- [ ] Logging configured
- [ ] Alerts configured
- [ ] Baseline metrics recorded

### Post-Test Checklist
- [ ] Collect all logs
- [ ] Generate reports
- [ ] Document issues
- [ ] Update baselines if needed
- [ ] Clean up test data
- [ ] Archive results

---

**Document Version**: 1.0  
**Created**: December 2025  
**Owner**: QA Team  
**Review Cycle**: Monthly

---

*This reliability testing plan ensures MultiAgent V2 meets production-grade stability requirements.*

