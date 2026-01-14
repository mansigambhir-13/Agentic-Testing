# MultiAgent V2 Reliability Testing Suite

Comprehensive reliability testing suite for the MultiAgent V2 E-commerce agent system.

## Overview

This testing suite validates the reliability, consistency, and recovery capabilities of the MultiAgent V2 orchestrator and its components.

## Test Categories

### 1. Agent Reliability Tests
- **Basic Response Reliability**: Tests agent responds reliably to basic queries
- **Session Persistence**: Validates session data persists across requests
- **Error Recovery**: Tests graceful handling of error scenarios
- **Concurrent Requests**: Tests handling of multiple sequential requests

### 2. Infrastructure Tests
- **Redis Connection Stability**: Tests Redis maintains stable connections
- **MongoDB Connection Pool**: Tests MongoDB connection pool handling
- **API Endpoint Availability**: Tests external API endpoints remain available

### 3. Recovery Tests
- **Graceful Degradation**: Tests agent degrades gracefully when services unavailable
- **Request Timeout Handling**: Tests appropriate timeout handling
- **State Recovery After Error**: Tests state recovery after errors

### 4. Continuous Operation Tests
- **Short Continuous Operation**: Tests operation for extended duration (2-24 hours)

## Quick Start

### Prerequisites
- Python 3.11+
- pytest
- Redis (optional, for full test coverage)
- OpenRouter API Key

### Running Tests

#### Quick Tests (Recommended for CI/CD)
```powershell
cd bench101_testing\reliability_testing
.\run_reliability_tests.ps1 --type quick
```

#### All Tests (Including Long-Running)
```powershell
cd bench101_testing\reliability_testing
.\run_reliability_tests.ps1 --type all
```

#### Specific Test Categories
```powershell
# Agent tests only
.\run_reliability_tests.ps1 --type agent

# Infrastructure tests only
.\run_reliability_tests.ps1 --type infrastructure

# Recovery tests only
.\run_reliability_tests.ps1 --type recovery
```

### Using Python Directly

```bash
cd bench101_testing/reliability_testing

# Quick tests
python scripts/run_reliability_tests.py --type quick -v

# All tests
python scripts/run_reliability_tests.py --type all -v

# Specific category
python scripts/run_reliability_tests.py --type agent -v
```

## Environment Variables

The tests require the following environment variables:

```powershell
# Required
$env:OPENROUTER_API_KEY = "sk-or-v1-..."

# Optional (for infrastructure tests)
$env:MONGODB_URI = "mongodb://localhost:27017"
```

## Test Results

Test results and logs are stored in:
- **Logs**: `bench101_testing/logs/reliability_testing/`
- **Summary Report**: `bench101_testing/reliability_testing/RELIABILITY_TEST_SUMMARY.md`

## Test Structure

```
reliability_testing/
├── tests/
│   ├── __init__.py
│   ├── test_agent_reliability.py      # Agent reliability tests
│   ├── test_infrastructure.py          # Infrastructure tests
│   ├── test_continuous_operation.py    # Long-running tests
│   └── test_recovery.py                # Recovery tests
├── scripts/
│   └── run_reliability_tests.py        # Test runner script
├── config/
│   └── test_config.yaml                # Test configuration
├── conftest.py                         # Pytest configuration
├── run_reliability_tests.ps1           # PowerShell runner
└── README.md                           # This file
```

## Success Criteria

Tests pass if:
- ✅ Availability ≥ 95% over test duration
- ✅ Error rate ≤ 5%
- ✅ All recovery scenarios handled gracefully
- ✅ No unrecoverable crashes
- ✅ Session state maintained correctly

## Continuous Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run Reliability Tests
  run: |
    cd bench101_testing/reliability_testing
    python scripts/run_reliability_tests.py --type quick -v
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

## Troubleshooting

### Tests Skipping
- **MongoDB/Redis tests skipped**: Expected if services not running locally
- **API endpoint tests skipped**: Expected if services not available
- **Long-running tests skipped**: Excluded in quick mode

### Common Issues
1. **Import errors**: Ensure project root is in Python path
2. **API key errors**: Set `OPENROUTER_API_KEY` environment variable
3. **Service connection errors**: Tests will skip gracefully if services unavailable

## Future Enhancements

- [ ] Extended 24-hour continuous operation tests
- [ ] Load testing under high concurrent load
- [ ] Chaos engineering scenarios
- [ ] Memory leak detection over extended periods
- [ ] Performance regression testing

## Documentation

- [Reliability Testing Plan](docs/RELIABILITY_TESTING_PLAN.md)
- [Test Summary Report](RELIABILITY_TEST_SUMMARY.md)

---

**Version**: 1.0  
**Last Updated**: December 28, 2025
