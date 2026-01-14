"""
Reliability Test Runner

Main script to run reliability tests and generate reports.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pytest

# Setup paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logs_dir = Path(__file__).parent.parent.parent / "logs" / "reliability_testing"
logs_dir.mkdir(parents=True, exist_ok=True)

log_file = logs_dir / f"reliability_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def run_tests(test_type="all", duration=None, verbose=False):
    """Run reliability tests."""
    
    tests_dir = Path(__file__).parent.parent / "tests"
    
    logger.info("=" * 80)
    logger.info("MultiAgent V2 - Reliability Testing Suite")
    logger.info("=" * 80)
    logger.info(f"Test Type: {test_type}")
    logger.info(f"Log File: {log_file}")
    logger.info("=" * 80)
    
    # Build pytest arguments
    pytest_args = [
        str(tests_dir),
        "-v" if verbose else "",
        "--tb=short",
        f"--log-file={log_file}",
        "--log-file-level=INFO",
    ]
    
    # Filter by test type
    if test_type == "infrastructure":
        pytest_args.extend(["-k", "TestInfrastructureReliability"])
    elif test_type == "agent":
        pytest_args.extend(["-k", "TestAgentReliability"])
    elif test_type == "continuous":
        pytest_args.extend(["-k", "TestContinuousOperation"])
    elif test_type == "recovery":
        pytest_args.extend(["-k", "TestRecovery"])
    elif test_type == "quick":
        # Run quick tests only (skip long-running)
        pytest_args.extend(["-k", "not long_running"])
    
    # Remove empty strings
    pytest_args = [arg for arg in pytest_args if arg]
    
    logger.info(f"Running pytest with args: {' '.join(pytest_args)}")
    
    try:
        exit_code = pytest.main(pytest_args)
        
        if exit_code == 0:
            logger.info("✅ All reliability tests passed!")
        else:
            logger.warning(f"⚠️ Some tests failed (exit code: {exit_code})")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run MultiAgent V2 reliability tests")
    parser.add_argument(
        "--type",
        choices=["all", "infrastructure", "agent", "continuous", "recovery", "quick"],
        default="quick",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Duration for continuous tests (minutes)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    exit_code = run_tests(
        test_type=args.type,
        duration=args.duration,
        verbose=args.verbose
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

