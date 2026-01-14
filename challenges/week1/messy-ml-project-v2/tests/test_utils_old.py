#!/usr/bin/env python3
"""
Old Utility Tests - DEPRECATED
These tests are outdated and may not pass
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# All tests in this file are deprecated and should be skipped


class DeprecatedTests:
    """Deprecated test class"""

    def test_old_function_1(self):
        """Old test 1 - deprecated"""
        pytest.skip("Deprecated test")

    def test_old_function_2(self):
        """Old test 2 - deprecated"""
        pytest.skip("Deprecated test")

    def test_old_function_3(self):
        """Old test 3 - deprecated"""
        pytest.skip("Deprecated test")


class LegacyTests:
    """Legacy tests from v1.0"""

    def test_legacy_1(self):
        pytest.skip("Legacy test")

    def test_legacy_2(self):
        pytest.skip("Legacy test")


def test_deprecated_util():
    """Deprecated utility test"""
    pytest.skip("This test is deprecated")


def test_old_helper():
    """Old helper test"""
    pytest.skip("Old test, needs update")


def test_broken_test():
    """This test is broken"""
    pytest.skip("Broken, don't run")


# These tests were never finished
def test_incomplete_1():
    """Incomplete test"""
    pass


def test_incomplete_2():
    """Another incomplete test"""
    pass


# Dead code - tests that can't run
if False:
    def test_never_runs():
        assert False, "This should never run"


# More deprecated tests
class VeryOldTests:
    """Very old tests - from v0.5"""

    def test_ancient_1(self):
        pytest.skip("Ancient test")

    def test_ancient_2(self):
        pytest.skip("Ancient test")


def _helper_test_function():
    """Helper that's never called"""
    pass


# Placeholder to make file non-empty
def test_file_exists():
    """Just to verify file exists"""
    assert True
