"""Tests for GOPH 419 Lab Assignment #1."""

import numpy as np

from launch import (
        arcsin,
        launch_angle,
        launch_angle_range,
)


def test_arcsin_value(x, result_expected):
    """Test a single valid argument of arcsin()."""
    result_str = [f"Testing x = {x}..."]
    result = arcsin(x)
    if not np.allclose(result, result_expected):
        failed = 1
        result_str.append("FAILED")
        msg = f"\n\tresult : {result}, expected : {result_expected}"
        result_str.append(msg)
    else:
        failed = 0
        result_str.append("PASSED")
    print("".join(result_str))
    return failed


def test_arcsin_value_error(x):
    """Test an invalid argument of arcsin().
    The argument x should be out of range abs(x) > 1.0.
    """
    result_str = [f"Testing x = {x}..."]
    try:
        result = arcsin(x)
    except ValueError:
        # this was the expected result
        failed = 0
        result_str.append("PASSED")
    except Exception as e:
        # catch unexpected exception type
        failed = 1
        result_str.append("FAILED")
        msg = f"\n\tresult : {e.__class__.__name__}, expected : ValueError"
        result_str.append(msg)
    else:
        # catch non-exception result
        failed = 1
        result_str.append("FAILED")
        msg = f"\n\tresult : {result}, expected : ValueError"
        result_str.append(msg)
    finally:
        print("".join(result_str))
        return failed


def test_arcsin():
    """Test arcsin() function."""

    num_tests = 0
    num_failed = 0

    print("\n***Testing arcsin(x)***")

    print("Testing with valid inputs [-1.0, 1.0]:")
    # Test: x = 0.0
    x = 0.0
    result_expected = 0.0
    num_tests += 1
    num_failed += test_arcsin_value(x, result_expected)
    # Test: x = small
    x = 1.e-8
    result_expected = 0.0
    num_tests += 1
    num_failed += test_arcsin_value(x, result_expected)
    # Test: x = -small
    x = -1.e-8
    result_expected = 0.0
    num_tests += 1
    num_failed += test_arcsin_value(x, result_expected)
    # Test: x = +mid
    x = 0.5
    result_expected = np.arcsin(x)
    num_tests += 1
    num_failed += test_arcsin_value(x, result_expected)
    # Test: x = -mid
    x = -0.7
    result_expected = np.arcsin(x)
    num_tests += 1
    num_failed += test_arcsin_value(x, result_expected)
    # Test: x = +big
    x = 0.9
    result_expected = np.arcsin(x)
    num_tests += 1
    num_failed += test_arcsin_value(x, result_expected)

    print("Testing with invalid inputs abs(x) > 1.0:")
    # Test: x = +invalid
    x = 1.1
    num_tests += 1
    num_failed += test_arcsin_value_error(x)
    # Test: x = -invalid
    x = -1.1
    num_tests += 1
    num_failed += test_arcsin_value_error(x)

    # Summary Output:
    if num_failed:
        print(f"{num_failed} / {num_tests} tests failed.")
    else:
        print("All tests passed!")


def test_launch_angle_value(ve_v0, alpha, result_expected):
    """Test launch_angle() with valid input."""
    result_str = [f"Testing ve_v0 = {ve_v0} and alpha = {alpha}..."]
    result = launch_angle(ve_v0, alpha)
    if not np.allclose(result, result_expected):
        failed = 1
        result_str.append("FAILED")
        msg = f"\n\tresult : {result}, expected : {result_expected}"
        result_str.append(msg)
    else:
        failed = 0
        result_str.append("PASSED")
    print("".join(result_str))
    return failed


def test_launch_angle_value_error(ve_v0, alpha):
    """Test invalid arguments to launch_angle().
    The arguments ve_v0 and alpha should result in a negative value of:
        1.0 - alpha / (1 + alpha) * ve_v0**2
    """
    result_str = [f"Testing ve_v0 = {ve_v0} and alpha = {alpha}..."]
    try:
        result = launch_angle(ve_v0, alpha)
    except ValueError:
        # this was the expected result
        failed = 0
        result_str.append("PASSED")
    except Exception as e:
        # catch unexpected exception type
        failed = 1
        result_str.append("FAILED")
        msg = f"\n\tresult : {e.__class__.__name__}, expected : ValueError"
        result_str.append(msg)
    else:
        # catch non-exception result
        failed = 1
        result_str.append("FAILED")
        msg = f"\n\tresult : {result}, expected : ValueError"
        result_str.append(msg)
    finally:
        print("".join(result_str))
        return failed


def test_launch_angle():
    """Test launch_angle() function."""

    num_tests = 0
    num_failed = 0

    print("\n***Testing launch_angle(x)***")

    print("Testing with valid inputs:")
    # Test: ve_v0 = 2.0, alpha = 0.25
    ve_v0 = 2.0
    alpha = 0.25
    result_expected = 0.593200
    num_tests += 1
    num_failed += test_launch_angle_value(ve_v0, alpha, result_expected)
    # Test: ve_v0 = 3.0, alpha = 0.10
    ve_v0 = 3.0
    alpha = 0.10
    result_expected = 0.488205
    num_tests += 1
    num_failed += test_launch_angle_value(ve_v0, alpha, result_expected)

    print("Testing with invalid inputs:")
    # Test: ve_v0 = 0.9
    ve_v0 = 0.9
    alpha = None    # value not relevant to this test
    num_tests += 1
    num_failed += test_launch_angle_value_error(ve_v0, alpha)
    # Test: ve_v0 = 2.0, alpha = 0.34
    ve_v0 = 2.0
    alpha = 0.34    # alpha too big for this ve_v0
    num_tests += 1
    num_failed += test_launch_angle_value_error(ve_v0, alpha)
    # Test: ve_v0 = 2.24, alpha = 0.25
    ve_v0 = 2.24    # ve_v0 too big for this alpha
    alpha = 0.25
    num_tests += 1
    num_failed += test_launch_angle_value_error(ve_v0, alpha)

    # Summary Output:
    if num_failed:
        print(f"{num_failed} / {num_tests} tests failed.")
    else:
        print("All tests passed!")


def test_launch_angle_range_value(ve_v0, alpha, tol_alpha, result_expected):
    """Test launch_angle_range() with valid input."""
    result_str = [(f"Testing ve_v0 = {ve_v0}, "
                   + f"alpha = {alpha}, "
                   + f"tol_alpha = {tol_alpha}...")]
    result = launch_angle_range(ve_v0, alpha, tol_alpha)
    if not np.allclose(result, result_expected):
        failed = 1
        result_str.append("FAILED")
        msg = f"\n\tresult : {result}, expected : {result_expected}"
        result_str.append(msg)
    else:
        failed = 0
        result_str.append("PASSED")
    print("".join(result_str))
    return failed


def test_launch_angle_range():
    """Test launch_angle_range() function."""

    num_tests = 0
    num_failed = 0

    print("\n***Testing launch_angle_range(x)***")
    # Test: ve_v0 = 2.0, alpha = 0.25, tol_alpha = 0.02
    ve_v0 = 2.0
    alpha = 0.25
    tol_alpha = 0.02
    result_expected = np.array([0.574089, 0.611860])
    num_tests += 1
    num_failed += test_launch_angle_range_value(ve_v0, alpha, tol_alpha,
                                                result_expected)
    # Test: ve_v0 = 3.0, alpha = 0.10, tol_alpha = 0.05
    ve_v0 = 3.0
    alpha = 0.10
    tol_alpha = 0.05
    result_expected = np.array([0.433970, 0.538257])
    num_tests += 1
    num_failed += test_launch_angle_range_value(ve_v0, alpha, tol_alpha,
                                                result_expected)

    # Summary Output:
    if num_failed:
        print(f"{num_failed} / {num_tests} tests failed.")
    else:
        print("All tests passed!")


if __name__ == "__main__":
    test_arcsin()
    test_launch_angle()
    test_launch_angle_range()
