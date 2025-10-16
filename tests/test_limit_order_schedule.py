"""
Unit tests for limit order schedule generation.

Tests the limit_order_schedule module's ability to:
- Generate correct order schedules for YES markets
- Generate correct order schedules for NO markets
- Verify cumulative costs match dcost functions
- Verify probabilities are reached correctly
"""

import pytest
import numpy as np
import pandas as pd

from make_market import (
    yes_order_schedule,
    no_order_schedule,
    dcost_yes,
    dcost_no,
    prob,
    qyes_from_pq,
    create_order_price_schedule,
)


class TestYesOrderSchedule:
    """Test suite for YES market order schedules."""

    def test_basic_schedule(self):
        """Test basic YES order schedule generation."""
        schedule = yes_order_schedule(
            p_init=0.7, Q=100, prob_schedule=[0.71, 0.8, 0.9], b=1.0
        )

        # Check DataFrame structure
        assert isinstance(schedule, pd.DataFrame)
        assert len(schedule) == 3
        assert all(
            col in schedule.columns
            for col in [
                "target_prob",
                "q_yes_incremental",
                "q_yes_cumulative",
                "cost_incremental",
                "cost_cumulative",
                "dcost_verification",
            ]
        )

        # Check probabilities are correct
        np.testing.assert_array_equal(schedule["target_prob"].values, [0.71, 0.8, 0.9])

        # Check cumulative cost matches dcost_yes
        for _, row in schedule.iterrows():
            expected_cost = dcost_yes(row["q_yes_cumulative"], 0.7, 100, b=1.0)
            np.testing.assert_allclose(
                row["cost_cumulative"], expected_cost, rtol=1e-10
            )
            np.testing.assert_allclose(
                row["dcost_verification"], expected_cost, rtol=1e-10
            )

    def test_probability_verification(self):
        """Test that target probabilities are actually reached."""
        schedule = yes_order_schedule(
            p_init=0.5, Q=200, prob_schedule=[0.6, 0.7, 0.8, 0.9], b=1.0
        )

        # Verify each probability is reached
        q_yes_init = qyes_from_pq(0.5, 200, b=1.0)
        q_no_init = 200 - q_yes_init

        for _, row in schedule.iterrows():
            actual_prob = prob(q_yes_init + row["q_yes_cumulative"], q_no_init, b=1.0)
            np.testing.assert_allclose(actual_prob, row["target_prob"], rtol=1e-10)

    def test_incremental_sums_to_cumulative(self):
        """Test that incremental orders sum to cumulative."""
        schedule = yes_order_schedule(
            p_init=0.3, Q=50, prob_schedule=[0.4, 0.5, 0.6], b=1.0
        )

        # Check q_yes incremental sums
        cumulative_sum = schedule["q_yes_incremental"].cumsum().values
        np.testing.assert_allclose(
            cumulative_sum, schedule["q_yes_cumulative"].values, rtol=1e-10
        )

        # Check cost incremental sums
        cost_cumulative_sum = schedule["cost_incremental"].cumsum().values
        np.testing.assert_allclose(
            cost_cumulative_sum, schedule["cost_cumulative"].values, rtol=1e-10
        )

    def test_different_b_values(self):
        """Test with different market depth parameters."""
        for b in [0.5, 1.0, 2.0, 5.0]:
            schedule = yes_order_schedule(
                p_init=0.6, Q=100, prob_schedule=[0.7, 0.8], b=b
            )

            # Verify costs match
            for _, row in schedule.iterrows():
                expected_cost = dcost_yes(row["q_yes_cumulative"], 0.6, 100, b=b)
                np.testing.assert_allclose(
                    row["cost_cumulative"], expected_cost, rtol=1e-10
                )

    def test_single_order(self):
        """Test with single order in schedule."""
        schedule = yes_order_schedule(p_init=0.5, Q=100, prob_schedule=[0.75], b=1.0)

        assert len(schedule) == 1
        assert (
            schedule.iloc[0]["q_yes_incremental"]
            == schedule.iloc[0]["q_yes_cumulative"]
        )
        assert (
            schedule.iloc[0]["cost_incremental"] == schedule.iloc[0]["cost_cumulative"]
        )

    def test_validation_errors(self):
        """Test input validation."""
        # Invalid p_init
        with pytest.raises(ValueError, match="p_init must be between 0 and 1"):
            yes_order_schedule(1.5, 100, [0.8])

        # Invalid Q
        with pytest.raises(ValueError, match="Q must be positive"):
            yes_order_schedule(0.5, -100, [0.8])

        # Invalid b
        with pytest.raises(ValueError, match="b must be positive"):
            yes_order_schedule(0.5, 100, [0.8], b=-1)

        # Target prob not greater than p_init
        with pytest.raises(ValueError, match="must be > p_init"):
            yes_order_schedule(0.7, 100, [0.65])

        # Schedule not sorted
        with pytest.raises(ValueError, match="must be in ascending order"):
            yes_order_schedule(0.5, 100, [0.8, 0.7])


class TestNoOrderSchedule:
    """Test suite for NO market order schedules."""

    def test_basic_schedule(self):
        """Test basic NO order schedule generation."""
        schedule = no_order_schedule(
            p_init=0.7, Q=100, prob_schedule=[0.65, 0.5, 0.3], b=1.0
        )

        # Check DataFrame structure
        assert isinstance(schedule, pd.DataFrame)
        assert len(schedule) == 3
        assert all(
            col in schedule.columns
            for col in [
                "target_prob",
                "q_no_incremental",
                "q_no_cumulative",
                "cost_incremental",
                "cost_cumulative",
                "dcost_verification",
            ]
        )

        # Check probabilities are correct
        np.testing.assert_array_equal(schedule["target_prob"].values, [0.65, 0.5, 0.3])

        # Check cumulative cost matches dcost_no
        for _, row in schedule.iterrows():
            expected_cost = dcost_no(row["q_no_cumulative"], 0.7, 100, b=1.0)
            np.testing.assert_allclose(
                row["cost_cumulative"], expected_cost, rtol=1e-10
            )
            np.testing.assert_allclose(
                row["dcost_verification"], expected_cost, rtol=1e-10
            )

    def test_probability_verification(self):
        """Test that target probabilities are actually reached."""
        schedule = no_order_schedule(
            p_init=0.8, Q=200, prob_schedule=[0.7, 0.5, 0.3, 0.1], b=1.0
        )

        # Verify each probability is reached
        q_yes_init = qyes_from_pq(0.8, 200, b=1.0)
        q_no_init = 200 - q_yes_init

        for _, row in schedule.iterrows():
            actual_prob = prob(q_yes_init, q_no_init + row["q_no_cumulative"], b=1.0)
            np.testing.assert_allclose(actual_prob, row["target_prob"], rtol=1e-10)

    def test_incremental_sums_to_cumulative(self):
        """Test that incremental orders sum to cumulative."""
        schedule = no_order_schedule(
            p_init=0.6, Q=50, prob_schedule=[0.5, 0.4, 0.3], b=1.0
        )

        # Check q_no incremental sums
        cumulative_sum = schedule["q_no_incremental"].cumsum().values
        np.testing.assert_allclose(
            cumulative_sum, schedule["q_no_cumulative"].values, rtol=1e-10
        )

        # Check cost incremental sums
        cost_cumulative_sum = schedule["cost_incremental"].cumsum().values
        np.testing.assert_allclose(
            cost_cumulative_sum, schedule["cost_cumulative"].values, rtol=1e-10
        )

    def test_different_b_values(self):
        """Test with different market depth parameters."""
        for b in [0.5, 1.0, 2.0, 5.0]:
            schedule = no_order_schedule(
                p_init=0.6, Q=100, prob_schedule=[0.5, 0.4], b=b
            )

            # Verify costs match
            for _, row in schedule.iterrows():
                expected_cost = dcost_no(row["q_no_cumulative"], 0.6, 100, b=b)
                np.testing.assert_allclose(
                    row["cost_cumulative"], expected_cost, rtol=1e-10
                )

    def test_single_order(self):
        """Test with single order in schedule."""
        schedule = no_order_schedule(p_init=0.75, Q=100, prob_schedule=[0.5], b=1.0)

        assert len(schedule) == 1
        assert (
            schedule.iloc[0]["q_no_incremental"] == schedule.iloc[0]["q_no_cumulative"]
        )
        assert (
            schedule.iloc[0]["cost_incremental"] == schedule.iloc[0]["cost_cumulative"]
        )

    def test_validation_errors(self):
        """Test input validation."""
        # Invalid p_init
        with pytest.raises(ValueError, match="p_init must be between 0 and 1"):
            no_order_schedule(1.5, 100, [0.3])

        # Invalid Q
        with pytest.raises(ValueError, match="Q must be positive"):
            no_order_schedule(0.5, -100, [0.3])

        # Invalid b
        with pytest.raises(ValueError, match="b must be positive"):
            no_order_schedule(0.5, 100, [0.3], b=-1)

        # Target prob not less than p_init
        with pytest.raises(ValueError, match="must be < p_init"):
            no_order_schedule(0.3, 100, [0.5])

        # Schedule not sorted descending
        with pytest.raises(ValueError, match="must be in descending order"):
            no_order_schedule(0.7, 100, [0.5, 0.6])


class TestSymmetry:
    """Test symmetry properties between YES and NO markets."""

    def test_perfect_symmetry_with_complement(self):
        """
        Test that NO schedule at p equals YES schedule at 1-p.

        This is a fundamental symmetry property:
        - NO orders from p_init to lower probabilities
        - Should equal YES orders from (1-p_init) to complementary probabilities

        Example: NO from 0.7 to [0.65, 0.5, 0.3]
                 = YES from 0.3 to [0.35, 0.5, 0.7]
        """
        p_init = 0.7
        Q = 100
        b = 1.0

        # NO market: decrease probability from 0.7
        prob_targets_no = [0.65, 0.5, 0.3]
        no_sched = no_order_schedule(p_init, Q, prob_targets_no, b=b)

        # YES market: increase probability from 1-p_init = 0.3
        # Target probabilities are complements: 1 - [0.65, 0.5, 0.3] = [0.35, 0.5, 0.7]
        # Need to reverse to get ascending order for YES
        prob_targets_yes = sorted([1 - p for p in prob_targets_no])
        yes_sched = yes_order_schedule(1 - p_init, Q, prob_targets_yes, b=b)

        # By symmetry, quantities should match exactly
        np.testing.assert_allclose(
            no_sched["q_no_cumulative"].values,
            yes_sched["q_yes_cumulative"].values,
            rtol=1e-10,
        )

        # Costs should also match exactly
        np.testing.assert_allclose(
            no_sched["cost_cumulative"].values,
            yes_sched["cost_cumulative"].values,
            rtol=1e-10,
        )

        # Incremental quantities should match
        np.testing.assert_allclose(
            no_sched["q_no_incremental"].values,
            yes_sched["q_yes_incremental"].values,
            rtol=1e-10,
        )

        # Incremental costs should match
        np.testing.assert_allclose(
            no_sched["cost_incremental"].values,
            yes_sched["cost_incremental"].values,
            rtol=1e-10,
        )

    def test_symmetry_multiple_scenarios(self):
        """Test symmetry across multiple initial probabilities and schedules."""
        test_cases = [
            (0.6, [0.55, 0.5, 0.4]),
            (0.8, [0.75, 0.6, 0.4]),
            (0.9, [0.85, 0.7, 0.5, 0.3]),
        ]

        for p_init, prob_schedule in test_cases:
            # NO market
            no_sched = no_order_schedule(p_init, 100, prob_schedule, b=1.0)

            # Complementary YES market
            # Need to sort in ascending order for YES
            yes_prob_schedule = sorted([1 - p for p in prob_schedule])
            yes_sched = yes_order_schedule(1 - p_init, 100, yes_prob_schedule, b=1.0)

            # Verify symmetry
            np.testing.assert_allclose(
                no_sched["q_no_cumulative"].values,
                yes_sched["q_yes_cumulative"].values,
                rtol=1e-10,
                err_msg=f"Symmetry failed for p_init={p_init}, schedule={prob_schedule}",
            )

            np.testing.assert_allclose(
                no_sched["cost_cumulative"].values,
                yes_sched["cost_cumulative"].values,
                rtol=1e-10,
                err_msg=f"Cost symmetry failed for p_init={p_init}, schedule={prob_schedule}",
            )


class TestCreateOrderPriceSchedule:
    """Test suite for create_order_price_schedule function."""

    def test_basic_schedule_creation(self):
        """Test basic price schedule generation."""
        yes_prices, no_prices = create_order_price_schedule(
            p=0.5, half_spread_bps=10, max_order_bps=100, num_orders=5
        )

        # Check we got the right number of orders
        assert len(yes_prices) == 5
        assert len(no_prices) == 5

        # YES prices should be ascending and > 0.5
        assert yes_prices == sorted(yes_prices)
        assert all(p > 0.5 for p in yes_prices)

        # NO prices should be descending and < 0.5
        assert no_prices == sorted(no_prices, reverse=True)
        assert all(p < 0.5 for p in no_prices)

    def test_log_spacing(self):
        """Test that prices are evenly spaced in log space."""
        yes_prices, no_prices = create_order_price_schedule(
            p=0.6, half_spread_bps=20, max_order_bps=500, num_orders=10
        )

        # YES prices should have constant log spacing
        log_yes = np.log(yes_prices)
        log_diffs_yes = np.diff(log_yes)
        np.testing.assert_allclose(log_diffs_yes, log_diffs_yes[0], rtol=1e-10)

        # NO prices (reversed) should have constant log spacing
        log_no = np.log(no_prices[::-1])  # Reverse to ascending
        log_diffs_no = np.diff(log_no)
        np.testing.assert_allclose(log_diffs_no, log_diffs_no[0], rtol=1e-10)

    def test_price_bounds(self):
        """Test that min/max prices match expected values from basis points."""
        p = 0.7
        half_spread_bps = 50
        max_order_bps = 1000

        yes_prices, no_prices = create_order_price_schedule(
            p=p,
            half_spread_bps=half_spread_bps,
            max_order_bps=max_order_bps,
            num_orders=5,
        )

        # Check YES bounds
        expected_yes_min = p * np.exp(half_spread_bps / 10000)
        expected_yes_max = p * np.exp(max_order_bps / 10000)
        np.testing.assert_allclose(yes_prices[0], expected_yes_min, rtol=1e-10)
        np.testing.assert_allclose(yes_prices[-1], expected_yes_max, rtol=1e-10)

        # Check NO bounds
        expected_no_max = p * np.exp(-half_spread_bps / 10000)
        expected_no_min = p * np.exp(-max_order_bps / 10000)
        np.testing.assert_allclose(no_prices[0], expected_no_max, rtol=1e-10)
        np.testing.assert_allclose(no_prices[-1], expected_no_min, rtol=1e-10)

    def test_single_order(self):
        """Test with num_orders=1."""
        yes_prices, no_prices = create_order_price_schedule(
            p=0.5, half_spread_bps=10, max_order_bps=100, num_orders=1
        )

        assert len(yes_prices) == 1
        assert len(no_prices) == 1

        # With single order, linspace returns the start point
        # For YES: start = half_spread (closest)
        # For NO: linspace goes from min to max, so start = min (furthest)
        expected_yes = 0.5 * np.exp(10 / 10000)
        expected_no = 0.5 * np.exp(-100 / 10000)  # Note: max_order, not half_spread
        np.testing.assert_allclose(yes_prices[0], expected_yes, rtol=1e-10)
        np.testing.assert_allclose(no_prices[0], expected_no, rtol=1e-10)

    def test_two_orders(self):
        """Test with num_orders=2 to verify endpoints."""
        yes_prices, no_prices = create_order_price_schedule(
            p=0.5, half_spread_bps=10, max_order_bps=100, num_orders=2
        )

        assert len(yes_prices) == 2
        assert len(no_prices) == 2

        # With two orders, should be at half_spread and max_order
        expected_yes_min = 0.5 * np.exp(10 / 10000)
        expected_yes_max = 0.5 * np.exp(100 / 10000)
        np.testing.assert_allclose(yes_prices[0], expected_yes_min, rtol=1e-10)
        np.testing.assert_allclose(yes_prices[1], expected_yes_max, rtol=1e-10)

        expected_no_max = 0.5 * np.exp(-10 / 10000)
        expected_no_min = 0.5 * np.exp(-100 / 10000)
        np.testing.assert_allclose(no_prices[0], expected_no_max, rtol=1e-10)
        np.testing.assert_allclose(no_prices[1], expected_no_min, rtol=1e-10)

    def test_integration_with_order_schedule(self):
        """Test that output can be used directly with yes/no_order_schedule."""
        p = 0.6
        Q = 100
        b = 1.0

        yes_prices, no_prices = create_order_price_schedule(
            p=p, half_spread_bps=20, max_order_bps=500, num_orders=3
        )

        # Should be able to create schedules without errors
        yes_sched = yes_order_schedule(p, Q, yes_prices, b=b)
        no_sched = no_order_schedule(p, Q, no_prices, b=b)

        # Verify schedules are valid
        assert len(yes_sched) == 3
        assert len(no_sched) == 3

        # Target probabilities should match input
        np.testing.assert_allclose(
            yes_sched["target_prob"].values, yes_prices, rtol=1e-10
        )
        np.testing.assert_allclose(
            no_sched["target_prob"].values, no_prices, rtol=1e-10
        )

    def test_validation_errors(self):
        """Test input validation."""
        # Invalid p
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            create_order_price_schedule(1.5, 10, 100, 5)

        # Invalid half_spread_bps
        with pytest.raises(ValueError, match="half_spread_bps must be positive"):
            create_order_price_schedule(0.5, -10, 100, 5)

        # max_order_bps <= half_spread_bps
        with pytest.raises(ValueError, match="max_order_bps .* must be >"):
            create_order_price_schedule(0.5, 100, 50, 5)

        # Invalid num_orders
        with pytest.raises(ValueError, match="num_orders must be at least 1"):
            create_order_price_schedule(0.5, 10, 100, 0)

    def test_automatic_capping(self):
        """Test that max_order_bps is automatically capped to keep probabilities in (0, 1)."""
        # Test case 1: High p, large max_order would push YES > 1
        yes_prices, no_prices = create_order_price_schedule(
            p=0.9, half_spread_bps=10, max_order_bps=50000, num_orders=10
        )

        # All prices should be in valid range
        assert all(
            0 < p < 1 for p in yes_prices
        ), f"YES prices out of range: {yes_prices}"
        assert all(0 < p < 1 for p in no_prices), f"NO prices out of range: {no_prices}"

        # YES should be close to but not reach 1
        assert yes_prices[-1] < 1.0
        assert yes_prices[-1] > 0.99  # Should be very close to 1

        # Test case 2: Low p, large max_order would push NO < 0
        yes_prices, no_prices = create_order_price_schedule(
            p=0.1, half_spread_bps=10, max_order_bps=50000, num_orders=10
        )

        # All prices should be in valid range
        assert all(
            0 < p < 1 for p in yes_prices
        ), f"YES prices out of range: {yes_prices}"
        assert all(0 < p < 1 for p in no_prices), f"NO prices out of range: {no_prices}"

        # NO should be close to but not reach 0
        assert no_prices[-1] > 0.0
        assert no_prices[-1] < 0.02  # Should be very close to 0

        # Test case 3: p=0.5 symmetric case
        yes_prices, no_prices = create_order_price_schedule(
            p=0.5, half_spread_bps=10, max_order_bps=100000, num_orders=5
        )

        # All prices should be in valid range
        assert all(0 < p < 1 for p in yes_prices)
        assert all(0 < p < 1 for p in no_prices)

        # Both should be capped symmetrically
        assert yes_prices[-1] > 0.99
        assert no_prices[-1] < 0.26  # log symmetric around 0.5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
