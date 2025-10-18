"""
Market Making Functions for LMSR (Logarithmic Market Scoring Rule)

This module provides core functions for LMSR-based prediction markets including:
- Cost calculations for buying shares
- Probability calculations from market state
- Conversion between probability/quantity representations
- Limit order placement schedule generation

The LMSR is a market maker that allows traders to buy and sell shares of outcomes
at prices that reflect the current market probability. The market depth parameter 'b'
controls liquidity - larger b means more liquid markets with smaller price impact.
"""

import numpy as np
import pandas as pd
from typing import List


def cost(q_yes, q_no, b=1):
    """
    Calculate total cost function for LMSR market maker.

    Args:
        q_yes: Number of YES shares outstanding
        q_no: Number of NO shares outstanding
        b: Market depth parameter (default: 1)

    Returns:
        Total cost of the market state
    """
    return b * np.log(np.exp(q_yes / b) + np.exp(q_no / b))


def qyes_from_pq(p, Q, b=1):
    """
    Calculate q_yes from probability p and total shares Q.

    Args:
        p: Market probability (between 0 and 1)
        Q: Total shares outstanding (q_yes + q_no)
        b: Market depth parameter (default: 1)

    Returns:
        Number of YES shares
    """
    x = np.sqrt((p / (1 - p) * np.exp(Q / b)))
    return (np.log(x) * b).item()


def qno_from_pq(p, Q, b=1):
    """
    Calculate q_no from probability p and total shares Q.

    Args:
        p: Market probability (between 0 and 1)
        Q: Total shares outstanding (q_yes + q_no)
        b: Market depth parameter (default: 1)

    Returns:
        Number of NO shares
    """
    return Q - qyes_from_pq(p, Q, b=b)


def prob(q_yes, q_no, b=1):
    """
    Probability of YES outcome given q_yes and q_no shares outstanding.

    Args:
        q_yes: Number of YES shares outstanding
        q_no: Number of NO shares outstanding
        b: Market depth parameter (default: 1)

    Returns:
        Market probability (between 0 and 1)
    """
    return np.exp(q_yes / b) / (np.exp(q_no / b) + np.exp(q_yes / b))


def dcost_yes(q_yes, p, Q, b=1):
    """
    Cost function for buying q_yes shares when the current market probability is p
    and total shares outstanding is Q.

    Args:
        q_yes: Number of YES shares to buy
        p: Current market probability
        Q: Total shares outstanding (q_yes + q_no)
        b: Market depth parameter (default: 1)

    Returns:
        Cost of buying q_yes shares
    """
    q_yes_init = qyes_from_pq(p, Q, b=b)
    q_no_init = Q - q_yes_init
    initial_cost = cost(q_yes=q_yes_init, q_no=q_no_init, b=b)

    return cost(q_yes=q_yes + q_yes_init, q_no=q_no_init, b=b) - initial_cost


def dcost_no(q_no, p, Q, b=1):
    """
    Cost function for buying q_no shares when the current market probability is p
    and total shares outstanding is Q.

    Args:
        q_no: Number of NO shares to buy
        p: Current market probability
        Q: Total shares outstanding (q_yes + q_no)
        b: Market depth parameter (default: 1)

    Returns:
        Cost of buying q_no shares
    """
    q_yes_init = qyes_from_pq(p, Q, b=b)
    q_no_init = Q - q_yes_init
    initial_cost = cost(q_yes=q_yes_init, q_no=q_no_init, b=b)

    return cost(q_yes=q_yes_init, q_no=q_no + q_no_init, b=b) - initial_cost


def create_order_price_schedule(
    p: float,
    half_spread_bps: float,
    max_order_bps: float,
    num_orders: int,
    min_tick_size: float = 0.001,
) -> tuple[list[float], list[float]]:
    """
    Create evenly spaced order price schedules for YES and NO orders in log space.

    This function generates price schedules for placing limit orders on both sides
    of the market. Prices are spaced evenly in log space from the initial price
    (determined by half_spread) to the maximum price (determined by max_order).

    All prices are discretized to multiples of min_tick_size. The first YES price
    is adjusted upward to the next allowed discrete level above p, and the first
    NO price is adjusted downward to the next allowed discrete level below p.

    Args:
        p: Initial market probability (between 0 and 1)
        half_spread_bps: Half spread in basis points (1 bps = 0.01%)
                        Defines the closest order to current price
                        Formula: half_spread_bps = log(p_min/p) * 10000
        max_order_bps: Maximum order distance in basis points
                      Defines the furthest order from current price
                      Formula: max_order_bps = log(p_max/p) * 10000
        num_orders: Number of orders to place on each side
        min_tick_size: Minimum price tick size for discretization (default: 0.001)
                      All prices will be rounded to multiples of this value

    Returns:
        Tuple of (yes_prices, no_prices):
        - yes_prices: List of probabilities for YES orders (ascending, all > p)
        - no_prices: List of probabilities for NO orders (descending, all < p)

    Example:
        >>> yes_prices, no_prices = create_order_price_schedule(
        ...     p=0.5,
        ...     half_spread_bps=10,      # 10 bps = 0.1% spread
        ...     max_order_bps=500,       # 500 bps = 5% max
        ...     num_orders=5,
        ...     min_tick_size=0.001
        ... )
        >>> print(f"YES orders at: {yes_prices}")
        >>> print(f"NO orders at: {no_prices}")

    Notes:
        - YES orders: Buy YES shares, increase probability, prices > p
        - NO orders: Buy NO shares, decrease probability, prices < p
        - Log spacing means more orders near current price, fewer far away
        - For p=0.5, half_spread_bps=10 gives p_min ≈ 0.4995, p_max ≈ 0.5005
        - All prices are discretized to multiples of min_tick_size
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be between 0 and 1, got {p}")
    if half_spread_bps <= 0:
        raise ValueError(f"half_spread_bps must be positive, got {half_spread_bps}")
    if max_order_bps <= half_spread_bps:
        raise ValueError(
            f"max_order_bps ({max_order_bps}) must be > half_spread_bps ({half_spread_bps})"
        )
    if num_orders < 1:
        raise ValueError(f"num_orders must be at least 1, got {num_orders}")
    if min_tick_size <= 0:
        raise ValueError(f"min_tick_size must be positive, got {min_tick_size}")

    # Convert basis points to log space
    # User specifies: half_spread_bps = 10000 * log(p_near/p)
    # So: log(p_near/p) = half_spread_bps / 10000
    # And: p_near = p * exp(half_spread_bps / 10000)

    log_half_spread = half_spread_bps / 10000
    log_max_order = max_order_bps / 10000

    # YES orders: prices increase from p (buy YES shares → prob goes up)
    # For YES: 10000 * log(p_yes / p) = bps
    # So: p_yes = p * exp(bps / 10000)

    # Minimum YES price (closest to p): half_spread_bps
    p_yes_min = p * np.exp(log_half_spread)

    # Maximum YES price (furthest from p): max_order_bps
    # User wants: 10000 * log(p_yes_max / p) = max_order_bps
    # So: p_yes_max = p * exp(max_order_bps / 10000)
    p_yes_max = p * np.exp(log_max_order)

    # Cap to ensure p_yes_max < 1
    if p_yes_max >= 1.0:
        p_yes_max = 0.9999999999  # Cap just below 1

    # Create evenly spaced points in log space
    # log(p_yes) goes from log(p_yes_min) to log(p_yes_max)
    log_p_yes = np.linspace(np.log(p_yes_min), np.log(p_yes_max), num_orders)
    yes_prices_raw = np.exp(log_p_yes)

    # Discretize YES prices to multiples of min_tick_size
    # For YES: ALWAYS round up to the NEXT allowed discrete value
    # We add a small epsilon to ensure we always move to the next tick, not the current one
    yes_prices_discretized = (
        np.ceil((yes_prices_raw + min_tick_size * 1e-9) / min_tick_size) * min_tick_size
    )

    # Ensure first YES price is strictly greater than p
    # We need to find the smallest discretized price > p
    # If p is on a tick: next tick is (p/min_tick_size + 1) * min_tick_size
    # If p is between ticks: next tick is ceil(p/min_tick_size) * min_tick_size
    # But ceil might equal p if p is on a tick, so we need: ceil(p/min_tick_size + epsilon) * min_tick_size
    # Simpler: always use floor(p/min_tick_size) + 1
    first_yes_candidate = (np.floor(p / min_tick_size + 1e-10) + 1) * min_tick_size
    if yes_prices_discretized[0] <= p:
        yes_prices_discretized[0] = first_yes_candidate

    # Cap to ensure all prices < 1
    yes_prices_discretized = np.minimum(yes_prices_discretized, 1.0 - min_tick_size)
    yes_prices = yes_prices_discretized.tolist()

    # NO orders: prices decrease from p (buy NO shares → prob goes down)
    # For NO: 10000 * log(p / p_no) = bps
    # So: p_no = p * exp(-bps / 10000)

    # Maximum NO price (closest to p): half_spread_bps
    p_no_max = p * np.exp(-log_half_spread)

    # Minimum NO price (furthest from p): max_order_bps
    # User wants: 10000 * log(p / p_no_min) = max_order_bps
    # So: log(p / p_no_min) = max_order_bps / 10000
    # So: p_no_min = p * exp(-max_order_bps / 10000)
    p_no_min = p * np.exp(-log_max_order)

    # Cap to ensure p_no_min > 0
    if p_no_min <= 0.0:
        p_no_min = 1e-10  # Cap just above 0

    # Create evenly spaced points in log space, then reverse for descending order
    # log(p_no) goes from log(p_no_min) to log(p_no_max)
    log_p_no = np.linspace(np.log(p_no_min), np.log(p_no_max), num_orders)
    no_prices_raw = np.exp(log_p_no)

    # Discretize NO prices to multiples of min_tick_size
    # For NO: ALWAYS round down to the PREVIOUS allowed discrete value
    # We subtract a small epsilon to ensure we always move to the previous tick, not the current one
    no_prices_discretized = (
        np.floor((no_prices_raw - min_tick_size * 1e-9) / min_tick_size) * min_tick_size
    )

    # Ensure first NO price is strictly less than p (after reversal, first is at end)
    # We need to find the largest discretized price < p
    # This is: (ceil(p / min_tick_size) - 1) * min_tick_size
    first_no_candidate = (np.ceil(p / min_tick_size) - 1) * min_tick_size
    if no_prices_discretized[-1] >= p:
        no_prices_discretized[-1] = first_no_candidate

    # Cap to ensure all prices > 0
    no_prices_discretized = np.maximum(no_prices_discretized, min_tick_size)
    no_prices = no_prices_discretized.tolist()
    no_prices.reverse()  # Descending order for NO market

    return yes_prices, no_prices


def yes_order_schedule(
    p_init: float,
    Q: float,
    prob_schedule: List[float],
    b: float = 1.0,
    min_order_size: float = None,
) -> pd.DataFrame:
    """
    Generate limit order schedule for YES market.

    Given an initial market state (p_init, Q, b) and a schedule of target probabilities,
    calculates the order sizes and costs for placing YES orders at each probability level.

    The key property: cumulative cost matches dcost_yes function.
    For target probability p_target, the cumulative q_yes needed is such that
    prob(q_yes_init + q_yes_cumulative, q_no_init, b) = p_target

    Args:
        p_init: Initial market probability (between 0 and 1)
        Q: Total shares outstanding (q_yes + q_no)
        prob_schedule: List of target probabilities in ascending order (all > p_init)
        b: Market depth parameter (default: 1.0)
        min_order_size: If set, rescale all incremental quantities so the smallest is exactly this value (default: None)

    Returns:
        DataFrame with columns:
        - target_prob: Target probability for this order
        - q_yes_incremental: Incremental YES shares to buy for this order
        - q_yes_cumulative: Cumulative YES shares bought so far
        - cost_incremental: Cost of this specific order
        - cost_cumulative: Total cost of all orders up to this point
        - dcost_verification: dcost_yes(q_yes_cumulative, p_init, Q, b) - should match cost_cumulative
        - prob_verification: Actual probability reached

    Raises:
        ValueError: If inputs are invalid or prob_schedule not ascending

    Example:
        >>> schedule = yes_order_schedule(p_init=0.7, Q=100, prob_schedule=[0.71, 0.8, 0.9])
        >>> print(schedule)
        target_prob  q_yes_incremental  q_yes_cumulative  cost_incremental  cost_cumulative
        0.71         0.048              0.048             0.034             0.034
        0.80         0.491              0.539             0.372             0.405
        0.90         0.811              1.350             0.693             1.099
    """
    # Validate inputs
    if not (0 < p_init < 1):
        raise ValueError(f"p_init must be between 0 and 1, got {p_init}")
    if Q <= 0:
        raise ValueError(f"Q must be positive, got {Q}")
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")

    for p_target in prob_schedule:
        if not (0 < p_target < 1):
            raise ValueError(
                f"All probabilities must be between 0 and 1, got {p_target}"
            )
        if p_target <= p_init:
            raise ValueError(
                f"All target probabilities must be > p_init={p_init}, got {p_target}"
            )

    # Check schedule is sorted
    if prob_schedule != sorted(prob_schedule):
        raise ValueError(f"prob_schedule must be in ascending order")

    # Calculate initial market state
    q_yes_init = qyes_from_pq(p_init, Q, b=b)
    q_no_init = Q - q_yes_init

    results = []
    prev_q_yes_cumulative = 0.0
    prev_cost_cumulative = 0.0

    for p_target in prob_schedule:
        # To reach p_target, we need to solve for q_yes such that:
        # prob(q_yes_init + q_yes, q_no_init, b) = p_target
        #
        # Using prob formula: exp((q_yes_init + q_yes)/b) / (exp((q_yes_init + q_yes)/b) + exp(q_no_init/b)) = p_target
        # Rearranging: exp((q_yes_init + q_yes)/b) = p_target / (1 - p_target) * exp(q_no_init/b)
        # Taking log: (q_yes_init + q_yes)/b = log(p_target / (1 - p_target)) + q_no_init/b
        # Solving: q_yes = b * log(p_target / (1 - p_target)) + q_no_init - q_yes_init

        q_yes_total = b * np.log(p_target / (1 - p_target)) + q_no_init - q_yes_init
        q_yes_cumulative = q_yes_total

        # Calculate incremental order size
        q_yes_incremental = q_yes_cumulative - prev_q_yes_cumulative

        # Calculate cost using dcost_yes
        cost_cumulative = dcost_yes(q_yes_cumulative, p_init, Q, b=b)
        cost_incremental = cost_cumulative - prev_cost_cumulative

        # Verify the probability is correct
        prob_verification = prob(q_yes_init + q_yes_cumulative, q_no_init, b=b)

        results.append(
            {
                "target_prob": p_target,
                "q_yes_incremental": q_yes_incremental,
                "q_yes_cumulative": q_yes_cumulative,
                "cost_incremental": cost_incremental,
                "cost_cumulative": cost_cumulative,
                "dcost_verification": dcost_yes(q_yes_cumulative, p_init, Q, b=b),
                "prob_verification": prob_verification,
            }
        )

        prev_q_yes_cumulative = q_yes_cumulative
        prev_cost_cumulative = cost_cumulative

    df = pd.DataFrame(results)

    # Apply min_order_size rescaling if specified
    if min_order_size is not None:
        if min_order_size <= 0:
            raise ValueError(f"min_order_size must be positive, got {min_order_size}")

        # Find the smallest incremental order
        min_incremental = df["q_yes_incremental"].min()

        if min_incremental <= 0:
            raise ValueError(
                "Cannot rescale: found non-positive incremental order size"
            )

        # Calculate scaling factor
        scale_factor = min_order_size / min_incremental

        # Rescale all quantities and costs
        df["q_yes_incremental"] = df["q_yes_incremental"] * scale_factor
        df["q_yes_cumulative"] = df["q_yes_cumulative"] * scale_factor
        df["cost_incremental"] = df["cost_incremental"] * scale_factor
        df["cost_cumulative"] = df["cost_cumulative"] * scale_factor
        df["dcost_verification"] = df["dcost_verification"] * scale_factor

    return df


def no_order_schedule(
    p_init: float,
    Q: float,
    prob_schedule: List[float],
    b: float = 1.0,
    min_order_size: float = None,
) -> pd.DataFrame:
    """
    Generate limit order schedule for NO market.

    Given an initial market state (p_init, Q, b) and a schedule of target probabilities,
    calculates the order sizes and costs for placing NO orders at each probability level.

    Note: For NO market, buying NO shares DECREASES the probability (probability of YES).
    So prob_schedule should be in descending order (all < p_init).

    The key property: cumulative cost matches dcost_no function.
    For target probability p_target, the cumulative q_no needed is such that
    prob(q_yes_init, q_no_init + q_no_cumulative, b) = p_target

    Args:
        p_init: Initial market probability (between 0 and 1)
        Q: Total shares outstanding (q_yes + q_no)
        prob_schedule: List of target probabilities in descending order (all < p_init)
        b: Market depth parameter (default: 1.0)
        min_order_size: If set, rescale all incremental quantities so the smallest is exactly this value (default: None)

    Returns:
        DataFrame with columns:
        - target_prob: Target probability for this order
        - q_no_incremental: Incremental NO shares to buy for this order
        - q_no_cumulative: Cumulative NO shares bought so far
        - cost_incremental: Cost of this specific order
        - cost_cumulative: Total cost of all orders up to this point
        - dcost_verification: dcost_no(q_no_cumulative, p_init, Q, b) - should match cost_cumulative
        - prob_verification: Actual probability reached

    Raises:
        ValueError: If inputs are invalid or prob_schedule not descending

    Example:
        >>> schedule = no_order_schedule(p_init=0.7, Q=100, prob_schedule=[0.65, 0.5, 0.3])
        >>> print(schedule)
        target_prob  q_no_incremental  q_no_cumulative  cost_incremental  cost_cumulative
        0.65         0.228             0.228             0.074             0.074
        0.50         0.619             0.847             0.262             0.336
        0.30         0.847             1.695             0.511             0.847
    """
    # Validate inputs
    if not (0 < p_init < 1):
        raise ValueError(f"p_init must be between 0 and 1, got {p_init}")
    if Q <= 0:
        raise ValueError(f"Q must be positive, got {Q}")
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")

    for p_target in prob_schedule:
        if not (0 < p_target < 1):
            raise ValueError(
                f"All probabilities must be between 0 and 1, got {p_target}"
            )
        if p_target >= p_init:
            raise ValueError(
                f"All target probabilities must be < p_init={p_init}, got {p_target}"
            )

    # Check schedule is sorted descending
    if prob_schedule != sorted(prob_schedule, reverse=True):
        raise ValueError(f"prob_schedule must be in descending order for NO market")

    # Calculate initial market state
    q_yes_init = qyes_from_pq(p_init, Q, b=b)
    q_no_init = Q - q_yes_init

    results = []
    prev_q_no_cumulative = 0.0
    prev_cost_cumulative = 0.0

    for p_target in prob_schedule:
        # To reach p_target, we need to solve for q_no such that:
        # prob(q_yes_init, q_no_init + q_no, b) = p_target
        #
        # Using prob formula: exp(q_yes_init/b) / (exp(q_yes_init/b) + exp((q_no_init + q_no)/b)) = p_target
        # Rearranging: exp(q_yes_init/b) = p_target * (exp(q_yes_init/b) + exp((q_no_init + q_no)/b))
        # Simplifying: exp(q_yes_init/b) * (1 - p_target) = p_target * exp((q_no_init + q_no)/b)
        # Taking log: q_yes_init/b + log(1 - p_target) - log(p_target) = (q_no_init + q_no)/b
        # Solving: q_no = b * (q_yes_init/b + log((1 - p_target) / p_target)) - q_no_init

        q_no_total = q_yes_init + b * np.log((1 - p_target) / p_target) - q_no_init
        q_no_cumulative = q_no_total

        # Calculate incremental order size
        q_no_incremental = q_no_cumulative - prev_q_no_cumulative

        # Calculate cost using dcost_no
        cost_cumulative = dcost_no(q_no_cumulative, p_init, Q, b=b)
        cost_incremental = cost_cumulative - prev_cost_cumulative

        # Verify the probability is correct
        prob_verification = prob(q_yes_init, q_no_init + q_no_cumulative, b=b)

        results.append(
            {
                "target_prob": p_target,
                "q_no_incremental": q_no_incremental,
                "q_no_cumulative": q_no_cumulative,
                "cost_incremental": cost_incremental,
                "cost_cumulative": cost_cumulative,
                "dcost_verification": dcost_no(q_no_cumulative, p_init, Q, b=b),
                "prob_verification": prob_verification,
            }
        )

        prev_q_no_cumulative = q_no_cumulative
        prev_cost_cumulative = cost_cumulative

    df = pd.DataFrame(results)

    # Apply min_order_size rescaling if specified
    if min_order_size is not None:
        if min_order_size <= 0:
            raise ValueError(f"min_order_size must be positive, got {min_order_size}")

        # Find the smallest incremental order
        min_incremental = df["q_no_incremental"].min()

        if min_incremental <= 0:
            raise ValueError(
                "Cannot rescale: found non-positive incremental order size"
            )

        # Calculate scaling factor
        scale_factor = min_order_size / min_incremental

        # Rescale all quantities and costs
        df["q_no_incremental"] = df["q_no_incremental"] * scale_factor
        df["q_no_cumulative"] = df["q_no_cumulative"] * scale_factor
        df["cost_incremental"] = df["cost_incremental"] * scale_factor
        df["cost_cumulative"] = df["cost_cumulative"] * scale_factor
        df["dcost_verification"] = df["dcost_verification"] * scale_factor

    return df


def create_lob_data(
    probability: float,
    capital: float,
    Q: float = 1000,
    B: float = 10000,
    half_spread_bps: float = 5,
    max_order_bps: float = 500,
    num_orders_coarse: int = 5,
    num_orders_fine: int = 50,
) -> dict:
    """
    Create LOB (Limit Order Book) data for a single market.

    This function generates both coarse (discrete points) and fine (continuous line)
    limit order book schedules for visualizing capital allocation across price levels.

    Args:
        probability: Market probability (between 0 and 1)
        capital: Allocated capital for this market
        Q: Total shares outstanding (default: 1000)
        B: Market depth parameter (default: 10000)
        half_spread_bps: Half spread in basis points (default: 5)
        max_order_bps: Maximum order distance in basis points (default: 500)
        num_orders_coarse: Number of discrete order points (default: 5)
        num_orders_fine: Number of points for fine schedule line (default: 50)

    Returns:
        dict with:
        - 'coarse_prices': Discrete price points (for scatter plot)
        - 'coarse_lob': LOB values at discrete points (for scatter plot)
        - 'fine_prices': Fine price points (for line plot)
        - 'fine_lob': LOB values for fine schedule (for line plot)
        - 'scale_factor': Capital scaling factor

    Example:
        >>> lob_data = create_lob_data(probability=0.635, capital=1000)
        >>> print(f"Scale factor: {lob_data['scale_factor']:.2f}")
        >>> # Use lob_data['coarse_prices'], lob_data['coarse_lob'] for scatter plot
        >>> # Use lob_data['fine_prices'], lob_data['fine_lob'] for line plot
    """
    # Create coarse schedule for discrete points (dots)
    yes_schedule_coarse, no_schedule_coarse = create_order_price_schedule(
        p=probability,
        half_spread_bps=half_spread_bps,
        max_order_bps=max_order_bps,
        num_orders=num_orders_coarse,
    )

    cum_yes_order_cost_coarse = yes_order_schedule(
        probability, Q, yes_schedule_coarse, b=B
    )["cost_cumulative"].values

    cum_no_order_cost_coarse = no_order_schedule(
        probability, Q, no_schedule_coarse, b=B
    )["cost_cumulative"].values

    # Calculate scale factor from coarse schedule
    unscaled_capital = max(cum_yes_order_cost_coarse) + max(cum_no_order_cost_coarse)
    scale_factor = capital / unscaled_capital

    # Coarse schedule for dots
    coarse_prices = np.concatenate([no_schedule_coarse[::-1], yes_schedule_coarse])
    coarse_lob = (
        np.concatenate([cum_no_order_cost_coarse[::-1], cum_yes_order_cost_coarse])
        * scale_factor
    )

    # Create fine schedule for line
    yes_schedule_fine, no_schedule_fine = create_order_price_schedule(
        p=probability,
        half_spread_bps=half_spread_bps,
        max_order_bps=max_order_bps,
        num_orders=num_orders_fine,
        min_tick_size=1e-6,
    )

    cum_yes_order_cost_fine = yes_order_schedule(
        probability, Q, yes_schedule_fine, b=B
    )["cost_cumulative"].values

    cum_no_order_cost_fine = no_order_schedule(probability, Q, no_schedule_fine, b=B)[
        "cost_cumulative"
    ].values

    # Fine schedule for line (using same scale factor)
    fine_prices = np.concatenate([no_schedule_fine[::-1], yes_schedule_fine])
    fine_lob = (
        np.concatenate([cum_no_order_cost_fine[::-1], cum_yes_order_cost_fine])
        * scale_factor
    )

    return {
        "coarse_prices": coarse_prices,
        "coarse_lob": coarse_lob,
        "fine_prices": fine_prices,
        "fine_lob": fine_lob,
        "scale_factor": scale_factor,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("YES Market Example")
    print("=" * 80)
    print("\nInitial conditions: p=0.7, Q=100, b=1")
    print("Target probabilities: [0.71, 0.8, 0.9]")
    print()

    yes_schedule = yes_order_schedule(
        p_init=0.7, Q=100, prob_schedule=[0.71, 0.8, 0.9], b=1.0
    )

    print(yes_schedule.to_string(index=False))
    print()
    print("Verification: cost_cumulative should match dcost_verification")
    print(
        f"Max difference: {(yes_schedule['cost_cumulative'] - yes_schedule['dcost_verification']).abs().max():.2e}"
    )
    print(
        f"Max prob error: {(yes_schedule['target_prob'] - yes_schedule['prob_verification']).abs().max():.2e}"
    )

    print("\n" + "=" * 80)
    print("NO Market Example")
    print("=" * 80)
    print("\nInitial conditions: p=0.7, Q=100, b=1")
    print("Target probabilities: [0.65, 0.5, 0.3]")
    print()

    no_schedule = no_order_schedule(
        p_init=0.7, Q=100, prob_schedule=[0.65, 0.5, 0.3], b=1.0
    )

    print(no_schedule.to_string(index=False))
    print()
    print("Verification: cost_cumulative should match dcost_verification")
    print(
        f"Max difference: {(no_schedule['cost_cumulative'] - no_schedule['dcost_verification']).abs().max():.2e}"
    )
    print(
        f"Max prob error: {(no_schedule['target_prob'] - no_schedule['prob_verification']).abs().max():.2e}"
    )
