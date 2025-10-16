# Limit Order Placement Schedule Generator

This module provides functions to generate optimal limit order placement schedules for betting markets using the cost functions from `make_market.py`.

## Overview

Given an initial market state `(p, Q, b)` and a desired schedule of target probabilities, these functions calculate exactly how many shares to buy at each price level such that the cumulative cost matches the `dcost_yes` and `dcost_no` functions.

## Key Concept

In a logarithmic market scoring rule (LMSR) betting market:
- **YES orders** increase the probability (buy YES shares → probability goes up)
- **NO orders** decrease the probability (buy NO shares → probability goes down)

The functions solve the inverse problem: given target probabilities, what order sizes are needed?

## Installation

No additional dependencies required beyond the base project requirements:
```bash
pip install -r requirements.txt
```

## Usage

### YES Market Orders

```python
from limit_order_schedule import yes_order_schedule

# Initial market: p=0.7, Q=100, b=1
# Want to place orders at probabilities: 0.71, 0.8, 0.9
schedule = yes_order_schedule(
    p_init=0.7,
    Q=100,
    prob_schedule=[0.71, 0.8, 0.9],
    b=1.0
)

print(schedule)
```

Output:
```
 target_prob  q_yes_incremental  q_yes_cumulative  cost_incremental  cost_cumulative
        0.71           0.048086          0.048086          0.033902         0.033902
        0.80           0.490910          0.538997          0.371564         0.405465
        0.90           0.810930          1.349927          0.693147         1.098612
```

**Interpretation:**
- To move probability from 0.7 → 0.71: buy 0.048 YES shares, costs 0.034
- To move probability from 0.71 → 0.8: buy 0.491 more YES shares, costs 0.372
- To move probability from 0.8 → 0.9: buy 0.811 more YES shares, costs 0.693
- Total cost to reach 0.9: 1.099 (matches `dcost_yes(1.350, p=0.7, Q=100)`)

### NO Market Orders

```python
from limit_order_schedule import no_order_schedule

# Initial market: p=0.7, Q=100, b=1
# Want to place orders at probabilities: 0.65, 0.5, 0.3 (descending!)
schedule = no_order_schedule(
    p_init=0.7,
    Q=100,
    prob_schedule=[0.65, 0.5, 0.3],
    b=1.0
)

print(schedule)
```

Output:
```
 target_prob  q_no_incremental  q_no_cumulative  cost_incremental  cost_cumulative
        0.65          0.228259         0.228259          0.074108         0.074108
        0.50          0.619039         0.847298          0.262364         0.336472
        0.30          0.847298         1.694596          0.510826         0.847298
```

**Interpretation:**
- To move probability from 0.7 → 0.65: buy 0.228 NO shares, costs 0.074
- To move probability from 0.65 → 0.5: buy 0.619 more NO shares, costs 0.262
- To move probability from 0.5 → 0.3: buy 0.847 more NO shares, costs 0.511
- Total cost to reach 0.3: 0.847 (matches `dcost_no(1.695, p=0.7, Q=100)`)

## Function Reference

### `yes_order_schedule(p_init, Q, prob_schedule, b=1.0)`

Generate limit order schedule for YES market.

**Parameters:**
- `p_init` (float): Initial market probability, between 0 and 1
- `Q` (float): Total shares outstanding (q_yes + q_no)
- `prob_schedule` (List[float]): Target probabilities in **ascending** order, all > p_init
- `b` (float): Market depth parameter (default: 1.0)

**Returns:**
- DataFrame with columns:
  - `target_prob`: Target probability for this order
  - `q_yes_incremental`: Incremental YES shares to buy
  - `q_yes_cumulative`: Cumulative YES shares bought so far
  - `cost_incremental`: Cost of this specific order
  - `cost_cumulative`: Total cost up to this point
  - `dcost_verification`: Verification that cost matches `dcost_yes()`
  - `prob_verification`: Verification that probability is reached

**Raises:**
- `ValueError`: If inputs are invalid or prob_schedule not ascending

### `no_order_schedule(p_init, Q, prob_schedule, b=1.0)`

Generate limit order schedule for NO market.

**Parameters:**
- `p_init` (float): Initial market probability, between 0 and 1
- `Q` (float): Total shares outstanding (q_yes + q_no)
- `prob_schedule` (List[float]): Target probabilities in **descending** order, all < p_init
- `b` (float): Market depth parameter (default: 1.0)

**Returns:**
- DataFrame with columns:
  - `target_prob`: Target probability for this order
  - `q_no_incremental`: Incremental NO shares to buy
  - `q_no_cumulative`: Cumulative NO shares bought so far
  - `cost_incremental`: Cost of this specific order
  - `cost_cumulative`: Total cost up to this point
  - `dcost_verification`: Verification that cost matches `dcost_no()`
  - `prob_verification`: Verification that probability is reached

**Raises:**
- `ValueError`: If inputs are invalid or prob_schedule not descending

## Mathematical Foundation

### LMSR Probability Formula

The market probability is determined by the ratio of shares:

```
p = exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))
```

### Solving for Required Shares

**For YES market** (increasing probability from p_init to p_target):

Given initial state `(q_yes_init, q_no_init)` from `p_init`, we need to find `q_yes` such that:

```
prob(q_yes_init + q_yes, q_no_init, b) = p_target
```

Solving:
```
q_yes = b * log(p_target / (1 - p_target)) + q_no_init - q_yes_init
```

**For NO market** (decreasing probability from p_init to p_target):

Given initial state `(q_yes_init, q_no_init)` from `p_init`, we need to find `q_no` such that:

```
prob(q_yes_init, q_no_init + q_no, b) = p_target
```

Solving:
```
q_no = q_yes_init + b * log((1 - p_target) / p_target) - q_no_init
```

### Cost Verification

The cumulative cost is verified against the `dcost` functions:

```python
# For YES market
cost_cumulative = dcost_yes(q_yes_cumulative, p_init, Q, b)

# For NO market
cost_cumulative = dcost_no(q_no_cumulative, p_init, Q, b)
```

These match to numerical precision (< 10^-10).

## Practical Applications

### Market Making Strategy

Place limit orders at specific probability levels to provide liquidity:

```python
# Start market at p=0.5
# Place YES orders at 0.55, 0.6, 0.65, 0.7
yes_sched = yes_order_schedule(0.5, 100, [0.55, 0.6, 0.65, 0.7])

# Place NO orders at 0.45, 0.4, 0.35, 0.3
no_sched = no_order_schedule(0.5, 100, [0.45, 0.4, 0.35, 0.3])

# Create limit order book
for _, row in yes_sched.iterrows():
    place_limit_order(
        side="YES",
        size=row['q_yes_incremental'],
        price=row['target_prob']
    )

for _, row in no_sched.iterrows():
    place_limit_order(
        side="NO",
        size=row['q_no_incremental'],
        price=row['target_prob']
    )
```

### Cost Planning

Calculate exact cost to move market to target probability:

```python
# How much does it cost to push market from 0.6 to 0.9?
schedule = yes_order_schedule(0.6, 100, [0.9])
total_cost = schedule.iloc[0]['cost_cumulative']
print(f"Cost to move from 0.6 to 0.9: ${total_cost:.2f}")
```

### Gradual Position Building

Build a position gradually across probability levels:

```python
# Want to buy YES shares, spread across levels
targets = [0.51, 0.52, 0.53, 0.54, 0.55]
schedule = yes_order_schedule(0.5, 100, targets)

# Shows incremental costs at each level
print(schedule[['target_prob', 'q_yes_incremental', 'cost_incremental']])
```

## Validation

The implementation includes comprehensive validation:

1. **Cost Verification**: Cumulative cost matches `dcost_yes`/`dcost_no` exactly
2. **Probability Verification**: Target probabilities are reached within numerical precision
3. **Incremental Sums**: Incremental values sum to cumulative values
4. **Input Validation**: All parameters are validated (ranges, ordering, etc.)

Run tests:
```bash
pytest tests/test_limit_order_schedule.py -v
```

All tests pass with:
- Cost matching: exact (difference < 10^-10)
- Probability matching: within numerical precision (< 10^-15)

## Important Notes

### YES vs NO Market Conventions

- **YES market**: `prob_schedule` must be **ascending** (all > p_init)
  - Buying YES shares increases probability
  - Example: [0.71, 0.8, 0.9] when p_init=0.7

- **NO market**: `prob_schedule` must be **descending** (all < p_init)
  - Buying NO shares decreases probability
  - Example: [0.65, 0.5, 0.3] when p_init=0.7

### Market Depth Parameter (b)

- Larger `b` → more liquid market (smaller price impact)
- Smaller `b` → less liquid market (larger price impact)
- Default: `b=1.0`

### Relationship to dcost Functions

The key property maintained:

```python
# For YES market
assert cost_cumulative == dcost_yes(q_yes_cumulative, p_init, Q, b)

# For NO market
assert cost_cumulative == dcost_no(q_no_cumulative, p_init, Q, b)
```

This ensures consistency with the market maker's cost functions.

## Examples

See `limit_order_schedule.py` for runnable examples:

```bash
python limit_order_schedule.py
```

## See Also

- `make_market.py` - Core LMSR market maker functions
- `tests/test_limit_order_schedule.py` - Comprehensive test suite
