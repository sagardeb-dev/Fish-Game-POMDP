"""
Config dicts that parameterize the POMDP model and the simulator.

Three configs:
  EASY_CONFIG  — all zones visible, well-separated distributions
  HARD_CONFIG  — noisier distributions, tighter budgets, all zones visible
  BENCHMARK_CONFIG — HARD distributions + only 2 of 4 zones report sensors/day

CONFIG = BENCHMARK_CONFIG (default used by runner, baselines, tests).

V5: 80 states = 2(storm) x 4(wind) x 5(equip_failure) x 2(tide)
  4 zones: A, B, C, D (ring: A-B-C-D-A)
  2 independent risks: storm + equipment failure
  4 causal traps: wave propagation, age-confounded equipment,
                  fish abundance bonus, water temperature confound
"""

import itertools

# ============================================================================
# Helper: generate 80 states as 4-tuples
# ============================================================================

def _generate_states():
    """Generate all (storm, wind, equip_failure, tide) 4-tuples.

    storm: 0 or 1
    wind: "N", "S", "E", "W"
    equip_failure: 0 (none), 1 (A), 2 (B), 3 (C), 4 (D)
    tide: 0 (low), 1 (high)

    Returns list of 80 tuples.
    """
    storms = [0, 1]
    winds = ["N", "S", "E", "W"]
    equips = [0, 1, 2, 3, 4]
    tides = [0, 1]
    states = [(s, w, e, t) for s in storms for w in winds for e in equips for t in tides]
    return states


def _generate_valid_allocations(max_boats=10, zones=None):
    """Generate all valid boat allocations across zones.

    An allocation is a dict {zone: n_boats} where sum in [1, max_boats]
    and each value >= 0.

    For 4 zones, max_boats=10: ~715 allocations.
    """
    zones = zones or ["A", "B", "C", "D"]
    n_zones = len(zones)
    allocations = []
    for total in range(1, max_boats + 1):
        # All ways to split `total` into n_zones non-negative parts
        for combo in itertools.combinations_with_replacement(range(n_zones), total):
            counts = [0] * n_zones
            for idx in combo:
                counts[idx] += 1
            alloc = {zones[i]: counts[i] for i in range(n_zones)}
            if alloc not in allocations:
                allocations.append(alloc)
    return allocations


_STATES = _generate_states()
_VALID_ALLOCATIONS = _generate_valid_allocations()


# ============================================================================
# Zone mappings
# ============================================================================

WIND_TO_ZONE = {"N": "A", "S": "B", "E": "C", "W": "D"}
EQUIP_TO_ZONE = {0: None, 1: "A", 2: "B", 3: "C", 4: "D"}
ZONE_TO_EQUIP = {"A": 1, "B": 2, "C": 3, "D": 4}

# Zone adjacency ring: A-B-C-D-A (for wave propagation)
ZONE_ADJACENCY = {
    "A": {"A": 0, "B": 1, "C": 2, "D": 1},
    "B": {"A": 1, "B": 0, "C": 1, "D": 2},
    "C": {"A": 2, "B": 1, "C": 0, "D": 1},
    "D": {"A": 1, "B": 2, "C": 1, "D": 0},
}


# ============================================================================
# Easy CONFIG — all zones visible, well-separated distributions
# ============================================================================

EASY_CONFIG = {
    # --- States: 80 = 2(storm) x 4(wind) x 5(equip) x 2(tide) ---
    "states": _STATES,

    # --- Factored transition sub-matrices ---
    "storm_transition": [
        [0.85, 0.15],   # from no-storm
        [0.30, 0.70],   # from storm
    ],
    "wind_transition": [
        #  N     S     E     W
        [0.65, 0.10, 0.15, 0.10],  # from N
        [0.10, 0.65, 0.10, 0.15],  # from S
        [0.15, 0.10, 0.65, 0.10],  # from E
        [0.10, 0.15, 0.10, 0.65],  # from W
    ],
    "equip_transition": [
        # none   A     B     C     D
        [0.80, 0.05, 0.05, 0.05, 0.05],  # from none
        [0.40, 0.45, 0.05, 0.05, 0.05],  # from A broken
        [0.40, 0.05, 0.45, 0.05, 0.05],  # from B broken
        [0.40, 0.05, 0.05, 0.45, 0.05],  # from C broken
        [0.40, 0.05, 0.05, 0.05, 0.45],  # from D broken
    ],
    "tide_transition": [
        [0.70, 0.30],   # from low tide
        [0.35, 0.65],   # from high tide
    ],

    # --- Tide labels and bonus ---
    "tide_to_label": {0: "low", 1: "high"},
    "tide_bonus": {0: 0, 1: 2},  # per-boat bonus for high tide on safe zones

    # --- Zone mappings ---
    "wind_to_zone": WIND_TO_ZONE,
    "equip_to_zone": EQUIP_TO_ZONE,
    "zone_to_equip": ZONE_TO_EQUIP,
    "zone_adjacency": ZONE_ADJACENCY,

    # --- Observation distributions ---
    # Sea color: P(color | storm) -- senses storm only
    "sea_color_probs": {
        0: {"green": 0.70, "murky": 0.25, "dark": 0.05},  # storm=0
        1: {"green": 0.05, "murky": 0.35, "dark": 0.60},  # storm=1
    },

    # Equipment indicator: P(level | equip_failure) -- senses any equip failure
    "equip_indicator_probs": {
        0: {"normal": 0.80, "warning": 0.15, "critical": 0.05},  # no failure
        1: {"normal": 0.10, "warning": 0.35, "critical": 0.55},  # any failure (equip>0)
    },

    # Barometer: Normal(mean, std) conditioned on storm
    "barometer_params": {
        0: {"mean": 1013.0, "std": 3.0},
        1: {"mean": 998.0, "std": 5.0},
    },

    # Buoy: Normal(mean, std) with wave propagation model
    # Reading depends on distance from storm source zone
    "buoy_params": {
        "normal":         {"mean": 1.2, "std": 0.3},   # no storm anywhere
        "source":         {"mean": 4.5, "std": 0.4},   # distance 0: storm source zone
        "propagated":     {"mean": 2.8, "std": 0.5},   # distance 1: adjacent to storm
        "far_propagated": {"mean": 1.6, "std": 0.4},   # distance 2: opposite to storm
    },

    # Equipment inspection: Normal(mean, std) with age confound
    # Reading = Normal(base + age * age_offset_factor, std)
    "equipment_inspection_params": {
        "broken": {"mean": 8.5, "std": 1.0},   # zone has equipment failure (before age offset)
        "ok":     {"mean": 2.0, "std": 0.5},   # zone ok (before age offset)
    },
    "equipment_age_offset_factor": 0.1,  # age * this = offset added to reading

    # Infrastructure age per zone (confounder for equipment readings)
    "zone_infrastructure_age": {"A": 25, "B": 15, "C": 5, "D": 2},

    # Maintenance alerts: Poisson(age * rate_factor + failure_signal)
    "maintenance_alert_params": {
        "age_rate_factor": 0.3,   # base rate per year of age
        "failure_signal": 5.0,    # additional rate when zone actually broken
    },

    # Water temperature: Normal(base + tide_effect * tide + zone_offset, std)
    # Causal trap 4: zone age confounds water temp (zone A always warm)
    "water_temp_params": {
        "base": {"mean": 15.0, "std": 1.0},
        "tide_effect": 1.5,         # high tide adds this
    },
    "zone_temp_offset": {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.2},

    # Fish abundance bonus: extra profit for zones adjacent to storm (currents bring fish)
    "fish_abundance_bonus": {0: 0, 1: 3, 2: 0},  # keyed by distance from storm

    # --- Weather signal emission (storm-based) ---
    "signal_tiers": {
        1: {
            "emission_prob": {0: 0.03, 1: 0.80},
            "headlines": [
                "STORM WARNING: gale force winds reported offshore",
                "Coast guard issues severe weather advisory for fishing zones",
                "Harbor master suspends departures due to storm conditions",
            ],
        },
        2: {
            "emission_prob": {0: 0.10, 1: 0.55},
            "headlines": [
                "Northern fleet reports unusual swell patterns this week",
                "Insurance premiums for fishing vessels trending upward",
                "Barometric pressure readings inconsistent across coastal stations",
            ],
        },
        3: {
            "emission_prob_always": 0.20,
            "headlines": [
                "Annual fishing quota review scheduled for next month",
                "New sonar technology tested by research vessel",
                "Local fishing tournament postponed for unrelated reasons",
                "Marine biologists report unusual jellyfish migration",
            ],
        },
    },

    # --- Equipment signal emission (equip-failure-based) ---
    "equipment_signal_tiers": {
        1: {
            "emission_prob": {0: 0.03, 1: 0.75},
            "headlines": [
                "EQUIPMENT ALERT: Critical net failure reported in fishing fleet",
                "Maintenance crew reports severe gear malfunction on vessels",
                "Emergency equipment inspection ordered for fishing boats",
            ],
        },
        2: {
            "emission_prob": {0: 0.08, 1: 0.50},
            "headlines": [
                "Fleet maintenance logs show unusual wear patterns",
                "Equipment insurance claims rising across fishing operations",
                "Vessel inspection backlogs causing concern among operators",
            ],
        },
        3: {
            "emission_prob_always": 0.20,
            "headlines": [
                "New fishing gear technology showcased at maritime expo",
                "Annual equipment certification process begins next quarter",
                "Fishing industry trade group publishes maintenance guidelines",
                "Marine equipment supplier announces new product line",
            ],
        },
    },

    "signal_sources": ["coast_guard", "market_data", "industry_news", "social_media"],
    "signals_per_step_range": (2, 5),

    # --- Reward function ---
    "safe_profit_per_boat": 7,
    "danger_loss_per_boat": -18,        # storm only
    "danger_loss_equip_per_boat": -10,  # equipment failure only
    "danger_loss_both_per_boat": -25,   # storm + equipment failure

    # --- Episode parameters ---
    "episode_length": 20,
    "historical_days": 30,
    "max_boats": 10,
    "zones": ["A", "B", "C", "D"],

    # --- Valid allocations (precomputed) ---
    "valid_allocations": _VALID_ALLOCATIONS,

    # --- Tool budgets (per day, no rollover) ---
    # Raw sensors (barometer, buoys, inspections, maintenance alerts, water_temp) are FREE.
    # Only interpretive/analytical tools are budget-gated.
    "tool_budgets": {
        "check_weather_reports": 2,
        "check_equipment_reports": 2,
        "query_fishing_log": 2,
        "query_maintenance_log": 2,
        "analyze_data": 1,
        "evaluate_options": 1,
        "forecast_scenario": 1,
    },

    # --- Initial belief (uniform over 80 states) ---
    "initial_belief": [1.0 / 80] * 80,
}


# ============================================================================
# Hard mode: noisier sensors, tighter budgets
# ============================================================================

HARD_CONFIG = {
    **EASY_CONFIG,

    # Storms less persistent, wind shifts more
    "storm_transition": [
        [0.75, 0.25],
        [0.40, 0.60],
    ],
    "wind_transition": [
        [0.50, 0.15, 0.20, 0.15],
        [0.15, 0.50, 0.15, 0.20],
        [0.20, 0.15, 0.50, 0.15],
        [0.15, 0.20, 0.15, 0.50],
    ],
    "equip_transition": [
        [0.65, 0.10, 0.08, 0.09, 0.08],
        [0.30, 0.50, 0.07, 0.06, 0.07],
        [0.30, 0.07, 0.50, 0.07, 0.06],
        [0.30, 0.06, 0.07, 0.50, 0.07],
        [0.30, 0.07, 0.06, 0.07, 0.50],
    ],
    # Tide transition same as easy
    "tide_transition": [
        [0.70, 0.30],
        [0.35, 0.65],
    ],

    # Tide bonus smaller in hard mode
    "tide_bonus": {0: 0, 1: 1},

    # Sea color less diagnostic
    "sea_color_probs": {
        0: {"green": 0.55, "murky": 0.35, "dark": 0.10},
        1: {"green": 0.10, "murky": 0.45, "dark": 0.45},
    },

    # Equipment indicator noisier
    "equip_indicator_probs": {
        0: {"normal": 0.60, "warning": 0.30, "critical": 0.10},
        1: {"normal": 0.20, "warning": 0.40, "critical": 0.40},
    },

    # Barometer: overlapping distributions
    "barometer_params": {
        0: {"mean": 1010.0, "std": 6.0},
        1: {"mean": 1002.0, "std": 6.0},
    },

    # Buoy: noisier propagation (tighter spreads make source harder to identify)
    "buoy_params": {
        "normal":         {"mean": 1.5, "std": 0.6},
        "source":         {"mean": 3.5, "std": 0.8},
        "propagated":     {"mean": 2.5, "std": 0.7},
        "far_propagated": {"mean": 1.8, "std": 0.6},
    },

    # Equipment inspection: noisier + age confound
    "equipment_inspection_params": {
        "broken": {"mean": 6.5, "std": 1.5},
        "ok":     {"mean": 3.0, "std": 1.0},
    },
    "equipment_age_offset_factor": 0.15,  # stronger age confound in hard mode

    # Water temperature: noisier, smaller tide effect, stronger confound
    "water_temp_params": {
        "base": {"mean": 15.0, "std": 1.5},
        "tide_effect": 1.0,
    },
    "zone_temp_offset": {"A": 1.5, "B": 0.8, "C": 0.0, "D": -0.3},

    # Weather signals noisier
    "signal_tiers": {
        1: {
            "emission_prob": {0: 0.08, 1: 0.60},
            "headlines": EASY_CONFIG["signal_tiers"][1]["headlines"],
        },
        2: {
            "emission_prob": {0: 0.15, 1: 0.45},
            "headlines": EASY_CONFIG["signal_tiers"][2]["headlines"],
        },
        3: {
            "emission_prob_always": 0.30,
            "headlines": EASY_CONFIG["signal_tiers"][3]["headlines"],
        },
    },

    # Equipment signals noisier
    "equipment_signal_tiers": {
        1: {
            "emission_prob": {0: 0.08, 1: 0.55},
            "headlines": EASY_CONFIG["equipment_signal_tiers"][1]["headlines"],
        },
        2: {
            "emission_prob": {0: 0.15, 1: 0.40},
            "headlines": EASY_CONFIG["equipment_signal_tiers"][2]["headlines"],
        },
        3: {
            "emission_prob_always": 0.30,
            "headlines": EASY_CONFIG["equipment_signal_tiers"][3]["headlines"],
        },
    },

    # Tighter tool budgets
    "tool_budgets": {
        "check_weather_reports": 1,
        "check_equipment_reports": 1,
        "query_fishing_log": 1,
        "query_maintenance_log": 1,
        "analyze_data": 1,
        "evaluate_options": 1,
        "forecast_scenario": 1,
    },
}


# ============================================================================
# Benchmark CONFIG — HARD distributions + limited sensor zone visibility
# ============================================================================

BENCHMARK_CONFIG = {
    **HARD_CONFIG,

    # Only 2 of 4 zones report sensors each day (randomly selected).
    # Forces agents to rely on transition model between steps.
    "sensor_zones_per_step": 2,

    # Revert sea_color and equip_indicator to HARD_CONFIG levels.
    # Current BENCHMARK raised these false positive rates unnecessarily.
    "sea_color_probs": {
        0: {"green": 0.55, "murky": 0.35, "dark": 0.10},
        1: {"green": 0.10, "murky": 0.45, "dark": 0.45},
    },
    "equip_indicator_probs": {
        0: {"normal": 0.60, "warning": 0.30, "critical": 0.10},
        1: {"normal": 0.20, "warning": 0.40, "critical": 0.40},
    },
}

# Default CONFIG used everywhere: benchmark difficulty
CONFIG = BENCHMARK_CONFIG
