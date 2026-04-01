"""
Single config dict that parameterizes both the POMDP model and the simulator.
Every probability, reward value, observation parameter, signal template,
and tool budget derives from this one config. Change it here, it changes everywhere.
"""

CONFIG = {
    # --- States ---
    # 4 states: (storm, wind) where storm in {0,1}, wind in {"N","S"}
    "states": [(0, "N"), (0, "S"), (1, "N"), (1, "S")],
    # state index mapping: 0=(0,N), 1=(0,S), 2=(1,N), 3=(1,S)

    # --- Transition matrix T(s'|s), row = from, col = to ---
    # Order: (0,N), (0,S), (1,N), (1,S)
    "transition_matrix": [
        [0.68, 0.17, 0.12, 0.03],  # from (0,N)
        [0.17, 0.68, 0.03, 0.12],  # from (0,S)
        [0.16, 0.04, 0.64, 0.16],  # from (1,N)
        [0.04, 0.16, 0.16, 0.64],  # from (1,S)
    ],

    # --- Observation distributions ---
    # Sea color: P(color | storm)
    "sea_color_probs": {
        0: {"green": 0.70, "murky": 0.25, "dark": 0.05},  # storm=0
        1: {"green": 0.05, "murky": 0.35, "dark": 0.60},  # storm=1
    },

    # Barometer: Normal(mean, std) conditioned on storm
    "barometer_params": {
        0: {"mean": 1013.0, "std": 3.0},
        1: {"mean": 998.0, "std": 5.0},
    },

    # Buoy: Normal(mean, std) conditioned on (zone_match AND storm)
    "buoy_params": {
        "danger": {"mean": 4.0, "std": 0.5},   # zone=affected AND storm=1
        "normal": {"mean": 1.2, "std": 0.3},    # otherwise
    },

    # --- Signal emission ---
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
    "signal_sources": ["coast_guard", "market_data", "industry_news", "social_media"],
    "signals_per_step_range": (2, 5),  # emit 2-5 signals per step

    # --- Reward function ---
    "safe_profit_per_boat": 7,
    "danger_loss_per_boat": -18,

    # --- Episode parameters ---
    "episode_length": 20,
    "max_boats": 3,
    "zones": ["A", "B"],

    # --- Tool budgets (per day, no rollover) ---
    "tool_budgets": {
        "check_weather_reports": 2,
        "query_fishing_log": 2,
        "read_barometer": 1,
        "read_buoy": 2,
        "analyze_data": 1,
        "evaluate_options": 1,
        "forecast_scenario": 1,
    },

    # --- Initial belief (uniform over 4 states) ---
    "initial_belief": [0.25, 0.25, 0.25, 0.25],

}


# Hard mode: noisier sensors, tighter budgets, less predictable transitions
HARD_CONFIG = {
    **CONFIG,

    # Storms less persistent (60% vs 80%), wind shifts more often
    "transition_matrix": [
        [0.60, 0.20, 0.12, 0.08],  # from (0,N)
        [0.20, 0.60, 0.08, 0.12],  # from (0,S)
        [0.20, 0.10, 0.40, 0.30],  # from (1,N) — storm breaks more, wind shifts more
        [0.10, 0.20, 0.30, 0.40],  # from (1,S)
    ],

    # Sea color less diagnostic
    "sea_color_probs": {
        0: {"green": 0.55, "murky": 0.35, "dark": 0.10},  # more dark false positives
        1: {"green": 0.10, "murky": 0.45, "dark": 0.45},  # less dark when stormy
    },

    # Barometer: overlapping distributions — can't tell from one reading
    "barometer_params": {
        0: {"mean": 1010.0, "std": 6.0},
        1: {"mean": 1002.0, "std": 6.0},
    },

    # Buoy: noisier, overlapping — danger zone not obvious
    "buoy_params": {
        "danger": {"mean": 2.8, "std": 0.8},
        "normal": {"mean": 1.5, "std": 0.6},
    },

    # Signals: more noise tier, less reliable storm warnings
    "signal_tiers": {
        1: {
            "emission_prob": {0: 0.08, 1: 0.60},  # more false positives, fewer true positives
            "headlines": CONFIG["signal_tiers"][1]["headlines"],
        },
        2: {
            "emission_prob": {0: 0.15, 1: 0.45},
            "headlines": CONFIG["signal_tiers"][2]["headlines"],
        },
        3: {
            "emission_prob_always": 0.30,  # more noise
            "headlines": CONFIG["signal_tiers"][3]["headlines"],
        },
    },

    # Tighter tool budgets: can only read ONE buoy per day
    "tool_budgets": {
        "check_weather_reports": 1,
        "query_fishing_log": 1,
        "read_barometer": 1,
        "read_buoy": 1,  # can only check ONE zone per day
        "analyze_data": 1,
        "evaluate_options": 1,
        "forecast_scenario": 1,
    },
}
