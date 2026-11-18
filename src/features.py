"""Feature groups used for modelling (see PLAN.md, section 4)."""

# known at the moment A enters the pits, available for every season (2018-2025)
PRE_DECISION_NUMERIC = [
    "gap_s", "a_was_ahead", "position_A", "same_team", "stop_number_A",
    "stint_age_A", "stint_age_B", "stint_age_delta",
    "pace_delta", "traffic_at_exit", "rejoin_gap_ahead",
    "laps_remaining", "race_progress", "neutral_before",
    "pit_loss_circuit", "overtakes_per_lap_circuit",
    "length_km", "n_turns", "street_circuit",
    "om_temp_c", "om_humidity", "om_precip_prev3h_mm", "om_cloud_cover", "om_wind_kmh", "rain_race",
    "era_ground_effect",
]

# tyre compound / true tyre age from OpenF1: only 2023+
TYRE_2023 = ["tyre_age_A", "tyre_age_B", "tyre_age_delta", "hard_A_old", "hard_B_old", "hard_A_new"]

# only known after the fact: used for analysis / ablation, never in the main model
POST_HOC = ["lap_delay_B", "pit_lane_A", "pit_lane_B", "hard_B_new"]

TARGET = "undercut_success"
