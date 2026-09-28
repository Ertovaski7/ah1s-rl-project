from __future__ import annotations

import math
import numpy as np

from helicopter_env_turn_goal import HelicopterEnvTurnGoal, wrap_deg


class HelicopterEnvTurnGoalV2(HelicopterEnvTurnGoal):
    """Safer reward shaping for PPO fine-tuning after turn behavior cloning.

    Dynamics and action mapping are unchanged from HelicopterEnvTurnGoal.
    Only the RL reward/termination logic is strengthened so PPO is strongly
    discouraged from overshooting a captured heading or sacrificing altitude.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v2_previous_remaining = float(self.target_turn_deg)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._v2_previous_remaining = float(info.get("remaining_turn_deg", self.target_turn_deg))
        return obs, info

    def step(self, action):
        prev_remaining = float(self.target_turn_deg - self.cumulative_turn_deg)
        prev_abs = abs(prev_remaining)

        obs, _old_reward, terminated, truncated, info = super().step(action)

        remaining = float(info["remaining_turn_deg"])
        current_abs = abs(remaining)
        altitude = float(info["altitude"])
        vertical_speed = float(info["vertical_speed"])
        forward_speed = float(info["forward_velocity"])
        roll_deg = abs(math.degrees(float(info["roll"])))
        yaw_rate_deg = abs(math.degrees(float(info["r_rate"])))

        # Main objective: continuously reduce unwrapped remaining turn error.
        progress = prev_abs - current_abs
        reward = 4.0 * progress

        # Dense terminal shaping. Close to the target should be much more
        # valuable than simply continuing to rotate in the correct direction.
        reward += 1.5 * math.exp(-current_abs / 12.0)
        reward += 2.0 * math.exp(-current_abs / 3.0)

        # Strong overshoot / moving-away penalty.
        if current_abs > prev_abs + 0.10:
            reward -= 3.0 * (current_abs - prev_abs)

        # Once close, leaving the capture region is expensive.
        if prev_abs <= 3.0 and current_abs > prev_abs:
            reward -= 2.0
        if current_abs > 8.0 and prev_abs <= 3.0:
            reward -= 20.0

        # Flight-quality penalties.
        alt_err = abs(300.0 - altitude)
        speed_err = abs(14.5 - forward_speed)
        reward -= 0.050 * alt_err
        reward -= 0.120 * abs(vertical_speed)
        reward -= 0.035 * speed_err
        reward -= 0.035 * roll_deg
        reward -= 0.020 * yaw_rate_deg

        # Extra protection around presentation-safe altitude band.
        if altitude < 285.0:
            reward -= 2.0 * (285.0 - altitude)
        elif altitude > 315.0:
            reward -= 2.0 * (altitude - 315.0)

        # Strong reward for actual stable target hold.
        hold_s = float(info.get("success_hold_s", 0.0))
        if current_abs <= 1.5:
            reward += 1.0
        if hold_s > 0.0:
            reward += 2.0 * self.CONTROL_DT
        if bool(info.get("success", False)):
            reward += 250.0

        # Safety failures must dominate all positive shaping.
        if bool(info.get("safety_failure", False)):
            reward -= 250.0

        # Additional terminal failure if the policy grossly overshoots.
        gross_overshoot = bool(
            (self.target_turn_deg > 0.0 and remaining < -45.0)
            or (self.target_turn_deg < 0.0 and remaining > 45.0)
        )
        if gross_overshoot:
            reward -= 150.0
            terminated = True
            info["safety_failure"] = True
            info["termination_reason_v2"] = "gross_heading_overshoot"

        self._v2_previous_remaining = remaining
        return obs, float(reward), bool(terminated), bool(truncated), info
