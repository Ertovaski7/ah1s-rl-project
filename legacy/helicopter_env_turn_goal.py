from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

from helicopter_env_stage2_refine_mapped import HelicopterEnvStage2RefineMapped


STAGE2_MODEL_PATH = Path("models_stage2_hybrid_final/AH1S_STAGE2_HYBRID_FINAL.zip")


def wrap_deg(x: float) -> float:
    return float((float(x) + 180.0) % 360.0 - 180.0)


class HelicopterEnvTurnGoal(HelicopterEnvStage2RefineMapped):
    """Goal-conditioned forward-flight turn environment.

    Base Stage-2 provides the forward-flight entry state.  Once the turn starts,
    all four actions are live and the observation is augmented with the turn goal.

    Observation = 12 Stage-2 features +
        [target_turn/360, remaining_turn/360, cumulative_turn/360, direction]

    Turn action mapping:
        a0 -> physical collective in [0.460, 0.620]
        a1 -> elevator in [-0.180, -0.110]
        a2 -> aileron residual around trim, scale 0.300
        a3 -> rudder residual around trim, scale 0.500
    """

    ENTRY_FORWARD_FT = 160.0
    TARGET_ALT_FT = 300.0
    TARGET_SPEED_FPS = 14.5
    TURN_COLLECTIVE_CENTER = 0.540
    TURN_COLLECTIVE_SCALE = 0.080
    TURN_AILERON_SCALE = 0.300
    TURN_RUDDER_SCALE = 0.500
    TURN_ROLL_AFCS = 0.15
    TURN_YAW_AFCS = 0.15
    MAX_TURN_TIME_S = 110.0

    def __init__(self, target_turn_deg=None, target_choices=None):
        super().__init__(aileron_scale=0.026, rudder_scale=0.040)

        if not STAGE2_MODEL_PATH.exists():
            raise FileNotFoundError(f"Stage-2 model missing: {STAGE2_MODEL_PATH}")

        self.stage2_model = PPO.load(str(STAGE2_MODEL_PATH))
        self.fixed_target_turn_deg = target_turn_deg
        self.target_choices = list(target_choices or [-50.0, 50.0, 90.0, 200.0, 360.0])

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(16,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.turn_active = False
        self.target_turn_deg = 50.0
        self.cumulative_turn_deg = 0.0
        self.prev_heading_deg = 0.0
        self.turn_steps = 0
        self.success_hold_s = 0.0
        self.previous_turn_action = np.zeros(4, dtype=np.float32)

    def _heading_deg(self) -> float:
        for key in ["attitude/heading-true-rad", "attitude/psi-rad"]:
            try:
                return math.degrees(float(self.fdm[key]))
            except Exception:
                pass
        return 0.0

    def _base_obs(self) -> np.ndarray:
        return np.asarray(super()._get_obs(), dtype=np.float32)

    def _get_obs(self):
        base = self._base_obs()
        if not self.turn_active:
            extras = np.zeros(4, dtype=np.float32)
        else:
            remaining = self.target_turn_deg - self.cumulative_turn_deg
            direction = 1.0 if self.target_turn_deg >= 0.0 else -1.0
            extras = np.array(
                [
                    np.clip(self.target_turn_deg / 360.0, -2.0, 2.0),
                    np.clip(remaining / 360.0, -2.0, 2.0),
                    np.clip(self.cumulative_turn_deg / 360.0, -2.0, 2.0),
                    direction,
                ],
                dtype=np.float32,
            )
        return np.clip(np.concatenate([base, extras]), -5.0, 5.0).astype(np.float32)

    @classmethod
    def collective_to_action(cls, collective: float) -> float:
        return float(np.clip(
            (float(collective) - cls.TURN_COLLECTIVE_CENTER) / cls.TURN_COLLECTIVE_SCALE,
            -1.0,
            +1.0,
        ))

    @classmethod
    def action_to_collective(cls, action0: float) -> float:
        return float(np.clip(
            cls.TURN_COLLECTIVE_CENTER + cls.TURN_COLLECTIVE_SCALE * float(action0),
            0.460,
            0.620,
        ))

    def _apply_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if action.shape[0] < 4:
            raise ValueError("Turn environment requires 4 actions")

        # Before the turn, preserve the exact Stage-2 mapping used by its model.
        if not self.turn_active:
            return super()._apply_action(action)

        collective = self.action_to_collective(action[0])
        elevator = float(np.clip(-0.145 + 0.035 * float(action[1]), -0.180, -0.110))
        aileron = float(np.clip(0.19095 + self.TURN_AILERON_SCALE * float(action[2]), -1.0, 1.0))
        rudder = float(np.clip(0.39 + self.TURN_RUDDER_SCALE * float(action[3]), -1.0, 1.0))

        self.fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
        self.fdm["ap/afcs/roll-channel-active-norm"] = self.TURN_ROLL_AFCS
        self.fdm["ap/afcs/yaw-channel-active-norm"] = self.TURN_YAW_AFCS

        self.fdm["fcs/collective-cmd-norm"] = collective
        self.fdm["fcs/elevator-cmd-norm"] = elevator
        self.fdm["fcs/aileron-cmd-norm"] = aileron
        self.fdm["fcs/rudder-cmd-norm"] = rudder

        return collective, elevator, aileron, rudder

    def reset(self, seed=None, options=None):
        # Parent builds the stable ~300 ft hover.
        self.turn_active = False
        obs, info = super().reset(seed=seed, options=options)

        # Use the already learned Stage-2 policy only to create the common
        # forward-flight entry condition. It is not the turn teacher.
        self.forward_distance = 0.0
        max_entry_steps = int(40.0 / self.CONTROL_DT)
        for _ in range(max_entry_steps):
            base_obs = self._base_obs()
            base_action, _ = self.stage2_model.predict(base_obs, deterministic=True)
            _, _, terminated, truncated, _ = super().step(base_action)
            if self.forward_distance >= self.ENTRY_FORWARD_FT:
                break
            if terminated or truncated:
                break

        if self.fixed_target_turn_deg is None:
            self.target_turn_deg = float(self.np_random.choice(self.target_choices))
        else:
            self.target_turn_deg = float(self.fixed_target_turn_deg)

        self.turn_active = True
        self.cumulative_turn_deg = 0.0
        self.prev_heading_deg = self._heading_deg()
        self.turn_steps = 0
        self.success_hold_s = 0.0
        self.previous_turn_action = np.zeros(4, dtype=np.float32)

        self.fdm["ap/afcs/roll-channel-active-norm"] = self.TURN_ROLL_AFCS
        self.fdm["ap/afcs/yaw-channel-active-norm"] = self.TURN_YAW_AFCS

        state = self._raw_state()
        info = {
            **state,
            "target_turn_deg": self.target_turn_deg,
            "cumulative_turn_deg": self.cumulative_turn_deg,
            "remaining_turn_deg": self.target_turn_deg,
            "success": False,
        }
        return self._get_obs(), info

    def step(self, action):
        self.turn_steps += 1
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)

        prev_remaining = self.target_turn_deg - self.cumulative_turn_deg
        self._apply_action(action)

        jsbsim_ok = True
        for _ in range(self.PHYSICS_STEPS):
            if not self.fdm.run():
                jsbsim_ok = False
                break

        s = self._raw_state()
        current_heading = self._heading_deg()
        self.cumulative_turn_deg += wrap_deg(current_heading - self.prev_heading_deg)
        self.prev_heading_deg = current_heading
        remaining = self.target_turn_deg - self.cumulative_turn_deg

        # Keep Stage-2 distance bookkeeping meaningful.
        self.forward_distance += float(s["forward_velocity"]) * self.CONTROL_DT

        alt_err = abs(self.TARGET_ALT_FT - s["altitude"])
        speed_err = abs(self.TARGET_SPEED_FPS - s["forward_velocity"])
        roll_deg = abs(math.degrees(s["roll"]))
        yaw_rate_deg = abs(math.degrees(s["r_rate"]))

        # Reward actual reduction in the unwrapped turn error.
        progress = abs(prev_remaining) - abs(remaining)
        reward = 2.0 * progress
        reward -= 0.020 * alt_err
        reward -= 0.060 * abs(s["vertical_speed"])
        reward -= 0.025 * speed_err
        reward -= 0.020 * roll_deg
        reward -= 0.010 * yaw_rate_deg
        reward -= 0.020 * float(np.mean((action - self.previous_turn_action) ** 2))
        self.previous_turn_action = action.copy()

        stable = bool(
            abs(remaining) <= 1.5
            and roll_deg <= 4.0
            and yaw_rate_deg <= 6.0
            and 285.0 <= s["altitude"] <= 315.0
            and abs(s["vertical_speed"]) <= 1.5
            and s["forward_velocity"] >= 8.0
        )
        self.success_hold_s = self.success_hold_s + self.CONTROL_DT if stable else 0.0
        success = self.success_hold_s >= 3.0
        if success:
            reward += 100.0

        safety_failure = bool(
            not jsbsim_ok
            or s["altitude"] < 275.0
            or s["altitude"] > 325.0
            or roll_deg > 18.0
            or abs(math.degrees(s["pitch"])) > 15.0
            or yaw_rate_deg > 30.0
            or s["forward_velocity"] < 1.0
        )
        if safety_failure:
            reward -= 100.0

        terminated = bool(success or safety_failure)
        truncated = bool(self.turn_steps * self.CONTROL_DT >= self.MAX_TURN_TIME_S)

        info = {
            **s,
            "target_turn_deg": self.target_turn_deg,
            "cumulative_turn_deg": self.cumulative_turn_deg,
            "remaining_turn_deg": remaining,
            "success_hold_s": self.success_hold_s,
            "success": success,
            "safety_failure": safety_failure,
        }
        return self._get_obs(), float(reward), terminated, truncated, info
