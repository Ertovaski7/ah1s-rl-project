from __future__ import annotations

import numpy as np

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2


class HelicopterEnvTurnGoalFullEntry(HelicopterEnvTurnGoalV2):
    """Turn environment whose reset finishes near the true full-mission handoff.

    The original turn environment begins the maneuver around ~310 ft because
    its standalone Stage-2 setup reaches 160 ft forward distance there.  In the
    true Stage1->Stage2 same-FDM mission, the turn handoff is around ~300-302 ft.

    This subclass keeps the same learned Stage-2 policy and physics, but after
    the normal reset it continues straight flight with turn_active=False until
    the aircraft reaches a full-mission-like vertical entry envelope.  No
    teacher/controller is used.
    """

    FULL_ENTRY_ALT_MIN = 299.0
    FULL_ENTRY_ALT_MAX = 303.0
    FULL_ENTRY_MAX_ABS_VS = 1.25
    FULL_ENTRY_MIN_SPEED = 8.0
    FULL_ENTRY_MAX_SPEED = 18.0
    EXTRA_ENTRY_MAX_S = 45.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Temporarily return to straight-flight semantics while preserving the
        # same live FDM created by the parent reset.
        self.turn_active = False
        self.success_hold_s = 0.0
        self.cumulative_turn_deg = 0.0
        self.turn_steps = 0
        self.previous_turn_action = np.zeros(4, dtype=np.float32)

        self.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        self.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        max_steps = int(self.EXTRA_ENTRY_MAX_S / self.CONTROL_DT)
        entry_ready = False

        for _ in range(max_steps):
            base_obs = self._base_obs()
            action, _ = self.stage2_model.predict(base_obs, deterministic=True)

            # turn_active=False makes _apply_action use exact Stage-2 mapping.
            self._apply_action(action)
            jsbsim_ok = True
            for _ in range(self.PHYSICS_STEPS):
                if not self.fdm.run():
                    jsbsim_ok = False
                    break
            if not jsbsim_ok:
                raise RuntimeError("JSBSim stopped while building full-entry state")

            s = self._raw_state()
            self.forward_distance += float(s["forward_velocity"]) * self.CONTROL_DT

            entry_ready = bool(
                self.FULL_ENTRY_ALT_MIN <= s["altitude"] <= self.FULL_ENTRY_ALT_MAX
                and abs(s["vertical_speed"]) <= self.FULL_ENTRY_MAX_ABS_VS
                and self.FULL_ENTRY_MIN_SPEED <= s["forward_velocity"] <= self.FULL_ENTRY_MAX_SPEED
                and abs(s["roll"]) <= 0.14
                and abs(s["pitch"]) <= 0.14
            )
            if entry_ready:
                break

        if not entry_ready:
            s = self._raw_state()
            raise RuntimeError(
                "Could not build full-mission-like turn entry: "
                f"alt={s['altitude']:.2f}, vs={s['vertical_speed']:+.2f}, "
                f"v={s['forward_velocity']:.2f}, fwd={self.forward_distance:.1f}"
            )

        # Re-enable the exact turn semantics used during V5 execution.
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
        self._v2_previous_remaining = float(self.target_turn_deg)

        self.fdm["ap/afcs/roll-channel-active-norm"] = self.TURN_ROLL_AFCS
        self.fdm["ap/afcs/yaw-channel-active-norm"] = self.TURN_YAW_AFCS

        s = self._raw_state()
        info = {
            **s,
            "target_turn_deg": self.target_turn_deg,
            "cumulative_turn_deg": 0.0,
            "remaining_turn_deg": self.target_turn_deg,
            "full_entry": True,
            "success": False,
        }
        return self._get_obs(), info
