"""Coordinator loop for agents, browser environment, and evaluation."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from agents.harness.semantic_controls import inspect_semantic_controls_output, map_semantic_controls_output
from tools.runtime_logger import RuntimeLogger

from .env import GameEnv
from .evaluator import Evaluator
from .game_agent import ActionPayload, Agent

LOGGER = logging.getLogger(__name__)

NO_GAME_STATE_SUMMARY = "(no game state)"
AGENT_LOOP_SLEEP_S = 0.05
FINAL_RT_ACTION_WINDOW_S = 0.2


def _final_rt_noop_action() -> dict[str, Any]:
    return {
        "action": "key_timeline",
        "slots": [[] for _ in range(6)],
        "slot_ms": FINAL_RT_ACTION_WINDOW_S * 1000.0 / 6,
    }


class Coordinator:
    """Coordinates agents, environment, and evaluation in the main loop."""

    def __init__(
        self,
        env: GameEnv,
        agents: list[Agent],
        evaluator: Evaluator | None = None,
    ):
        if not agents:
            raise ValueError("Coordinator requires at least one agent.")

        self.env = env
        self.agents = agents
        self.evaluator = evaluator
        self._clock_mode = str(getattr(env.config, "clock_mode", "paused"))
        if self._clock_mode == "final-rt" and len(agents) > 1:
            endpoints = [
                str(getattr(agent.client, "_endpoint", "") or "")
                for agent in agents
            ]
            if not all(endpoints) or len(set(endpoints)) != len(endpoints):
                raise ValueError(
                    "multi-agent final-rt requires one distinct endpoint per agent "
                    "so every vLLM replica has at most one in-flight request"
                )
        # Frozen-capture loop: keep the game PAUSED except during the action's
        # ~200ms execution, so screenshot + state capture don't let the game
        # advance uncontrolled (~150ms/step otherwise). Matters for precision
        # games (e.g. stack). Only meaningful with pause_during_inference.
        self._frozen_capture = bool(os.environ.get("GW_FROZEN_CAPTURE")) and bool(
            getattr(env, "pause_during_inference", False)
        )
        self._stop_event = asyncio.Event()
        self._final_rt_pending_actions: dict[str, ActionPayload] = {
            agent.agent_id: _final_rt_noop_action() for agent in agents
        }
        self._final_rt_pending_is_noop: dict[str, bool] = {
            agent.agent_id: True for agent in agents
        }
        self._final_rt_cycles: dict[str, int] = {agent.agent_id: 0 for agent in agents}
        self.agent_loggers: dict[str, RuntimeLogger] = {}
        self._evaluation_agent_id = self.agents[0].agent_id if self.agents else None
        self._init_agent_loggers()

    @staticmethod
    def _build_runtime_logger(agent: Agent, env: GameEnv) -> RuntimeLogger:
        client = agent.client
        return RuntimeLogger(
            log_dir=client.config.log_dir,
            session_id=client.config.log_session_id,
            game_name=env.config.game_id,
            model_name=client.config.model,
            agent_id=agent.agent_id,
            session_root=getattr(client.config, "log_root", None),
            memory_screenshot_mode=getattr(client.config, "memory_screenshot_mode", "path"),
        )

    def _init_agent_loggers(self) -> None:
        for agent in self.agents:
            self.agent_loggers[agent.agent_id] = self._build_runtime_logger(agent, self.env)

    def _log_model_interaction(
        self,
        agent: Agent,
        agent_logger: RuntimeLogger | None,
    ) -> dict[str, Any] | None:
        trace = agent.client.pop_logged_interaction()
        if agent_logger:
            agent_logger.log_interaction_from_trace(trace)
        return trace

    @staticmethod
    def _count_proposed_actions(action: ActionPayload) -> int:
        return len(Coordinator._collect_proposed_actions(action))

    @staticmethod
    def _collect_proposed_actions(action: ActionPayload) -> list[Any]:
        if isinstance(action, list):
            return list(action) or [None]

        if isinstance(action, dict):
            return [action]

        return [action]

    def _build_action_validity_record(
        self,
        agent: Agent,
        raw_action: ActionPayload,
        resolved_action: ActionPayload,
        trace: dict[str, Any] | None,
    ) -> dict[str, Any]:
        trace = trace or {}
        executor = self.env._get_executor(agent)

        if agent.agent_type == "generalist":
            proposed_tool_call = trace.get("tool_call")
            semantic = inspect_semantic_controls_output(
                proposed_tool_call,
                agent.semantic_controls_map,
            )
            low_level = executor.inspect_action(
                resolved_action if isinstance(resolved_action, dict) else None
            )
            is_valid = bool(semantic.get("is_valid")) and bool(low_level.get("is_valid"))
            invalid_kind = None
            reason = "valid"
            if not semantic.get("is_valid"):
                invalid_kind = semantic.get("invalid_kind")
                reason = str(semantic.get("reason") or "invalid_semantic_action")
            elif not low_level.get("is_valid"):
                invalid_kind = low_level.get("invalid_kind")
                reason = str(low_level.get("reason") or "invalid_low_level_action")

            return {
                "agent_type": agent.agent_type,
                "is_valid": is_valid,
                "reason": reason,
                "invalid_kind": invalid_kind,
                "proposed_action_count": 1,
                "valid_action_count": 1 if is_valid else 0,
                "raw_action": raw_action,
                "raw_tool_call": proposed_tool_call,
                "resolved_action": resolved_action,
                "normalized_action": low_level.get("normalized_action"),
                "semantic_control_id": semantic.get("control_id"),
            }

        proposed_count = self._count_proposed_actions(raw_action)
        if trace.get("error"):
            return {
                "agent_type": agent.agent_type,
                "is_valid": False,
                "reason": "client_parse_error",
                "invalid_kind": "no_function_call",
                "proposed_action_count": proposed_count,
                "valid_action_count": 0,
                "raw_action": raw_action,
                "resolved_action": resolved_action,
                "normalized_action": None,
            }

        proposed_actions = self._collect_proposed_actions(raw_action)
        inspections = [executor.inspect_action(item) for item in proposed_actions]
        valid_count = sum(1 for item in inspections if item.get("is_valid"))
        first_invalid = next((item for item in inspections if not item.get("is_valid")), None)
        is_valid = valid_count == proposed_count
        return {
            "agent_type": agent.agent_type,
            "is_valid": is_valid,
            "reason": (
                "valid"
                if is_valid
                else str((first_invalid or {}).get("reason") or "invalid_action")
            ),
            "invalid_kind": None if is_valid else (first_invalid or {}).get("invalid_kind"),
            "proposed_action_count": proposed_count,
            "valid_action_count": valid_count,
            "raw_action": raw_action,
            "resolved_action": resolved_action,
            "normalized_action": (inspections[0].get("normalized_action") if inspections else None),
        }

    async def _capture_state(self) -> tuple[dict[str, Any] | None, str]:
        snapshot = await self.env.capture_state()
        if not snapshot:
            return None, NO_GAME_STATE_SUMMARY
        return snapshot.state, snapshot.summary

    @staticmethod
    def _extract_completion_progress(state: dict[str, Any] | None) -> object:
        if not isinstance(state, dict):
            return None
        game_state = state.get("game_state")
        if not isinstance(game_state, dict):
            return None
        return game_state.get("completion_progress")

    @staticmethod
    def _extract_game_status(state: dict[str, Any] | None) -> object:
        if not isinstance(state, dict):
            return None
        return state.get("status")

    def _build_evaluation_payload(
        self,
        agent: Agent,
        result: Any,
        game_status: object,
        progress: object,
        game_completion_progress: object,
    ) -> dict[str, Any]:
        return {
            "interaction_id": agent.step_index,
            "step": agent.step_index,
            "max_steps": self.env.config.max_steps,
            "task_status": result.status,
            "game_status": game_status,
            "summary": result.summary,
            "should_stop": result.should_stop,
            "should_reset": result.should_reset,
            "stop_reason": result.stop_reason,
            "finalized": result.finalized,
            "progress": progress,
            "game_completion_progress": game_completion_progress,
            "metrics": dict(agent.eval_metrics),
        }

    async def _evaluate_step(
        self,
        agent: Agent,
        state: dict[str, Any] | None,
        agent_logger: RuntimeLogger | None,
    ):
        if not self.evaluator:
            return None
        if self._evaluation_agent_id and agent.agent_id != self._evaluation_agent_id:
            return None

        result = await self.evaluator.evaluate(agent, state)
        if not result:
            return None
        if result.should_stop and not result.should_reset:
            result = await self.evaluator.summarize(agent, state)

        progress = result.metrics.get("progress") if isinstance(result.metrics, dict) else None
        game_completion_progress = self._extract_completion_progress(state)
        game_status = self._extract_game_status(state)
        evaluation_payload = self._build_evaluation_payload(
            agent,
            result,
            game_status,
            progress,
            game_completion_progress,
        )
        if agent_logger:
            agent_logger.log_task_evaluation(evaluation_payload)

        LOGGER.task(
            "Task eval (%s): step=%s/%s task_status=%s game_status=%s "
            "stop=%s reset=%s finalized=%s reason=%s progress=%s "
            "game_completion_progress=%s metrics=%s",
            agent.agent_id,
            agent.step_index,
            self.env.config.max_steps,
            result.status,
            game_status,
            result.should_stop,
            result.should_reset,
            result.finalized,
            result.stop_reason,
            progress,
            game_completion_progress,
            evaluation_payload["metrics"],
        )
        return result

    async def _handle_eval_controls(self, agent: Agent, result) -> None:
        if result and result.should_reset:
            reset_ok = await self.env.reset_game()
            if reset_ok:
                if self.evaluator:
                    agent.eval_metrics = self.evaluator.reset_metrics(agent.eval_metrics)
                LOGGER.task(
                    "Task eval: auto-reset after fail; continuing at step=%s",
                    agent.step_index,
                )
            else:
                LOGGER.task(
                    "Task eval: auto-reset failed; stopping run at step=%s",
                    agent.step_index,
                )
                self._stop_event.set()
        if result and result.should_stop:
            self._stop_event.set()

    async def _get_raw_action(self, agent: Agent) -> ActionPayload:
        if self._frozen_capture:
            # Pause FIRST so the screenshot is the frozen frame the action will
            # apply to, and the game does not advance during capture/inference.
            # Stays paused on return; _run_agent_step resumes only for the action.
            await self.env.pause_game()
            screenshot_path = await self.env.capture_screenshot(agent.agent_id)
            return await asyncio.to_thread(agent.client.get_action, screenshot_path)

        screenshot_path = await self.env.capture_screenshot(agent.agent_id)

        paused = False
        if self.env.pause_during_inference:
            await self.env.pause_game()
            paused = True

        try:
            return await asyncio.to_thread(agent.client.get_action, screenshot_path)
        finally:
            if paused:
                await self.env.resume_game()

    def _resolve_action(self, agent: Agent, raw_action: ActionPayload) -> ActionPayload:
        if agent.agent_type == "generalist" and agent.semantic_controls_map:
            action = map_semantic_controls_output(raw_action, agent.semantic_controls_map)
            LOGGER.model(
                "Agent %s raw action %s -> semantic_controls mapped action: %s",
                agent.agent_id,
                raw_action,
                action,
            )
            return action

        LOGGER.model("Agent %s raw action %s", agent.agent_id, raw_action)
        return raw_action

    async def _run_agent_step(self, agent: Agent, agent_logger: RuntimeLogger | None) -> None:
        if self._clock_mode == "final-rt":
            await self._run_final_rt_step(agent, agent_logger)
            return

        raw_action = await self._get_raw_action(agent)
        trace = self._log_model_interaction(agent, agent_logger)
        action = self._resolve_action(agent, raw_action)
        action_validity = self._build_action_validity_record(agent, raw_action, action, trace)
        if agent_logger:
            agent_logger.log_action_validity(action_validity)
        # Frozen-capture: game is paused here; resume ONLY for the ~200ms action,
        # then pause again so state capture doesn't advance the game.
        if self._frozen_capture:
            await self.env.resume_game()
        await self.env.execute_action(agent, action)
        if self._frozen_capture:
            await self.env.pause_game()
        if agent_logger:
            agent_logger.log_executed_action(action)

        state, state_summary = await self._capture_state()
        if agent_logger:
            agent_logger.log_game_state(state)
        LOGGER.game("Game state: %s", state_summary)

        agent.step_index += 1
        result = await self._evaluate_step(agent, state, agent_logger)
        await self._handle_eval_controls(agent, result)
        if agent_logger:
            agent_logger.finalize_step()

    async def _run_final_rt_step(
        self,
        agent: Agent,
        agent_logger: RuntimeLogger | None,
    ) -> None:
        """Pipeline the previous 200 ms action with the current model decision."""
        loop = asyncio.get_running_loop()
        cycle_index = self._final_rt_cycles[agent.agent_id]
        applied_action = self._final_rt_pending_actions[agent.agent_id]
        applied_is_noop = self._final_rt_pending_is_noop[agent.agent_id]
        cycle_started = loop.time()

        async def decide() -> tuple[ActionPayload, float, float]:
            screenshot_path = await self.env.capture_screenshot(agent.agent_id)
            frame_captured = loop.time()
            raw = await asyncio.to_thread(agent.client.get_action, screenshot_path)
            return raw, frame_captured, loop.time()

        async def execute_applied_action() -> float:
            await self.env.execute_action(
                agent,
                applied_action,
                allow_concurrent_capture=True,
            )
            return loop.time()

        # The old action and the next decision run concurrently. KeyTimeline keeps
        # the final slot physically held after its 6 slots, so a slow decision
        # naturally extends only the final key state and never repeats mouse delta.
        action_task = asyncio.create_task(execute_applied_action())
        decision_task = asyncio.create_task(decide())
        try:
            (
                raw_action,
                frame_captured,
                decision_ready,
            ), action_finished = await asyncio.gather(
                decision_task,
                action_task,
            )
        except BaseException:
            for task in (decision_task, action_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(decision_task, action_task, return_exceptions=True)
            raise

        earliest_boundary = cycle_started + FINAL_RT_ACTION_WINDOW_S
        if decision_ready < earliest_boundary:
            await asyncio.sleep(earliest_boundary - decision_ready)
        boundary_reached = loop.time()

        trace_started = loop.time()
        trace = self._log_model_interaction(agent, agent_logger)
        trace_logged = loop.time()
        issued_action = self._resolve_action(agent, raw_action)
        action_validity = self._build_action_validity_record(
            agent,
            raw_action,
            issued_action,
            trace,
        )
        action_resolved = loop.time()
        self._final_rt_pending_actions[agent.agent_id] = issued_action
        self._final_rt_pending_is_noop[agent.agent_id] = False
        self._final_rt_cycles[agent.agent_id] = cycle_index + 1

        decision_latency_s = decision_ready - cycle_started
        action_residence_s = boundary_reached - cycle_started
        hold_overrun_s = max(0.0, action_residence_s - FINAL_RT_ACTION_WINDOW_S)
        screenshot_ms = (frame_captured - cycle_started) * 1000.0
        client_timing = (trace or {}).get("timing")
        breakdown = {
            str(key): round(float(value), 3)
            for key, value in (client_timing or {}).items()
            if isinstance(value, (int, float))
        }
        client_total_ms = breakdown.get("client_total_ms", 0.0)
        breakdown.update(
            {
                "engine_screenshot_ms": round(screenshot_ms, 3),
                "thread_bridge_ms": round(
                    max(
                        0.0,
                        decision_latency_s * 1000.0
                        - screenshot_ms
                        - client_total_ms,
                    ),
                    3,
                ),
                "runtime_log_materialize_ms": round(
                    (trace_logged - trace_started) * 1000.0,
                    3,
                ),
                "engine_action_resolve_ms": round(
                    (action_resolved - trace_logged) * 1000.0,
                    3,
                ),
                "engine_action_window_execute_ms": round(
                    (action_finished - cycle_started) * 1000.0,
                    3,
                ),
                "engine_action_hold_extension_ms": round(
                    max(0.0, boundary_reached - action_finished) * 1000.0,
                    3,
                ),
                "critical_path_ms": round(action_residence_s * 1000.0, 3),
            }
        )
        clock_record = {
            "mode": "final-rt",
            "cycle": cycle_index,
            "startup_noop": cycle_index == 0,
            "reset_noop": cycle_index > 0 and applied_is_noop,
            "action_window_ms": round(FINAL_RT_ACTION_WINDOW_S * 1000.0, 3),
            "decision_latency_ms": round(decision_latency_s * 1000.0, 3),
            "frame_age_ms": round((boundary_reached - frame_captured) * 1000.0, 3),
            "action_residence_ms": round(action_residence_s * 1000.0, 3),
            "hold_overrun_ms": round(hold_overrun_s * 1000.0, 3),
            "decision_rate": min(
                1.0,
                FINAL_RT_ACTION_WINDOW_S / max(action_residence_s, 1e-9),
            ),
            "latency_breakdown_ms": breakdown,
            "issued_action": issued_action,
            "applied_action": applied_action,
        }
        if agent_logger:
            agent_logger.log_action_validity(action_validity)
            agent_logger.log_executed_action(applied_action)
            agent_logger.log_clock(clock_record)

        LOGGER.info(
            "[final-rt] cycle=%s startup_noop=%s decision=%.1fms "
            "frame_age=%.1fms residence=%.1fms hold_overrun=%.1fms decision_rate=%.4f",
            cycle_index,
            cycle_index == 0,
            decision_latency_s * 1000.0,
            clock_record["frame_age_ms"],
            action_residence_s * 1000.0,
            hold_overrun_s * 1000.0,
            clock_record["decision_rate"],
        )

        state, state_summary = await self._capture_state()
        if agent_logger:
            agent_logger.log_game_state(state)
        LOGGER.game("Game state: %s", state_summary)

        agent.step_index += 1
        result = await self._evaluate_step(agent, state, agent_logger)
        await self._handle_eval_controls(agent, result)
        if result and result.should_reset:
            self._final_rt_pending_actions[agent.agent_id] = _final_rt_noop_action()
            self._final_rt_pending_is_noop[agent.agent_id] = True
        if agent_logger:
            agent_logger.finalize_step()

    async def _run_final_rt_joint_step(self) -> None:
        """Run all roles from one shared frame and one synchronized action boundary."""
        loop = asyncio.get_running_loop()
        cycle_indices = {self._final_rt_cycles[agent.agent_id] for agent in self.agents}
        if len(cycle_indices) != 1:
            raise RuntimeError(f"joint final-rt cycle indices diverged: {cycle_indices}")
        cycle_index = cycle_indices.pop()
        cycle_started = loop.time()
        applied = {
            agent.agent_id: self._final_rt_pending_actions[agent.agent_id]
            for agent in self.agents
        }
        applied_is_noop = {
            agent.agent_id: self._final_rt_pending_is_noop[agent.agent_id]
            for agent in self.agents
        }

        async def execute_applied_action(agent: Agent) -> tuple[str, float]:
            await self.env.execute_action(
                agent,
                applied[agent.agent_id],
                allow_concurrent_capture=True,
            )
            return agent.agent_id, loop.time()

        action_tasks = [
            asyncio.create_task(execute_applied_action(agent))
            for agent in self.agents
        ]
        decision_tasks: list[asyncio.Task] = []
        try:
            # Both roles see the exact same game frame. Their role-specific system
            # prompts and disjoint controls still produce independent decisions.
            screenshot_path = await self.env.capture_screenshot("joint")
            frame_captured = loop.time()

            async def decide(agent: Agent) -> tuple[ActionPayload, float]:
                raw = await asyncio.to_thread(agent.client.get_action, screenshot_path)
                return raw, loop.time()

            decision_tasks = [
                asyncio.create_task(decide(agent)) for agent in self.agents
            ]
            decision_results = await asyncio.gather(*decision_tasks)
            action_results = await asyncio.gather(*action_tasks)
        except BaseException:
            for task in [*decision_tasks, *action_tasks]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                *decision_tasks,
                *action_tasks,
                return_exceptions=True,
            )
            raise

        latest_decision_ready = max(ready for _, ready in decision_results)
        earliest_boundary = cycle_started + FINAL_RT_ACTION_WINDOW_S
        if latest_decision_ready < earliest_boundary:
            await asyncio.sleep(earliest_boundary - latest_decision_ready)
        boundary_reached = loop.time()
        action_residence_s = boundary_reached - cycle_started
        hold_overrun_s = max(0.0, action_residence_s - FINAL_RT_ACTION_WINDOW_S)
        action_finished = dict(action_results)

        for agent, (raw_action, decision_ready) in zip(self.agents, decision_results):
            agent_logger = self.agent_loggers.get(agent.agent_id)
            trace_started = loop.time()
            trace = self._log_model_interaction(agent, agent_logger)
            trace_logged = loop.time()
            issued_action = self._resolve_action(agent, raw_action)
            action_validity = self._build_action_validity_record(
                agent,
                raw_action,
                issued_action,
                trace,
            )
            action_resolved = loop.time()
            self._final_rt_pending_actions[agent.agent_id] = issued_action
            self._final_rt_pending_is_noop[agent.agent_id] = False
            self._final_rt_cycles[agent.agent_id] = cycle_index + 1
            decision_latency_s = decision_ready - cycle_started
            screenshot_ms = (frame_captured - cycle_started) * 1000.0
            client_timing = (trace or {}).get("timing")
            breakdown = {
                str(key): round(float(value), 3)
                for key, value in (client_timing or {}).items()
                if isinstance(value, (int, float))
            }
            client_total_ms = breakdown.get("client_total_ms", 0.0)
            agent_action_finished = action_finished[agent.agent_id]
            breakdown.update(
                {
                    "engine_screenshot_ms": round(screenshot_ms, 3),
                    "thread_bridge_ms": round(
                        max(
                            0.0,
                            decision_latency_s * 1000.0
                            - screenshot_ms
                            - client_total_ms,
                        ),
                        3,
                    ),
                    "runtime_log_materialize_ms": round(
                        (trace_logged - trace_started) * 1000.0,
                        3,
                    ),
                    "engine_action_resolve_ms": round(
                        (action_resolved - trace_logged) * 1000.0,
                        3,
                    ),
                    "engine_action_window_execute_ms": round(
                        (agent_action_finished - cycle_started) * 1000.0,
                        3,
                    ),
                    "engine_action_hold_extension_ms": round(
                        max(0.0, boundary_reached - agent_action_finished) * 1000.0,
                        3,
                    ),
                    "critical_path_ms": round(action_residence_s * 1000.0, 3),
                }
            )
            clock_record = {
                "mode": "final-rt",
                "cycle": cycle_index,
                "joint_agents": len(self.agents),
                "startup_noop": cycle_index == 0,
                "reset_noop": cycle_index > 0 and applied_is_noop[agent.agent_id],
                "action_window_ms": round(FINAL_RT_ACTION_WINDOW_S * 1000.0, 3),
                "decision_latency_ms": round(decision_latency_s * 1000.0, 3),
                "frame_age_ms": round(
                    (boundary_reached - frame_captured) * 1000.0,
                    3,
                ),
                "action_residence_ms": round(action_residence_s * 1000.0, 3),
                "hold_overrun_ms": round(hold_overrun_s * 1000.0, 3),
                "decision_rate": min(
                    1.0,
                    FINAL_RT_ACTION_WINDOW_S / max(action_residence_s, 1e-9),
                ),
                "latency_breakdown_ms": breakdown,
                "issued_action": issued_action,
                "applied_action": applied[agent.agent_id],
            }
            if agent_logger:
                agent_logger.log_action_validity(action_validity)
                agent_logger.log_executed_action(applied[agent.agent_id])
                agent_logger.log_clock(clock_record)
            LOGGER.info(
                "[final-rt-joint] agent=%s cycle=%s decision=%.1fms "
                "residence=%.1fms hold_overrun=%.1fms",
                agent.agent_id,
                cycle_index,
                decision_latency_s * 1000.0,
                action_residence_s * 1000.0,
                hold_overrun_s * 1000.0,
            )

        state, state_summary = await self._capture_state()
        LOGGER.game("Game state: %s", state_summary)
        for agent in self.agents:
            agent.step_index += 1
            agent_logger = self.agent_loggers.get(agent.agent_id)
            if agent_logger:
                agent_logger.log_game_state(state)

        evaluation_agent = self.agents[0]
        evaluation_logger = self.agent_loggers.get(evaluation_agent.agent_id)
        result = await self._evaluate_step(
            evaluation_agent,
            state,
            evaluation_logger,
        )
        await self._handle_eval_controls(evaluation_agent, result)
        if result and result.should_reset:
            for agent in self.agents:
                self._final_rt_pending_actions[agent.agent_id] = _final_rt_noop_action()
                self._final_rt_pending_is_noop[agent.agent_id] = True
        for agent_logger in self.agent_loggers.values():
            agent_logger.finalize_step()

    async def _agent_loop(self, agent: Agent) -> None:
        agent_logger = self.agent_loggers.get(agent.agent_id)
        while not self._stop_event.is_set():
            try:
                await self._run_agent_step(agent, agent_logger)
            except Exception as exc:
                if agent_logger:
                    agent_logger.flush_pending_step()
                LOGGER.exception("Agent %s loop error: %s", agent.agent_id, exc)
                self._stop_event.set()
                break

            if self._clock_mode != "final-rt":
                await asyncio.sleep(AGENT_LOOP_SLEEP_S)

    async def _final_rt_joint_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._run_final_rt_joint_step()
            except Exception as exc:
                for agent_logger in self.agent_loggers.values():
                    agent_logger.flush_pending_step()
                LOGGER.exception("Joint final-rt loop error: %s", exc)
                self._stop_event.set()
                break

    async def run(self) -> None:
        tasks: list[asyncio.Task] = []
        try:
            await self.env.start()
            if self._clock_mode == "final-rt" and len(self.agents) > 1:
                tasks = [asyncio.create_task(self._final_rt_joint_loop())]
            else:
                tasks = [asyncio.create_task(self._agent_loop(agent)) for agent in self.agents]
            await self._stop_event.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for agent_logger in self.agent_loggers.values():
                agent_logger.flush_pending_step()
            await self.env.close_game()


__all__ = ["Coordinator"]
