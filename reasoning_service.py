"""Core orchestration logic for executing reasoning workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from api_base import APIFactory, APIError
from cot_reasoning import (
    VisualizationConfig as CoTVisualizationConfig,
    create_mermaid_diagram as create_cot_diagram,
    parse_cot_response,
)
from tot_reasoning import (
    create_mermaid_diagram as create_tot_diagram,
    parse_tot_response,
)
from bs_reasoning import (
    create_mermaid_diagram as create_bs_diagram,
    parse_bs_response,
)
from l2m_reasoning import (
    create_mermaid_diagram as create_l2m_diagram,
    parse_l2m_response,
)
from selfrefine_reasoning import (
    create_mermaid_diagram as create_selfrefine_diagram,
    parse_selfrefine_response,
)
from selfconsistency_reasoning import (
    create_mermaid_diagram as create_scr_diagram,
    parse_scr_response,
)
from plain_text_reasoning import parse_plain_text_response


logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Structured result for a reasoning run."""

    raw_output: str
    visualization: Optional[Dict[str, Any]] = None
    parsed: Optional[Any] = None


class ReasoningService:
    """Executes reasoning workflows using the selected provider and method."""

    def __init__(self) -> None:
        self._factory = APIFactory

    def run(
        self,
        *,
        provider: str,
        api_key: str,
        model: Optional[str],
        question: str,
        reasoning_method: str,
        prompt_template: Optional[str],
        max_tokens: int,
        chars_per_line: int,
        max_lines: int,
    ) -> ReasoningResult:
        """Runs the selected reasoning method and returns the structured result."""

        api_client = self._factory.create_api(provider, api_key, model)

        logger.info(
            "Executing reasoning",
            extra={
                "provider": provider,
                "model": model,
                "method": reasoning_method,
                "max_tokens": max_tokens,
            },
        )

        raw_output = api_client.generate_response(
            prompt=question,
            max_tokens=max_tokens,
            prompt_format=prompt_template,
        )

        viz_config = CoTVisualizationConfig(
            max_chars_per_line=chars_per_line,
            max_lines=max_lines,
        )

        method = reasoning_method or "plain"
        method = method.lower()

        if method in {"cot", "chain_of_thought", "chain-of-thought"}:
            return self._handle_cot(raw_output, question, viz_config)
        if method in {"tot", "tree_of_thoughts", "tree-of-thoughts"}:
            return self._handle_tot(raw_output, question, viz_config)
        if method in {"bs", "beam_search", "beam-search"}:
            return self._handle_bs(raw_output, question, viz_config)
        if method in {"l2m", "least_to_most", "least-to-most"}:
            return self._handle_l2m(raw_output, question, viz_config)
        if method in {"srf", "self_refine", "self-refine"}:
            return self._handle_self_refine(raw_output, question, viz_config)
        if method in {"scr", "self_consistency", "self-consistency"}:
            return self._handle_self_consistency(raw_output, question, viz_config)
        if method in {"plain", "plain_text", "plain-text"}:
            return self._handle_plain(raw_output, question)
        if method in {"long_reasoning", "long-reasoning"}:
            return self._handle_plain(raw_output, question)

        # Fallback to plain text if method is unrecognised
        logger.warning("Unknown reasoning method '%s', falling back to plain text", method)
        return self._handle_plain(raw_output, question)

    @staticmethod
    def _visualization_payload(code: str, config: CoTVisualizationConfig) -> Dict[str, Any]:
        return {
            "type": "mermaid",
            "code": code,
            "chars_per_line": config.max_chars_per_line,
            "max_lines": config.max_lines,
        }

    def _handle_plain(self, raw_output: str, question: str) -> ReasoningResult:
        parsed = parse_plain_text_response(raw_output, question)
        return ReasoningResult(raw_output=parsed)

    def _handle_cot(
        self,
        raw_output: str,
        question: str,
        viz_config: CoTVisualizationConfig,
    ) -> ReasoningResult:
        try:
            parsed = parse_cot_response(raw_output, question)
            diagram = create_cot_diagram(parsed, viz_config)
            return ReasoningResult(
                raw_output=raw_output,
                parsed=parsed,
                visualization=self._visualization_payload(diagram, viz_config),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse CoT response, fallback to plain text: %s", exc)
            return self._handle_plain(raw_output, question)

    def _handle_tot(
        self,
        raw_output: str,
        question: str,
        viz_config: CoTVisualizationConfig,
    ) -> ReasoningResult:
        try:
            parsed = parse_tot_response(raw_output, question)
            diagram = create_tot_diagram(parsed, viz_config)
            return ReasoningResult(
                raw_output=raw_output,
                parsed=parsed,
                visualization=self._visualization_payload(diagram, viz_config),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse ToT response, fallback to plain text: %s", exc)
            return self._handle_plain(raw_output, question)

    def _handle_bs(
        self,
        raw_output: str,
        question: str,
        viz_config: CoTVisualizationConfig,
    ) -> ReasoningResult:
        try:
            parsed = parse_bs_response(raw_output, question)
            diagram = create_bs_diagram(parsed, viz_config)
            return ReasoningResult(
                raw_output=raw_output,
                parsed=parsed,
                visualization=self._visualization_payload(diagram, viz_config),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse Beam Search response, fallback to plain text: %s", exc)
            return self._handle_plain(raw_output, question)

    def _handle_l2m(
        self,
        raw_output: str,
        question: str,
        viz_config: CoTVisualizationConfig,
    ) -> ReasoningResult:
        try:
            parsed = parse_l2m_response(raw_output, question)
            diagram = create_l2m_diagram(parsed, viz_config)
            return ReasoningResult(
                raw_output=raw_output,
                parsed=parsed,
                visualization=self._visualization_payload(diagram, viz_config),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse L2M response, fallback to plain text: %s", exc)
            return self._handle_plain(raw_output, question)

    def _handle_self_refine(
        self,
        raw_output: str,
        question: str,
        viz_config: CoTVisualizationConfig,
    ) -> ReasoningResult:
        try:
            parsed = parse_selfrefine_response(raw_output, question)
            diagram = create_selfrefine_diagram(parsed, viz_config)
            return ReasoningResult(
                raw_output=raw_output,
                parsed=parsed,
                visualization=self._visualization_payload(diagram, viz_config),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse Self-Refine response, fallback to plain text: %s", exc)
            return self._handle_plain(raw_output, question)

    def _handle_self_consistency(
        self,
        raw_output: str,
        question: str,
        viz_config: CoTVisualizationConfig,
    ) -> ReasoningResult:
        try:
            parsed = parse_scr_response(raw_output, question)
            diagram = create_scr_diagram(parsed, viz_config)
            return ReasoningResult(
                raw_output=raw_output,
                parsed=parsed,
                visualization=self._visualization_payload(diagram, viz_config),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse Self-Consistency response, fallback to plain text: %s", exc)
            return self._handle_plain(raw_output, question)


__all__ = ["ReasoningService", "ReasoningResult"]
