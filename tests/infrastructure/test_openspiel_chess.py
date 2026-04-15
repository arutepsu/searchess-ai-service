from __future__ import annotations

from unittest.mock import MagicMock

from searchess_ai.infrastructure.inference._openspiel_chess import (
    get_legal_move_candidates,
    get_legal_move_strings,
    select_move,
)


def _make_mock_state(legal_move_strings: list[str]) -> MagicMock:
    state = MagicMock()
    state.current_player.return_value = 0
    state.legal_actions.return_value = list(range(len(legal_move_strings)))
    state.action_to_string.side_effect = (
        lambda player, action: legal_move_strings[action]
    )
    return state


class TestGetLegalMoveStrings:
    def test_returns_uci_strings_for_all_legal_actions(self) -> None:
        state = _make_mock_state(["e2e4", "d2d4", "g1f3"])
        assert get_legal_move_strings(state) == ["e2e4", "d2d4", "g1f3"]

    def test_passes_current_player_to_action_to_string(self) -> None:
        state = _make_mock_state(["e2e4"])
        state.current_player.return_value = 1
        get_legal_move_strings(state)
        state.action_to_string.assert_called_with(1, 0)

    def test_returns_empty_list_when_no_legal_actions(self) -> None:
        state = MagicMock()
        state.current_player.return_value = 0
        state.legal_actions.return_value = []
        assert get_legal_move_strings(state) == []


class TestGetLegalMoveCandidates:
    def test_returns_candidates_in_legal_action_order(self) -> None:
        state = _make_mock_state(["e2e4", "d2d4", "g1f3"])

        candidates = get_legal_move_candidates(state)

        assert [candidate.move for candidate in candidates] == ["e2e4", "d2d4", "g1f3"]

    def test_marks_promotion_from_uci_suffix(self) -> None:
        state = _make_mock_state(["e7e8q", "e2e4"])

        candidates = get_legal_move_candidates(state)

        assert candidates[0].is_promotion is True
        assert candidates[1].is_promotion is False

    def test_marks_capture_when_state_exposes_is_capture_action_only(self) -> None:
        state = _make_mock_state(["d4e5", "e2e4"])
        state.is_capture.side_effect = lambda action: action == 0

        candidates = get_legal_move_candidates(state)

        assert candidates[0].is_capture is True
        assert candidates[1].is_capture is False

    def test_marks_capture_when_state_exposes_is_capture_player_action(self) -> None:
        class CustomState:
            def current_player(self) -> int:
                return 0

            def legal_actions(self) -> list[int]:
                return [0, 1]

            def action_to_string(self, player: int, action: int) -> str:
                return ["d4e5", "e2e4"][action]

            def is_capture(self, player: int, action: int) -> bool:
                return action == 0

        candidates = get_legal_move_candidates(CustomState())

        assert candidates[0].is_capture is True
        assert candidates[1].is_capture is False

    def test_returns_empty_list_when_no_legal_actions(self) -> None:
        state = MagicMock()
        state.current_player.return_value = 0
        state.legal_actions.return_value = []

        candidates = get_legal_move_candidates(state)

        assert candidates == []