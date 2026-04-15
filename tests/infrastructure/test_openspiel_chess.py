from searchess_ai.infrastructure.inference._openspiel_chess import (
    get_legal_move_candidates,
    get_legal_move_strings,
    select_move,
)
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
        state = _make_mock_state(["d4e5", "e2e4"])

        def is_capture(player: int, action: int) -> bool:
            return player == 0 and action == 0

        state.is_capture.side_effect = TypeError("wrong signature")
        del state.is_capture

        class CustomState:
            def current_player(self) -> int:
                return 0

            def legal_actions(self) -> list[int]:
                return [0, 1]

            def action_to_string(self, player: int, action: int) -> str:
                return ["d4e5", "e2e4"][action]

            def is_capture(self, player: int, action: int) -> bool:
                return is_capture(player, action)

        custom_state = CustomState()

        candidates = get_legal_move_candidates(custom_state)

        assert candidates[0].is_capture is True
        assert candidates[1].is_capture is False

    def test_returns_empty_list_when_no_legal_actions(self) -> None:
        state = MagicMock()
        state.current_player.return_value = 0
        state.legal_actions.return_value = []

        candidates = get_legal_move_candidates(state)

        assert candidates == []