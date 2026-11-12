import unittest

from src.agents import RandomAgent
from src.game import DRAW, O, X, Board
from src.mcts import MCTSAgent, Node
from src.minimax import WIN_SCORE, MinimaxAgent


def replay_caro(turns: int) -> Board:
    """Dựng lại thế cờ từ ván Alpha-Beta đấu MCTS 1000 mô phỏng."""
    moves = (
        (5, 5), (6, 5), (4, 4), (4, 6), (5, 4), (4, 5), (6, 6), (4, 3),
        (5, 6), (5, 3), (6, 4), (5, 2), (6, 3), (3, 4), (5, 7),
    )
    board = Board()
    for turn, (row, col) in enumerate(moves[:turns]):
        board.play(row - 1, col - 1, X if turn % 2 == 0 else O)
    return board


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.b = Board()
        for col in range(2, 6):
            self.b.play(4, col, X)
        self.b.play(3, 3, O)
        self.b.play(3, 4, O)

    def test_alphabeta_immediate_win(self):
        decision = MinimaxAgent(2).choose_move(self.b, X)
        self.assertIn(decision.move, [(4, 1), (4, 6)])
        self.assertEqual(self.b.move_count, 6)

    def test_minimax_unpruned(self):
        decision = MinimaxAgent(2, alpha_beta=False).choose_move(self.b, X)
        self.assertIn(decision.move, [(4, 1), (4, 6)])

    def test_alphabeta_block(self):
        decision = MinimaxAgent(2).choose_move(self.b, O)
        self.assertIn(decision.move, [(4, 1), (4, 6)])

    def test_mcts_visits_sum(self):
        agent = MCTSAgent(iterations=35, c=1.4, seed=7)
        move = agent.choose_move(self.b, X)
        self.assertTrue(self.b.is_legal(*move.move))
        self.assertEqual(agent.last_root.visits, 35)
        self.assertEqual(sum(ch.visits for ch in agent.last_root.children), 35)
        self.assertTrue(all(ch.visits > 0 for ch in agent.last_root.children))

    def test_mcts_seed_reproducible(self):
        a = MCTSAgent(30, seed=8).choose_move(self.b, X)
        b = MCTSAgent(30, seed=8).choose_move(self.b, X)
        self.assertEqual(a.move, b.move)

    def test_uct_unvisited_priority(self):
        node = Node(self.b, O)
        child = Node(self.b, X, parent=node)
        node.visits = 1
        node.children.append(child)
        self.assertIs(node.best_uct_child(1.4), child)

    def test_random_valid(self):
        d = RandomAgent(8).choose_move(self.b, O)
        self.assertTrue(self.b.is_legal(*d.move))

    def test_pruned_and_unpruned_search_same_tree(self):
        board = Board()
        for row, col, player in ((4, 4, X), (4, 5, O), (5, 4, X), (3, 4, O)):
            board.play(row, col, player)
        before = board.copy()
        raw = MinimaxAgent(depth=2, alpha_beta=False).choose_move(board, X)
        pruned = MinimaxAgent(depth=2).choose_move(board, X)
        self.assertEqual(raw.search_value, pruned.search_value)
        self.assertEqual(raw.move, pruned.move)
        self.assertLess(pruned.nodes, raw.nodes)
        self.assertEqual(pruned.depth, 2)
        self.assertEqual(board.cells, before.cells)
        self.assertEqual(board.last_move, before.last_move)
        self.assertEqual(board.status, before.status)

    def test_minimax_counts_root_and_leaves(self):
        agent = MinimaxAgent(depth=1, alpha_beta=False, max_candidates=3)
        decision = agent.choose_move(self.b, X)
        self.assertEqual(decision.nodes, 4)
        self.assertIn(decision.move, self.b.winning_moves(X))

    def test_candidate_cap_keeps_all_immediate_blocks(self):
        moves = MinimaxAgent(depth=2, max_candidates=1)._ordered_moves(self.b, O)
        self.assertEqual(set(moves), {(4, 1), (4, 6)})

    def test_backpropagation_rewards_both_player_perspectives(self):
        root = Node(self.b, O)
        child = Node(self.b, X, parent=root)
        child.backpropagate(X)
        child.backpropagate(DRAW)
        self.assertEqual((root.visits, root.reward), (2, 0.5))
        self.assertEqual((child.visits, child.reward), (2, 1.5))

    def test_mcts_selects_most_visited_child(self):
        agent = MCTSAgent(iterations=75, seed=19)
        snapshot = self.b.copy()
        decision = agent.choose_move(self.b, O)
        children = agent.last_root.children
        most_visited = max(children, key=lambda child: child.visits)
        self.assertEqual(decision.move, most_visited.move)
        self.assertEqual(decision.iterations, 75)
        self.assertLessEqual(decision.nodes, 76)
        self.assertEqual(self.b.cells, snapshot.cells)
        self.assertEqual(self.b.last_move, snapshot.last_move)

    def test_terminal_node_has_no_untried_moves(self):
        self.b.play(4, 6, X)
        self.assertEqual(Node(self.b, X).untried, [])
        with self.assertRaises(ValueError):
            MCTSAgent().choose_move(self.b, O)

    def test_nonfinite_uct_coefficient_rejected(self):
        for invalid in (0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MCTSAgent(c=invalid)

    def test_alphabeta_prefers_fastest_forced_win(self):
        board = replay_caro(8)
        agent = MinimaxAgent(depth=5, prefer_quick_wins=True)
        decision = agent.choose_move(board, X)
        self.assertEqual(decision.move, (2, 2))
        self.assertEqual(decision.search_value, WIN_SCORE - 3)
        board.play(*decision.move, X)
        self.assertEqual(set(board.winning_moves(X)), {(1, 1), (6, 6)})
        self.assertEqual(board.winning_moves(O), [])

    def test_mcts_considers_only_the_unique_required_block(self):
        board = replay_caro(15)
        block = (4, 7)
        self.assertEqual(board.winning_moves(X), [block])
        self.assertNotIn(block, board.candidate_moves(limit=1))
        agent = MCTSAgent(iterations=4, max_candidates=1, tactical_root=True)
        self.assertEqual(agent.choose_move(board, O).move, block)
        self.assertEqual(agent.last_root.children[0].visits, 4)

    def test_mcts_prioritizes_winning_over_blocking(self):
        board = Board()
        for col in range(4):
            board.play(4, col, X)
            board.play(5, col, O)
        agent = MCTSAgent(iterations=4, tactical_root=True)
        self.assertEqual(agent.choose_move(board, X).move, (4, 4))
        self.assertEqual(agent.last_root.children[0].visits, 4)
