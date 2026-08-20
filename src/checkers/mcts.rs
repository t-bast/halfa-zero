//! Monte-Carol Tree Search for the Chinese Checkers game (inspired by AlphaZero).

use crate::checkers::board::{Board, GameState};

/// One legal move out of a node. Statistics live on edges, not on child nodes, and children are
/// created lazily.
struct Edge {
    /// Move made when going through this edge (from, to).
    mv: (u8, u8),
    /// Evaluator's opinion about this move, prior to actually running the search algorithm: this is
    /// used to decide where the search should start, which can greatly improve learning efficiency
    /// compared to using a uniform distribution over all moves.
    prior: f32,
    /// Number of times that this edge was visited during a simulation.
    visits: u32,
    /// Sum of backed-up values, all in *this node's mover's* frame.
    value_sum: f32,
    /// Child node (created lazily when using that edge during a simulation).
    child: Option<u32>,
}

/// We build a tree representing potential moves.
struct Node {
    /// Current state of the game, associated with this node.
    state: GameState,
    /// Number of moves remaining at this position before the game is stopped (if nobody wins).
    remaining_moves: u16,
    /// Empty until the node is expanded. A non-terminal position always has at least one
    /// legal move, so "empty and not terminal" unambiguously means "not yet expanded".
    edges: Vec<Edge>,
    /// Number of times this node was visited during the search.
    total_visits: u32,
    /// Outcome value from this node's mover's perspective (`None` if the game isn't finished).
    outcome: Option<f32>,
}

/// Anything that can score a position for the search. It can be implemented by stubs (as done
/// below) or by a neural network. Comparing results between different evaluators makes sure the
/// neural network improves: it should beat stubs quickly, and it should beat evaluators based on
/// earlier generations of the neural network.
pub trait Evaluator {
    /// The vector returned contains the probability distribution associated with each move in the
    /// `moves` provided. The sum of values in this vector must sum to 1.
    ///
    /// The lone value returned scores the current state from the perspective of the player about
    /// to move: a positive value means they believe they're winning, while a negative value means
    /// that they believe they're losing. The value must be in [-1;1].
    fn evaluate(&self, board: &Board, state: &GameState, moves: &[(u8, u8)]) -> (Vec<f32>, f32);
}

/// A uniform evaluator, where every move has the same probability and the evaluator doesn't know
/// whether it's winning or losing (always return 0.0).
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&self, _: &Board, _: &GameState, moves: &[(u8, u8)]) -> (Vec<f32>, f32) {
        let move_probability = 1.0 / moves.len() as f32;
        let priors = vec![move_probability; moves.len()];
        (priors, 0.0)
    }
}

/// A distance-based evaluator, which values a position by how much closer the mover is to its
/// target triangle than its opponent.
pub struct DistanceEvaluator {
    /// We use a scaling factor to set how quickly the evaluator saturates (e.g. 0.1 is reasonable
    /// for n = 2, where distances run 0..19).
    pub scale: f32,
    /// Controls the *priors* that will be applied to edges before search. If `None`, we just use
    /// a uniform set of priors. If `Some(t)`, we softmax each move's distance gain with the
    /// temperature `t`: a small `t` concentrates on the best-looking move, a large `t` flattens
    /// back toward uniform.
    pub prior_temperature: Option<f32>,
}

impl DistanceEvaluator {
    /// Lattice steps of progress this move makes for `player`. Positive is closer to home.
    fn gain(&self, board: &Board, player: u8, (from, to): (u8, u8)) -> f32 {
        let from_dist = board.distance_to_target(player, from) as f32;
        let to_dist = board.distance_to_target(player, to) as f32;
        from_dist - to_dist
    }

    fn priors(&self, board: &Board, state: &GameState, moves: &[(u8, u8)]) -> Vec<f32> {
        match self.prior_temperature {
            // If no temperature is provided, we use uniform priors.
            None => vec![1.0 / moves.len() as f32; moves.len()],
            Some(temperature) => {
                debug_assert!(temperature > 0.0, "temperature must be positive");
                let player = state.side();
                let scores: Vec<f32> = moves
                    .iter()
                    .map(|&mv| self.gain(board, player, mv) / temperature)
                    .collect();
                // Subtract the max before exponentiating to avoid overflows.
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let scaled: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
                let sum = scaled.iter().sum::<f32>();
                scaled.iter().map(|&s| s / sum).collect()
            }
        }
    }
}

impl Evaluator for DistanceEvaluator {
    fn evaluate(&self, board: &Board, state: &GameState, moves: &[(u8, u8)]) -> (Vec<f32>, f32) {
        let priors = self.priors(board, state, moves);
        let mover = board.remaining_distance(&state, state.side());
        let adversary = board.remaining_distance(&state, state.side() ^ 1);
        // We use `tanh` to ensure that the returned value is in [-1, 1].
        let value = (((adversary as f32) - (mover as f32)) * self.scale).tanh();
        (priors, value)
    }
}

/// Monte-Carlo Tree Search: we simulate parts of the game over and over again, using the given
/// evaluator to rank our available moves and then choose moves that have been visited the most
/// often.
pub struct Mcts<'a, E: Evaluator> {
    board: &'a Board,
    evaluator: &'a E,
    nodes: Vec<Node>,
    /// "Predictor + Upper Confidence bounds applied to Trees": actually just a tuning parameter
    /// at that point.
    c_puct: f32,
}

impl<'a, E: Evaluator> Mcts<'a, E> {
    pub fn new(board: &'a Board, evaluator: &'a E, c_puct: f32) -> Self {
        Mcts {
            board,
            evaluator,
            nodes: Vec::new(),
            c_puct,
        }
    }

    /// Run simulations from the given starting state and return `(move, visit count)` for
    /// every legal move at the root.
    pub fn search(
        &mut self,
        state: GameState,
        remaining_moves: u16,
        simulations_count: u32,
    ) -> Vec<((u8, u8), u32)> {
        self.nodes.clear();
        let outcome = self.board.outcome_score(&state, remaining_moves);
        debug_assert!(outcome.is_none(), "searching a finished game");
        self.nodes.push(Node {
            state,
            remaining_moves,
            edges: Vec::new(),
            total_visits: 0,
            outcome,
        });
        for _ in 0..simulations_count {
            // We don't care about the value of the root node since it's our starting point.
            self.simulate(0);
        }
        // We return the edges, which represent moves statistics for each legal move that can be
        // taken from our starting state.
        self.nodes[0]
            .edges
            .iter()
            .map(|e| (e.mv, e.visits))
            .collect()
    }

    /// Run a single simulation starting at the given node. Returns the value of that node's
    /// position **from the perspective of the player about to move there**.
    fn simulate(&mut self, node: usize) -> f32 {
        if let Some(value) = self.nodes[node].outcome {
            return value;
        }
        if self.nodes[node].edges.is_empty() {
            // First visit: evaluate and expand, but do *not* descend. This is what makes
            // the tree grow by exactly one node per simulation.
            return self.expand(node);
        }
        // Select the best edge to visit.
        let edge_index = self.select(node);
        // Note that we use indices in the global nodes vector to work around the borrow checker.
        let child = match self.nodes[node].edges[edge_index].child {
            Some(c) => c as usize,
            None => self.create_child(node, edge_index),
        };
        // When simulating the next state, it will be the adversary's turn.
        // The value returned is thus from their point of view, so we simply need to negate it
        // (both players use the same evaluator so values are symmetric).
        let value = -self.simulate(child);
        self.nodes[node].total_visits += 1;
        self.nodes[node].edges[edge_index].visits += 1;
        self.nodes[node].edges[edge_index].value_sum += value;
        value
    }

    /// Pick the edge maximising Q + U, where:
    ///
    ///     U = c_puct * P * sqrt(total_visits) / (1 + visits)
    ///
    /// Q is 0 for an unvisited edge, which makes untried moves look exactly average.
    fn select(&self, node: usize) -> usize {
        let n = &self.nodes[node];
        // Important note: the first time we descend *from* a node its total is still 0, and
        // sqrt(0) would zero out every U, collapsing the argmax onto whichever edge came first
        // rather than onto the prior. We fix that by clamping to 1.
        let sqrt_total = (n.total_visits as f32).max(1.0).sqrt();
        let mut best_edge_idx: usize = 0;
        let mut best_edge_score = f32::NEG_INFINITY;
        for i in 0..n.edges.len() {
            let e = &n.edges[i];
            let q = match e.visits {
                0 => 0.0,
                _ => e.value_sum / e.visits as f32,
            };
            let score = q + self.c_puct * e.prior * sqrt_total / (e.visits + 1) as f32;
            if score > best_edge_score {
                best_edge_score = score;
                best_edge_idx = i;
            }
        }
        best_edge_idx
    }

    /// Evaluate `node`, create its edges, return the value.
    fn expand(&mut self, node: usize) -> f32 {
        let state = self.nodes[node].state;
        let evaluator = self.evaluator;
        let moves = self.board.available_moves(state.mover(), state.adversary());
        debug_assert!(
            !moves.is_empty(),
            "non-terminal position with no legal move"
        );
        let (priors, value) = evaluator.evaluate(self.board, &state, &moves);
        debug_assert_eq!(priors.len(), moves.len());
        let mut edges: Vec<Edge> = Vec::with_capacity(moves.len());
        for i in 0..moves.len() {
            edges.push(Edge {
                mv: moves[i],
                prior: priors[i],
                visits: 0,
                value_sum: 0.0,
                child: None,
            });
        }
        self.nodes[node].edges = edges;
        value
    }

    /// Materialise the child on `edge_index` and return its arena index.
    fn create_child(&mut self, node: usize, edge_index: usize) -> usize {
        let (from, to) = self.nodes[node].edges[edge_index].mv;
        let state_after_move = self.nodes[node].state.apply(from, to);
        // Note that we use saturating_sub to ensure that an arithmetic underflow cannot be silently
        // performed, which would mess up the whole state and would be very hard to debug.
        let remaining_moves = self.nodes[node].remaining_moves.saturating_sub(1);
        let outcome = self.board.outcome_score(&state_after_move, remaining_moves);
        let child_node = Node {
            state: state_after_move,
            remaining_moves,
            edges: Vec::new(),
            total_visits: 0,
            outcome,
        };
        let node_idx = self.nodes.len();
        self.nodes.push(child_node);
        self.nodes[node].edges[edge_index].child = Some(node_idx as u32);
        node_idx
    }

    /// Mean value of an edge at the root, for tests and debugging.
    pub fn root_q(&self, mv: (u8, u8)) -> Option<f32> {
        self.nodes[0]
            .edges
            .iter()
            .find(|&e| e.mv == mv)
            .and_then(|e| match e.visits {
                0 => None,
                _ => Some(e.value_sum / e.visits as f32),
            })
    }
}

/// The move with the most root visits. Visit counts, not Q: counts reflect where the
/// search actually chose to spend its budget and are far less noisy than means at low
/// visit counts.
pub fn best_move(visits: &[((u8, u8), u32)]) -> (u8, u8) {
    let mut best_move: (u8, u8) = visits[0].0;
    let mut best_move_score = visits[0].1;
    for i in 1..visits.len() {
        if visits[i].1 > best_move_score {
            best_move = visits[i].0;
            best_move_score = visits[i].1;
        }
    }
    best_move
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkers::board::Outcome;
    use rand::distr::{Distribution, Uniform};

    #[test]
    fn finds_a_win_in_one() {
        let board = Board::new(2);
        let evaluator = UniformEvaluator;
        let mut mcts = Mcts::new(&board, &evaluator, 1.5);
        // Player 0 to move; (31, 36) hops over 35 and completes the target triangle.
        let state = GameState::state_from(&[34, 35, 31], &[16, 17, 18], 0);
        let visits = mcts.search(state, 100, 200);
        assert_eq!(best_move(&visits), (31, 36));
        // Every simulation through that edge hits a terminal win, so its mean is exactly +1.
        assert_eq!(mcts.root_q((31, 36)), Some(1.0));
    }

    #[test]
    fn the_search_is_side_symmetric() {
        let board = Board::new(2);
        let evaluator = UniformEvaluator;
        let mut mcts = Mcts::new(&board, &evaluator, 1.5);
        // The rotated image of `finds_a_win_in_one`, with player 1 to move.
        let state = GameState::state_from(&[18, 19, 20], &[1, 2, 5], 1);
        let visits = mcts.search(state, 100, 200);
        assert_eq!(best_move(&visits), (5, 0));
        assert_eq!(mcts.root_q((5, 0)), Some(1.0));
    }

    #[test]
    fn visit_counts_are_consistent() {
        let board = Board::new(2);
        let evaluator = UniformEvaluator;
        let mut mcts = Mcts::new(&board, &evaluator, 1.5);
        let visits = mcts.search(board.starting_state(), 400, 200);
        let total: u32 = visits.iter().map(|&(_, n)| n).sum();
        // One simulation expanded the root without descending.
        assert_eq!(total, 199);
        assert_eq!(
            visits.len(),
            board
                .available_moves(
                    board.starting_state().mover(),
                    board.starting_state().adversary()
                )
                .len()
        );
        // With a uniform prior and value 0 everywhere, no move can distinguish itself:
        // visits should be spread, not concentrated.
        let max = visits.iter().map(|&(_, n)| n).max().unwrap();
        assert!(
            max < 100,
            "search concentrated with nothing to concentrate on: {max}"
        );
    }

    #[test]
    fn priors_rank_moves_by_progress() {
        let board = Board::new(2);
        let evaluator = DistanceEvaluator {
            scale: 0.1,
            prior_temperature: Some(1.0),
        };
        let state = board.starting_state();
        let moves = board.available_moves(state.mover(), state.adversary());
        let (priors, _) = evaluator.evaluate(&board, &state, &moves);

        let sum: f32 = priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "priors must be a distribution: {sum}"
        );
        // No move may be assigned zero: a zero prior is unreachable in PUCT, not merely unlikely.
        assert!(priors.iter().all(|&p| p > 0.0));

        // The move with the most progress must carry the most weight.
        let best = moves
            .iter()
            .enumerate()
            .map(|(i, &m)| (i, evaluator.gain(&board, state.side(), m)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let heaviest = priors
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(best, heaviest);
    }

    /// Play multiple games between two move-choosing policies, swapping seats each game.
    /// Returns (wins for `a`, wins for `b`, draws).
    fn play_match(
        board: &Board,
        cap: u16,
        games: u32,
        opening_moves: u16,
        mut a: impl FnMut(&Board, &GameState, u16) -> (u8, u8),
        mut b: impl FnMut(&Board, &GameState, u16) -> (u8, u8),
    ) -> (u32, u32, u32) {
        debug_assert_eq!(cap % 2, 0, "an odd cap gives one seat an extra move");
        let mut rng = rand::rng();
        let mut wins_a: u32 = 0;
        let mut wins_b: u32 = 0;
        let mut draws: u32 = 0;
        for _ in 0..games {
            // We start with random opening moves, otherwise every game would be exactly the same
            // when using a deterministic evaluator such as the DistanceEvaluator.
            let mut opening: Vec<(u8, u8)> = Vec::new();
            let mut state = board.starting_state();
            for _ in 0..opening_moves {
                let moves = board.available_moves(state.mover(), state.adversary());
                let distribution = Uniform::new(0, moves.len()).unwrap();
                let (from, to) = moves[distribution.sample(&mut rng)];
                state = state.apply(from, to);
                opening.push((from, to));
            }
            // We rewind the opening moves, because we play two games: one where a starts and one
            // where b starts. In both cases we want to use the same opening moves for fairness.
            for first_player_is_a in [true, false] {
                state = board.starting_state();
                let mut remaining_moves = cap;
                // Replay the opening moves.
                for &(from, to) in opening.iter() {
                    state = state.apply(from, to);
                    remaining_moves -= 1;
                }
                let result = loop {
                    if let Some(outcome) = board.outcome(&state, remaining_moves) {
                        break outcome;
                    }
                    let (from, to) = match state.side() {
                        0 if first_player_is_a => a(&board, &state, remaining_moves),
                        1 if !first_player_is_a => a(&board, &state, remaining_moves),
                        _ => b(&board, &state, remaining_moves),
                    };
                    state = state.apply(from, to);
                    remaining_moves -= 1;
                };
                match result {
                    Outcome::Win(0) if first_player_is_a => wins_a += 1,
                    Outcome::Win(1) if !first_player_is_a => wins_a += 1,
                    Outcome::Win(_) => wins_b += 1,
                    Outcome::CapWin(0) if first_player_is_a => wins_a += 1,
                    Outcome::CapWin(1) if !first_player_is_a => wins_a += 1,
                    Outcome::CapWin(_) => wins_b += 1,
                    Outcome::Draw => draws += 1,
                }
            }
        }
        (wins_a, wins_b, draws)
    }

    #[test]
    fn search_beats_random() {
        let board = Board::new(2);
        let evaluator = DistanceEvaluator {
            scale: 0.1,
            prior_temperature: None,
        };
        let mut rng = rand::rng();
        let mut mcts_a = Mcts::new(&board, &evaluator, 1.5);
        let (wins, losses, draws) = play_match(
            &board,
            400,
            25,
            4,
            |_, s, r| best_move(&mcts_a.search(*s, r, 200)),
            |b, s, _| {
                let moves = b.available_moves(s.mover(), s.adversary());
                let distribution = Uniform::new(0, moves.len()).unwrap();
                moves[distribution.sample(&mut rng)]
            },
        );
        println!("search {wins} / random {losses} / drawn {draws}");
        assert!(wins > losses * 4, "search should dominate random play");
    }

    #[test]
    fn more_simulations_play_better() {
        // Same evaluator on both sides: the only difference is search depth.
        let board = Board::new(2);
        let evaluator = DistanceEvaluator {
            scale: 0.1,
            prior_temperature: None,
        };
        let mut mcts_a = Mcts::new(&board, &evaluator, 1.5);
        let mut mcts_b = Mcts::new(&board, &evaluator, 1.5);
        let (wins, losses, draws) = play_match(
            &board,
            400,
            30,
            4,
            |_, s, r| best_move(&mcts_a.search(*s, r, 400)),
            |_, s, r| best_move(&mcts_b.search(*s, r, 100)),
        );
        println!("400 sims {wins} / 100 sims {losses} / drawn {draws}");
        assert!(wins > losses, "more search should be at least as strong");
    }

    #[test]
    fn informative_priors_beat_uniform_priors() {
        let board = Board::new(2);
        // We use the same number of simulations for the guided and uniform evaluator.
        let guided = DistanceEvaluator {
            scale: 0.1,
            prior_temperature: Some(1.0),
        };
        let mut mcts_a = Mcts::new(&board, &guided, 1.5);
        let uniform = DistanceEvaluator {
            scale: 0.1,
            prior_temperature: None,
        };
        let mut mcts_b = Mcts::new(&board, &uniform, 1.5);
        // Identical value function, identical simulation count. Only the priors differ.
        let (wins, losses, draws) = play_match(
            &board,
            400,
            30,
            4,
            |_, s, r| best_move(&mcts_a.search(*s, r, 100)),
            |_, s, r| best_move(&mcts_b.search(*s, r, 100)),
        );
        println!("guided priors {wins} / uniform priors {losses} / drawn {draws}");
        assert!(wins > losses);
    }

    #[test]
    fn priors_are_worth_more_than_simulations() {
        let board = Board::new(2);
        // We use a smaller number of simulations for the guided evaluator, but it should still
        // perform better.
        let guided = DistanceEvaluator {
            scale: 0.1,
            prior_temperature: Some(1.0),
        };
        let mut mcts_a = Mcts::new(&board, &guided, 1.5);
        let uniform = DistanceEvaluator {
            scale: 0.1,
            prior_temperature: None,
        };
        let mut mcts_b = Mcts::new(&board, &uniform, 1.5);
        // Guided gets a quarter of the budget.
        let (wins, losses, draws) = play_match(
            &board,
            400,
            30,
            4,
            |_, s, r| best_move(&mcts_a.search(*s, r, 100)),
            |_, s, r| best_move(&mcts_b.search(*s, r, 400)),
        );
        println!("guided@100 {wins} / uniform@400 {losses} / drawn {draws}");
    }
}
