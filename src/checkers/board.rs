//! Board geometry for Chinese Checkers, parameterized by the board size `N`. Real Chinese Checkers
//! use `N = 4`, but smaller sizes can be useful for early training and experimentation.
//!
//! The board is a six-pointed star: a central hexagon of "radius" `N` with six triangles of side
//! `N` attached to its edges. That gives `6N(N+1) + 1` holes and `N(N+1)/2` pieces per player
//! (exactly the number of holes in one triangle, since a player's pieces fill one exactly).
//! `N = 4` is the standard board: 121 holes, 10 pieces each.
//! `N = 2` gives 37 holes and 3 pieces, which is small enough to hand-check and fast enough for
//! random games to actually terminate.
//!
//! Cube coordinates are *build-time scaffolding only*. We use them here to enumerate the holes and
//! to precompute flat lookup tables (neighbours, jumps, symmetry permutations), but everything
//! downstream (move generation, bitboards, MCTS) works purely on `u8` hole indices and never
//! performs coordinate arithmetic in a hot loop.

use std::fmt;
use std::ops::Range;

/// The Chinese Checkers board uses rows that are offset (it's not a rectangular grid). Each hole
/// on the board is connected to six adjacent holes that form the tips of a hexagon centered on
/// the current hole. With N = 2, it looks like this:
///
///                   o
///                 o   o
///       o   o   o   o   o   o   o
///         o   o   o   o   o   o
///           o   o   o   o   o
///         o   o   o   o   o   o
///       o   o   o   o   o   o   o
///                 o   o
///                   o
///
/// We use "cubic" coordinates to represent holes on the board. The X axis is vertical, then the Y
/// axis is obtained by rotating the X axis by 120° counter-clockwise. The Z axis is obtained
/// by rotating the X axis by 120° clockwise. The distance between two adjacent points in those
/// coordinates is sqrt(2).
///
/// Using those coordinates with the invariant `x + y + z == 0` lets us represent all the holes in
/// the board plane (what we're doing is actually taking the intersection between a plane and a 3D
/// space, which lets us represent a 2D board) with the additional constraint that at least two of
/// |x|, |y| and |z| are <= N. If all three are <= N, we are inside the central hexagon: if only two
/// of them are <= N, we are inside one of the external triangles.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Coordinate {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Coordinate {
    /// Build a coordinate from two free components; the third is forced by `x + y + z == 0`.
    pub const fn new(x: i32, y: i32) -> Self {
        Coordinate { x, y, z: -x - y }
    }

    /// Translation. Adding two sum-zero triples gives a sum-zero triple, so this can never leave
    /// the lattice plane.
    pub fn add(self, other: Coordinate) -> Coordinate {
        Coordinate {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Scalar multiple. Used to reach the landing hole of a hop: `c + 2d`.
    pub fn scale(self, k: i32) -> Coordinate {
        Coordinate {
            x: self.x * k,
            y: self.y * k,
            z: self.z * k,
        }
    }

    /// Number of single steps between two holes, ignoring occupancy and board edges.
    pub fn distance(self, other: Coordinate) -> i32 {
        // A single step changes one coordinate by +1 and another one by -1.
        // It reduces the sum of absolute coordinates by at most 2.
        // Since there is always a "most direct" path, it is thus easy to compute.
        let dx = (self.x - other.x).abs();
        let dy = (self.y - other.y).abs();
        let dz = (self.z - other.z).abs();
        (dx + dy + dz) / 2
    }

    /// Is this point one of the holes of the board?
    pub fn on_board(self, n: i32) -> bool {
        // If all three coordinates are smaller than `N`, we are inside the central hexagon.
        // If only two coordinates are smaller than `N`, we are inside one of the external triangles.
        // Otherwise, we're outside the board.
        let mut count = 0;
        count += (self.x.abs() <= n) as i32;
        count += (self.y.abs() <= n) as i32;
        count += (self.z.abs() <= n) as i32;
        count >= 2
    }

    /// Rotation by 180° about the center of the board.
    pub fn rotate_180(self) -> Coordinate {
        Coordinate {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Reflection in the axis through the two home triangles.
    pub fn mirror(self) -> Coordinate {
        Coordinate {
            x: self.x,
            y: self.z,
            z: self.y,
        }
    }
}

/// The game state is very small: it can be used in a hash map to detect duplicates, which is very
/// important during the learning phase since we don't care about the history of the moves when
/// deciding the next move, we only care about the board state.
///
/// We thus keep this struct minimal: don't add any field that would mess up duplicate states
/// detection.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GameState {
    /// Since the board contains 121 holes, we encode the state of a player as a 128-bit integer
    /// interpreted as a bitfield of occupied holes. This very cheap to copy, mutate and check.
    players: [u128; 2],
    /// Index of the player whose turn it is inside the `players` list.
    side: u8,
}

impl GameState {
    pub fn side(&self) -> u8 {
        self.side
    }

    /// Pieces of the player about to move.
    pub fn mover(&self) -> u128 {
        match self.side {
            0 => self.players[0],
            _ => self.players[1],
        }
    }

    /// Pieces of the player who just moved.
    pub fn adversary(&self) -> u128 {
        match self.side {
            0 => self.players[1],
            _ => self.players[0],
        }
    }

    pub fn player(&self, p: u8) -> u128 {
        self.players[p as usize]
    }

    /// Apply a legal move for the side to move and hand the turn to the next player.
    pub fn apply(&self, from: u8, to: u8) -> GameState {
        let mut moving = self.mover();
        debug_assert!(Board::is_occupied(moving, from), "no piece at origin");
        debug_assert!(
            !Board::is_occupied(self.players[0] | self.players[1], to),
            "destination occupied"
        );
        moving ^= 1u128 << from; // unset the previous position
        moving |= 1u128 << to; // set the new position
        GameState {
            players: match self.side {
                0 => [moving, self.players[1]],
                _ => [self.players[0], moving],
            },
            side: self.side ^ 1,
        }
    }

    /// Helper function that sets the state to occupy the given positions (for tests).
    pub fn state_from(p0: &[u8], p1: &[u8], side: u8) -> GameState {
        let mask = |holes: &[u8]| holes.iter().fold(0u128, |m, &i| m | (1u128 << i));
        let state = GameState {
            players: [mask(p0), mask(p1)],
            side,
        };
        assert_eq!(
            state.players[0] & state.players[1],
            0,
            "overlapping pieces in fixture"
        );
        state
    }
}

/// How a finished game ended. `Win` and `CapWin` produce the same training target, but keeping them
/// distinct matters for diagnostics: the fraction of games decided by a real win/loss (instead of
/// an unofficial "win" after reaching the maximum number of rounds) is the clearest signal of
/// whether self-play is actually learning to finish.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// This player filled their target triangle (official win).
    Win(u8),
    /// We've reached the maximum number of rounds without an official winner.
    /// We compute which player was strictly closer to winning.
    CapWin(u8),
    /// The maximum number of rounds was reached with both players equally far from winning.
    Draw,
}

impl Outcome {
    pub fn winner(&self) -> Option<u8> {
        match self {
            Outcome::Win(player) => Some(*player),
            Outcome::CapWin(player) => Some(*player),
            Outcome::Draw => None,
        }
    }
}

/// Immutable board geometry, built once at startup and shared for the lifetime of the program.
/// We precompute potential moves in the `neighbours` and `jumps` fields for every position.
pub struct Board {
    /// Radius of the board (N = 4 for the official game).
    n: i32,
    /// Hole `i` in cubic coordinates, ordered lexicographically by `(x, y)`.
    holes: Vec<Coordinate>,
    /// Reverse map that returns the index of a given [Coordinate] in the `holes` array, or `None`
    /// for off-board points.
    /// The index is simply `(x + 2n, y + 2n)` (since `z` is implied by `x + y + z = 0`).
    lookup: Vec<Option<u8>>,
    /// `neighbours[i][d]` = hole reached by one step from `i` in direction `d`, if on board.
    neighbours: Vec<[Option<u8>; 6]>,
    /// `neighbour_mask[i]` = bitmask of every hole adjacent to `i`.
    /// The same information as `neighbours[i]`, but in a form you can AND against occupancy.
    neighbour_mask: Vec<u128>,
    /// `jumps[i][d]` = `(hole jumped over, landing hole)` for a hop from `i` in direction `d`.
    jumps: Vec<[Option<(u8, u8)>; 6]>,
    /// `rotation[i]` = index of hole `i` after a 180° rotation.
    rotation: Vec<u8>,
    /// `reflection[i]` = index of hole `i` after mirroring.
    reflection: Vec<u8>,
    /// We precompute the target triangle that each player must fill to win.
    /// This is the *opposite* home to where they start.
    target_mask: [u128; 2],
    /// For each player, we precompute a static table with the distance from each hole in the board
    /// to the nearest hole of the player's target triangle (or 0 for holes already in the target).
    distance_to_target: [Vec<u8>; 2],
}

impl Board {
    pub fn new(n: i32) -> Self {
        assert!(n >= 1, "board size must be at least 1");
        // We don't allow creating boards larger than the official board, which contains 121 holes
        // and lets us efficiently encode game state in 128-bit integers.
        assert!(n <= 4, "board size must be at most 4");

        let span = 4 * n + 1;
        let mut holes: Vec<Coordinate> = Vec::new();
        let mut lookup: Vec<Option<u8>> = vec![None; (span * span) as usize];

        // Enumerate lexicographically by (x, y). Since home membership is exactly `x >= n+1` or
        // `x <= -(n+1)`, ascending x puts one home triangle at the very start of the index range
        // and the other at the very end: both target masks end up as contiguous bit ranges, which
        // makes them easy to construct and easy to eyeball.
        for x in -2 * n..=2 * n {
            for y in -2 * n..=2 * n {
                let c = Coordinate::new(x, y);
                if !c.on_board(n) {
                    continue;
                }
                let index = holes.len();
                assert!(index < u8::MAX as usize, "too many holes for u8 indices");
                lookup[Self::lookup_key(n, c)] = Some(index as u8);
                holes.push(c);
            }
        }

        let mut board = Board {
            n,
            holes,
            lookup,
            neighbours: Vec::new(),
            neighbour_mask: Vec::new(),
            target_mask: [0u128, 0u128],
            distance_to_target: [Vec::new(), Vec::new()],
            jumps: Vec::new(),
            rotation: Vec::new(),
            reflection: Vec::new(),
        };

        board.neighbours = board.build_neighbours();
        board.neighbour_mask = board.build_neighbour_masks();
        board.target_mask = board.build_target_masks();
        board.distance_to_target = board.build_distance_tables();
        board.jumps = board.build_jumps();
        board.rotation = board.build_permutation(Coordinate::rotate_180);
        board.reflection = board.build_permutation(Coordinate::mirror);
        board
    }

    /// The six directions: every permutation of `(+1, -1, 0)`.
    ///
    /// These are forced, not chosen: they are the shortest integer vectors lying in the plane
    /// `x + y + z = 0`. Note that no direction changes only one coordinate, which would break the
    /// `x + y + z = 0` invariant.
    ///
    /// Ordered so that `DIRECTIONS[(d + 3) % 6]` is the opposite of `DIRECTIONS[d]`. That relation
    /// is worth preserving; it makes the neighbour-symmetry test below trivial to write.
    pub const DIRECTIONS: [Coordinate; 6] = [
        Coordinate { x: 1, y: -1, z: 0 },
        Coordinate { x: 1, y: 0, z: -1 },
        Coordinate { x: 0, y: 1, z: -1 },
        Coordinate { x: -1, y: 1, z: 0 },
        Coordinate { x: -1, y: 0, z: 1 },
        Coordinate { x: 0, y: -1, z: 1 },
    ];

    /// Index of the direction opposite to `d`.
    pub const fn opposite(d: usize) -> usize {
        (d + 3) % 6
    }

    /// Flat offset into the `lookup` reverse map use to obtain the corresponding index in `holes`.
    /// Only valid for `|x|, |y| <= 2n`; callers must bounds-check first.
    fn lookup_key(n: i32, c: Coordinate) -> usize {
        let span = 4 * n + 1;
        ((c.x + 2 * n) * span + (c.y + 2 * n)) as usize
    }

    /// Returns the [Self::lookup_key] for on-board holes and `None` otherwise.
    fn safe_lookup_key(n: i32, c: Coordinate) -> Option<usize> {
        match c.on_board(n) {
            false => None,
            true => Some(Self::lookup_key(n, c)),
        }
    }

    /// For every hole and every direction, we pre-compute the neighbour hole index if it belongs
    /// to the board.
    fn build_neighbours(&self) -> Vec<[Option<u8>; 6]> {
        let mut neighbours: Vec<[Option<u8>; 6]> = Vec::new();
        for i in 0..self.len() {
            let current = self.holes[i];
            neighbours.push(Self::DIRECTIONS.map(|direction| {
                let neighbour = current.add(direction);
                Self::safe_lookup_key(self.n, neighbour).and_then(|idx| self.lookup[idx])
            }));
        }
        neighbours
    }

    fn build_neighbour_masks(&self) -> Vec<u128> {
        let mut neighbour_masks: Vec<u128> = Vec::new();
        for i in 0..self.len() {
            let neighbour_mask = self.neighbours[i]
                .iter()
                .flatten()
                .fold(0u128, |mask, &j| mask | (1u128 << j));
            neighbour_masks.push(neighbour_mask);
        }
        neighbour_masks
    }

    fn build_target_masks(&self) -> [u128; 2] {
        let (bottom, top) = self.home_ranges();
        let mask = |r: Range<usize>| r.fold(0u128, |mask, i| mask | (1u128 << i));
        [mask(top), mask(bottom)]
    }

    fn build_distance_tables(&self) -> [Vec<u8>; 2] {
        let (bottom, top) = self.home_ranges();
        let table = |target: Range<usize>| -> Vec<u8> {
            (0..self.len())
                .map(|i| {
                    if target.contains(&i) {
                        0u8
                    } else {
                        target
                            .clone()
                            .map(|j| self.holes[i].distance(self.holes[j]) as u8)
                            .min()
                            .unwrap()
                    }
                })
                .collect()
        };
        [table(top), table(bottom)]
    }

    /// For every hole and every direction, we pre-compute the jump over the neighbour in that
    /// direction and store the jumped-over neighbour and the landing hole.
    /// Both must exist as holes: you cannot jump over a point that isn't on the board, and you
    /// cannot land off the board. We store both, because move generation needs to test whether the
    /// jumped hole is *occupied* and the landing hole is *empty*.
    ///
    /// Note that `jumps[i][d]` being `Some` and `neighbours[i][d]` being `Some` are not the same
    /// condition: the second is implied by the first, but not the reverse.
    fn build_jumps(&self) -> Vec<[Option<(u8, u8)>; 6]> {
        let mut jumps: Vec<[Option<(u8, u8)>; 6]> = Vec::new();
        for i in 0..self.len() {
            let current = self.holes[i];
            jumps.push(Self::DIRECTIONS.map(|direction| {
                let neighbour = current.add(direction);
                match Self::safe_lookup_key(self.n, neighbour).and_then(|i| self.lookup[i]) {
                    None => None, // cannot jump over a neighbour that isn't on the board
                    Some(neighbour_idx) => {
                        // We land after that neighbour, in the same direction.
                        let landing = neighbour.add(direction);
                        match Self::safe_lookup_key(self.n, landing).and_then(|i| self.lookup[i]) {
                            None => None, // cannot jump outside the board
                            Some(landing_idx) => Some((neighbour_idx, landing_idx)),
                        }
                    }
                }
            }))
        }
        jumps
    }

    /// `perm[i]` = index of the hole that `map` sends hole `i` to.
    /// Note that `map` must be a valid permutation, otherwise we'll throw.
    fn build_permutation(&self, map: fn(Coordinate) -> Coordinate) -> Vec<u8> {
        let mut permutation: Vec<u8> = Vec::new();
        for i in 0..self.len() {
            let rotated = map(self.holes[i]);
            assert!(
                rotated.on_board(self.n),
                "permutation yielded an off-board hole"
            );
            let rotated_idx = self.lookup[Self::lookup_key(self.n, rotated)];
            // Any genuine symmetry of the star must map holes to holes, so every lookup here should
            // succeed: if one returns `None`, either `map` isn't a symmetry or `on_board` is wrong.
            // Assert rather than silently skipping: a partial permutation would corrupt every
            // canonicalized position downstream in a way that's very hard to trace.
            assert!(!rotated_idx.is_none(), "invalid permutation");
            permutation.push(rotated_idx.unwrap());
        }
        permutation
    }

    pub fn n(&self) -> i32 {
        self.n
    }

    /// Total number of holes.
    pub fn len(&self) -> usize {
        self.holes.len()
    }

    /// Number of pieces each player owns (= size of one triangle).
    pub fn pieces_per_player(&self) -> usize {
        (self.n * (self.n + 1) / 2) as usize
    }

    pub fn hole(&self, index: usize) -> Coordinate {
        self.holes[index]
    }

    pub fn index_of(&self, c: Coordinate) -> Option<usize> {
        debug_assert_eq!(c.x + c.y + c.z, 0, "coordinates must sum to zero");
        let n = self.n;
        if c.x.abs() > 2 * n || c.y.abs() > 2 * n || c.z.abs() > 2 * n {
            return None;
        }
        self.lookup[Self::lookup_key(n, c)].map(|i| i as usize)
    }

    pub fn neighbour(&self, index: usize, direction: usize) -> Option<usize> {
        self.neighbours[index][direction].map(|i| i as usize)
    }

    /// `(jumped over, landing in)` for a hop from `index` in `direction`.
    pub fn jump(&self, index: usize, direction: usize) -> Option<(usize, usize)> {
        self.jumps[index][direction].map(|(o, l)| (o as usize, l as usize))
    }

    pub fn rotate(&self, index: usize) -> usize {
        self.rotation[index] as usize
    }

    pub fn reflect(&self, index: usize) -> usize {
        self.reflection[index] as usize
    }

    /// The two home triangles, as contiguous index ranges: `(bottom, top)`.
    pub fn home_ranges(&self) -> (Range<usize>, Range<usize>) {
        // We ordered holes so that the first player's triangle is simply the first elements, while
        // the second player's triangle is simply the last elements.
        let range1 = 0..self.pieces_per_player();
        let range2 = (self.len() - self.pieces_per_player())..self.len();
        (range1, range2)
    }

    /// Return the initial states of the two players.
    /// Note that the initial state of the first player is the winning state of the second player,
    /// and the other way around.
    pub fn starting_state(&self) -> GameState {
        let (r1, r2) = self.home_ranges();
        let mut player1: u128 = 0;
        r1.for_each(|i| player1 |= 1u128 << i);
        let mut player2: u128 = 0;
        r2.for_each(|i| player2 |= 1u128 << i);
        GameState {
            players: [player1, player2],
            side: 0,
        }
    }

    /// Returns the target winning state for the given player.
    pub fn winning_state(&self, player: u8) -> u128 {
        self.target_mask[player as usize]
    }

    /// We compute a lower-bound on the distance of a player to its winning state.
    /// Note that this is just an estimate, which depends on game conditions.
    pub fn remaining_distance(&self, state: &GameState, player: u8) -> u32 {
        let mut player_state = state.player(player);
        let mut total_distance = 0u32;
        while player_state != 0 {
            let from = player_state.trailing_zeros() as u8;
            player_state &= player_state - 1;
            total_distance += self.distance_to_target[player as usize][from as usize] as u32;
        }
        total_distance
    }

    /// Lattice distance from `hole` to the nearest hole of `player`'s target triangle.
    pub fn distance_to_target(&self, player: u8, hole: u8) -> u8 {
        self.distance_to_target[player as usize][hole as usize]
    }

    /// All legal moves for `player` as `(from, to)` pairs (optimized implementation using binary
    /// operations, avoiding heap-allocated collections).
    ///
    /// A move is either one step into an adjacent empty hole, or a chain of one or more hops.
    /// Because hops never remove the jumped piece, the board is *static* for the whole duration
    /// of a chain — so enumerating chains is plain reachability on a fixed graph, not a search.
    /// And because the path is unobservable in the resulting position, the destination alone
    /// identifies the move: we return a set of destinations, never paths.
    pub fn available_moves(&self, player: u128, adversary: u128) -> Vec<(u8, u8)> {
        debug_assert!(self.len() <= u8::MAX as usize + 1);
        debug_assert_eq!(player & adversary, 0, "overlapping pieces");

        // Positions occupied by us or our adversary.
        let occupied_positions = player | adversary;
        let mut moves: Vec<(u8, u8)> = Vec::with_capacity(64);

        let mut pieces = player;
        while pieces != 0 {
            // trailing_zeroes provides the lowest bit set: x &= x - 1 clears it (subtracting 1
            // flips the lowest set bit to 0 and turns every zero below it into a 1, so the AND
            // wipes exactly that one bit).
            let from = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
            let from_bit = 1u128 << from;

            // Lift the moving piece off its origin: for the duration of this move the origin is
            // an empty hole, so it can be neither jumped over nor treated as blocked.
            let occupied = occupied_positions & !from_bit;

            // We first compute moves to direct neighbours that aren't already occupied (either by
            // us or our adversary).
            let mut available_neighbours = self.neighbour_mask[from] & !occupied;
            while available_neighbours != 0 {
                let to = available_neighbours.trailing_zeros() as usize;
                available_neighbours &= available_neighbours - 1;
                moves.push((from as u8, to as u8));
            }

            // Then we compute jumps recursively (we can chain several jumps in a single move).
            // Note that the landing hole must not be occupied (either by us or our adversary).
            // The `reached` variable tracks every hole this piece can land on, while `frontier`
            // tracks the holes we still have to use as starting hole.
            // Seeding `reached` with the origin does double duty: it stops the chain re-entering
            // its own starting hole, and it keeps the null move out of the result.
            let mut reached = from_bit;
            let mut frontier = from_bit;
            while frontier != 0 {
                let idx = frontier.trailing_zeros() as usize;
                frontier &= frontier - 1;
                // We check each jump direction.
                for d in 0..6 {
                    if let Some((over, to)) = self.jumps[idx][d] {
                        let to_bit = 1u128 << to;
                        let over_occupied = occupied & (1u128 << over) != 0;
                        let landing_free = (occupied | reached) & to_bit == 0;
                        if over_occupied && landing_free {
                            reached |= to_bit;
                            frontier |= to_bit;
                        }
                    }
                }
            }
            // At that point, reached contains all the holes we can reach by jumps.
            // We simply need to add them to our available moves.
            let mut chains = reached & !from_bit;
            while chains != 0 {
                let to = chains.trailing_zeros() as usize;
                chains &= chains - 1;
                moves.push((from as u8, to as u8));
            }
        }

        moves
    }

    /// The result of the game, or `None` if it is still running.
    /// This should be checked after every move.
    pub fn outcome(&self, state: &GameState, remaining_moves: u16) -> Option<Outcome> {
        // Only the player who just moved can have completed their triangle.
        let just_moved = state.side ^ 1;
        debug_assert!(
            state.mover() != self.winning_state(state.side),
            "side to move has already won; an outcome() check was missed"
        );
        match remaining_moves {
            // If the player who just moved has completed their triangle, they win.
            _ if state.adversary() == self.winning_state(just_moved) => {
                Some(Outcome::Win(just_moved))
            }
            // If we've exhausted all rounds without a winner, we let the player that is closer to
            // its winning state win (if any).
            0 => match [0u8, 1u8].map(|i| self.remaining_distance(state, i)) {
                [dist0, dist1] if dist0 < dist1 => Some(Outcome::CapWin(0)),
                [dist0, dist1] if dist0 > dist1 => Some(Outcome::CapWin(1)),
                _ => Some(Outcome::Draw),
            },
            _ => None, // the game is still running
        }
    }

    /// The score associated with a game's outcome from the perspective of the side to move:
    /// `+1` win, `-1` loss, `0` draw. `None` if the game is still running.
    pub fn outcome_score(&self, state: &GameState, remaining_moves: u16) -> Option<f32> {
        self.outcome(state, remaining_moves)
            .map(|outcome| match outcome {
                Outcome::Win(winner) if winner == state.side => 1.0,
                Outcome::Win(_) => -1.0,
                Outcome::CapWin(winner) if winner == state.side => 1.0,
                Outcome::CapWin(_) => -1.0,
                Outcome::Draw => 0.0,
            })
    }

    pub fn is_occupied(state: u128, position: u8) -> bool {
        (state & (1u128 << position)) != 0
    }

    /// Render the board, using `glyph(index)` to choose the character for each hole.
    ///
    /// The layout falls out of the screen mapping for cubic coordinates.
    /// With the `x` axis pointing straight up:
    ///
    /// ```text
    ///     screen_x  proportional to  (z - y)
    ///     screen_y  proportional to  -x        (using y + z = -x to eliminate y and z)
    /// ```
    ///
    /// Since `screen_y` depends only on `x`, sets of constant `x` are straight horizontal rows —
    /// so one row of output per `x`, descending, which puts the top home triangle at the top.
    ///
    /// Within a row, `y` changes by 1 between adjacent holes, so `z - y` changes by 2: holes land
    /// on every *other* character column. And because `z - y = -x - 2y`, the parity of `z - y`
    /// equals the parity of `x` — which is exactly the half-cell stagger between consecutive rows
    /// of a triangular lattice, reproduced for free.
    ///
    /// `z - y` ranges over `[-3n, 3n]` (the extremes are the apexes of the four side triangles),
    /// so a row is `6n + 1` characters wide.
    pub fn render<F: Fn(usize) -> char>(&self, glyph: F) -> String {
        let n = self.n;
        let width = (6 * n + 1) as usize;
        let mut out = String::new();

        for x in (-2 * n..=2 * n).rev() {
            let mut cells = vec![' '; width];
            for y in -2 * n..=2 * n {
                let c = Coordinate::new(x, y);
                if let Some(index) = self.index_of(c) {
                    let column = (c.z - c.y + 3 * n) as usize;
                    debug_assert!(column < width, "column {} outside row width", column);
                    cells[column] = glyph(index);
                }
            }
            let row: String = cells.into_iter().collect();
            out.push_str(&format!("{:>3}  {}\n", x, row.trim_end()));
        }
        out
    }

    /// Render with the two home triangles marked, for verifying the geometry by eye.
    pub fn render_homes(&self) -> String {
        let n = self.n;
        self.render(|index| {
            let c = self.hole(index);
            if c.x >= n + 1 {
                'A'
            } else if c.x <= -(n + 1) {
                'B'
            } else {
                '.'
            }
        })
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render(|_| '.'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::distr::Uniform;
    use rand::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn ascii_layout() {
        let g = Board::new(2);
        const STAR_N2: &str = concat!(
            "  4        .\n",
            "  3       . .\n",
            "  2  . . . . . . .\n",
            "  1   . . . . . .\n",
            "  0    . . . . .\n",
            " -1   . . . . . .\n",
            " -2  . . . . . . .\n",
            " -3       . .\n",
            " -4        .\n",
        );
        assert_eq!(g.render(|_| '.'), STAR_N2);
        // We print the real board for manual verification as well.
        println!("{}", Board::new(4).render_homes());
    }

    #[test]
    fn hole_counts() {
        for n in 1..=4 {
            let g = Board::new(n);
            assert_eq!(
                g.len() as i32,
                6 * n * (n + 1) + 1,
                "hole count for n = {}",
                n
            );
        }
        assert_eq!(Board::new(4).len(), 121);
        assert_eq!(Board::new(4).pieces_per_player(), 10);
        assert_eq!(Board::new(2).len(), 37);
        assert_eq!(Board::new(2).pieces_per_player(), 3);
    }

    #[test]
    fn notches_are_excluded() {
        let g = Board::new(4);
        // Apexes of the six triangles: two coordinates at the limit, one at 2n. On board.
        assert!(g.index_of(Coordinate::new(8, -4)).is_some()); // top home apex
        assert!(g.index_of(Coordinate::new(-8, 4)).is_some()); // bottom home apex
        assert!(g.index_of(Coordinate::new(-4, 8)).is_some()); // side triangle apex
        // Notches between adjacent points: only one coordinate within n.
        assert!(g.index_of(Coordinate::new(5, -5)).is_none());
        assert!(g.index_of(Coordinate::new(8, -8)).is_none());
        assert!(g.index_of(Coordinate::new(-5, 0)).is_none());
        // Well outside.
        assert!(g.index_of(Coordinate::new(20, -10)).is_none());
    }

    #[test]
    fn lookup_roundtrips() {
        let g = Board::new(4);
        for i in 0..g.len() {
            assert_eq!(g.index_of(g.hole(i)), Some(i));
        }
    }

    #[test]
    fn neighbours_are_symmetric() {
        let g = Board::new(4);
        for i in 0..g.len() {
            for d in 0..6 {
                if let Some(j) = g.neighbour(i, d) {
                    assert_eq!(g.neighbour(j, Board::opposite(d)), Some(i));
                    assert_ne!(i, j);
                }
            }
        }
    }

    #[test]
    fn neighbour_degrees() {
        let g = Board::new(4);
        let degree = |c: Coordinate| {
            let i = g.index_of(c).unwrap();
            (0..6).filter(|&d| g.neighbour(i, d).is_some()).count()
        };
        assert_eq!(degree(Coordinate::new(0, 0)), 6); // interior
        assert_eq!(degree(Coordinate::new(8, -4)), 2); // triangle apex
        // Vertex of the central hexagon, where two triangles meet. Five neighbours: the only
        // missing direction is the one pointing straight out at the notch, (5, 0, -5).
        assert_eq!(degree(Coordinate::new(4, 0)), 5);
    }

    #[test]
    fn jumps_agree_with_neighbours() {
        let g = Board::new(4);
        for i in 0..g.len() {
            for d in 0..6 {
                if let Some((over, landing)) = g.jump(i, d) {
                    // A hop is two steps in the same direction.
                    assert_eq!(g.neighbour(i, d), Some(over));
                    assert_eq!(g.neighbour(over, d), Some(landing));
                    assert_eq!(g.jump(landing, Board::opposite(d)), Some((over, i)));
                }
            }
        }
    }

    #[test]
    fn symmetries_are_involutions() {
        let g = Board::new(4);
        for i in 0..g.len() {
            assert_eq!(g.rotate(g.rotate(i)), i);
            assert_eq!(g.reflect(g.reflect(i)), i);
        }
    }

    #[test]
    fn symmetries_preserve_adjacency() {
        // A symmetry of the board must send neighbours to neighbours, though not necessarily in
        // the same direction slot. If this fails, the permutation is wrong even if it is a valid
        // involution.
        let g = Board::new(4);
        for i in 0..g.len() {
            for d in 0..6 {
                if let Some(j) = g.neighbour(i, d) {
                    let (ri, rj) = (g.rotate(i), g.rotate(j));
                    assert!((0..6).any(|e| g.neighbour(ri, e) == Some(rj)));
                    let (mi, mj) = (g.reflect(i), g.reflect(j));
                    assert!((0..6).any(|e| g.neighbour(mi, e) == Some(mj)));
                }
            }
        }
    }

    #[test]
    fn homes_are_contiguous_and_swapped_by_rotation() {
        let g = Board::new(4);
        let (bottom, top) = g.home_ranges();
        assert_eq!(bottom.len(), g.pieces_per_player());
        assert_eq!(top.len(), g.pieces_per_player());
        assert_eq!(bottom.start, 0);
        assert_eq!(top.end, g.len());
        for i in bottom.clone() {
            assert!(g.hole(i).x <= -(g.n() + 1));
            assert!(top.contains(&g.rotate(i)));
        }
        for i in top.clone() {
            assert!(g.hole(i).x >= g.n() + 1);
            assert!(bottom.contains(&g.rotate(i)));
        }
        // The mirror fixes each home triangle, which is why it preserves the side to move.
        for i in top.clone() {
            assert!(top.contains(&g.reflect(i)));
        }
    }

    #[test]
    fn distances() {
        let origin = Coordinate::new(0, 0);
        assert_eq!(origin.distance(origin), 0);
        for d in Board::DIRECTIONS {
            assert_eq!(origin.distance(origin.add(d)), 1);
            assert_eq!(origin.distance(origin.add(d.scale(3))), 3);
        }
        // Apex to apex, straight through the centre.
        assert_eq!(Coordinate::new(8, -4).distance(Coordinate::new(-8, 4)), 16);
        // Distance must agree with breadth-first search on the neighbour table for holes that are
        // connected by a straight line inside the board.
        assert_eq!(Coordinate::new(0, 0).distance(Coordinate::new(2, -1)), 2);
    }

    #[test]
    fn distance_tables_agree_with_rotation() {
        // The two targets are related by the 180° rotation, so the two distance tables must
        // be too. If this fails, one player is being scored on a different scale from the
        // other and every capped game is decided unfairly.
        for n in 1..=4 {
            let board = Board::new(n);
            for i in 0..board.len() {
                assert_eq!(
                    board.distance_to_target[1][i],
                    board.distance_to_target[0][board.rotate(i)]
                );
            }
            // Distance is zero exactly on the target holes, and nowhere else.
            for i in 0..board.len() {
                let in_target = Board::is_occupied(board.winning_state(0), i as u8);
                assert_eq!(board.distance_to_target[0][i] == 0, in_target);
            }
        }
    }

    #[test]
    fn game_state() {
        let g = Board::new(4);
        // Players start at opposing sides of the board.
        let state_0 = g.starting_state();
        let alice_0 = state_0.players[0];
        let bob_0 = state_0.players[1];
        assert_eq!(10, alice_0.count_ones());
        (0..10).for_each(|i| assert!(Board::is_occupied(alice_0, i)));
        assert_eq!(10, bob_0.count_ones());
        (111..121).for_each(|i| assert!(Board::is_occupied(bob_0, i)));
        // Moving correctly sets the state.
        // Note that we support moving to arbitrary places (for test setup).
        let state_1 = state_0.apply(3, 17);
        let alice_1 = state_1.adversary();
        assert_eq!(10, alice_1.count_ones());
        assert!(Board::is_occupied(alice_1, 17));
        assert!(!Board::is_occupied(alice_1, 3));
        let state_2 = state_1.apply(113, 97);
        let bob_1 = state_2.adversary();
        assert_eq!(10, bob_1.count_ones());
        assert!(Board::is_occupied(bob_1, 97));
        assert!(!Board::is_occupied(bob_1, 113));
    }

    #[test]
    fn starting_position_is_symmetric_and_maximal() {
        for n in 1..=4 {
            let board = Board::new(n);
            let start = board.starting_state();
            let (d0, d1) = (
                board.remaining_distance(&start, 0),
                board.remaining_distance(&start, 1),
            );
            assert_eq!(d0, d1, "the start must not favour either player");
            let moves = board.available_moves(start.mover(), start.adversary());
            moves.iter().for_each(|&(from, to)| {
                assert!(board.remaining_distance(&start.apply(from, to), 0) < d0)
            });
        }
    }

    #[test]
    fn target_is_the_opposite_home() {
        for n in 1..=4 {
            let board = Board::new(n);
            let start = board.starting_state();
            // Each player's target is exactly where the other player begins.
            assert_eq!(board.winning_state(0), start.player(1));
            assert_eq!(board.winning_state(1), start.player(0));
            assert_eq!(board.winning_state(0) & board.winning_state(1), 0);
            assert_eq!(
                board.winning_state(0).count_ones() as usize,
                board.pieces_per_player()
            );
        }
    }

    #[test]
    fn game_is_open_at_the_start() {
        let board = Board::new(2);
        let start = board.starting_state();
        assert_eq!(board.outcome(&start, 100), None);
        assert_eq!(board.outcome_score(&start, 100), None);
    }

    #[test]
    fn win_is_detected_and_signed_correctly() {
        let board = Board::new(2);
        // Player 0 has filled the top triangle (34, 35, 36); player 1 sits in the middle.
        // side = 1 because player 0 has just moved.
        let state = GameState::state_from(&[34, 35, 36], &[16, 17, 18], 1);
        assert_eq!(board.outcome(&state, 100), Some(Outcome::Win(0)));
        // From the side to move's perspective this is a loss.
        assert_eq!(board.outcome_score(&state, 100), Some(-1.0));
        // The win takes precedence over any remaining budget, including none.
        assert_eq!(board.outcome(&state, 0), Some(Outcome::Win(0)));
    }

    #[test]
    fn cap_awards_the_closer_player() {
        let board = Board::new(2);
        // Player 0 has advanced up the board; player 1 has barely left home.
        let advanced = GameState::state_from(&[26, 30, 31], &[34, 35, 2], 0);
        let d0 = board.remaining_distance(&advanced, 0);
        let d1 = board.remaining_distance(&advanced, 1);
        assert_ne!(d0, d1, "fixture is a tie; pick different holes");
        let expected = if d0 < d1 { 0 } else { 1 };
        assert_eq!(board.outcome(&advanced, 0), Some(Outcome::CapWin(expected)));

        // The sign of the value depends on who is to move, not on who won.
        let as_mover = GameState::state_from(&[26, 30, 31], &[34, 35, 2], expected);
        let as_waiter = GameState::state_from(&[26, 30, 31], &[34, 35, 2], expected ^ 1);
        assert_eq!(board.outcome_score(&as_mover, 0), Some(1.0));
        assert_eq!(board.outcome_score(&as_waiter, 0), Some(-1.0));

        // A position symmetric under the rotation must draw.
        let start = board.starting_state();
        assert_eq!(board.outcome(&start, 0), Some(Outcome::Draw));
    }

    #[test]
    fn refusing_to_leave_home_loses() {
        // The spoiling strategy: player 1 never moves, so player 0 can never fill the top triangle
        // and the game ends after rounds have been exhausted.
        let board = Board::new(2);
        let mut state = board.starting_state();
        let mut rng = rand::rng();
        // Let player 0 shuffle around for a while; player 1 never moves.
        for _ in 0..40 {
            if state.side() == 1 {
                state = GameState {
                    players: state.players,
                    side: 0,
                };
                continue;
            }
            let moves = board.available_moves(state.mover(), state.adversary());
            let distribution = Uniform::new(0, moves.len()).unwrap();
            let (from, to) = moves[distribution.sample(&mut rng)];
            state = state.apply(from, to);
        }
        assert_eq!(state.player(1), board.starting_state().player(1));
        assert_eq!(board.outcome(&state, 0), Some(Outcome::CapWin(0)));
    }

    /// Naive, deliberately inefficient move generator used to verify the correctness of the
    /// optimized implementation of `available_moves`.
    ///
    /// The point is *independence*: this shares nothing with the optimized generator except the
    /// primitives that already have their own tests. It works directly in cube coordinates on a
    /// `HashSet` of occupied holes and enumerates hop *paths* explicitly.
    ///
    /// It returns coordinate pairs rather than indices so that comparing against it also
    /// exercises the neighbour/jump tables and the index enumeration.
    fn reference_moves(
        board: &Board,
        player: u128,
        adversary: u128,
    ) -> HashSet<(Coordinate, Coordinate)> {
        let n = board.n();
        let all: HashSet<Coordinate> = (0..board.len())
            .filter(|&i| {
                Board::is_occupied(player, i as u8) || Board::is_occupied(adversary, i as u8)
            })
            .map(|i| board.hole(i))
            .collect();

        let mut moves = HashSet::new();
        for i in 0..board.len() {
            if !Board::is_occupied(player, i as u8) {
                continue;
            }
            let from = board.hole(i);

            // Lift the moving piece: for the duration of this move its origin is an empty hole.
            let mut occupied = all.clone();
            occupied.remove(&from);

            // Single steps: one hole in any of the six directions, if on board and empty.
            let mut steps = HashSet::new();
            for d in Board::DIRECTIONS {
                let to = from.add(d);
                if to.on_board(n) && !occupied.contains(&to) {
                    steps.insert(to);
                }
            }

            // Jump chains, by explicit depth-first path enumeration.
            let mut chains = HashSet::new();
            let mut path = vec![from];
            walk_paths(n, &occupied, from, &mut path, &mut chains);

            // We verify that neighbour steps and jumps can never land on the same holes.
            assert!(
                steps.is_disjoint(&chains),
                "step and chain destinations overlap at {:?}",
                from
            );

            for to in steps.into_iter().chain(chains) {
                moves.insert((from, to));
            }
        }
        moves
    }

    /// Enumerate every jump path from `current`, recording the destinations reached along the way.
    ///
    /// Restricting to simple paths (never revisiting a hole) is what guarantees termination, and it
    /// loses nothing: the board is static during a chain, so arriving at a hole a second time
    /// offers exactly the same onward hops as the first time.
    ///
    /// Both `over` and `landing` need an on-board check. The star is not convex, so it is not
    /// enough to verify the landing hole: you can have an on-board origin whose neighbour falls in
    /// one of the notches between the points.
    fn walk_paths(
        n: i32,
        occupied: &HashSet<Coordinate>,
        current: Coordinate,
        path: &mut Vec<Coordinate>,
        found: &mut HashSet<Coordinate>,
    ) {
        for d in Board::DIRECTIONS {
            let over = current.add(d);
            let landing = over.add(d);
            // We must stay on the board.
            if !over.on_board(n) || !landing.on_board(n) {
                continue;
            }
            // We can only jump above an occupied hole towards an unoccupied hole.
            if !occupied.contains(&over) || occupied.contains(&landing) {
                continue;
            }
            // We avoid loops inside a path, otherwise we won't terminate.
            if path.contains(&landing) {
                continue;
            }
            found.insert(landing);
            path.push(landing);
            walk_paths(n, occupied, landing, path, found);
            path.pop();
        }
    }

    /// Compare both move generators at one position, rendering the board if they disagree.
    fn assert_move_generators_agree(board: &Board, player: u128, adversary: u128) {
        let fast = board.available_moves(player, adversary);
        let fast_set: HashSet<(Coordinate, Coordinate)> = fast
            .iter()
            .map(|&(f, t)| (board.hole(f as usize), board.hole(t as usize)))
            .collect();
        assert_eq!(
            fast.len(),
            fast_set.len(),
            "fast generator emitted duplicate moves"
        );
        let expected = reference_moves(board, player, adversary);
        if fast_set != expected {
            let rendered = board.render(|i| match i as u8 {
                i if Board::is_occupied(player, i) => 'A',
                i if Board::is_occupied(adversary, i) => 'B',
                _ => '.',
            });
            let only_fast: Vec<_> = fast_set.difference(&expected).collect();
            let only_ref: Vec<_> = expected.difference(&fast_set).collect();
            panic!(
                "generators disagree\n{}\nfast only: {:?}\nreference only: {:?}",
                rendered, only_fast, only_ref
            );
        }
    }

    #[test]
    fn generate_valid_moves() {
        let g = Board::new(2);
        // We simulate the following state which contains interesting moves:
        const SAMPLE_BOARD: &str = concat!(
            "  4        .\n",
            "  3       . .\n",
            "  2  . . . . B . .\n",
            "  1   . . . B . .\n",
            "  0    . . A B .\n",
            " -1   . . . A . .\n",
            " -2  . . . . . . .\n",
            " -3       A .\n",
            " -4        .\n",
        );
        let mut state = g.starting_state();
        state = state.apply(0, 12);
        state = state.apply(34, 17);
        state = state.apply(2, 18);
        state = state.apply(35, 23);
        state = state.apply(1, 2);
        state = state.apply(36, 29);
        let rendered = g.render(|i| match i {
            i if Board::is_occupied(state.player(0), i as u8) => 'A',
            i if Board::is_occupied(state.player(1), i as u8) => 'B',
            _ => '.',
        });
        assert_eq!(rendered, SAMPLE_BOARD);
        let mut expected_moves: HashSet<(u8, u8)> = HashSet::new();
        // The bottom-most piece can only go to its direct neighbours:
        expected_moves.insert((2, 0));
        expected_moves.insert((2, 1));
        expected_moves.insert((2, 6));
        expected_moves.insert((2, 7));
        // The middle piece can go to its direct neighbours:
        expected_moves.insert((12, 11));
        expected_moves.insert((12, 13));
        expected_moves.insert((12, 5));
        expected_moves.insert((12, 6));
        // And also make a few jumps:
        expected_moves.insert((12, 24));
        expected_moves.insert((12, 22));
        expected_moves.insert((12, 34));
        // The upper piece can go to its direct neighbours:
        expected_moves.insert((18, 19));
        expected_moves.insert((18, 13));
        expected_moves.insert((18, 24));
        // And also make a few jumps:
        expected_moves.insert((18, 5));
        expected_moves.insert((18, 16));
        let (alice, bob) = (state.player(0), state.player(1));
        let computed_moves = HashSet::from_iter(g.available_moves(alice, bob));
        assert_eq!(expected_moves, computed_moves);
    }

    #[test]
    fn generate_valid_moves_during_random_games() {
        let board = Board::new(2);
        let mut rng = rand::rng();
        for _ in 0..100 {
            let mut state = board.starting_state();
            for _ in 0..60 {
                assert_move_generators_agree(&board, state.mover(), state.adversary());
                let moves = board.available_moves(state.mover(), state.adversary());
                if moves.is_empty() {
                    break;
                }
                let (from, to) = moves[rng.next_u32() as usize % moves.len()];
                state = state.apply(from, to);
            }
        }
    }

    #[test]
    fn generators_agree_on_random_positions() {
        let mut rng = rand::rng();
        for n in 1..=4 {
            let board = Board::new(n);
            let distribution = Uniform::new(0, board.len()).unwrap();
            let half_distribution = Uniform::new(0, board.len() / 2).unwrap();
            for _ in 0..200 {
                // Vary density. Dense clusters produce long hop chains, which is where the
                // reached/frontier logic gets stressed — legal play from the opening keeps
                // pieces spread out and rarely generates them.
                let count = 2 + half_distribution.sample(&mut rng);
                let (mut a, mut b) = (0u128, 0u128);
                let mut placed = 0;
                while placed < count {
                    let bit = 1u128 << distribution.sample(&mut rng);
                    if (a | b) & bit != 0 {
                        continue;
                    }
                    if placed % 2 == 0 {
                        a |= bit
                    } else {
                        b |= bit
                    }
                    placed += 1;
                }
                assert_move_generators_agree(&board, a, b);
            }
        }
    }

    #[test]
    fn random_games_terminate_decisively() {
        for n in [2, 4] {
            let board = Board::new(n);
            let cap: u16 = 400; // even, so both players get the same number of moves
            let mut rng = rand::rng();
            let (mut settled, mut on_cap, mut drawn) = (0, 0, 0);
            for _ in 0..200 {
                let mut state = board.starting_state();
                let mut remaining_moves = cap;
                let result = loop {
                    if let Some(outcome) = board.outcome(&state, remaining_moves) {
                        break outcome;
                    }
                    let moves = board.available_moves(state.mover(), state.adversary());
                    assert!(
                        !moves.is_empty(),
                        "no legal move: a pass rule would be needed"
                    );
                    let distribution = Uniform::new(0, moves.len()).unwrap();
                    let (from, to) = moves[distribution.sample(&mut rng)];
                    state = state.apply(from, to);
                    remaining_moves -= 1;
                };
                match result {
                    Outcome::Win(_) => settled += 1,
                    Outcome::CapWin(_) => on_cap += 1,
                    Outcome::Draw => drawn += 1,
                }
            }
            println!("n={n}: {settled} settled, {on_cap} on the cap, {drawn} drawn");
            // The graded cap exists so that random play still produces a learning signal.
            assert!(drawn < 50, "too many draws: something is wrong");
        }
    }
}
