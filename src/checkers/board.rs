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
//! downstream (move generation, bitboards, MCTS) works purely on `u16` hole indices and never
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
    lookup: Vec<Option<u16>>,
    /// `neighbours[i][d]` = hole reached by one step from `i` in direction `d`, if on board.
    neighbours: Vec<[Option<u16>; 6]>,
    /// `jumps[i][d]` = `(hole jumped over, landing hole)` for a hop from `i` in direction `d`.
    jumps: Vec<[Option<(u16, u16)>; 6]>,
    /// `rotation[i]` = index of hole `i` after a 180° rotation.
    rotation: Vec<u16>,
    /// `reflection[i]` = index of hole `i` after mirroring.
    reflection: Vec<u16>,
}

impl Board {
    pub fn new(n: i32) -> Self {
        assert!(n >= 1, "board size must be at least 1");

        let span = 4 * n + 1;
        let mut holes: Vec<Coordinate> = Vec::new();
        let mut lookup: Vec<Option<u16>> = vec![None; (span * span) as usize];

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
                assert!(index < u16::MAX as usize, "too many holes for u16 indices");
                lookup[Self::lookup_key(n, c)] = Some(index as u16);
                holes.push(c);
            }
        }

        let mut board = Board {
            n,
            holes,
            lookup,
            neighbours: Vec::new(),
            jumps: Vec::new(),
            rotation: Vec::new(),
            reflection: Vec::new(),
        };

        board.neighbours = board.build_neighbours();
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
    fn build_neighbours(&self) -> Vec<[Option<u16>; 6]> {
        let mut neighbours: Vec<[Option<u16>; 6]> = Vec::new();
        for i in 0..self.len() {
            let current = self.holes[i];
            neighbours.push(Self::DIRECTIONS.map(|direction| {
                let neighbour = current.add(direction);
                Self::safe_lookup_key(self.n, neighbour).and_then(|idx| self.lookup[idx])
            }));
        }
        neighbours
    }

    /// For every hole and every direction, we pre-compute the jump over the neighbour in that
    /// direction and store the jumped-over neighbour and the landing hole.
    /// Both must exist as holes: you cannot jump over a point that isn't on the board, and you
    /// cannot land off the board. We store both, because move generation needs to test whether the
    /// jumped hole is *occupied* and the landing hole is *empty*.
    ///
    /// Note that `jumps[i][d]` being `Some` and `neighbours[i][d]` being `Some` are not the same
    /// condition: the second is implied by the first, but not the reverse.
    fn build_jumps(&self) -> Vec<[Option<(u16, u16)>; 6]> {
        let mut jumps: Vec<[Option<(u16, u16)>; 6]> = Vec::new();
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
    fn build_permutation(&self, map: fn(Coordinate) -> Coordinate) -> Vec<u16> {
        let mut permutation: Vec<u16> = Vec::new();
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
        // We verify that each range has a correct `x` field, otherwise there's a serious bug.
        for i in range1.clone() {
            let x = self.holes[i].x;
            assert!(
                x >= -2 * self.n && x < -self.n,
                "first home triangle not correctly computed"
            );
        }
        for j in range2.clone() {
            let x = self.holes[j].x;
            assert!(
                x > self.n && x <= 2 * self.n,
                "second home triangle not correctly computed"
            );
        }
        (range1, range2)
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

    /// `N = 2`: 37 holes, rows of 1, 2, 7, 6, 5, 6, 7, 2, 1. Small enough to check by eye: this is
    /// the expected output of `render`, so it can be used as a regression test on the layout.
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

    #[test]
    fn ascii_layout() {
        let g = Board::new(2);
        assert_eq!(g.render(|_| '.'), STAR_N2);
        // We print the real board for manual verification as well.
        println!("{}", Board::new(4).render_homes());
    }

    #[test]
    fn hole_counts() {
        for n in 1..=6 {
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
}
