//! Crossword layout search, called by `src/layout.py`: fill a grid greedily, then run simulated
//! annealing over ruin-and-recreate steps (drop a few words, refill, keep or undo).
//!
//! Protocol, UTF-8 with LF line ends.
//!
//! stdin, first line:  `size seed steps max_words`
//!   size       side of the square the layout must fit in
//!   seed       unsigned 64-bit seed of the built-in generator
//!   steps      number of ruin-and-recreate steps
//!   max_words  cap on placed words, 0 for none
//! stdin, then one line per candidate: `score<TAB>answer`. The caller has already filtered,
//! deduplicated and ordered the answers (best first; the first one is the starting word) and
//! normalised the scores, so they are used as given.
//!
//! stdout, one line per placed word, in placement order: `answer<TAB>row<TAB>col<TAB>A` or `D`.
//! row and col are the word's first cell, shifted so the layout's top row and left column are 0.
//!
//! Malformed input prints a message to stderr and exits with status 2. The output depends only on
//! the input (nothing here iterates a hash map), and the generator and the logarithm are built in
//! from exact IEEE 754 operations only (no libm), so a seed gives the same layout on every run and
//! every platform.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{Read, Write};
use std::process::ExitCode;

/// Search value of a layout: 1 per word, this bonus per crossing...
const CROSSING_VALUE: f64 = 1.5;
/// ...and this bonus per (squared, 0..1) answer score.
const SCORE_VALUE: f64 = 3.0;
/// A step drops 1 to this many random words (plus whatever their removal strands) before refilling.
const RUIN_MAX: usize = 3;
/// Metropolis temperature at the first step; it cools linearly to zero at the last.
const START_TEMPERATURE: f64 = 3.0;
/// Refill raises each move's gain by a random factor of up to 1 + this, so refills differ.
const PICK_NOISE: f64 = 1.0;

struct Word {
    text: String,
    /// Letter ids, 1-based, so that 0 can mean an empty cell.
    letters: Vec<u32>,
    score: f64,
}

struct Problem {
    size: i64,
    seed: u64,
    steps: u64,
    max_words: usize,
    words: Vec<Word>,
    alphabet: usize,
}

fn parse(input: &str) -> Result<Problem, String> {
    let mut lines = input.split('\n');
    let head: Vec<&str> = lines.next().unwrap_or("").split_whitespace().collect();
    let [size, seed, steps, max_words] = head[..] else {
        return Err("first line must be `size seed steps max_words`".into());
    };
    let number = |name: &str, text: &str| format!("bad {name} `{text}`");
    let size: i64 = size.parse().map_err(|_| number("size", size))?;
    if size < 1 {
        return Err(number("size", &size.to_string()));
    }
    let seed: u64 = seed.parse().map_err(|_| number("seed", seed))?;
    let steps: u64 = steps.parse().map_err(|_| number("steps", steps))?;
    let max_words: usize = max_words
        .parse()
        .map_err(|_| number("max_words", max_words))?;

    let mut raw = Vec::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let (score, text) = line
            .split_once('\t')
            .ok_or("candidate line without a tab")?;
        let score: f64 = score.parse().map_err(|_| number("score", score))?;
        raw.push((score, text));
    }
    let (words, alphabet) = build_words(&raw);
    let max_words = if max_words == 0 {
        usize::MAX
    } else {
        max_words
    };
    Ok(Problem {
        size,
        seed,
        steps,
        max_words,
        words,
        alphabet,
    })
}

/// The words with their letters numbered, and the number of distinct letters.
fn build_words(raw: &[(f64, &str)]) -> (Vec<Word>, usize) {
    let mut alphabet: Vec<char> = raw.iter().flat_map(|(_, text)| text.chars()).collect();
    alphabet.sort_unstable();
    alphabet.dedup();
    let words = raw
        .iter()
        .map(|&(score, text)| Word {
            text: text.to_string(),
            letters: text
                .chars()
                .map(|ch| alphabet.binary_search(&ch).unwrap() as u32 + 1)
                .collect(),
            score,
        })
        .collect();
    (words, alphabet.len())
}

/// xoshiro256** seeded through SplitMix64.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng([next(), next(), next(), next()])
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in 0..n for n below 2^32.
    fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }

    /// `k` distinct items of `pool`, in random order.
    fn sample(&mut self, pool: &[usize], k: usize) -> Vec<usize> {
        let mut items = pool.to_vec();
        for i in 0..k {
            let j = i + self.below(items.len() - i);
            items.swap(i, j);
        }
        items.truncate(k);
        items
    }
}

/// Natural logarithm of x in (0, 1], from `+ - * /` only: libm's `ln` may differ in the last bit
/// between platforms, and the accept test below would then follow different paths. Doubles x into
/// [1/sqrt 2, 1], then ln x = 2 atanh(s), s = (x - 1) / (x + 1) <= 0.172, summed as a series.
fn ln_exact(mut x: f64) -> f64 {
    let mut doublings = 0.0;
    while x < std::f64::consts::FRAC_1_SQRT_2 {
        x *= 2.0;
        doublings += 1.0;
    }
    let s = (x - 1.0) / (x + 1.0);
    let mut series = 0.0;
    for j in (0..=10).rev() {
        series = series * (s * s) + 1.0 / (2 * j + 1) as f64;
    }
    2.0 * s * series - doublings * std::f64::consts::LN_2
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Spot {
    row: i64,
    col: i64,
    across: bool,
}

impl Spot {
    /// Step from one cell of the word to the next, as (row, column).
    fn step(self) -> (i64, i64) {
        if self.across {
            (0, 1)
        } else {
            (1, 0)
        }
    }
}

const DOWN: usize = 0;
const ACROSS: usize = 1;

/// Board with unbounded coordinates, built from and cut back by whole words.
///
/// Invariant: the board is always a valid layout inside a `span` x `span` box, so a move only
/// needs local checks. Cells live in a small torus: every cell a check can touch lies within
/// `span + 2` consecutive rows and columns, which a torus at least that wide keeps distinct, so
/// the layout can drift anywhere while the arrays stay fixed. (Coordinates are i64; a step moves
/// the layout by at most the word length, so they cannot overflow.)
struct Board<'a> {
    words: &'a [Word],
    span: i64,
    bits: u32,
    mask: i64,
    letters: Vec<u32>,
    /// Indexed by `DOWN` / `ACROSS`: the word id + 1 owning each cell in that direction, 0 for none.
    owner: [Vec<u32>; 2],
    at: Vec<Option<Spot>>,
    /// Placed word ids, oldest first.
    placed: Vec<usize>,
    crossings: i64,
    /// min row, max row, min col, max col of the placed words. Stale while the board is empty.
    bounds: [i64; 4],
}

impl<'a> Board<'a> {
    fn new(words: &'a [Word], span: i64) -> Self {
        let side = (span as usize + 2).next_power_of_two();
        Board {
            words,
            span,
            bits: side.trailing_zeros(),
            mask: side as i64 - 1,
            letters: vec![0; side * side],
            owner: [vec![0; side * side], vec![0; side * side]],
            at: vec![None; words.len()],
            placed: Vec::new(),
            crossings: 0,
            bounds: [0; 4],
        }
    }

    fn cell(&self, row: i64, col: i64) -> usize {
        (((row & self.mask) << self.bits) | (col & self.mask)) as usize
    }

    fn letter(&self, row: i64, col: i64) -> u32 {
        self.letters[self.cell(row, col)]
    }

    /// The crossings the word would make at `spot`, or None if the move is illegal.
    fn fit(&self, id: usize, spot: Spot) -> Option<i64> {
        let letters = &self.words[id].letters;
        let n = letters.len() as i64;
        let Spot {
            row: r,
            col: c,
            across,
        } = spot;
        let [lo_r, hi_r, lo_c, hi_c] = self.bounds;
        let reach = self.span - 1;
        let outside = if across {
            r < hi_r - reach || r > lo_r + reach || c < hi_c - reach || c + n - 1 > lo_c + reach
        } else {
            c < hi_c - reach || c > lo_c + reach || r < hi_r - reach || r + n - 1 > lo_r + reach
        };
        if outside {
            return None;
        }
        let (dr, dc) = spot.step();
        if self.letter(r - dr, c - dc) != 0 || self.letter(r + dr * n, c + dc * n) != 0 {
            return None;
        }
        let mine = &self.owner[across as usize];
        let mut crossings = 0;
        for (i, &ch) in letters.iter().enumerate() {
            let (rr, cc) = (r + dr * i as i64, c + dc * i as i64);
            let k = self.cell(rr, cc);
            let have = self.letters[k];
            if have == 0 {
                if self.letter(rr + dc, cc + dr) != 0 || self.letter(rr - dc, cc - dr) != 0 {
                    return None;
                }
            } else if have != ch || mine[k] != 0 {
                return None;
            } else {
                crossings += 1;
            }
        }
        (crossings > 0).then_some(crossings)
    }

    fn place(&mut self, id: usize, spot: Spot) {
        self.at[id] = Some(spot);
        self.placed.push(id);
        let (dr, dc) = spot.step();
        for (i, &ch) in self.words[id].letters.iter().enumerate() {
            let k = self.cell(spot.row + dr * i as i64, spot.col + dc * i as i64);
            if self.letters[k] != 0 {
                self.crossings += 1;
            } else {
                self.letters[k] = ch;
            }
            self.owner[spot.across as usize][k] = id as u32 + 1;
        }
        self.rebound();
    }

    /// Take a word off, and with it any word its crossings would leave touching a neighbour: two
    /// crossing words that sit side by side are only legal while the word across their ends is
    /// there. Returns every word taken off, in order, with where it stood.
    fn remove(&mut self, id: usize) -> Vec<(usize, Spot)> {
        let mut gone = Vec::new();
        let mut doomed = vec![id];
        let mut next = 0;
        while next < doomed.len() {
            let name = doomed[next];
            next += 1;
            let spot = self.at[name].take().expect("removed word is placed");
            self.placed.retain(|&other| other != name);
            gone.push((name, spot));
            let (dr, dc) = spot.step();
            let (mine, theirs) = (spot.across as usize, !spot.across as usize);
            let mut kept = Vec::new();
            for i in 0..self.words[name].letters.len() as i64 {
                let (r, c) = (spot.row + dr * i, spot.col + dc * i);
                let k = self.cell(r, c);
                self.owner[mine][k] = 0;
                if self.owner[theirs][k] != 0 {
                    self.crossings -= 1;
                    kept.push((r, c));
                } else {
                    self.letters[k] = 0;
                }
            }
            for (r, c) in kept {
                if self.letter(r - dr, c - dc) != 0 {
                    let other = self.owner[theirs][self.cell(r, c)] as usize - 1;
                    if !doomed.contains(&other) {
                        doomed.push(other);
                    }
                }
            }
        }
        if !self.placed.is_empty() {
            self.rebound();
        }
        gone
    }

    fn rebound(&mut self) {
        let mut b = [i64::MAX, i64::MIN, i64::MAX, i64::MIN];
        for &id in &self.placed {
            let spot = self.at[id].expect("placed word has a spot");
            let (dr, dc) = spot.step();
            let end = self.words[id].letters.len() as i64 - 1;
            b[0] = b[0].min(spot.row);
            b[1] = b[1].max(spot.row + dr * end);
            b[2] = b[2].min(spot.col);
            b[3] = b[3].max(spot.col + dc * end);
        }
        self.bounds = b;
    }

    /// Groups of words joined by crossings, each in discovery order, in order of first word.
    fn components(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.words.len()];
        let mut groups = Vec::new();
        for &first in &self.placed {
            if seen[first] {
                continue;
            }
            seen[first] = true;
            let mut group = vec![first];
            let mut next = 0;
            while next < group.len() {
                let id = group[next];
                next += 1;
                let spot = self.at[id].expect("placed word has a spot");
                let (dr, dc) = spot.step();
                let theirs = &self.owner[!spot.across as usize];
                for i in 0..self.words[id].letters.len() as i64 {
                    let other = theirs[self.cell(spot.row + dr * i, spot.col + dc * i)] as usize;
                    if other != 0 && !seen[other - 1] {
                        seen[other - 1] = true;
                        group.push(other - 1);
                    }
                }
            }
            groups.push(group);
        }
        groups
    }

    /// Adds the single-owner cells of a placed word to `out`, by letter.
    fn add_anchors(&self, id: usize, out: &mut Anchors) {
        let spot = self.at[id].expect("placed word has a spot");
        let (dr, dc) = spot.step();
        for i in 0..self.words[id].letters.len() as i64 {
            let (r, c) = (spot.row + dr * i, spot.col + dc * i);
            let k = self.cell(r, c);
            let down = self.owner[DOWN][k] != 0;
            if down != (self.owner[ACROSS][k] != 0) {
                out.push(
                    self.letters[k],
                    Anchor {
                        row: r,
                        col: c,
                        across: down,
                    },
                );
            }
        }
    }

    fn value(&self) -> f64 {
        let scores: f64 = self.placed.iter().map(|&id| self.words[id].score).sum();
        self.placed.len() as f64 + CROSSING_VALUE * self.crossings as f64 + SCORE_VALUE * scores
    }

    fn snapshot(&self) -> Vec<(usize, Spot)> {
        self.placed
            .iter()
            .map(|&id| (id, self.at[id].expect("placed word has a spot")))
            .collect()
    }
}

/// A single-owner cell a new word can cross; `across` is the direction the new word would run.
#[derive(Clone, Copy)]
struct Anchor {
    row: i64,
    col: i64,
    across: bool,
}

/// Anchors grouped by the letter in their cell.
struct Anchors {
    by_letter: Vec<Vec<Anchor>>,
    used: Vec<usize>,
}

impl Anchors {
    fn new(alphabet: usize) -> Self {
        Anchors {
            by_letter: vec![Vec::new(); alphabet + 1],
            used: Vec::new(),
        }
    }

    fn push(&mut self, letter: u32, anchor: Anchor) {
        let slot = &mut self.by_letter[letter as usize];
        if slot.is_empty() {
            self.used.push(letter as usize);
        }
        slot.push(anchor);
    }

    fn clear(&mut self) {
        for letter in self.used.drain(..) {
            self.by_letter[letter].clear();
        }
    }
}

/// A legal placement waiting in the refill heap. Best gain first, then oldest.
struct Move {
    gain: f64,
    ticket: u64,
    word: usize,
    spot: Spot,
}

impl Ord for Move {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain
            .total_cmp(&other.gain)
            .then(other.ticket.cmp(&self.ticket))
    }
}

impl PartialOrd for Move {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Move {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Move {}

struct Search<'a> {
    board: Board<'a>,
    rng: Rng,
    limit: usize,
    heap: BinaryHeap<Move>,
    everywhere: Anchors,
    fresh: Anchors,
}

/// Push every legal placement of `id` through `anchors`.
fn offer(
    board: &Board,
    id: usize,
    anchors: &Anchors,
    rng: &mut Rng,
    heap: &mut BinaryHeap<Move>,
    ticket: &mut u64,
) {
    let word = &board.words[id];
    for (i, &ch) in word.letters.iter().enumerate() {
        for anchor in &anchors.by_letter[ch as usize] {
            let i = i as i64;
            let (row, col) = if anchor.across {
                (anchor.row, anchor.col - i)
            } else {
                (anchor.row - i, anchor.col)
            };
            let spot = Spot {
                row,
                col,
                across: anchor.across,
            };
            let Some(crossings) = board.fit(id, spot) else {
                continue;
            };
            let gain = (1.0 + CROSSING_VALUE * crossings as f64 + SCORE_VALUE * word.score)
                / ((word.letters.len() as i64 - crossings).max(1) as f64).sqrt()
                * (1.0 + PICK_NOISE * rng.unit());
            heap.push(Move {
                gain,
                ticket: *ticket,
                word: id,
                spot,
            });
            *ticket += 1;
        }
    }
}

impl<'a> Search<'a> {
    fn new(problem: &'a Problem) -> Self {
        Search {
            board: Board::new(&problem.words, problem.size),
            rng: Rng::new(problem.seed),
            limit: problem.max_words.min(problem.words.len()),
            heap: BinaryHeap::new(),
            everywhere: Anchors::new(problem.alphabet),
            fresh: Anchors::new(problem.alphabet),
        }
    }

    /// Place unused words best-first until none fits; returns them in placement order.
    ///
    /// One scan finds every legal move; after that a placement only adds the moves that cross its
    /// own new letters, and stale moves are re-checked when they come off the heap.
    fn refill(&mut self) -> Vec<usize> {
        let mut ticket = 0;
        let board = &self.board;
        let unused: Vec<usize> = (0..board.words.len())
            .filter(|&id| board.at[id].is_none())
            .collect();
        self.heap.clear();
        self.everywhere.clear();
        for &id in &board.placed {
            board.add_anchors(id, &mut self.everywhere);
        }
        for &id in &unused {
            offer(
                board,
                id,
                &self.everywhere,
                &mut self.rng,
                &mut self.heap,
                &mut ticket,
            );
        }
        let mut added = Vec::new();
        while self.board.placed.len() < self.limit {
            let Some(Move { word, spot, .. }) = self.heap.pop() else {
                break;
            };
            if self.board.at[word].is_some() || self.board.fit(word, spot).is_none() {
                continue;
            }
            self.board.place(word, spot);
            added.push(word);
            self.fresh.clear();
            self.board.add_anchors(word, &mut self.fresh);
            for &other in &unused {
                if self.board.at[other].is_none() {
                    offer(
                        &self.board,
                        other,
                        &self.fresh,
                        &mut self.rng,
                        &mut self.heap,
                        &mut ticket,
                    );
                }
            }
        }
        added
    }

    fn run(&mut self, steps: u64) -> Vec<(usize, Spot)> {
        self.board.place(
            0,
            Spot {
                row: 0,
                col: 0,
                across: true,
            },
        );
        self.refill();
        let mut value = self.board.value();
        let mut best_value = value;
        let mut best = self.board.snapshot();
        for step in 0..steps {
            if self.board.placed.len() < 2 {
                break;
            }
            let temperature = START_TEMPERATURE * (1.0 - step as f64 / steps as f64);
            let mut dropped = Vec::new();
            let ruined = 1 + self.rng.below(RUIN_MAX.min(self.board.placed.len() - 1));
            for id in self.rng.sample(&self.board.placed, ruined) {
                if self.board.at[id].is_some() {
                    dropped.extend(self.board.remove(id));
                }
            }
            let mut groups = self.board.components();
            groups.sort_by_key(|group| std::cmp::Reverse(group.len()));
            for &id in groups.iter().skip(1).flatten() {
                if self.board.at[id].is_some() {
                    dropped.extend(self.board.remove(id));
                }
            }
            let added = self.refill();
            let trial = self.board.value();
            if trial >= value + temperature * ln_exact(1.0 - self.rng.unit()) {
                value = trial;
                if value > best_value {
                    best_value = value;
                    best = self.board.snapshot();
                }
            } else {
                for &id in added.iter().rev() {
                    self.board.remove(id);
                }
                for &(id, spot) in &dropped {
                    self.board.place(id, spot);
                }
            }
        }
        best
    }
}

fn solve(problem: &Problem) -> String {
    if problem.words.is_empty() {
        return String::new();
    }
    let best = Search::new(problem).run(problem.steps);
    let top = best.iter().map(|(_, spot)| spot.row).min().unwrap_or(0);
    let left = best.iter().map(|(_, spot)| spot.col).min().unwrap_or(0);
    let mut out = String::new();
    for (id, spot) in best {
        let direction = if spot.across { 'A' } else { 'D' };
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            problem.words[id].text,
            spot.row - top,
            spot.col - left,
            direction
        ));
    }
    out
}

fn main() -> ExitCode {
    let mut input = Vec::new();
    let problem = std::io::stdin()
        .read_to_end(&mut input)
        .map_err(|e| e.to_string())
        .and_then(|_| String::from_utf8(input).map_err(|e| e.to_string()))
        .and_then(|text| parse(&text));
    match problem {
        Ok(problem) => {
            let out = solve(&problem);
            if std::io::stdout().write_all(out.as_bytes()).is_err() {
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("layout-rs: {message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORDS: [&str; 20] = [
        "HEAT",
        "ENERGY",
        "ENTROPY",
        "WORK",
        "TEMPERATURE",
        "PRESSURE",
        "VOLUME",
        "GAS",
        "ENGINE",
        "CARNOT",
        "KELVIN",
        "JOULE",
        "WATT",
        "FLUID",
        "PHASE",
        "STATE",
        "CYCLE",
        "LAW",
        "ZERO",
        "BOILING",
    ];

    fn words(texts: &[&str]) -> Vec<Word> {
        let raw: Vec<(f64, &str)> = texts.iter().map(|&text| (0.5, text)).collect();
        build_words(&raw).0
    }

    fn across(row: i64, col: i64) -> Spot {
        Spot {
            row,
            col,
            across: true,
        }
    }

    fn down(row: i64, col: i64) -> Spot {
        Spot {
            row,
            col,
            across: false,
        }
    }

    type State = (
        Vec<u32>,
        [Vec<u32>; 2],
        Vec<Option<Spot>>,
        Vec<usize>,
        i64,
        [i64; 4],
    );

    fn state(board: &Board) -> State {
        let owner = [board.owner[0].clone(), board.owner[1].clone()];
        (
            board.letters.clone(),
            owner,
            board.at.clone(),
            board.placed.clone(),
            board.crossings,
            board.bounds,
        )
    }

    /// Every letter belongs to a word, side-by-side letters belong to the same word, the crossing
    /// count matches the cells with two owners, and the words fit the span.
    fn assert_valid(board: &Board) {
        let [lo_r, hi_r, lo_c, hi_c] = board.bounds;
        assert!(
            hi_r - lo_r < board.span && hi_c - lo_c < board.span,
            "{:?}",
            board.bounds
        );
        let mut crossing_cells = 0;
        for r in lo_r..=hi_r {
            for c in lo_c..=hi_c {
                let k = board.cell(r, c);
                if board.letters[k] == 0 {
                    continue;
                }
                assert!(
                    board.owner[DOWN][k] + board.owner[ACROSS][k] != 0,
                    "orphan at {r},{c}"
                );
                crossing_cells += (board.owner[DOWN][k] != 0 && board.owner[ACROSS][k] != 0) as i64;
                let right = board.cell(r, c + 1);
                if board.letters[right] != 0 {
                    assert_ne!(board.owner[ACROSS][k], 0, "non-word run at {r},{c}");
                    assert_eq!(board.owner[ACROSS][k], board.owner[ACROSS][right]);
                }
                let below = board.cell(r + 1, c);
                if board.letters[below] != 0 {
                    assert_ne!(board.owner[DOWN][k], 0, "non-word run at {r},{c}");
                    assert_eq!(board.owner[DOWN][k], board.owner[DOWN][below]);
                }
            }
        }
        assert_eq!(crossing_cells, board.crossings);
    }

    #[test]
    fn remove_undoes_place() {
        let words = words(&["CAT", "CAR", "TOE"]);
        let mut board = Board::new(&words, 15);
        board.place(0, across(0, 0));
        board.place(1, down(0, 0));
        let before = state(&board);
        board.place(2, down(0, 2));
        assert_eq!(board.crossings, 2);
        board.remove(2);
        assert_eq!(state(&board), before);
    }

    #[test]
    fn bounds_follow_the_placed_words() {
        let words = words(&["CAT", "TOE"]);
        let mut board = Board::new(&words, 15);
        board.place(0, across(4, 4));
        board.place(1, down(4, 6));
        assert_eq!(board.bounds, [4, 6, 4, 6]);
        board.remove(1);
        assert_eq!(board.bounds, [4, 4, 4, 6]);
    }

    fn check_fit_rules(row: i64, col: i64) {
        let words = words(&["CAT", "CAR", "ARE", "TOE", "TOES"]);
        let mut board = Board::new(&words, 3);
        board.place(0, across(row, col));
        board.place(1, down(row, col));
        assert_eq!(board.fit(2, down(row, col + 1)), None, "runs alongside CAR");
        assert_eq!(board.fit(3, down(row, col + 2)), Some(1));
        assert_eq!(board.fit(4, down(row, col + 2)), None, "past the span");
        assert_eq!(
            board.fit(3, down(row, col + 3)),
            None,
            "touches the end of CAT"
        );
    }

    #[test]
    fn fit_refuses_words_running_alongside_or_past_the_span() {
        check_fit_rules(0, 0);
    }

    #[test]
    fn board_works_however_far_the_layout_has_drifted() {
        check_fit_rules(-1_000_000_007, 3_000_000_011);
    }

    #[test]
    fn remove_never_leaves_letters_touching_side_by_side() {
        let words = words(&["BELL", "PHASE", "LIGHT"]);
        let mut board = Board::new(&words, 15);
        board.place(0, across(0, 0));
        board.place(1, down(-4, 1)); // ends on BELL's E
        board.place(2, down(0, 2)); // starts on BELL's L, one column over
        assert_valid(&board);
        let gone: Vec<usize> = board.remove(0).into_iter().map(|(id, _)| id).collect();
        assert_eq!(gone, [0, 2], "LIGHT would touch PHASE once BELL is gone");
        assert_valid(&board);
    }

    #[test]
    fn components_split_when_a_bridge_word_goes() {
        let words = words(&["CAT", "TOE", "EAR"]);
        let mut board = Board::new(&words, 15);
        board.place(0, across(0, 0));
        board.place(1, down(0, 2));
        board.place(2, across(2, 2));
        assert_eq!(board.components().len(), 1);
        board.remove(1);
        let mut groups = board.components();
        groups.sort();
        assert_eq!(groups, [[0], [2]]);
    }

    fn input(seed: u64, steps: u64, max_words: usize) -> String {
        let mut text = format!("13 {seed} {steps} {max_words}\n");
        for (i, word) in WORDS.iter().enumerate() {
            text.push_str(&format!("{}\t{word}\n", 1.0 - i as f64 / 100.0));
        }
        text
    }

    #[test]
    fn ln_exact_matches_ln() {
        let mut rng = Rng::new(1);
        let mut inputs = vec![1.0, 0.5, 0.7, 0.999_999_999_999, 1.0 / (1u64 << 53) as f64];
        inputs.extend((0..100_000).map(|_| 1.0 - rng.unit()));
        for x in inputs {
            let (got, want) = (ln_exact(x), x.ln());
            assert!(
                (got - want).abs() <= 1e-15 * want.abs().max(1e-3),
                "{x}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn same_input_same_layout() {
        let first = solve(&parse(&input(7, 2000, 0)).unwrap());
        assert_eq!(first, solve(&parse(&input(7, 2000, 0)).unwrap()));
        assert!(first.lines().count() >= 2, "{first}");
        assert_ne!(first, solve(&parse(&input(8, 2000, 0)).unwrap()));
    }

    #[test]
    fn searched_layout_obeys_the_board_invariant() {
        for seed in 0..6 {
            let problem = parse(&input(seed, 400, 0)).unwrap();
            let best = Search::new(&problem).run(problem.steps);
            let mut board = Board::new(&problem.words, problem.size);
            for &(id, spot) in &best {
                board.place(id, spot);
            }
            assert!(best.len() >= 2);
            assert_valid(&board);
            assert_eq!(board.components().len(), 1, "seed {seed}");
        }
    }

    #[test]
    fn max_words_caps_the_layout() {
        let out = solve(&parse(&input(3, 300, 5)).unwrap());
        assert_eq!(out.lines().count(), 5);
    }

    #[test]
    fn letters_are_characters_not_bytes() {
        let out = solve(
            &parse("5 1 10 0\n1\t\u{391}\u{392}\u{393}\n0.5\t\u{393}\u{391}\u{3a4}\n").unwrap(),
        );
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("\u{391}\u{392}\u{393}\t"));
        assert!(lines[1].starts_with("\u{393}\u{391}\u{3a4}\t"));
    }

    #[test]
    fn malformed_input_is_an_error() {
        assert!(parse("15 1 100\n").is_err());
        assert!(parse("15 1 100 0\n0.5 NOTAB\n").is_err());
        assert!(parse("15 x 100 0\n").is_err());
    }
}
