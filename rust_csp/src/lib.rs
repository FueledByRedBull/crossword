use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::wrap_pyfunction;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone)]
struct Slot {
    id: usize,
    length: usize,
    cells: Vec<(usize, usize)>,
}

fn get_field<T: for<'a> FromPyObject<'a>>(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<T> {
    if let Ok(value) = obj.getattr(name) {
        return value.extract();
    }
    if let Ok(value) = obj.get_item(name) {
        return value.extract();
    }
    Err(PyKeyError::new_err(format!("Missing field: {name}")))
}

fn slot_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Slot> {
    let id: usize = get_field(obj, "id")?;
    let length: usize = get_field(obj, "length")?;
    let cells: Vec<(usize, usize)> = get_field(obj, "cells")?;
    Ok(Slot { id, length, cells })
}

fn build_intersections(
    slots: &[Slot],
) -> (
    Vec<Vec<Option<(usize, usize)>>>,
    Vec<Vec<(usize, usize, usize)>>,
) {
    let slot_count = slots.len();
    let mut cell_map: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for (slot_idx, slot) in slots.iter().enumerate() {
        for (cell_idx, cell) in slot.cells.iter().enumerate() {
            cell_map.entry(*cell).or_default().push((slot_idx, cell_idx));
        }
    }

    let mut intersections = vec![vec![None; slot_count]; slot_count];
    for positions in cell_map.values() {
        if positions.len() < 2 {
            continue;
        }
        for (a_id, a_idx) in positions {
            for (b_id, b_idx) in positions {
                if a_id == b_id {
                    continue;
                }
                intersections[*a_id][*b_id] = Some((*a_idx, *b_idx));
            }
        }
    }

    let mut neighbors: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); slot_count];
    for i in 0..slot_count {
        for j in 0..slot_count {
            if let Some((a_idx, b_idx)) = intersections[i][j] {
                neighbors[i].push((j, a_idx, b_idx));
            }
        }
    }

    (intersections, neighbors)
}

fn revise(
    domains: &mut [Vec<usize>],
    xi: usize,
    xj: usize,
    intersections: &[Vec<Option<(usize, usize)>>],
    word_bytes: &[Vec<u8>],
) -> bool {
    let Some((a_idx, b_idx)) = intersections[xi][xj] else {
        return false;
    };
    let mut revised = false;
    let mut filtered = Vec::with_capacity(domains[xi].len());
    for &word_idx in domains[xi].iter() {
        let a = word_bytes[word_idx][a_idx];
        let mut ok = false;
        for &other_idx in domains[xj].iter() {
            if a == word_bytes[other_idx][b_idx] {
                ok = true;
                break;
            }
        }
        if ok {
            filtered.push(word_idx);
        } else {
            revised = true;
        }
    }
    if revised {
        domains[xi] = filtered;
    }
    revised
}

fn ac3(
    domains: &mut [Vec<usize>],
    intersections: &[Vec<Option<(usize, usize)>>],
    neighbors: &[Vec<(usize, usize, usize)>],
    word_bytes: &[Vec<u8>],
) -> bool {
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for (xi, entries) in neighbors.iter().enumerate() {
        for (xj, _, _) in entries.iter() {
            queue.push_back((xi, *xj));
        }
    }
    while let Some((xi, xj)) = queue.pop_front() {
        if revise(domains, xi, xj, intersections, word_bytes) {
            if domains[xi].is_empty() {
                return false;
            }
            for (xk, _, _) in neighbors[xi].iter() {
                if *xk != xj {
                    queue.push_back((*xk, xi));
                }
            }
        }
    }
    true
}

fn is_consistent(
    assignments: &[Option<usize>],
    used_words: &HashSet<usize>,
    slot_id: usize,
    word_idx: usize,
    neighbors: &[Vec<(usize, usize, usize)>],
    word_bytes: &[Vec<u8>],
) -> bool {
    if used_words.contains(&word_idx) {
        return false;
    }
    for (neighbor_id, a_idx, b_idx) in neighbors[slot_id].iter() {
        if let Some(neighbor_word) = assignments[*neighbor_id] {
            if word_bytes[word_idx][*a_idx] != word_bytes[neighbor_word][*b_idx] {
                return false;
            }
        }
    }
    true
}

fn forward_check(
    assignments: &[Option<usize>],
    used_words: &HashSet<usize>,
    slot_id: usize,
    word_idx: usize,
    domains: &mut [Vec<usize>],
    neighbors: &[Vec<(usize, usize, usize)>],
    word_bytes: &[Vec<u8>],
) -> bool {
    for (neighbor_id, a_idx, b_idx) in neighbors[slot_id].iter() {
        if assignments[*neighbor_id].is_some() {
            continue;
        }
        let mut allowed: Vec<usize> = Vec::new();
        for &candidate in domains[*neighbor_id].iter() {
            if used_words.contains(&candidate) {
                continue;
            }
            if word_bytes[candidate][*b_idx] == word_bytes[word_idx][*a_idx] {
                allowed.push(candidate);
            }
        }
        if allowed.is_empty() {
            return false;
        }
        domains[*neighbor_id] = allowed;
    }
    true
}

fn choose_slot(assignments: &[Option<usize>], domains: &[Vec<usize>], neighbors: &[Vec<(usize, usize, usize)>]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_key: (usize, isize) = (usize::MAX, isize::MIN);
    for (slot_id, assigned) in assignments.iter().enumerate() {
        if assigned.is_some() {
            continue;
        }
        let domain_size = domains[slot_id].len();
        let degree = neighbors[slot_id].len() as isize;
        let key = (domain_size, -degree);
        if key < best_key {
            best_key = key;
            best = Some(slot_id);
        }
    }
    best
}

fn value_score(
    slot_id: usize,
    word_idx: usize,
    assignments: &[Option<usize>],
    domains: &[Vec<usize>],
    neighbors: &[Vec<(usize, usize, usize)>],
    word_bytes: &[Vec<u8>],
    score_lookup: &[f64],
) -> f64 {
    let mut support = 0usize;
    for (neighbor_id, a_idx, b_idx) in neighbors[slot_id].iter() {
        if assignments[*neighbor_id].is_some() {
            continue;
        }
        for &candidate in domains[*neighbor_id].iter() {
            if word_bytes[candidate][*b_idx] == word_bytes[word_idx][*a_idx] {
                support += 1;
            }
        }
    }
    support as f64 + (2.0 * score_lookup[word_idx])
}

fn state_rank(
    assignments: &[Option<usize>],
    domains: &[Vec<usize>],
    quality: f64,
) -> (usize, f64, isize) {
    let mut assigned_count = 0usize;
    let mut domain_pressure = 0usize;
    for (slot_id, assigned) in assignments.iter().enumerate() {
        if assigned.is_some() {
            assigned_count += 1;
        } else {
            domain_pressure += domains[slot_id].len();
        }
    }
    (assigned_count, quality, -(domain_pressure as isize))
}

fn better_candidate(
    candidate_count: usize,
    candidate_quality: f64,
    incumbent_count: usize,
    incumbent_quality: f64,
) -> bool {
    if candidate_count != incumbent_count {
        return candidate_count > incumbent_count;
    }
    candidate_quality > incumbent_quality
}

#[derive(Clone)]
struct State {
    assignments: Vec<Option<usize>>,
    used_words: HashSet<usize>,
    domains: Vec<Vec<usize>>,
    quality: f64,
    assigned_count: usize,
}

#[pyfunction]
#[pyo3(
    signature = (
        grid,
        slots,
        words,
        min_len=3,
        max_steps=20000,
        max_restarts=2,
        random_seed=13,
        use_ac3=true,
        word_scores=None,
        beam_width=32,
        enable_local_repair=true,
        repair_steps=300
    )
)]
fn solve_crossword(
    py: Python<'_>,
    grid: Vec<Vec<String>>,
    slots: Vec<Py<PyAny>>,
    words: Vec<String>,
    min_len: usize,
    max_steps: usize,
    max_restarts: usize,
    random_seed: u64,
    use_ac3: bool,
    word_scores: Option<HashMap<String, f64>>,
    beam_width: usize,
    enable_local_repair: bool,
    repair_steps: usize,
) -> PyResult<PyObject> {
    let _ = grid;
    let _ = min_len;
    let mut slot_records: Vec<Slot> = Vec::with_capacity(slots.len());
    for slot in slots {
        slot_records.push(slot_from_py(&slot.bind(py))?);
    }

    let slot_count = slot_records.len();
    let beam_width = beam_width.max(1);
    let repair_steps = repair_steps.max(0);

    let word_scores = word_scores.unwrap_or_default();
    let mut score_lookup: Vec<f64> = Vec::with_capacity(words.len());
    let mut word_bytes: Vec<Vec<u8>> = Vec::with_capacity(words.len());
    let mut word_len: Vec<usize> = Vec::with_capacity(words.len());
    for word in words.iter() {
        score_lookup.push(*word_scores.get(word).unwrap_or(&0.0));
        let bytes = word.as_bytes().to_vec();
        word_len.push(bytes.len());
        word_bytes.push(bytes);
    }

    let mut length_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, len) in word_len.iter().enumerate() {
        length_map.entry(*len).or_default().push(idx);
    }

    let mut base_domains: Vec<Vec<usize>> = vec![Vec::new(); slot_count];
    for (idx, slot) in slot_records.iter().enumerate() {
        if let Some(domain) = length_map.get(&slot.length) {
            base_domains[idx] = domain.clone();
        }
    }

    let (intersections, neighbors) = build_intersections(&slot_records);

    let mut best_overall: Vec<Option<usize>> = vec![None; slot_count];
    let mut best_overall_quality = -1e9;
    let mut best_overall_count = 0usize;
    let mut total_steps = 0usize;
    let mut solved = false;
    let mut restarts_used = 0usize;
    let mut local_repair_applied = false;

    for restart_idx in 0..=max_restarts {
        restarts_used += 1;
        let mut rng = StdRng::seed_from_u64(random_seed + (restart_idx as u64 * 7919));
        let mut local_domains = base_domains.clone();
        if use_ac3 && !ac3(&mut local_domains, &intersections, &neighbors, &word_bytes) {
            local_domains = base_domains.clone();
        }

        let mut states = vec![State {
            assignments: vec![None; slot_count],
            used_words: HashSet::new(),
            domains: local_domains,
            quality: 0.0,
            assigned_count: 0,
        }];

        while !states.is_empty() && total_steps < max_steps {
            let mut next_states: Vec<State> = Vec::new();
            for state in states.into_iter() {
                if state.assigned_count == slot_count {
                    solved = true;
                    if better_candidate(
                        state.assigned_count,
                        state.quality,
                        best_overall_count,
                        best_overall_quality,
                    ) {
                        best_overall = state.assignments.clone();
                        best_overall_quality = state.quality;
                        best_overall_count = state.assigned_count;
                    }
                    break;
                }

                let slot_id = choose_slot(&state.assignments, &state.domains, &neighbors);
                if slot_id.is_none() {
                    solved = true;
                    if better_candidate(
                        state.assigned_count,
                        state.quality,
                        best_overall_count,
                        best_overall_quality,
                    ) {
                        best_overall = state.assignments.clone();
                        best_overall_quality = state.quality;
                        best_overall_count = state.assigned_count;
                    }
                    break;
                }
                let slot_id = slot_id.unwrap();
                let mut candidates = state.domains[slot_id].clone();
                if candidates.is_empty() {
                    continue;
                }
                candidates.shuffle(&mut rng);
                candidates.sort_by(|a, b| {
                    value_score(slot_id, *b, &state.assignments, &state.domains, &neighbors, &word_bytes, &score_lookup)
                        .partial_cmp(
                            &value_score(
                                slot_id,
                                *a,
                                &state.assignments,
                                &state.domains,
                                &neighbors,
                                &word_bytes,
                                &score_lookup,
                            ),
                        )
                        .unwrap_or(Ordering::Equal)
                });
                let branch_limit = std::cmp::max(8, std::cmp::min(candidates.len(), beam_width * 2));

                for &word_idx in candidates.iter().take(branch_limit) {
                    total_steps += 1;
                    if total_steps > max_steps {
                        break;
                    }
                    if !is_consistent(
                        &state.assignments,
                        &state.used_words,
                        slot_id,
                        word_idx,
                        &neighbors,
                        &word_bytes,
                    ) {
                        continue;
                    }
                    let mut next_assignments = state.assignments.clone();
                    next_assignments[slot_id] = Some(word_idx);
                    let mut next_used_words = state.used_words.clone();
                    next_used_words.insert(word_idx);
                    let mut next_domains = state.domains.clone();
                    if !forward_check(
                        &next_assignments,
                        &next_used_words,
                        slot_id,
                        word_idx,
                        &mut next_domains,
                        &neighbors,
                        &word_bytes,
                    ) {
                        continue;
                    }
                    let quality = state.quality + score_lookup[word_idx];
                    let assigned_count = state.assigned_count + 1;
                    if better_candidate(
                        assigned_count,
                        quality,
                        best_overall_count,
                        best_overall_quality,
                    ) {
                        best_overall = next_assignments.clone();
                        best_overall_quality = quality;
                        best_overall_count = assigned_count;
                    }
                    next_states.push(State {
                        assignments: next_assignments,
                        used_words: next_used_words,
                        domains: next_domains,
                        quality,
                        assigned_count,
                    });
                }
            }

            if solved {
                break;
            }
            if next_states.is_empty() {
                break;
            }

            next_states.sort_by(|a, b| {
                let ar = state_rank(&a.assignments, &a.domains, a.quality);
                let br = state_rank(&b.assignments, &b.domains, b.quality);
                br.0
                    .cmp(&ar.0)
                    .then_with(|| br.1.partial_cmp(&ar.1).unwrap_or(Ordering::Equal))
                    .then_with(|| br.2.cmp(&ar.2))
            });

            let mut deduped: Vec<State> = Vec::new();
            let mut seen: HashSet<Vec<(usize, usize)>> = HashSet::new();
            for state in next_states.into_iter() {
                let signature: Vec<(usize, usize)> = state
                    .assignments
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, value)| value.map(|word_idx| (idx, word_idx)))
                    .collect();
                if seen.contains(&signature) {
                    continue;
                }
                seen.insert(signature);
                deduped.push(state);
                if deduped.len() >= beam_width {
                    break;
                }
            }
            states = deduped;
        }

        if solved {
            break;
        }
    }

    if !solved && enable_local_repair && repair_steps > 0 {
        let mut repaired = best_overall.clone();
        let mut used_words: HashSet<usize> = repaired.iter().filter_map(|w| *w).collect();
        let mut repair_budget = repair_steps;
        while repair_budget > 0 {
            let mut unassigned: Vec<usize> = repaired
                .iter()
                .enumerate()
                .filter_map(|(idx, value)| if value.is_none() { Some(idx) } else { None })
                .collect();
            if unassigned.is_empty() {
                break;
            }
            unassigned.sort_by(|a, b| {
                let domain_a = base_domains[*a].len();
                let domain_b = base_domains[*b].len();
                let degree_a = neighbors[*a].len();
                let degree_b = neighbors[*b].len();
                (domain_a, -(degree_a as isize)).cmp(&(domain_b, -(degree_b as isize)))
            });
            let mut progress = false;
            for slot_id in unassigned {
                let mut candidates: Vec<usize> = Vec::new();
                for &candidate in base_domains[slot_id].iter() {
                    if used_words.contains(&candidate) {
                        continue;
                    }
                    if is_consistent(
                        &repaired,
                        &used_words,
                        slot_id,
                        candidate,
                        &neighbors,
                        &word_bytes,
                    ) {
                        candidates.push(candidate);
                    }
                }
                if candidates.is_empty() {
                    continue;
                }
                candidates.sort_by(|a, b| {
                    let score_a = score_lookup[*a];
                    let score_b = score_lookup[*b];
                    let degree_a = neighbors[slot_id].len();
                    let degree_b = neighbors[slot_id].len();
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| degree_b.cmp(&degree_a))
                });
                let chosen = candidates[0];
                repaired[slot_id] = Some(chosen);
                used_words.insert(chosen);
                repair_budget -= 1;
                progress = true;
                if repair_budget == 0 {
                    break;
                }
            }
            if !progress {
                break;
            }
        }

        let repaired_quality: f64 = repaired
            .iter()
            .filter_map(|value| value.map(|idx| score_lookup[idx]))
            .sum();
        let repaired_count = repaired.iter().filter(|value| value.is_some()).count();
        if better_candidate(
            repaired_count,
            repaired_quality,
            best_overall_count,
            best_overall_quality,
        ) {
            best_overall = repaired;
        }
        local_repair_applied = true;
    }

    let assignments = PyDict::new_bound(py);
    for (slot_idx, value) in best_overall.iter().enumerate() {
        if let Some(word_idx) = value {
            let slot_id = slot_records[slot_idx].id;
            assignments.set_item(slot_id, &words[*word_idx])?;
        }
    }

    let result = PyDict::new_bound(py);
    result.set_item("solved", solved)?;
    result.set_item("assignments", assignments)?;
    result.set_item("steps", total_steps)?;
    result.set_item("restarts", restarts_used)?;
    result.set_item("local_repair_applied", local_repair_applied)?;
    Ok(result.into_any().unbind())
}

#[pymodule]
fn rust_csp(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_crossword, m)?)?;
    Ok(())
}
