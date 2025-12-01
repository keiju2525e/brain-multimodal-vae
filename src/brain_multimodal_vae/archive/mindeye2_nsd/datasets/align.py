"""
Utilities for aligning per-subject WebDataset samples on shared stimulus presentations.

The helpers here assume each entry in ``all_subj_data[subject]`` matches the tuple
structure produced by ``webdataset`` in ``dataset_loading_preview.ipynb``:
    (behav, past_behav, future_behav, olds_behav)

The ``behav`` tensor is expected to have shape ``(1, 17)`` where:
    - index 0 stores the COCO stimulus id (``cocoidx``)
    - index 5 stores the global trial id (``global_trial``)

Examples
--------
>>> from alignment_utils import align_subject_trials, alignment_to_dataframe
>>> alignment = align_subject_trials(all_subj_data, anchor_subject="subj01")
>>> df = alignment_to_dataframe(alignment)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple


SubjectSample = Tuple  # we only rely on index access, so keep type loose
SubjectData = Iterable[SubjectSample]
AlignedTrial = MutableMapping[str, object]


def _to_scalar(value) -> float:
    """Return a plain Python scalar from a tensor/ndarray element."""
    if hasattr(value, "item"):
        return value.item()
    return float(value)


def _extract_behav_field(behav, column: int) -> int:
    """Safely extract a single column from the (1, 17) behav tensor."""
    if behav is None:
        raise ValueError("behav tensor is required to extract metadata.")
    candidate = behav[0, column] if getattr(behav, "ndim", 2) == 2 else behav[column]
    return int(_to_scalar(candidate))


def _index_subject_occurrences(samples: SubjectData) -> Dict[Tuple[int, int], Dict[str, int]]:
    """
    Build a lookup mapping (cocoidx, occurrence_index) to metadata for a subject.

    occurrence_index counts how many times the subject has seen the same ``cocoidx``,
    starting at 1 the first time the image appears.
    """
    counts: Dict[int, int] = defaultdict(int)
    lookup: Dict[Tuple[int, int], Dict[str, int]] = {}

    for subj_data_index, sample in enumerate(samples):
        if not sample:
            continue

        behav = sample[0]
        cocoidx = _extract_behav_field(behav, 0)
        global_trial = _extract_behav_field(behav, 5)

        counts[cocoidx] += 1
        occurrence = counts[cocoidx]
        lookup[(cocoidx, occurrence)] = {
            "subj_data_index": subj_data_index,
            "global_trial": global_trial,
        }

    return lookup


def align_subject_trials(
    all_subj_data: Dict[str, SubjectData],
    anchor_subject: str = "subj01",
    subjects: Optional[Iterable[str]] = None,
) -> List[AlignedTrial]:
    """
    Align trials across subjects based on shared cocoidx occurrences.

    Parameters
    ----------
    all_subj_data
        Mapping subject id -> iterable of WebDataset samples (ordered by presentation).
    anchor_subject
        Subject whose presentation order defines the master timeline.
    subjects
        Optional subset of subject ids to include. Defaults to all keys in all_subj_data.

    Returns
    -------
    List of aligned trials. Each entry contains:
        - ``cocoidx``: stimulus id
        - ``occurrence``: 1-based repetition count for anchor_subject
        - For each subject: ``{"subj_data_index": int, "global_trial": int}`` or ``None``
    """
    if anchor_subject not in all_subj_data:
        raise KeyError(f"Anchor subject '{anchor_subject}' not found in all_subj_data keys.")

    subject_ids = list(subjects) if subjects is not None else list(all_subj_data.keys())
    if anchor_subject not in subject_ids:
        subject_ids.insert(0, anchor_subject)

    anchor_subj_data = all_subj_data[anchor_subject]
    anchor_counts: Dict[int, int] = defaultdict(int)
    aligned_trials: List[AlignedTrial] = []

    for anchor_index, sample in enumerate(anchor_subj_data):
        behav = sample[0]
        cocoidx = _extract_behav_field(behav, 0)
        global_trial = _extract_behav_field(behav, 5)

        anchor_counts[cocoidx] += 1
        occurrence = anchor_counts[cocoidx]

        aligned_trials.append(
            {
                "cocoidx": cocoidx,
                "occurrence": occurrence,
                anchor_subject: {
                    "subj_data_index": anchor_index,
                    "global_trial": global_trial,
                },
            }
        )

    # Pre-compute lookup tables for the remaining subjects
    lookups: Dict[str, Dict[Tuple[int, int], Dict[str, int]]] = {}
    for subject in subject_ids:
        if subject == anchor_subject:
            continue
        if subject not in all_subj_data:
            raise KeyError(f"Subject '{subject}' requested but not found in all_subj_data keys.")
        lookups[subject] = _index_subject_occurrences(all_subj_data[subject])

    # Populate aligned trials with per-subject matches
    for trial in aligned_trials:
        cocoidx = trial["cocoidx"]
        occurrence = trial["occurrence"]
        for subject in subject_ids:
            if subject == anchor_subject:
                continue
            subject_lookup = lookups.get(subject, {})
            trial[subject] = subject_lookup.get((cocoidx, occurrence))

    return aligned_trials