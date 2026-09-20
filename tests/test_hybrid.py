import numpy as np

from audio_debrief import hybrid


def W(i, s, e):
    return {"text": f"w{i}", "start": s, "end": e}


def test_units_split_on_gap_and_cap_on_word_boundary():
    words = [W(0, 0.0, 0.5), W(1, 0.6, 1.0),           # gap 0.5 -> split
             W(2, 1.5, 2.0), W(3, 2.1, 2.6), W(4, 2.7, 3.5),
             W(5, 3.6, 4.9),                            # would make unit 3.4 s -> split
             W(6, 5.0, 5.4)]
    units = hybrid.make_units(words, gap_s=0.3, max_len_s=3.0)
    assert [u["word_idx"] for u in units] == [[0, 1], [2, 3, 4], [5, 6]]
    assert units[0] == {"start": 0.0, "end": 1.0, "word_idx": [0, 1]}
    assert units[2]["start"] == 3.6 and units[2]["end"] == 5.4


def test_embedding_window_pads_short_units_and_clips_to_file():
    assert hybrid.embed_window({"start": 10.0, "end": 10.4}, 100.0, 1.0, 0.25) == (9.75, 10.65)
    assert hybrid.embed_window({"start": 10.0, "end": 12.0}, 100.0, 1.0, 0.25) == (10.0, 12.0)
    assert hybrid.embed_window({"start": 0.1, "end": 0.5}, 100.0, 1.0, 0.25) == (0.0, 0.75)
    assert hybrid.embed_window({"start": 99.8, "end": 99.9}, 100.0, 1.0, 0.25) == (99.55, 100.0)


def test_label_units_by_voiceprint_with_ambiguous_inheritance():
    scores = [0.8, 0.4, -0.1, 0.45, 0.6, 0.36]
    assert hybrid.label_units(scores, match=0.5, ambiguous_low=0.35) == [True, True, False, False, True, True]
    assert hybrid.label_units([0.4, 0.7], 0.5, 0.35) == [False, True]  # ambiguous first unit -> not self


def test_cluster_units_separates_two_and_collapses_one():
    rng = np.random.default_rng(0)
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    embs = np.array([a + rng.normal(0, 0.05, 3) for _ in range(5)] + [b + rng.normal(0, 0.05, 3) for _ in range(5)])
    labels = hybrid.cluster_units(embs, threshold=0.5, durations=[1.0] * 10, min_cluster_s=0.0)
    assert len(set(labels)) == 2 and len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1
    one = np.array([a + rng.normal(0, 0.05, 3) for _ in range(6)])
    assert len(set(hybrid.cluster_units(one, 0.5, [1.0] * 6, 0.0))) == 1
    assert hybrid.cluster_units(np.array([a]), 0.5, [1.0], 0.0) == [0]
    assert hybrid.cluster_units(np.zeros((0, 3)), 0.5, [], 0.0) == []


def test_cluster_units_absorbs_tiny_cluster_into_nearest():
    a, b, c = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    embs = np.array([a] * 5 + [b] * 5 + [c])
    assert len(set(hybrid.cluster_units(embs, 0.5, [2.0] * 10 + [0.5], min_cluster_s=0.0))) == 3
    labels = hybrid.cluster_units(embs, 0.5, [2.0] * 10 + [0.5], min_cluster_s=1.0)
    assert len(set(labels)) == 2 and labels[10] in (labels[0], labels[5])


def test_cluster_units_merges_anchors_whose_centroids_agree():
    # one speaker split by noise into two tight clusters (cos 0.9 apart) + a second speaker
    a1, a2, b = np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.436, 0.0]), np.array([0.0, 1.0, 0.0])
    embs = np.array([a1] * 5 + [a2] * 5 + [b] * 5)
    durs = [10.0] * 15
    assert len(set(hybrid.cluster_units(embs, 0.05, durs, min_cluster_s=30.0, merge_sim=0.95))) == 3
    labels = hybrid.cluster_units(embs, 0.05, durs, min_cluster_s=30.0, merge_sim=0.5)
    assert len(set(labels)) == 2 and labels[0] == labels[5] and labels[0] != labels[10]


def test_cluster_units_keeps_biggest_cluster_when_nothing_reaches_min():
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    labels = hybrid.cluster_units(np.array([a, a, a, b]), 0.5, [1.0, 1.0, 1.0, 1.0], min_cluster_s=100.0)
    assert len(set(labels)) == 1


def test_refine_labels_by_nearest_centroid():
    vp = np.array([1.0, 0.0, 0.0])
    other = np.array([0.0, 1.0, 0.0])
    embs = np.array([
        [0.9, 0.1, 0.0],    # strong self (0.9): untouched
        [0.31, -0.05, 0.9],  # weak self-ish, below ambiguous floor but closer to self than other -> self
        [0.1, 0.05, 0.99],   # noise: both low, below refine_min -> stays non-self
        [0.4, 0.8, 0.0],     # inherited-self (ambiguous band) but closer to other -> flipped to other
    ])
    scores = (embs @ vp).tolist()
    flags = [True, False, False, True]
    cluster_ids = [None, 0, 0, None]
    centroids = {0: other}
    new_flags, new_ids, stats = hybrid.refine_labels(embs, scores, flags, cluster_ids, centroids,
                                                     match=0.5, refine_min=0.25)
    assert new_flags == [True, True, False, False]
    assert new_ids == [None, None, 0, 0]
    assert stats == {"refined_to_self": 1, "refined_from_self": 1}


def test_assign_labels_self_first_then_clusters_by_first_appearance():
    labels = hybrid.assign_labels(self_flags=[True, False, False, True, False], cluster_ids=[None, 2, 2, None, 1])
    assert labels == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_00", "SPEAKER_02"]
    assert hybrid.assign_labels(self_flags=None, cluster_ids=[3, 1, 3]) == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_units_to_turns_merges_consecutive_same_speaker():
    units = [{"start": 0.0, "end": 1.0}, {"start": 1.2, "end": 2.0}, {"start": 3.0, "end": 4.0}, {"start": 4.5, "end": 5.0}]
    labels = ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert hybrid.units_to_turns(units, labels) == [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
        {"speaker": "SPEAKER_01", "start": 3.0, "end": 4.0},
        {"speaker": "SPEAKER_00", "start": 4.5, "end": 5.0},
    ]


def test_assemble_diarization_shape_with_voiceprint():
    units = [{"start": 0.0, "end": 1.0, "word_idx": [0]}, {"start": 1.5, "end": 2.5, "word_idx": [1]},
             {"start": 3.0, "end": 4.0, "word_idx": [2]}]
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    vp = np.array([1.0, 0.0])
    cfg = {"match_threshold": 0.5, "ambiguous_low": 0.35, "cluster_threshold": 0.5, "min_cluster_s": 0.0,
           "centroid_merge_sim": 0.5, "refine_min": 0.25}
    d = hybrid.assemble(units, embs, vp, cfg)
    assert d["method"] == "hybrid" and d["overlaps"] == [] and d["self_label"] == "SPEAKER_00"
    assert d["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert d["turns"] == [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}, {"speaker": "SPEAKER_01", "start": 1.5, "end": 4.0}]
    assert set(d["embeddings"]) == {"SPEAKER_00", "SPEAKER_01"} and len(d["embeddings"]["SPEAKER_01"]) == 2
    assert d["unit_count"] == 3 and d["scores"]["self_units"] == 1
    assert d["scores"]["refined_to_self"] == 0 and d["scores"]["refined_from_self"] == 0
    d2 = hybrid.assemble(units, embs, None, cfg)
    assert d2["self_label"] is None and d2["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
