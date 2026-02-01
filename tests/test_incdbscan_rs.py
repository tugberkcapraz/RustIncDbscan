"""Tests for incdbscan_rs.

Covers: noise, cluster creation, absorption, merge, splits, duplicates,
multi-dimensional data, deletion, and cross-validation against sklearn.
"""

import numpy as np
import pytest

from incdbscan_rs import IncrementalDBSCAN

EPS = 1.5
NOISE = -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def labels_of(db, points):
    return db.get_cluster_labels(np.atleast_2d(points))


def assert_all_same_cluster(db, points):
    labs = labels_of(db, points)
    assert len(set(labs)) == 1 and labs[0] >= 0, f"Expected one cluster, got {labs}"


def assert_all_noise(db, points):
    labs = labels_of(db, points)
    assert np.all(labs == NOISE), f"Expected all noise, got {labs}"


def assert_all_nan(db, points):
    labs = labels_of(db, points)
    assert np.all(np.isnan(labs)), f"Expected all NaN, got {labs}"


def are_lists_isomorphic(list_1, list_2):
    if len(list_1) != len(list_2):
        return False
    distinct_1 = set(list_1)
    distinct_2 = set(list_2)
    if len(distinct_1) != len(distinct_2):
        return False
    mappings = set(zip(list_1, list_2))
    return len(distinct_1) == len(mappings)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_params(self):
        db = IncrementalDBSCAN()
        assert db is not None

    def test_custom_params(self):
        db = IncrementalDBSCAN(eps=2.0, min_pts=10, p=1.0)
        assert db is not None

    def test_invalid_eps(self):
        with pytest.raises(ValueError):
            IncrementalDBSCAN(eps=-1.0)

    def test_invalid_eps_zero(self):
        with pytest.raises(ValueError):
            IncrementalDBSCAN(eps=0.0)

    def test_invalid_min_pts(self):
        with pytest.raises((ValueError, OverflowError)):
            IncrementalDBSCAN(min_pts=0)

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            IncrementalDBSCAN(p=0.5)


# ---------------------------------------------------------------------------
# Insertion: noise and cluster creation
# ---------------------------------------------------------------------------

class TestNoiseAndCreation:
    def test_single_point_is_noise(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        p = np.array([[0.0, 0.0]])
        db.insert(p)
        assert_all_noise(db, p)

    def test_two_points_are_noise(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        db.insert(pts)
        assert_all_noise(db, pts)

    def test_three_close_points_form_cluster(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        db.insert(pts)
        assert_all_same_cluster(db, pts)

    def test_far_point_stays_noise(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        cluster = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        far = np.array([[10.0, 10.0]])
        db.insert(cluster)
        db.insert(far)
        assert_all_same_cluster(db, cluster)
        assert_all_noise(db, far)


# ---------------------------------------------------------------------------
# Insertion: absorption
# ---------------------------------------------------------------------------

class TestAbsorption:
    def test_noise_absorbed_into_cluster(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        db.insert(pts)
        assert_all_noise(db, pts)

        trigger = np.array([[0.5, 0.5]])
        db.insert(trigger)
        assert_all_same_cluster(db, np.vstack([pts, trigger]))

    def test_border_point_absorbed(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        core = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
        db.insert(core)

        border = np.array([[EPS, 0.0]])
        db.insert(border)

        core_label = labels_of(db, core)[0]
        border_label = labels_of(db, border)[0]
        assert border_label == core_label or border_label == NOISE


# ---------------------------------------------------------------------------
# Insertion: two clusters
# ---------------------------------------------------------------------------

class TestTwoClusters:
    def test_two_separate_clusters(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        c1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        c2 = np.array([[10.0, 10.0], [11.0, 10.0], [10.5, 10.5]])
        db.insert(c1)
        db.insert(c2)

        l1 = labels_of(db, c1)
        l2 = labels_of(db, c2)
        assert len(set(l1)) == 1 and l1[0] >= 0
        assert len(set(l2)) == 1 and l2[0] >= 0
        assert l1[0] != l2[0]


# ---------------------------------------------------------------------------
# Insertion: merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_bridge_merges_two_clusters(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        left = np.array([[-EPS, 0], [-EPS * 2, 0], [-EPS * 3, 0]])
        right = np.array([[EPS, 0], [EPS * 2, 0], [EPS * 3, 0]])
        db.insert(left)
        db.insert(right)

        l_before = labels_of(db, np.vstack([left, right]))
        assert l_before[0] != l_before[3], "Clusters should be separate"

        bridge = np.array([[0.0, 0.0]])
        db.insert(bridge)

        l_after = labels_of(db, np.vstack([left, right]))
        assert len(set(l_after)) == 1, f"Should be merged, got {l_after}"


# ---------------------------------------------------------------------------
# Insertion: duplicates
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_three_identical_points_form_cluster(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        p = np.array([[0.0, 0.0]])
        db.insert(p)
        db.insert(p)
        db.insert(p)
        assert labels_of(db, p)[0] == 0

    def test_duplicate_increments_neighbor_count(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=4)
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        db.insert(pts)
        assert_all_noise(db, pts)

        # Fourth insert of an existing point should push it to min_pts
        db.insert(np.array([[0.0, 0.0]]))
        lab = labels_of(db, pts)
        assert any(l >= 0 for l in lab), f"Expected some clustering, got {lab}"


# ---------------------------------------------------------------------------
# Deletion: basic
# ---------------------------------------------------------------------------

class TestDeletion:
    def test_delete_existing_returns_true(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        p = np.array([[0.0, 0.0]])
        db.insert(p)
        result = db.delete(p)
        assert result == [True]

    def test_delete_nonexistent_returns_false(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        result = db.delete(np.array([[99.0, 99.0]]))
        assert result == [False]

    def test_deleted_point_becomes_nan(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        p = np.array([[0.0, 0.0]])
        db.insert(p)
        db.delete(p)
        assert_all_nan(db, p)

    def test_unknown_point_is_nan(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        assert_all_nan(db, np.array([[42.0, 42.0]]))

    def test_delete_duplicate_decrements(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        p = np.array([[0.0, 0.0]])
        db.insert(p)
        db.insert(p)
        db.insert(p)
        assert labels_of(db, p)[0] >= 0  # cluster

        db.delete(p)  # count 3 -> 2
        db.delete(p)  # count 2 -> 1
        # Still exists but may not be a cluster anymore
        lab = labels_of(db, p)[0]
        assert not np.isnan(lab), "Point should still exist"

        db.delete(p)  # count 1 -> 0, fully removed
        assert_all_nan(db, p)


# ---------------------------------------------------------------------------
# Deletion: splits
# ---------------------------------------------------------------------------

class TestSplit:
    def test_two_way_split(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        left = np.array([[-EPS, 0], [-EPS * 2, 0], [-EPS * 3, 0]])
        right = np.array([[EPS, 0], [EPS * 2, 0], [EPS * 3, 0]])
        bridge = np.array([[0.0, 0.0]])

        db.insert(left)
        db.insert(right)
        db.insert(bridge)

        all_pts = np.vstack([left, right, bridge])
        assert len(set(labels_of(db, all_pts))) == 1, "Should be one cluster"

        db.delete(bridge)
        labs = labels_of(db, np.vstack([left, right]))

        left_labs = set(labs[:3])
        right_labs = set(labs[3:])
        assert len(left_labs) == 1, f"Left not uniform: {labs[:3]}"
        assert len(right_labs) == 1, f"Right not uniform: {labs[3:]}"
        assert left_labs != right_labs, "Left and right should be different clusters"

    def test_three_way_split(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        left = np.array([[-EPS, 0], [-EPS * 2, 0], [-EPS * 3, 0]])
        top = np.array([[0, EPS], [0, EPS * 2], [0, EPS * 3]])
        bottom = np.array([[0, -EPS], [0, -EPS * 2], [0, -EPS * 3]])
        bridge = np.array([[0.0, 0.0]])

        db.insert(left)
        db.insert(top)
        db.insert(bottom)
        db.insert(bridge)

        db.delete(bridge)
        labs = labels_of(db, np.vstack([left, top, bottom]))

        left_labs = set(labs[:3])
        top_labs = set(labs[3:6])
        bottom_labs = set(labs[6:])
        all_unique = {labs[0], labs[3], labs[6]}

        assert len(left_labs) == 1
        assert len(top_labs) == 1
        assert len(bottom_labs) == 1
        assert len(all_unique) == 3, f"Expected 3 clusters, got {all_unique}"
        assert NOISE not in all_unique


# ---------------------------------------------------------------------------
# Deletion: reinsert after delete
# ---------------------------------------------------------------------------

class TestReinsert:
    def test_delete_then_reinsert(self):
        db = IncrementalDBSCAN(eps=EPS, min_pts=3)
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        db.insert(pts)
        assert_all_same_cluster(db, pts)

        db.delete(np.array([[0.5, 0.5]]))
        db.insert(np.array([[0.5, 0.5]]))
        assert_all_same_cluster(db, pts)


# ---------------------------------------------------------------------------
# Multi-dimensional
# ---------------------------------------------------------------------------

class TestMultiDimensional:
    @pytest.mark.parametrize("n_dims", [1, 2, 3, 10, 50])
    def test_various_dimensions(self, n_dims):
        rng = np.random.RandomState(42)
        db = IncrementalDBSCAN(eps=3.0, min_pts=3)
        pts = rng.randn(20, n_dims) * 0.5
        db.insert(pts)
        labs = labels_of(db, pts)
        assert not np.any(np.isnan(labs)), f"No NaN expected in {n_dims}D"


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

class TestDistanceMetrics:
    def test_manhattan_distance(self):
        db = IncrementalDBSCAN(eps=2.0, min_pts=3, p=1.0)
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        db.insert(pts)
        labs = labels_of(db, pts)
        assert np.all(labs == labs[0]) and labs[0] >= 0

    def test_chebyshev_distance(self):
        db = IncrementalDBSCAN(eps=1.5, min_pts=3, p=float("inf"))
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        db.insert(pts)
        labs = labels_of(db, pts)
        assert np.all(labs == labs[0]) and labs[0] >= 0


# ---------------------------------------------------------------------------
# Cross-validation with sklearn DBSCAN
# ---------------------------------------------------------------------------

class TestSklearnCrossValidation:
    @pytest.fixture(autouse=True)
    def _skip_if_no_sklearn(self):
        pytest.importorskip("sklearn")

    @pytest.mark.parametrize(
        "n_samples,n_centers,eps,min_pts",
        [
            (50, 3, 1.5, 5),
            (30, 1, 1.5, 5),
            (100, 4, 2.0, 5),
            (200, 5, 2.0, 5),
        ],
    )
    def test_matches_sklearn(self, n_samples, n_centers, eps, min_pts):
        from sklearn.cluster import DBSCAN
        from sklearn.datasets import make_blobs

        X, _ = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=2,
            cluster_std=0.5,
            random_state=42,
        )

        rust_db = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
        rust_db.insert(X)
        rust_labels = [int(l) for l in rust_db.get_cluster_labels(X)]

        sklearn_labels = list(DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X))

        assert are_lists_isomorphic(rust_labels, sklearn_labels), (
            f"Labels mismatch: rust={rust_labels[:10]}... sklearn={sklearn_labels[:10]}..."
        )


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------

class TestStress:
    def test_many_batches_no_crash(self):
        """Insert and delete in batches to verify no crash or corruption."""
        rng = np.random.RandomState(42)
        db = IncrementalDBSCAN(eps=2.0, min_pts=5)

        for _ in range(5):
            batch = rng.randn(200, 2) * 10
            db.insert(batch)
            db.delete(batch[:40])

        remaining = rng.randn(50, 2) * 10
        db.insert(remaining)
        labs = labels_of(db, remaining)
        assert len(labs) == 50
        assert not np.any(np.isnan(labs))
