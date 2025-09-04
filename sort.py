"""
Minimal, pure-Python SORT tracker.

Dependencies:
- Required: numpy
- Optional: scipy (for Hungarian). Falls back to greedy matching if not present.
- Optional: filterpy (KalmanFilter). If not present, a simple numpy-based KF is used.

API:
- class Sort(max_age=1, min_hits=3, iou_threshold=0.3)
- update(dets): dets as ndarray of shape (N, 4) or (N, 5) [x1,y1,x2,y2,(score)]
  returns ndarray of shape (M, 5) [x1,y1,x2,y2,track_id]

This follows the original SORT paper formulation (x, y, s, r) with a constant
velocity model on (x, y, s).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

try:  # Optional scipy for optimal assignment
    from scipy.optimize import linear_sum_assignment as _lsa

    def _linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
        rows, cols = _lsa(cost_matrix)
        return np.stack([rows, cols], axis=1)

except Exception:  # Fallback to greedy if scipy is not available

    def _linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
        cm = cost_matrix.copy()
        matches: List[Tuple[int, int]] = []
        used_rows, used_cols = set(), set()

        while True:
            # pick best available pair
            idx = np.unravel_index(np.argmin(cm, axis=None), cm.shape)
            r, c = int(idx[0]), int(idx[1])
            v = cm[r, c]
            if not np.isfinite(v):
                break
            if r in used_rows or c in used_cols:
                cm[r, c] = np.inf
                continue
            matches.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
            cm[r, :] = np.inf
            cm[:, c] = np.inf
        if not matches:
            return np.empty((0, 2), dtype=int)
        return np.asarray(matches, dtype=int)


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Computes IoU between two sets of boxes.

    bb_test: (N, 4) [x1,y1,x2,y2]
    bb_gt:   (M, 4) [x1,y1,x2,y2]
    returns IoU matrix (N, M)
    """
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]), dtype=float)

    xx1 = np.maximum(bb_test[:, None, 0], bb_gt[None, :, 0])
    yy1 = np.maximum(bb_test[:, None, 1], bb_gt[None, :, 1])
    xx2 = np.minimum(bb_test[:, None, 2], bb_gt[None, :, 2])
    yy2 = np.minimum(bb_test[:, None, 3], bb_gt[None, :, 3])

    w = np.clip(xx2 - xx1, a_min=0.0, a_max=None)
    h = np.clip(yy2 - yy1, a_min=0.0, a_max=None)
    inter = w * h

    area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
    union = area_test[:, None] + area_gt[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert box [x1,y1,x2,y2] to z = [x,y,s,r].
    x,y is center, s is scale/area, r is aspect ratio.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / (h + 1e-6)
    return np.array([x, y, s, r], dtype=float)


def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert KF state x=[x,y,s,r,dx,dy,ds] to [x1,y1,x2,y2]."""
    w = math.sqrt(max(x[2], 1e-6) * x[3])
    h = max(x[2], 1e-6) / (w + 1e-9)
    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)


class _SimpleKalman:
    """Fallback simple Kalman filter if filterpy is not available."""

    def __init__(self):
        self.dim_x = 7
        self.dim_z = 4
        self.x = np.zeros((self.dim_x, 1), dtype=float)
        self.F = np.eye(self.dim_x)
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.P = np.eye(self.dim_x) * 10.0
        self.Q = np.eye(self.dim_x)
        self.R = np.eye(self.dim_z)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        z = z.reshape(-1, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P


def _make_kalman_filter() -> object:
    try:
        from filterpy.kalman import KalmanFilter  # type: ignore

        kf = KalmanFilter(dim_x=7, dim_z=4)
        return kf
    except Exception:
        return _SimpleKalman()


class KalmanBoxTracker:
    """Represents the internal state of an individual tracked object."""

    count = 0

    def __init__(self, bbox: np.ndarray):
        self.kf = _make_kalman_filter()

        # State transition for constant velocity on (x,y,s)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.kf.R *= 0.01
        self.kf.P *= 10.0
        self.kf.Q *= 0.01

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count + 1
        KalmanBoxTracker.count = self.id
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        z = convert_bbox_to_z(np.asarray(bbox, dtype=float))
        # Initialize state [x,y,s,r,dx,dy,ds]
        self.kf.x[:4, 0] = z
        # velocities start at zero
        self.kf.x[4:, 0] = 0.0

    def update(self, bbox: np.ndarray):
        """Update state with observed bbox."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        z = convert_bbox_to_z(np.asarray(bbox, dtype=float))
        self.kf.update(z)

    def predict(self) -> np.ndarray:
        """Advance the state and return the predicted bbox."""
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x[:, 0])

    def get_state(self) -> np.ndarray:
        return convert_x_to_bbox(self.kf.x[:, 0])


def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assigns detections to tracked object (both represented as [x1,y1,x2,y2]).

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers.
    """
    if trackers.size == 0 or detections.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(detections.shape[0], dtype=int),
            np.arange(trackers.shape[0], dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)
    # Convert to cost (we minimize cost)
    cost_matrix = 1.0 - iou_matrix
    matched_indices = _linear_assignment(cost_matrix)

    unmatched_dets = []
    unmatched_trks = []
    for d in range(detections.shape[0]):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)
    for t in range(trackers.shape[0]):
        if t not in matched_indices[:, 1]:
            unmatched_trks.append(t)

    matches = []
    for m in matched_indices:
        det_idx, trk_idx = int(m[0]), int(m[1])
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            unmatched_dets.append(det_idx)
            unmatched_trks.append(trk_idx)
        else:
            matches.append([det_idx, trk_idx])

    return (
        np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int),
        np.asarray(unmatched_dets, dtype=int) if unmatched_dets else np.empty((0,), dtype=int),
        np.asarray(unmatched_trks, dtype=int) if unmatched_trks else np.empty((0,), dtype=int),
    )


class Sort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        """Update tracker with current frame detections.

        dets: ndarray (N, 4) or (N, 5) with [x1,y1,x2,y2,(score)]
        Returns: ndarray (M, 5): [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1

        dets = np.asarray(dets, dtype=float)
        if dets.size == 0:
            dets = np.empty((0, 4), dtype=float)
        if dets.shape[1] >= 4:
            dets_xyxy = dets[:, :4]
        else:
            raise ValueError("detections must have shape (N, 4) or (N, 5)")

        trks = np.zeros((len(self.trackers), 4), dtype=float)
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = pos

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_xyxy, trks, self.iou_threshold
        )

        # update matched trackers
        for m in matched:
            self.trackers[int(m[1])].update(dets_xyxy[int(m[0])])

        # create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets_xyxy[int(i)])
            self.trackers.append(trk)

        # collect results and prune dead trackers
        ret: List[np.ndarray] = []
        new_trackers: List[KalmanBoxTracker] = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                d = trk.get_state()
                if (trk.hits >= self.min_hits) or (self.frame_count <= self.min_hits):
                    ret.append(np.concatenate([d, np.array([trk.id], dtype=float)]))
                new_trackers.append(trk)
        self.trackers = new_trackers

        if ret:
            return np.stack(ret, axis=0)
        return np.empty((0, 5), dtype=float)

