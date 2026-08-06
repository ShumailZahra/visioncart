"""
tracker.py
----------
A minimal centroid tracker used to satisfy the proposal's "Real-Time
Multi-Object Detection" + temporal tracking requirement: an item sitting in
front of the camera across many frames must be counted into the cart
*once*, not once per frame.

This is intentionally simple (nearest-centroid + IoU gate, no Kalman
filter) so it runs at high FPS on CPU-only edge hardware such as a
Raspberry Pi, per the proposal's "Edge AI Deployment" requirement. Swap in
a heavier tracker (ByteTrack / DeepSORT) if the deployment target has more
headroom.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    class_name: str
    box: tuple
    hits: int = 1
    misses: int = 0
    counted: bool = False  # whether this track has already been logged to cart


@dataclass
class CentroidTracker:
    iou_threshold: float = 0.3
    max_misses: int = 8  # frames a track can go undetected before it's dropped
    min_hits_to_count: int = 3  # frames of confirmation before adding to cart
    _next_id: int = field(default=0, init=False)
    tracks: dict[int, Track] = field(default_factory=dict)

    def update(self, detections: list[tuple[str, tuple, float]]):
        """
        detections: list of (class_name, (x1,y1,x2,y2), confidence)
        returns: list of Track objects that just crossed the "count into cart"
                 threshold on this update (i.e. new, confirmed items)
        """
        unmatched = list(range(len(detections)))
        matched_track_ids = set()

        for track_id, track in list(self.tracks.items()):
            best_iou, best_idx = 0.0, -1
            for i in unmatched:
                cls, box, _ = detections[i]
                if cls != track.class_name:
                    continue
                iou = _iou(track.box, box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx != -1 and best_iou >= self.iou_threshold:
                cls, box, _ = detections[best_idx]
                track.box = box
                track.hits += 1
                track.misses = 0
                unmatched.remove(best_idx)
                matched_track_ids.add(track_id)
            else:
                track.misses += 1

        # drop stale tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].misses > self.max_misses:
                del self.tracks[track_id]

        # start new tracks for anything left unmatched
        for i in unmatched:
            cls, box, _ = detections[i]
            self.tracks[self._next_id] = Track(self._next_id, cls, box)
            self._next_id += 1

        newly_confirmed = []
        for track in self.tracks.values():
            if not track.counted and track.hits >= self.min_hits_to_count:
                track.counted = True
                newly_confirmed.append(track)
        return newly_confirmed

    def reset(self):
        self.tracks.clear()
        self._next_id = 0
