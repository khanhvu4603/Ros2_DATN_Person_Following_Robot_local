import numpy as np
import time
from mecanum_control.tracking import DeepSORTTracker
from .detector import iou, center_of, enhanced_body_feature

class PersonTracker:
    def __init__(self, config, ort_sess, logger=None):
        self.logger = logger
        self.ort_sess = ort_sess
        self.config = config
        
        # DeepSORT Tracker
        self.deepsort = DeepSORTTracker(
            max_age=config.get('max_age', 60),
            n_init=config.get('n_init', 5),
            max_cosine_distance=config.get('max_cosine_distance', 0.08),
            lambda_weight=config.get('lambda_weight', 0.85)
        )
        
        # State
        self.current_track_id = None
        self.target_feature = None
        self.original_target_feature = None
        self.current_similarity = 0.0
        
    def _log_warn(self, msg):
        if self.logger: self.logger.warn(msg)
        else: print(f"[WARN] {msg}")

    def _log_info(self, msg):
        if self.logger: self.logger.info(msg)
        else: print(f"[INFO] {msg}")

    def update(self, boxes, features):
        return self.deepsort.update(boxes, features)
        
    def get_confirmed_tracks(self):
        return self.deepsort.get_confirmed_tracks()
        
    def get_track_by_id(self, track_id):
        return self.deepsort.get_track_by_id(track_id)

    def find_best_track_by_reid(self, tracks):
        """
        Tìm track có similarity cao nhất với ANCHOR feature.
        Bao gồm TRACK SWITCHING PREVENTION.
        """
        if self.original_target_feature is None:
            return None
        
        anchor = self.original_target_feature
        accept_thr = self.config.get('accept_threshold', 0.73)
        switch_margin = self.config.get('track_switch_margin', 0.2)
        
        best_track = None
        best_score = -1.0
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_feature = track.get_feature()
            if track_feature is None:
                continue
            
            score = float(np.dot(track_feature, anchor))
            if score > best_score:
                best_score = score
                best_track = track
        
        # === ANTI-ID-SWITCHING: Track Switching Prevention ===
        if self.current_track_id is not None:
            current_track = self.deepsort.get_track_by_id(self.current_track_id)
            if current_track is not None and not current_track.is_deleted():
                current_feature = current_track.get_feature()
                if current_feature is not None:
                    current_score = float(np.dot(current_feature, anchor))
                    
                    if best_track is not None and best_track.track_id != self.current_track_id:
                        if best_score < current_score + switch_margin:
                            if current_score > accept_thr:
                                self.current_similarity = current_score
                                return current_track
        
        if best_score > accept_thr:
            self.current_similarity = best_score
            return best_track
        
        return None

    def find_best_match_by_reid(self, boxes, frame, depth_frame, color_weight):
        best_box, best_score = None, -1.0
        anchor = self.original_target_feature if self.original_target_feature is not None else self.target_feature
        if anchor is None:
            return None, -1.0
            
        for box in boxes:
            feat = enhanced_body_feature(frame, box, depth_frame, self.ort_sess, color_weight=color_weight)
            if feat is None: continue
            
            score = np.dot(feat, anchor)
            if score > best_score:
                best_score = score
                best_box = box
        return best_box, best_score

    def update_target_model(self, new_feature):
        """
        Cập nhật target_feature với ANCHOR protection.
        Formula: target = 0.6 × ANCHOR + 0.3 × current + 0.1 × new
        """
        if self.target_feature is None:
            self.target_feature = new_feature.astype(np.float32)
            return
        
        if self.original_target_feature is None:
            alpha = 0.2
            self.target_feature = (1.0 - alpha) * self.target_feature + alpha * new_feature
        else:
            anchor_weight = 0.6
            current_weight = 0.3
            new_weight = 0.1
            
            self.target_feature = (
                anchor_weight * self.original_target_feature +
                current_weight * self.target_feature +
                new_weight * new_feature
            )

        self.target_feature /= (np.linalg.norm(self.target_feature) + 1e-8)

    def adaptive_model_update(self, box, frame, depth_frame, color_weight):
        """Cập nhật model thông minh với ANCHOR protection."""
        if box is None or self.target_feature is None:
            return

        candidate_feat = enhanced_body_feature(
            frame, box, depth_frame, self.ort_sess,
            color_weight=color_weight
        )
        if candidate_feat is None:
            return

        anchor = self.original_target_feature if self.original_target_feature is not None else self.target_feature
        similarity_with_anchor = float(np.dot(candidate_feat, anchor))
        
        update_min = 0.88
        if similarity_with_anchor < update_min:
            self._log_warn(f"Update rejected: similarity {similarity_with_anchor:.2f} < {update_min}")
            return

        if similarity_with_anchor > 0.99:
            return

        self.update_target_model(candidate_feat)
        self._log_info(f"Model updated. Anchor similarity: {similarity_with_anchor:.2f}")

    def locked_mode_tracking(self, frame, depth_frame, all_detections, all_features, last_known_depth, get_median_depth_func):
        """
        Custom tracking logic khi ở trạng thái LOCKED.
        """
        if self.current_track_id is None:
            return None, None
        
        target_track = self.deepsort.get_track_by_id(self.current_track_id)
        if target_track is None or target_track.is_deleted():
            return None, None
        
        predicted_box = tuple(map(int, target_track.to_tlbr()))
        
        best_detection_idx = None
        best_score = -1.0
        
        anchor = self.original_target_feature if self.original_target_feature is not None else self.target_feature
        
        for idx, (det_box, det_feat) in enumerate(zip(all_detections, all_features)):
            appearance_score = np.dot(det_feat, anchor) if anchor is not None else 0.0
            iou_score = iou(det_box, predicted_box)
            
            det_depth = get_median_depth_func(det_box, depth_frame)
            depth_score = 0.0
            if det_depth is not None and last_known_depth is not None:
                depth_diff = abs(det_depth - last_known_depth)
                depth_score = max(0, 1.0 - depth_diff / 1.0)
            
            combined_score = (
                0.60 * appearance_score +
                0.25 * iou_score +
                0.15 * depth_score
            )
            
            MIN_APPEARANCE = 0.70
            MIN_IOU = 0.20
            MIN_COMBINED = 0.65
            
            if (appearance_score >= MIN_APPEARANCE and 
                iou_score >= MIN_IOU and 
                combined_score > best_score and
                combined_score >= MIN_COMBINED):
                best_score = combined_score
                best_detection_idx = idx
        
        if best_detection_idx is not None:
            matched_det = [all_detections[best_detection_idx]]
            matched_feat = [all_features[best_detection_idx]]
            return matched_det, matched_feat
        else:
            self._log_warn("LOCKED: No matching detection, predict only")
            return [], []
