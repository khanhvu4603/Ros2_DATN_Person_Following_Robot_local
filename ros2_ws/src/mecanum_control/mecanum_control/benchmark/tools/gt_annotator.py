#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground Truth Annotation Tool for Single-Target Tracking.

A simple OpenCV-based tool to annotate bounding boxes for the target person
in each frame of a video. Supports interpolation between keyframes.

Usage:
    python gt_annotator.py --video path/to/video.mp4 --output annotations.json

Controls:
    Left Click + Drag : Draw bounding box
    Right Click       : Delete current box
    → / D             : Next frame
    ← / A             : Previous frame
    Space             : Play/Pause
    I                 : Interpolate between keyframes
    S                 : Save annotations
    Q / Esc           : Quit (auto-saves)
    J                 : Jump to frame (enter frame number)
    + / =             : Increase playback speed
    - / _             : Decrease playback speed
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time


@dataclass
class Annotation:
    """Single frame annotation."""
    box: Optional[Tuple[int, int, int, int]]  # x1, y1, x2, y2
    is_keyframe: bool  # True if manually annotated, False if interpolated
    target_id: int = 1


class GroundTruthAnnotator:
    """
    Interactive tool for annotating ground truth bounding boxes.
    """
    
    WINDOW_NAME = "Ground Truth Annotator"
    
    def __init__(self, video_path: str, output_path: str):
        self.video_path = video_path
        self.output_path = output_path
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Annotations: frame_id -> Annotation
        self.annotations: Dict[int, Annotation] = {}
        self.keyframes: List[int] = []  # Sorted list of keyframe indices
        
        # Current state
        self.current_frame_idx = 0
        self.current_frame = None
        self.is_playing = False
        self.play_speed = 1.0
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_box = None
        
        # UI settings
        self.box_color = (0, 255, 0)  # Green for GT
        self.keyframe_color = (0, 255, 255)  # Yellow for keyframes
        self.interpolated_color = (255, 165, 0)  # Orange for interpolated
        
        # Load existing annotations if file exists
        if Path(output_path).exists():
            self._load_annotations()
            print(f"Loaded existing annotations from {output_path}")
    
    def _load_annotations(self):
        """Load annotations from JSON file."""
        with open(self.output_path, 'r') as f:
            data = json.load(f)
        
        self.annotations = {}
        self.keyframes = []
        
        for frame_data in data.get('frames', []):
            frame_id = frame_data['frame_id']
            box = tuple(frame_data['box']) if frame_data.get('box') else None
            is_keyframe = frame_data.get('is_keyframe', True)
            
            self.annotations[frame_id] = Annotation(
                box=box,
                is_keyframe=is_keyframe,
                target_id=frame_data.get('target_id', 1)
            )
            
            if is_keyframe and box is not None:
                self.keyframes.append(frame_id)
        
        self.keyframes.sort()
    
    def _save_annotations(self):
        """Save annotations to JSON file."""
        frames = []
        for frame_id in sorted(self.annotations.keys()):
            ann = self.annotations[frame_id]
            frames.append({
                'frame_id': frame_id,
                'box': list(ann.box) if ann.box else None,
                'is_keyframe': ann.is_keyframe,
                'target_id': ann.target_id
            })
        
        data = {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frames': frames,
            'keyframe_count': len(self.keyframes),
            'annotated_frame_count': len([a for a in self.annotations.values() if a.box])
        }
        
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.annotations)} annotations to {self.output_path}")
    
    def _read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read a specific frame from video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay on frame."""
        display = frame.copy()
        
        # Draw current annotation
        ann = self.annotations.get(self.current_frame_idx)
        if ann and ann.box:
            x1, y1, x2, y2 = ann.box
            color = self.keyframe_color if ann.is_keyframe else self.interpolated_color
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            label = "KEYFRAME" if ann.is_keyframe else "INTERPOLATED"
            cv2.putText(display, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw temporary box while drawing
        if self.temp_box:
            x1, y1, x2, y2 = self.temp_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Info bar at top
        info_bar = np.zeros((60, display.shape[1], 3), dtype=np.uint8)
        info_bar[:] = (40, 40, 40)
        
        # Frame info
        frame_text = f"Frame: {self.current_frame_idx}/{self.total_frames-1}"
        cv2.putText(info_bar, frame_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress bar
        progress = self.current_frame_idx / max(1, self.total_frames - 1)
        bar_width = display.shape[1] - 20
        bar_start = 10
        bar_y = 45
        cv2.rectangle(info_bar, (bar_start, bar_y), (bar_start + bar_width, bar_y + 8),
                     (100, 100, 100), -1)
        cv2.rectangle(info_bar, (bar_start, bar_y), 
                     (bar_start + int(bar_width * progress), bar_y + 8),
                     (0, 200, 0), -1)
        
        # Draw keyframe markers on progress bar
        for kf in self.keyframes:
            kf_x = bar_start + int(bar_width * kf / max(1, self.total_frames - 1))
            cv2.line(info_bar, (kf_x, bar_y - 2), (kf_x, bar_y + 10), (0, 255, 255), 2)
        
        # Stats
        stats_text = f"Keyframes: {len(self.keyframes)} | Annotated: {len([a for a in self.annotations.values() if a.box])}"
        cv2.putText(info_bar, stats_text, (300, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Speed indicator
        if self.is_playing:
            speed_text = f"Playing {self.play_speed}x"
            cv2.putText(info_bar, speed_text, (display.shape[1] - 150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Help bar at bottom
        help_bar = np.zeros((35, display.shape[1], 3), dtype=np.uint8)
        help_bar[:] = (40, 40, 40)
        help_text = "←→:Nav | Drag:Draw | RClick:Delete | I:Interpolate | S:Save | Q:Quit"
        cv2.putText(help_bar, help_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Combine
        display = np.vstack([info_bar, display, help_bar])
        
        return display
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes."""
        # Adjust y for info bar offset
        y = y - 60
        
        if y < 0 or y >= self.height:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                self.temp_box = (x1, y1, x2, y2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                
                # Only save if box is meaningful size
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self._set_annotation(self.current_frame_idx, (x1, y1, x2, y2), is_keyframe=True)
                
                self.temp_box = None
                self.start_point = None
                self.end_point = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete annotation
            self._delete_annotation(self.current_frame_idx)
    
    def _set_annotation(self, frame_idx: int, box: Tuple[int, int, int, int], is_keyframe: bool = True):
        """Set annotation for a frame."""
        self.annotations[frame_idx] = Annotation(box=box, is_keyframe=is_keyframe)
        
        if is_keyframe and frame_idx not in self.keyframes:
            self.keyframes.append(frame_idx)
            self.keyframes.sort()
    
    def _delete_annotation(self, frame_idx: int):
        """Delete annotation for a frame."""
        if frame_idx in self.annotations:
            del self.annotations[frame_idx]
        if frame_idx in self.keyframes:
            self.keyframes.remove(frame_idx)
    
    def _interpolate_keyframes(self):
        """Interpolate boxes between all keyframes."""
        if len(self.keyframes) < 2:
            print("Need at least 2 keyframes to interpolate")
            return
        
        interpolated_count = 0
        
        for i in range(len(self.keyframes) - 1):
            start_kf = self.keyframes[i]
            end_kf = self.keyframes[i + 1]
            
            start_box = self.annotations[start_kf].box
            end_box = self.annotations[end_kf].box
            
            if start_box is None or end_box is None:
                continue
            
            # Linear interpolation between keyframes
            for frame_idx in range(start_kf + 1, end_kf):
                t = (frame_idx - start_kf) / (end_kf - start_kf)
                
                interp_box = tuple(int(
                    start_box[j] + t * (end_box[j] - start_box[j])
                ) for j in range(4))
                
                self._set_annotation(frame_idx, interp_box, is_keyframe=False)
                interpolated_count += 1
        
        print(f"Interpolated {interpolated_count} frames between {len(self.keyframes)} keyframes")
    
    def _go_to_frame(self, frame_idx: int):
        """Navigate to a specific frame."""
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.current_frame_idx = frame_idx
        self.current_frame = self._read_frame(frame_idx)
    
    def run(self):
        """Main loop."""
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        
        # Read first frame
        self.current_frame = self._read_frame(0)
        
        print("\n" + "="*50)
        print("     GROUND TRUTH ANNOTATOR")
        print("="*50)
        print(f"Video: {self.video_path}")
        print(f"Frames: {self.total_frames} @ {self.fps:.1f} FPS")
        print(f"Resolution: {self.width}x{self.height}")
        print("\nControls:")
        print("  Drag        : Draw box")
        print("  Right Click : Delete box")
        print("  ← →         : Previous/Next frame")
        print("  Space       : Play/Pause")
        print("  I           : Interpolate keyframes")
        print("  J           : Jump to frame")
        print("  S           : Save")
        print("  Q/Esc       : Quit")
        print("="*50 + "\n")
        
        last_frame_time = time.time()
        
        while True:
            if self.current_frame is None:
                self.current_frame = self._read_frame(self.current_frame_idx)
            
            if self.current_frame is not None:
                display = self._draw_ui(self.current_frame)
                cv2.imshow(self.WINDOW_NAME, display)
            
            # Handle playback
            if self.is_playing:
                current_time = time.time()
                frame_duration = 1.0 / (self.fps * self.play_speed)
                if current_time - last_frame_time >= frame_duration:
                    if self.current_frame_idx < self.total_frames - 1:
                        self._go_to_frame(self.current_frame_idx + 1)
                    else:
                        self.is_playing = False
                    last_frame_time = current_time
            
            # Wait for key
            wait_time = 1 if self.is_playing else 0
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or Esc
                self._save_annotations()
                break
            
            elif key == ord('d') or key == 83:  # D or Right arrow
                self._go_to_frame(self.current_frame_idx + 1)
            
            elif key == ord('a') or key == 81:  # A or Left arrow
                self._go_to_frame(self.current_frame_idx - 1)
            
            elif key == ord(' '):  # Space - play/pause
                self.is_playing = not self.is_playing
                last_frame_time = time.time()
            
            elif key == ord('i'):  # Interpolate
                self._interpolate_keyframes()
            
            elif key == ord('s'):  # Save
                self._save_annotations()
            
            elif key == ord('j'):  # Jump to frame
                print("Enter frame number: ", end='', flush=True)
                try:
                    frame_num = int(input())
                    self._go_to_frame(frame_num)
                except ValueError:
                    print("Invalid frame number")
            
            elif key == ord('+') or key == ord('='):  # Increase speed
                self.play_speed = min(4.0, self.play_speed * 1.5)
                print(f"Playback speed: {self.play_speed}x")
            
            elif key == ord('-') or key == ord('_'):  # Decrease speed
                self.play_speed = max(0.25, self.play_speed / 1.5)
                print(f"Playback speed: {self.play_speed}x")
        
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Ground Truth Annotation Tool for Single-Target Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Drag        : Draw bounding box
  Right Click : Delete box
  ← →         : Navigate frames
  Space       : Play/Pause
  I           : Interpolate between keyframes
  S           : Save annotations
  Q/Esc       : Quit (auto-saves)
        """
    )
    parser.add_argument('--video', '-v', required=True, help='Path to input video')
    parser.add_argument('--output', '-o', default=None, 
                        help='Output JSON path (default: video_name_gt.json)')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_gt.json")
    
    annotator = GroundTruthAnnotator(args.video, args.output)
    annotator.run()


if __name__ == '__main__':
    main()
