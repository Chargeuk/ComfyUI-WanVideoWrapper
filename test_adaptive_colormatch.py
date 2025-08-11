#!/usr/bin/env python3
"""
Test script for the adaptive color matching functionality
"""

import torch
import numpy as np

class TestAdaptiveColorMatch:
    def __init__(self):
        self.color_reference_buffer = []
        self.buffer_size = 5

    def detect_scene_change(self, prev_frame, current_frame, threshold=0.3):
        """
        Detect if there's a significant scene change between frames
        Returns a value between 0 (no change) and 1 (complete change)
        """
        # Convert to LAB color space for perceptual difference
        prev_lab = self.rgb_to_lab(prev_frame)
        curr_lab = self.rgb_to_lab(current_frame)
        
        # Calculate histogram differences
        prev_hist = torch.histc(prev_lab.flatten(), bins=256, min=0, max=1)
        curr_hist = torch.histc(curr_lab.flatten(), bins=256, min=0, max=1)
        hist_diff = torch.abs(prev_hist - curr_hist).mean()
        
        # Calculate mean difference
        mean_diff = torch.abs(prev_lab.mean() - curr_lab.mean())
        
        # Combine metrics
        scene_change_score = (hist_diff + mean_diff) / 2
        
        return min(scene_change_score.item(), 1.0)

    def rgb_to_lab(self, rgb_tensor):
        """Simple RGB to LAB conversion approximation"""
        # This is a simplified version - you might want to use a proper color space conversion
        r, g, b = rgb_tensor[..., 0], rgb_tensor[..., 1], rgb_tensor[..., 2]
        l = 0.299 * r + 0.587 * g + 0.114 * b
        a = (r - g) * 0.5
        b_comp = (r + g - 2 * b) * 0.25
        return torch.stack([l, a, b_comp], dim=-1)

    def temporal_color_smooth(self, current_frames, prev_frames, smoothing_factor=0.1):
        """
        Apply temporal smoothing to reduce sudden color changes
        """
        if prev_frames is None or smoothing_factor <= 0.0:
            return current_frames
        
        # Convert to LAB for better perceptual smoothing
        current_lab = self.rgb_to_lab(current_frames)
        prev_lab = self.rgb_to_lab(prev_frames[-current_frames.shape[0]:])
        
        # Apply exponential smoothing
        smoothed_lab = (1 - smoothing_factor) * current_lab + smoothing_factor * prev_lab
        
        # Convert back to RGB (simplified approximation)
        l, a, b_comp = smoothed_lab[..., 0], smoothed_lab[..., 1], smoothed_lab[..., 2]
        r = l + a
        g = l - a
        b = l - 4 * b_comp
        
        # Clamp values to valid range
        rgb_smoothed = torch.stack([r, g, b], dim=-1)
        rgb_smoothed = torch.clamp(rgb_smoothed, 0.0, 1.0)
        
        return rgb_smoothed

def test_scene_change_detection():
    """Test the scene change detection functionality"""
    tester = TestAdaptiveColorMatch()
    
    # Create test frames
    # Similar frames (beach scene)
    frame1 = torch.rand(1, 256, 256, 3) * 0.5 + 0.3  # Light colors (beach)
    frame2 = frame1 + torch.rand_like(frame1) * 0.1  # Similar with noise
    
    # Very different frames (dark indoor scene)
    frame3 = torch.rand(1, 256, 256, 3) * 0.3  # Dark colors
    
    # Test similar frames
    score1 = tester.detect_scene_change(frame1, frame2)
    print(f"Similar frames scene change score: {score1:.3f}")
    
    # Test different frames
    score2 = tester.detect_scene_change(frame1, frame3)
    print(f"Different frames scene change score: {score2:.3f}")
    
    # Test adaptive strength calculation
    base_strength = 0.8
    adaptive_strength1 = base_strength * (1.0 - score1 * 0.8)
    adaptive_strength2 = base_strength * (1.0 - score2 * 0.8)
    
    print(f"Base strength: {base_strength}")
    print(f"Adaptive strength for similar frames: {adaptive_strength1:.3f}")
    print(f"Adaptive strength for different frames: {adaptive_strength2:.3f}")

def test_temporal_smoothing():
    """Test the temporal smoothing functionality"""
    tester = TestAdaptiveColorMatch()
    
    # Create test frames with different colors
    prev_frames = torch.ones(5, 256, 256, 3) * 0.5  # Gray frames
    current_frames = torch.ones(2, 256, 256, 3) * 0.9  # Bright frames
    
    # Test smoothing
    smoothed = tester.temporal_color_smooth(current_frames, prev_frames, 0.3)
    
    print(f"Original current frame mean: {current_frames.mean():.3f}")
    print(f"Previous frames mean: {prev_frames.mean():.3f}")
    print(f"Smoothed frame mean: {smoothed.mean():.3f}")

if __name__ == "__main__":
    print("Testing adaptive color matching functionality...")
    print("\n=== Scene Change Detection Test ===")
    test_scene_change_detection()
    
    print("\n=== Temporal Smoothing Test ===")
    test_temporal_smoothing()
    
    print("\nTests completed!")
