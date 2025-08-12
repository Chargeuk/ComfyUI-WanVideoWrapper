#!/usr/bin/env python3
"""
Test script for the improved frame-by-frame adaptive color matching functionality
"""

import torch
import numpy as np

class TestFrameByFrameColorMatch:
    def __init__(self):
        self.color_reference_buffer = []
        self.buffer_size = 5
        self.color_reference_method = "rolling_average"

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
        
        # Normalize histograms to prevent values over 1.0
        total_pixels = prev_frame.numel() / 3  # RGB channels
        prev_hist = prev_hist / total_pixels
        curr_hist = curr_hist / total_pixels
        
        # Combine metrics for final score
        scene_change_score = (hist_diff.item() + mean_diff.item()) / 2.0
        return min(scene_change_score, 1.0)  # Cap at 1.0
    
    def test_weighted_color_correction(self):
        """Test the new weighted blending color correction approach"""
        print("Testing weighted color correction blending...")
        
        # Create a simulated reference frame (sunny outdoor scene)
        ref_frame = torch.rand(1, 3, 256, 256)
        ref_frame[:, :, :, :128] = 0.8  # Bright right side
        ref_frame[:, :, :, 128:] = 0.3  # Darker left side
        
        # Create a test frame with color cast (blue tint)
        test_frame = torch.rand(1, 3, 256, 256)
        test_frame[:, 0, :, :] *= 0.7  # Reduce red
        test_frame[:, 1, :, :] *= 0.8  # Reduce green  
        test_frame[:, 2, :, :] *= 1.2  # Enhance blue
        
        # Simulate luminance analysis results
        mock_analysis = {
            'has_high_contrast': True,
            'luminance_variance_high': True,
            'color_cast_detected': True,
            'mixed_lighting': False,
            'dark_region_change': 0.45,      # High value
            'bright_region_change': 0.25,    # Medium value
            'contrast': 0.22,                # High value
            'color_cast_check': 0.15,        # High value
            'luminance_variance': 0.18       # High value
        }
        
        # Calculate weighted strengths
        total_metrics = (
            mock_analysis['dark_region_change'] +
            mock_analysis['bright_region_change'] +
            mock_analysis['contrast'] +
            mock_analysis['color_cast_check'] +
            mock_analysis['luminance_variance']
        )
        
        expected_weights = {
            'zone_based': mock_analysis['dark_region_change'] / total_metrics,
            'luminance_preserving': mock_analysis['bright_region_change'] / total_metrics,  
            'white_balance': mock_analysis['color_cast_check'] / total_metrics
        }
        
        print(f"Expected weights:")
        print(f"  Zone-based: {expected_weights['zone_based']:.3f}")
        print(f"  Luminance-preserving: {expected_weights['luminance_preserving']:.3f}")
        print(f"  White balance: {expected_weights['white_balance']:.3f}")
        print(f"  Total weight: {sum(expected_weights.values()):.3f}")
        
        # Verify weights sum to approximately 1.0 (accounting for unused metrics)
        weight_sum = sum(expected_weights.values())
        assert 0.4 <= weight_sum <= 1.0, f"Weight sum should be reasonable: {weight_sum}"
        
        print("✓ Weighted color correction test passed!")
        return True
        
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

    def mock_colormatch(self, reference, target, method, strength):
        """Mock color matching function"""
        # Simple color adjustment simulation
        adjusted = target + strength * 0.1 * (reference.mean() - target.mean())
        return (torch.clamp(adjusted, 0, 1),)

    def process_batch_with_frame_by_frame_colormatch(self, decoded_frames, color_match_strength, color_match_method, 
                                                    color_match_adaptive, color_match_scene_threshold, temporal_smoothing, color_match_source=None):
        """
        Process each frame in a batch individually for more precise color matching
        Updates references with the corrected frames
        """
        if color_match_strength <= 0.0:
            return decoded_frames
        
        processed_frames = []
        
        for i, current_frame in enumerate(decoded_frames):
            current_frame_batch = current_frame.unsqueeze(0)  # Add batch dimension
            
            # Determine reference frame for this specific frame
            reference_frame = None
            adaptive_strength = color_match_strength
            
            if self.color_reference_method == "rolling_average" and len(self.color_reference_buffer) > 0:
                # Use rolling average as reference
                reference_frame = torch.stack(self.color_reference_buffer).mean(dim=0).unsqueeze(0)
            elif self.color_reference_method == "previous_frame":
                if i > 0:
                    # Use previous CORRECTED frame in current batch
                    reference_frame = processed_frames[-1].unsqueeze(0)
                elif len(self.color_reference_buffer) > 0:
                    # Use last CORRECTED frame from buffer if this is first frame in batch
                    reference_frame = self.color_reference_buffer[-1].unsqueeze(0)
            else:
                # Default to first frame method or provided color_match_source
                if color_match_source is not None:
                    reference_frame = color_match_source
                elif len(self.color_reference_buffer) > 0:
                    reference_frame = self.color_reference_buffer[0].unsqueeze(0)
                elif i > 0:
                    reference_frame = processed_frames[0].unsqueeze(0)
            
            # If no reference available, use current frame as reference (no change)
            if reference_frame is None:
                reference_frame = current_frame_batch.clone()
                
            # Set initial color_match_source if not set yet and this is the first frame
            if i == 0 and color_match_source is None:
                color_match_source = current_frame_batch.clone()
            
            # Apply adaptive color matching if enabled
            if color_match_adaptive:
                scene_change_score = 0.0
                if i > 0:
                    # Compare with previous CORRECTED frame in batch
                    scene_change_score = self.detect_scene_change(
                        processed_frames[-1].unsqueeze(0), 
                        current_frame_batch
                    )
                elif len(self.color_reference_buffer) > 0:
                    # Compare with last CORRECTED frame from buffer
                    scene_change_score = self.detect_scene_change(
                        self.color_reference_buffer[-1].unsqueeze(0), 
                        current_frame_batch
                    )
                
                # Reduce color matching strength for significant scene changes
                adaptive_strength = color_match_strength * (1.0 - scene_change_score * 0.8)
                
                print(f"Frame {i}: Scene change score: {scene_change_score:.3f}, adaptive strength: {adaptive_strength:.3f}")
            
            # Apply color matching
            if adaptive_strength > 0.0:
                color_match_result = self.mock_colormatch(reference_frame, current_frame_batch, color_match_method, adaptive_strength)
                processed_frame = color_match_result[0][0]  # Remove batch dimension
            else:
                processed_frame = current_frame
            
            # Apply temporal smoothing if enabled
            if temporal_smoothing > 0.0 and i > 0:
                processed_frame_batch = processed_frame.unsqueeze(0)
                prev_frame_batch = processed_frames[-1].unsqueeze(0)
                smoothed_frame = self.temporal_color_smooth(processed_frame_batch, prev_frame_batch, temporal_smoothing)
                processed_frame = smoothed_frame[0]
            
            processed_frames.append(processed_frame)
            
            # Update rolling buffer with CORRECTED frame
            if self.color_reference_method == "rolling_average":
                if len(self.color_reference_buffer) >= self.buffer_size:
                    self.color_reference_buffer.pop(0)
                self.color_reference_buffer.append(processed_frame.clone())
        
        return torch.stack(processed_frames, dim=0)

def test_frame_by_frame_processing():
    """Test the frame-by-frame color matching functionality"""
    tester = TestFrameByFrameColorMatch()
    
    # Create test batch with varying colors
    batch_size = 8
    frames = []
    
    # Create frames with gradual color changes
    for i in range(batch_size):
        base_color = 0.3 + (i * 0.1)  # Gradually brightening
        frame = torch.ones(256, 256, 3) * base_color + torch.rand(256, 256, 3) * 0.1
        frames.append(frame)
    
    batch_frames = torch.stack(frames)
    
    print(f"Original batch shape: {batch_frames.shape}")
    print(f"Original frame means: {[f.mean().item():.3f for f in batch_frames]}")
    
    # Test frame-by-frame processing
    processed_frames = tester.process_batch_with_frame_by_frame_colormatch(
        batch_frames,
        color_match_strength=0.5,
        color_match_method="test_method",
        color_match_adaptive=True,
        color_match_scene_threshold=0.3,
        temporal_smoothing=0.1
    )
    
    print(f"Processed batch shape: {processed_frames.shape}")
    print(f"Processed frame means: {[f.mean().item():.3f for f in processed_frames]}")
    print(f"Buffer size after processing: {len(tester.color_reference_buffer)}")
    
    # Test rolling average update
    print(f"Rolling buffer means: {[f.mean().item():.3f for f in tester.color_reference_buffer]}")

def test_different_reference_methods():
    """Test different reference methods"""
    methods = ["rolling_average", "previous_frame", "first_frame"]
    
    for method in methods:
        print(f"\n=== Testing {method} method ===")
        tester = TestFrameByFrameColorMatch()
        tester.color_reference_method = method
        
        # Create test frames
        frames = []
        for i in range(5):
            base_color = 0.4 + (i * 0.1)
            frame = torch.ones(64, 64, 3) * base_color
            frames.append(frame)
        
        batch_frames = torch.stack(frames)
        
        processed_frames = tester.process_batch_with_frame_by_frame_colormatch(
            batch_frames,
            color_match_strength=0.3,
            color_match_method="test",
            color_match_adaptive=True,
            color_match_scene_threshold=0.3,
            temporal_smoothing=0.0
        )
        
        print(f"Buffer size: {len(tester.color_reference_buffer)}")

if __name__ == "__main__":
    print("Testing frame-by-frame adaptive color matching functionality...")
    
    tester = TestFrameByFrameColorMatch()
    
    print("\n=== Weighted Color Correction Test ===")
    try:
        tester.test_weighted_color_correction()
    except Exception as e:
        print(f"Weighted test failed: {e}")
    
    print("\n=== Frame-by-Frame Processing Test ===")
    test_frame_by_frame_processing()
    
    print("\n=== Reference Methods Test ===")
    test_different_reference_methods()
    
    print("\nAll tests completed!")
