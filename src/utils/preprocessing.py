"""Video and frame preprocessing utilities."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .face_detection import FaceDetector


class VideoProcessor:
    """Process videos to extract frames."""
    
    def __init__(self, sample_rate: int = 2, temporal_length: int = 16):
        """
        Initialize video processor.
        
        Args:
            sample_rate: Frames per second to extract
            temporal_length: Number of frames per sequence
        """
        self.sample_rate = sample_rate
        self.temporal_length = temporal_length
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frames (BGR format)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.sample_rate))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def extract_sequences(self, video_path: str) -> List[List[np.ndarray]]:
        """
        Extract temporal sequences from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame sequences
        """
        frames = self.extract_frames(video_path)
        
        sequences = []
        for i in range(0, len(frames), self.temporal_length):
            sequence = frames[i:i+self.temporal_length]
            if len(sequence) == self.temporal_length:
                sequences.append(sequence)
        
        return sequences


class FrameProcessor:
    """Process individual frames with augmentation."""
    
    def __init__(self, config: dict, is_training: bool = True):
        """
        Initialize frame processor.
        
        Args:
            config: Configuration dictionary
            is_training: Whether to apply training augmentations
        """
        self.config = config
        self.is_training = is_training
        self.face_detector = FaceDetector(
            method=config['preprocessing']['face_detector'],
            face_size=config['preprocessing']['face_size'],
            min_face_size=config['preprocessing']['min_face_size']
        )
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        """Build augmentation pipeline."""
        aug_config = self.config['augmentation']
        
        transforms = []
        
        if self.is_training:
            # Photometric augmentations
            if 'photometric' in aug_config:
                phot = aug_config['photometric']
                if phot.get('brightness_range'):
                    transforms.append(A.RandomBrightnessContrast(
                        brightness_limit=phot['brightness_range'],
                        contrast_limit=phot['contrast_range'],
                        p=0.5
                    ))
                if phot.get('hue_shift'):
                    transforms.append(A.HueSaturationValue(
                        hue_shift_limit=int(phot['hue_shift'] * 255),
                        sat_shift_limit=phot.get('saturation_range', [0.8, 1.2]),
                        val_shift_limit=phot.get('brightness_range', [0.8, 1.2]),
                        p=0.5
                    ))
                if phot.get('jpeg_quality_range'):
                    transforms.append(A.ImageCompression(
                        quality_lower=phot['jpeg_quality_range'][0],
                        quality_upper=phot['jpeg_quality_range'][1],
                        p=0.3
                    ))
                if phot.get('blur_prob', 0) > 0:
                    transforms.append(A.GaussianBlur(
                        blur_limit=(3, 7),
                        sigma_limit=phot.get('blur_sigma', [0.5, 2.0]),
                        p=phot['blur_prob']
                    ))
            
            # Geometric augmentations
            if 'geometric' in aug_config:
                geom = aug_config['geometric']
                if geom.get('rotation_range'):
                    transforms.append(A.Rotate(
                        limit=geom['rotation_range'],
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5
                    ))
                if geom.get('translation_range'):
                    transforms.append(A.ShiftScaleRotate(
                        shift_limit=geom['translation_range'],
                        scale_limit=0.1,
                        rotate_limit=0,
                        p=0.5
                    ))
                if geom.get('horizontal_flip', False):
                    transforms.append(A.HorizontalFlip(p=geom.get('flip_prob', 0.5)))
        
        # Normalization and tensor conversion
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def process_frame(self, frame: np.ndarray, detect_face: bool = True) -> Optional[torch.Tensor]:
        """
        Process a single frame.
        
        Args:
            frame: Input frame (BGR format)
            detect_face: Whether to detect and crop face
            
        Returns:
            Processed tensor or None if face detection fails
        """
        if detect_face:
            result = self.face_detector.detect_face(frame)
            if result is None:
                return None
            face, _ = result
        else:
            face = cv2.resize(frame, (self.config['preprocessing']['face_size'], 
                                     self.config['preprocessing']['face_size']))
        
        # Apply augmentations
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=face_rgb)
        return transformed['image']
    
    def process_sequence(self, frames: List[np.ndarray]) -> Optional[torch.Tensor]:
        """
        Process a sequence of frames.
        
        Args:
            frames: List of frames
            
        Returns:
            Tensor of shape [T, C, H, W] or None
        """
        processed_frames = []
        for frame in frames:
            processed = self.process_frame(frame)
            if processed is None:
                return None
            processed_frames.append(processed)
        
        return torch.stack(processed_frames)

