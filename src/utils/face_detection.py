"""Face detection and alignment utilities."""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import torch
from facenet_pytorch import MTCNN
import dlib


class FaceDetector:
    """Face detection and alignment using multiple backends."""
    
    def __init__(self, method: str = "retinaface", face_size: int = 224, min_face_size: int = 50):
        """
        Initialize face detector.
        
        Args:
            method: Detection method ('mtcnn', 'retinaface', 'dlib')
            face_size: Output face size
            min_face_size: Minimum face size to accept
        """
        self.method = method
        self.face_size = face_size
        self.min_face_size = min_face_size
        
        if method == "mtcnn":
            self.detector = MTCNN(image_size=face_size, margin=0)
        elif method == "dlib":
            # Initialize dlib face detector
            self.detector = dlib.get_frontal_face_detector()
            predictor_path = Path(__file__).parent.parent.parent / "models" / "shape_predictor_68_face_landmarks.dat"
            if predictor_path.exists():
                self.predictor = dlib.shape_predictor(str(predictor_path))
            else:
                self.predictor = None
        else:  # retinaface or opencv fallback
            try:
                from retinaface import RetinaFace
                self.detector = RetinaFace
            except ImportError:
                # Fallback to OpenCV Haar Cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
                self.method = "opencv"
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect and align face in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (aligned_face, bbox) or None if no face detected
        """
        if self.method == "mtcnn":
            return self._detect_mtcnn(image)
        elif self.method == "dlib":
            return self._detect_dlib(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_mtcnn(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect face using MTCNN."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face
        boxes, probs, landmarks = self.detector.detect(rgb_image, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            return None
        
        # Get the face with highest confidence
        best_idx = np.argmax(probs)
        box = boxes[best_idx]
        
        # Extract and align face
        face = self.detector.extract(rgb_image, [box], save_path=None)
        if face is None or len(face) == 0:
            return None
        
        face = face[0]
        # Convert RGB back to BGR for consistency
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        
        bbox = box.astype(int)
        return face, bbox
    
    def _detect_dlib(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect face using dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        if face.width() < self.min_face_size or face.height() < self.min_face_size:
            return None
        
        # Extract face region
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        
        # Align if predictor available
        if self.predictor is not None:
            landmarks = self.predictor(gray, face)
            face_aligned = self._align_face(image, landmarks)
        else:
            face_aligned = image[y1:y2, x1:x2]
        
        # Resize to target size
        face_aligned = cv2.resize(face_aligned, (self.face_size, self.face_size))
        bbox = np.array([x1, y1, x2, y2])
        
        return face_aligned, bbox
    
    def _detect_opencv(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect face using OpenCV Haar Cascade (fallback)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_size, self.min_face_size)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Extract and resize
        face_roi = image[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (self.face_size, self.face_size))
        bbox = np.array([x, y, x+w, y+h])
        
        return face_roi, bbox
    
    def _align_face(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Align face using facial landmarks."""
        # Simple alignment: use eye positions
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        
        # Calculate angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        return aligned
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Detect faces in a batch of images."""
        results = []
        for image in images:
            result = self.detect_face(image)
            results.append(result)
        return results

