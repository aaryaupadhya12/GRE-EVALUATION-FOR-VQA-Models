
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    # Dataset sizes
    TOTAL_SAMPLES: int = 1000
    GRE_SUBSET: int = 300
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    
    # Image properties
    IMAGE_SIZE: Tuple[int, int] = (400, 400)
    MIN_OBJECTS: int = 1
    MAX_OBJECTS: int = 5
    
    # Random seed
    SEED: int = 42
    
    # Data paths
    DATA_DIR: str = "data"
    IMAGES_DIR: str = os.path.join(DATA_DIR, "images")
    ANNOTATIONS_DIR: str = os.path.join(DATA_DIR, "annotations")


@dataclass
class ObjectConfig:
    
    OBJECTS: Dict[str, List[str]] = None
    SCENES: Dict[str, Tuple[int, int, int]] = None
    COLORS: Dict[str, Tuple[int, int, int]] = None
    
    def __post_init__(self):
        if self.OBJECTS is None:
            self.OBJECTS = {
                "cup": ["red", "blue", "green", "yellow", "white"],
                "ball": ["red", "blue", "green", "orange", "purple"],
                "car": ["red", "blue", "black", "white", "silver"],
                "apple": ["red", "green", "yellow"],
                "banana": ["yellow", "green", "brown"],
                "book": ["red", "blue", "green", "brown", "black"],
                "bottle": ["red", "blue", "green", "clear"],
                "phone": ["black", "white", "silver", "blue"],
            }
        
        if self.SCENES is None:
            self.SCENES = {
                "table": (200, 180, 150),
                "grass": (100, 180, 100),
                "beach": (240, 220, 180),
                "sky": (135, 206, 235),
                "road": (80, 80, 80),
                "carpet": (150, 100, 100),
            }
        
        if self.COLORS is None:
            self.COLORS = {
                'red': (255, 0, 0),
                'blue': (0, 0, 255),
                'green': (0, 180, 0),
                'yellow': (255, 255, 0),
                'orange': (255, 165, 0),
                'purple': (150, 0, 150),
                'white': (255, 255, 255),
                'black': (30, 30, 30),
                'brown': (139, 69, 19),
                'silver': (180, 180, 180),
                'clear': (240, 240, 240),
            }


@dataclass
class ModelConfig:
    """Configuration for model evaluation"""
    MODEL_NAME: str = "Salesforce/blip-vqa-base"
    DEVICE: str = "cuda"  # Will be set automatically
    BATCH_SIZE: int = 16
    MAX_LENGTH: int = 20


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    RESULTS_DIR: str = "results"
    METRICS_FILE: str = os.path.join(RESULTS_DIR, "metrics.json")
    DETAILED_RESULTS: str = os.path.join(RESULTS_DIR, "detailed_results.json")
    VISUALIZATION_FILE: str = os.path.join(RESULTS_DIR, "gre_results.png")
    
    # Question types and their weights
    QUESTION_TYPES: List[str] = None
    
    def __post_init__(self):
        if self.QUESTION_TYPES is None:
            self.QUESTION_TYPES = [
                "color",
                "count",
                "spatial",
                "existence",
                "compare",
                "relative_position",
                "size_comparison",
            ]


class Config:
    
    def __init__(self):
        self.dataset = DatasetConfig()
        self.objects = ObjectConfig()
        self.model = ModelConfig()
        self.evaluation = EvaluationConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        
        dirs = [
            self.dataset.DATA_DIR,
            self.dataset.IMAGES_DIR,
            self.dataset.ANNOTATIONS_DIR,
            self.evaluation.RESULTS_DIR,
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def to_dict(self):
        
        return {
            "dataset": self.dataset.__dict__,
            "objects": {
                "OBJECTS": self.objects.OBJECTS,
                "SCENES": self.objects.SCENES,
                "COLORS": self.objects.COLORS,
            },
            "model": self.model.__dict__,
            "evaluation": self.evaluation.__dict__,
        }


# Global config instance
config = Config()