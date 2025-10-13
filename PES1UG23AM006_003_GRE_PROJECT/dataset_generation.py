
import random
import math
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from config import config


class EnhancedSyntheticGenerator:
    """
    Advanced synthetic data generator for GRE evaluation
    Features:
    - Multi-object scenes with relationships
    - Textured rendering with lighting
    - Complex spatial arrangements
    - Comprehensive metadata tracking
    - Dataset persistence (images + annotations)
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        self.objects = config.objects.OBJECTS
        self.scenes = config.objects.SCENES
        self.color_map = config.objects.COLORS
        self.image_size = config.dataset.IMAGE_SIZE
        
        # Question templates for different reasoning types
        self.question_templates = {
            "color": [
                "What color is the {obj}?",
                "What is the color of the {obj}?",
            ],
            "count": [
                "How many {obj}s are there?",
                "How many {obj}s are in the image?",
                "Count the number of {obj}s.",
            ],
            "existence": [
                "Is there a {obj} in the image?",
                "Can you see a {obj}?",
            ],
            "spatial": [
                "Is the {obj} near the center?",
                "Where is the {obj} located?",
            ],
            "compare": [
                "Is the {obj1} larger than the {obj2}?",
                "Which is bigger, the {obj1} or the {obj2}?",
            ],
            "relative_position": [
                "Is the {obj1} to the left of the {obj2}?",
                "Is the {obj1} above the {obj2}?",
            ],
        }
    
    def create_complex_image(
        self,
        main_obj: str,
        main_color: str,
        scene: str,
        add_distractors: bool = True,
        num_distractors: Optional[int] = None
    ) -> Tuple[Image.Image, List[Dict]]:
        """Create synthetic scene with controlled complexity"""
        bg_color = self.scenes[scene]
        img = Image.new("RGB", self.image_size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Apply lighting gradient
        self._apply_lighting_gradient(img)
        
        # Track all objects
        obj_metadata = []
        
        # Main object (centered)
        cx, cy = self.image_size[0] // 2, self.image_size[1] // 2
        main_scale = self.rng.uniform(1.2, 1.8)
        self._draw_object(draw, main_obj, main_color, cx, cy, main_scale)
        obj_metadata.append({
            "type": main_obj,
            "color": main_color,
            "x": cx,
            "y": cy,
            "scale": main_scale,
            "is_main": True
        })
        
        # Distractor objects
        if add_distractors:
            if num_distractors is None:
                num_distractors = self.rng.randint(
                    config.dataset.MIN_OBJECTS,
                    config.dataset.MAX_OBJECTS
                )
            
            for _ in range(num_distractors):
                obj_type = self.rng.choice(list(self.objects.keys()))
                obj_color = self.rng.choice(self.objects[obj_type])
                
                # Avoid center
                x, y = self._random_position_away_from_center(
                    cx, cy, 100, self.image_size[0]
                )
                scale = self.rng.uniform(0.6, 1.2)
                
                self._draw_object(draw, obj_type, obj_color, x, y, scale)
                obj_metadata.append({
                    "type": obj_type,
                    "color": obj_color,
                    "x": x,
                    "y": y,
                    "scale": scale,
                    "is_main": False
                })
        
        # Post-processing
        img = self._postprocess_image(img)
        
        return img, obj_metadata
    
    def _draw_object(
        self,
        draw: ImageDraw,
        obj_type: str,
        color: str,
        x: int,
        y: int,
        scale: float
    ):
        """Enhanced object rendering with more detail"""
        c = self.color_map[color]
        s = int(40 * scale)
        
        if obj_type in ["cup", "book"]:
            # Rectangle-based objects
            draw.rectangle(
                [x - s, y - s, x + s, y + s // 2],
                fill=c,
                outline=(0, 0, 0),
                width=2
            )
            # Add shading
            shade_color = tuple(max(0, c_val - 30) for c_val in c)
            draw.rectangle(
                [x - s + 5, y - s + 5, x + s - 5, y + s // 2 - 5],
                fill=shade_color
            )
        
        elif obj_type in ["ball", "apple"]:
            # Circular objects with gradient
            draw.ellipse(
                [x - s, y - s, x + s, y + s],
                fill=c,
                outline=(0, 0, 0),
                width=2
            )
            # Highlight
            highlight_color = tuple(min(255, c_val + 50) for c_val in c)
            draw.ellipse(
                [x - s // 3, y - s // 3, x + s // 3, y + s // 3],
                fill=highlight_color
            )
        
        elif obj_type == "car":
            # Car body
            draw.rectangle(
                [x - s, y - s // 2, x + s, y + s // 2],
                fill=c,
                outline=(0, 0, 0),
                width=2
            )
            # Wheels
            wheel_positions = [
                (x - s * 0.6, y + s * 0.3),
                (x + s * 0.6, y + s * 0.3)
            ]
            for wx, wy in wheel_positions:
                draw.ellipse(
                    [wx - s * 0.2, wy, wx + s * 0.2, wy + s * 0.4],
                    fill=(30, 30, 30)
                )
        
        elif obj_type == "banana":
            # Curved banana
            draw.arc(
                [x - s, y - s, x + s, y + s],
                start=30,
                end=150,
                fill=c,
                width=int(s * 0.4)
            )
        
        elif obj_type == "bottle":
            # Bottle shape
            neck_width = s // 3
            draw.rectangle(
                [x - neck_width, y - s, x + neck_width, y - s // 2],
                fill=c,
                outline=(0, 0, 0),
                width=1
            )
            draw.rectangle(
                [x - s, y - s // 2, x + s, y + s],
                fill=c,
                outline=(0, 0, 0),
                width=2
            )
        
        elif obj_type == "phone":
            # Smartphone rectangle
            draw.rectangle(
                [x - s // 2, y - s, x + s // 2, y + s],
                fill=c,
                outline=(0, 0, 0),
                width=2
            )
            # Screen
            screen_color = (220, 220, 255)
            draw.rectangle(
                [x - s // 3, y - s * 0.8, x + s // 3, y + s * 0.8],
                fill=screen_color
            )
    
    def _apply_lighting_gradient(self, img: Image.Image):
        """Apply realistic lighting gradient"""
        arr = np.array(img).astype(np.float32)
        h, w, _ = arr.shape
        
        # Create gradient
        lx = self.rng.uniform(-0.5, 0.5)
        ly = self.rng.uniform(-0.5, 0.5)
        
        x_gradient = np.linspace(0.8 + lx, 1.2 + lx, w)
        y_gradient = np.linspace(0.8 + ly, 1.2 + ly, h)
        
        gradient = np.outer(y_gradient, x_gradient)
        gradient = gradient[:, :, np.newaxis]
        
        arr = np.clip(arr * gradient, 0, 255)
        
        # Apply back to image
        result = Image.fromarray(arr.astype(np.uint8))
        img.paste(result)
    
    def _random_position_away_from_center(
        self,
        cx: int,
        cy: int,
        min_dist: int,
        max_dim: int
    ) -> Tuple[int, int]:
        """Sample position avoiding center"""
        margin = 60
        for _ in range(100):  # Max attempts
            x = self.rng.randint(margin, max_dim - margin)
            y = self.rng.randint(margin, max_dim - margin)
            if math.dist((x, y), (cx, cy)) > min_dist:
                return x, y
        return x, y  # Fallback
    
    def _postprocess_image(self, img: Image.Image) -> Image.Image:
        """Apply realistic image effects"""
        # Occasional blur
        if self.rng.random() < 0.3:
            blur_radius = self.rng.uniform(0.3, 1.2)
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.rng.uniform(0.9, 1.1))
        
        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.rng.uniform(0.95, 1.05))
        
        return img
    
    def generate_qa_pairs(
        self,
        objects: List[Dict],
        main_obj: str,
        num_questions: int = 3
    ) -> List[Tuple[str, str, str]]:
        """
        Generate multiple QA pairs per image
        Returns: List of (question, answer, type) tuples
        """
        qa_pairs = []
        main_object = [o for o in objects if o["is_main"]][0]
        other_objects = [o for o in objects if not o["is_main"]]
        
        # Available question types
        available_types = ["color", "count", "spatial", "existence"]
        if other_objects:
            available_types.extend(["compare", "relative_position"])
        
        # Generate diverse questions
        selected_types = self.rng.sample(
            available_types,
            min(num_questions, len(available_types))
        )
        
        for q_type in selected_types:
            if q_type == "color":
                q = self.rng.choice(self.question_templates["color"]).format(
                    obj=main_obj
                )
                a = main_object["color"]
            
            elif q_type == "count":
                count = sum(1 for o in objects if o["type"] == main_obj)
                q = self.rng.choice(self.question_templates["count"]).format(
                    obj=main_obj
                )
                a = str(count)
            
            elif q_type == "existence":
                q = self.rng.choice(self.question_templates["existence"]).format(
                    obj=main_obj
                )
                a = "yes"
            
            elif q_type == "spatial":
                cx, cy = self.image_size[0] // 2, self.image_size[1] // 2
                dist = math.dist((main_object["x"], main_object["y"]), (cx, cy))
                is_center = dist < 80
                q = self.rng.choice(self.question_templates["spatial"]).format(
                    obj=main_obj
                )
                a = "yes" if is_center else "no"
            
            elif q_type == "compare" and other_objects:
                other = self.rng.choice(other_objects)
                q = self.rng.choice(self.question_templates["compare"]).format(
                    obj1=main_obj,
                    obj2=other["type"]
                )
                a = "yes" if main_object["scale"] > other["scale"] else "no"
            
            elif q_type == "relative_position" and other_objects:
                other = self.rng.choice(other_objects)
                if self.rng.random() < 0.5:
                    # Left/right question
                    q = self.question_templates["relative_position"][0].format(
                        obj1=main_obj,
                        obj2=other["type"]
                    )
                    a = "yes" if main_object["x"] < other["x"] else "no"
                else:
                    # Above/below question
                    q = self.question_templates["relative_position"][1].format(
                        obj1=main_obj,
                        obj2=other["type"]
                    )
                    a = "yes" if main_object["y"] < other["y"] else "no"
            
            qa_pairs.append((q, a, q_type))
        
        return qa_pairs
    
    def generate_sample(
        self,
        sample_id: int,
        use_distractors: bool = True
    ) -> Dict:
        """Generate single sample with complete metadata"""
        obj_type = self.rng.choice(list(self.objects.keys()))
        obj_color = self.rng.choice(self.objects[obj_type])
        scene_type = self.rng.choice(list(self.scenes.keys()))
        
        img, metadata = self.create_complex_image(
            obj_type,
            obj_color,
            scene_type,
            use_distractors
        )
        
        qa_pairs = self.generate_qa_pairs(metadata, obj_type)
        
        return {
            "id": f"sample_{sample_id:05d}",
            "image": img,
            "object_type": obj_type,
            "object_color": obj_color,
            "scene_type": scene_type,
            "qa_pairs": qa_pairs,
            "metadata": metadata,
            "complexity": len(metadata),  # Number of objects
        }
    
    def save_dataset(
        self,
        samples: List[Dict],
        split_name: str = "full"
    ):
        """Save dataset to disk with images and annotations"""
        images_dir = Path(config.dataset.IMAGES_DIR) / split_name
        annotations_dir = Path(config.dataset.ANNOTATIONS_DIR)
        
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        annotations = []
        
        for sample in samples:
            sample_id = sample["id"]
            
            # Save image
            img_path = images_dir / f"{sample_id}.png"
            sample["image"].save(img_path)
            
            # Prepare annotation
            annotation = {
                "id": sample_id,
                "image_path": str(img_path),
                "object_type": sample["object_type"],
                "object_color": sample["object_color"],
                "scene_type": sample["scene_type"],
                "qa_pairs": sample["qa_pairs"],
                "metadata": sample["metadata"],
                "complexity": sample["complexity"],
            }
            annotations.append(annotation)
        
        # Save annotations as JSON
        ann_path = annotations_dir / f"{split_name}_annotations.json"
        with open(ann_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Saved {len(samples)} samples to {split_name}")
        print(f"  Images: {images_dir}")
        print(f"  Annotations: {ann_path}")
        
        return annotations