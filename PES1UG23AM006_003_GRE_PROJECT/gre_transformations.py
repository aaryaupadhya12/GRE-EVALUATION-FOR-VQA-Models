
import random
from typing import Dict, List
from config import config


class GRETransformer:
    
    
    def __init__(self, generator):
        self.gen = generator
        self.rng = random.Random(generator.rng.random())
    
    def generalizability(self, sample: Dict) -> Dict:
        
        obj_type = sample["object_type"]
        scene = sample["scene_type"]
        
        # Choose different color within same category
        available_colors = [
            c for c in self.gen.objects[obj_type]
            if c != sample["object_color"]
        ]
        if not available_colors:
            available_colors = self.gen.objects[obj_type]
        
        new_color = self.rng.choice(available_colors)
        
        # Generate new image
        img, meta = self.gen.create_complex_image(
            obj_type,
            new_color,
            scene,
            add_distractors=True
        )
        
        # Generate new QA pairs
        qa_pairs = self.gen.generate_qa_pairs(meta, obj_type)
        
        return self._package(
            sample, img, meta, qa_pairs,
            transform="G",
            new_obj=obj_type,
            new_color=new_color,
            new_scene=scene
        )
    
    def robustness(self, sample: Dict) -> Dict:
       
        obj_type = sample["object_type"]
        obj_color = sample["object_color"]
        
        # Choose different scene
        available_scenes = [
            s for s in self.gen.scenes.keys()
            if s != sample["scene_type"]
        ]
        if not available_scenes:
            available_scenes = list(self.gen.scenes.keys())
        
        new_scene = self.rng.choice(available_scenes)
        
        # Generate new image
        img, meta = self.gen.create_complex_image(
            obj_type,
            obj_color,
            new_scene,
            add_distractors=True
        )
        
        # Generate new QA pairs
        qa_pairs = self.gen.generate_qa_pairs(meta, obj_type)
        
        return self._package(
            sample, img, meta, qa_pairs,
            transform="R",
            new_obj=obj_type,
            new_color=obj_color,
            new_scene=new_scene
        )
    
    def extensibility(self, sample: Dict) -> Dict:
       
        scene = sample["scene_type"]
        
        # Choose different object type
        available_objs = [
            o for o in self.gen.objects.keys()
            if o != sample["object_type"]
        ]
        if not available_objs:
            available_objs = list(self.gen.objects.keys())
        
        new_obj = self.rng.choice(available_objs)
        new_color = self.rng.choice(self.gen.objects[new_obj])
        
        # Generate new image
        img, meta = self.gen.create_complex_image(
            new_obj,
            new_color,
            scene,
            add_distractors=True
        )
        
        # Generate new QA pairs
        qa_pairs = self.gen.generate_qa_pairs(meta, new_obj)
        
        return self._package(
            sample, img, meta, qa_pairs,
            transform="E",
            new_obj=new_obj,
            new_color=new_color,
            new_scene=scene
        )
    
    def _package(
        self,
        original: Dict,
        img,
        meta: List[Dict],
        qa_pairs: List,
        transform: str,
        new_obj: str,
        new_color: str,
        new_scene: str
    ) -> Dict:
        
        return {
            "id": f"{transform}_{original['id']}",
            "image": img,
            "object_type": new_obj,
            "object_color": new_color,
            "scene_type": new_scene,
            "qa_pairs": qa_pairs,
            "transform": transform,
            "original_id": original["id"],
            "metadata": meta,
            "complexity": len(meta),
        }
    
    def apply_all_transforms(
        self,
        samples: List[Dict]
    ) -> Dict[str, List[Dict]]:
        
        transforms = {
            'G': [],
            'R': [],
            'E': []
        }
        
        print(f"\nApplying GRE transformations to {len(samples)} samples...")
        
        for i, sample in enumerate(samples):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(samples)}")
            
            transforms['G'].append(self.generalizability(sample))
            transforms['R'].append(self.robustness(sample))
            transforms['E'].append(self.extensibility(sample))
        
        print(f"✓ GRE transformations complete")
        print(f"  G (Generalizability): {len(transforms['G'])} samples")
        print(f"  R (Robustness): {len(transforms['R'])} samples")
        print(f"  E (Extensibility): {len(transforms['E'])} samples")
        
        return transforms