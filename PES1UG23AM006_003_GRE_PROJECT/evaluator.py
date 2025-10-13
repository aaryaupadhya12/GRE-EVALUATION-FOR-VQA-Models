
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
from config import config
import json
from pathlib import Path


class VQAEvaluator:
    
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None
    ):
        if model_name is None:
            model_name = config.model.MODEL_NAME
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        print(f"\nLoading model: {model_name}")
        print(f"Device: {device}")
        
        # Load model and processor
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
        self.model.eval()
        print("Model loaded successfully.\n")

    def predict(self, image, question: str) -> str:
        """
        Generate answer for a single image-question pair
        """
        # Process inputs
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        
        # Use generate() for inference instead of forward()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=20)
            answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return answer

    def evaluate(
        self,
        dataset: List[Dict],
        metrics: List[str] = ["accuracy"]
    ) -> Dict[str, float]:
        
        results = defaultdict(list)
        
        print(f"Evaluating on {len(dataset)} samples...")
        for i, item in enumerate(dataset):
            pred = self.predict(item['image'], item['question'])
            results['pred'].append(pred)
            results['gt'].append(item['answer'])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples")
        
        scores = {}
        if "accuracy" in metrics:
            correct = sum([p.strip().lower() == g.strip().lower() 
                          for p, g in zip(results['pred'], results['gt'])])
            scores['accuracy'] = correct / len(dataset)
            print(f"\nAccuracy: {scores['accuracy']:.4f} ({correct}/{len(dataset)})")
        
        # Add other metrics if needed (BLEU, ROUGE, etc.)
        return scores

    def save_results(self, dataset: List[Dict], save_path: str = "vqa_results.json"):
        """
        Save predictions and ground truths for analysis
        """
        results = []
        for item in dataset:
            pred = self.predict(item['image'], item['question'])
            results.append({
                "question": item['question'],
                "ground_truth": item['answer'],
                "prediction": pred
            })
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {save_path}")