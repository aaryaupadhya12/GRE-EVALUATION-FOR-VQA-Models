"""
GRE Benchmark - Main Execution Script
Minimal orchestration for dataset generation, transformation, and evaluation
"""

import json
from pathlib import Path
from dataset_generation import EnhancedSyntheticGenerator
from gre_transformations import GRETransformer
from evaluator import VQAEvaluator
from config import config


def generate_dataset():

    gen = EnhancedSyntheticGenerator(seed=config.dataset.SEED)
    
    # Generate all samples
    samples = [gen.generate_sample(i) for i in range(config.dataset.TOTAL_SAMPLES)]
    
    # Split dataset
    n = len(samples)
    train_end = int(n * config.dataset.TRAIN_SPLIT)
    val_end = train_end + int(n * config.dataset.VAL_SPLIT)
    
    splits = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }
    
    # Save splits
    for split_name, split_data in splits.items():
        gen.save_dataset(split_data, split_name)
    
    return splits


def apply_gre_transforms(test_samples):

    
    gen = EnhancedSyntheticGenerator(seed=config.dataset.SEED)
    transformer = GRETransformer(gen)
    
    # Select subset for GRE evaluation
    gre_subset = test_samples[:config.dataset.GRE_SUBSET]
    
    # Apply all transforms
    transformed = transformer.apply_all_transforms(gre_subset)
    
    # Save transformed datasets
    for transform_type, samples in transformed.items():
        gen.save_dataset(samples, f"gre_{transform_type}")
    
    return gre_subset, transformed


def evaluate_model(base_samples, transformed_samples):

    evaluator = VQAEvaluator()
    results = {}
    
    # Prepare evaluation datasets
    datasets = {
        'Base': base_samples,
        'G': transformed_samples['G'],
        'R': transformed_samples['R'],
        'E': transformed_samples['E']
    }
    
    # Evaluate each dataset
    for name, samples in datasets.items():
        print(f"\nEvaluating {name}...")
        
        # Convert to VQA format
        vqa_data = []
        for sample in samples:
            for q, a, q_type in sample['qa_pairs']:
                vqa_data.append({
                    'image': sample['image'],
                    'question': q,
                    'answer': a,
                    'type': q_type
                })
        
        # Evaluate
        scores = evaluator.evaluate(vqa_data, metrics=['accuracy'])
        results[name] = {
            'accuracy': scores['accuracy'],
            'num_samples': len(vqa_data)
        }
        
        print(f"  Accuracy: {scores['accuracy']:.2%} ({len(vqa_data)} QA pairs)")
    
    return results


def save_results(results):

    Path(config.evaluation.RESULTS_DIR).mkdir(exist_ok=True)
    
    # Save metrics
    with open(config.evaluation.METRICS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    

    

    for name, metrics in results.items():
        print(f"{name:15s}: {metrics['accuracy']:.2%} ({metrics['num_samples']} samples)")
    
    # Calculate GRE retention
    base_acc = results['Base']['accuracy']
    gre_avg = sum(results[t]['accuracy'] for t in ['G', 'R', 'E']) / 3
    retention = (gre_avg / base_acc * 100) if base_acc > 0 else 0
    
    print(f"\n{'GRE Retention':15s}: {retention:.1f}%")
  


def main():

    # Step 1: Generate dataset
    splits = generate_dataset()
    
    # Step 2: Apply GRE transforms
    base_samples, transformed = apply_gre_transforms(splits['test'])
    
    # Step 3: Evaluate model
    results = evaluate_model(base_samples, transformed)
    
    # Step 4: Save results
    save_results(results)
    



if __name__ == "__main__":
    main()