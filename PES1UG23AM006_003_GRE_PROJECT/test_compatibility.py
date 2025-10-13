"""
Compatibility Test Script
Quick test to verify all modules work together
"""

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    try:
        from config import config
        from dataset_generation import EnhancedSyntheticGenerator
        from gre_transformations import GRETransformer
        from evaluator import VQAEvaluator
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test config structure"""
    print("\nTesting config...")
    try:
        from config import config
        assert hasattr(config, 'dataset')
        assert hasattr(config, 'objects')
        assert hasattr(config, 'model')
        assert hasattr(config, 'evaluation')
        print(f"✓ Config loaded: {config.dataset.TOTAL_SAMPLES} samples")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False


def test_generator():
    """Test dataset generation"""
    print("\nTesting generator...")
    try:
        from dataset_generation import EnhancedSyntheticGenerator
        gen = EnhancedSyntheticGenerator(seed=42)
        sample = gen.generate_sample(0, use_distractors=True)
        
        assert 'image' in sample
        assert 'qa_pairs' in sample
        assert len(sample['qa_pairs']) > 0
        
        # Check QA pair structure
        q, a, qtype = sample['qa_pairs'][0]
        assert isinstance(q, str)
        assert isinstance(a, str)
        assert isinstance(qtype, str)
        
        print(f"✓ Generated sample with {len(sample['qa_pairs'])} QA pairs")
        return True
    except Exception as e:
        print(f"✗ Generator error: {e}")
        return False


def test_transformer():
    """Test GRE transformations"""
    print("\nTesting GRE transformer...")
    try:
        from dataset_generation import EnhancedSyntheticGenerator
        from gre_transformations import GRETransformer
        
        gen = EnhancedSyntheticGenerator(seed=42)
        transformer = GRETransformer(gen)
        sample = gen.generate_sample(0)
        
        # Test each transform
        g_sample = transformer.generalizability(sample)
        r_sample = transformer.robustness(sample)
        e_sample = transformer.extensibility(sample)
        
        assert g_sample['transform'] == 'G'
        assert r_sample['transform'] == 'R'
        assert e_sample['transform'] == 'E'
        
        print("✓ All GRE transforms working")
        return True
    except Exception as e:
        print(f"✗ Transformer error: {e}")
        return False


def test_evaluator():
    """Test evaluator (without actual model loading)"""
    print("\nTesting evaluator structure...")
    try:
        from evaluator import VQAEvaluator
        print("✓ Evaluator class loaded (skipping model download for speed)")
        return True
    except Exception as e:
        print(f"✗ Evaluator error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("GRE BENCHMARK COMPATIBILITY TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_generator,
        test_transformer,
        test_evaluator
    ]
    
    passed = sum([test() for test in tests])
    total = len(tests)
    
    print("\n" + "=" * 60)
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("Your setup is ready to run!")
    else:
        print(f"⚠️ SOME TESTS FAILED ({passed}/{total} passed)")
        print("Please fix errors before running main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()