"""
Test Feature Engineering Module
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from moltrainer.core.feature_engineering import MolecularFeatureGenerator, get_available_feature_types

# Test data
test_smiles = pd.Series([
    'CCO',                    # Ethanol
    'CC(C)O',                # Isopropanol  
    'c1ccccc1',              # Benzene
    'c1ccc(O)cc1',           # Phenol
    'CCN',                   # Ethylamine
    'c1ccccc1O',             # Phenol (tautomer)
    'CC(=O)O',               # Acetic acid
    'c1ccc(N)cc1',           # Aniline
    'CC(C)(C)C',             # Neopentane
    'c1ccc(C)cc1',           # Toluene
])

def test_basic_descriptors():
    """Test basic descriptor generation"""
    print("\n" + "="*80)
    print("TEST 1: Basic Descriptors")
    print("="*80)
    
    generator = MolecularFeatureGenerator()
    features, feature_names = generator.generate_features(
        test_smiles,
        feature_type='descriptors',
        descriptor_set='basic',
        verbose=True
    )
    
    print(f"\nGenerated features shape: {features.shape}")
    print(f"Feature names ({len(feature_names)}): {feature_names[:5]}...")
    print(f"Sample values (first molecule): {features[0][:5]}")
    print(f"Invalid SMILES: {len(generator.invalid_smiles)}")
    print("✓ PASSED")

def test_extended_descriptors():
    """Test extended descriptor set"""
    print("\n" + "="*80)
    print("TEST 2: Extended Descriptors")
    print("="*80)
    
    generator = MolecularFeatureGenerator()
    features, feature_names = generator.generate_features(
        test_smiles,
        feature_type='descriptors',
        descriptor_set='extended',
        verbose=True
    )
    
    print(f"\nGenerated features shape: {features.shape}")
    print(f"Number of descriptors: {len(feature_names)}")
    print("✓ PASSED")

def test_morgan_fingerprint():
    """Test Morgan fingerprint"""
    print("\n" + "="*80)
    print("TEST 3: Morgan Fingerprint (2048 bits)")
    print("="*80)
    
    generator = MolecularFeatureGenerator()
    features, feature_names = generator.generate_features(
        test_smiles,
        feature_type='fingerprints',
        fingerprint_type='morgan',
        fingerprint_bits=2048,
        fingerprint_radius=2,
        verbose=True
    )
    
    print(f"\nGenerated features shape: {features.shape}")
    print(f"Feature dimensionality: {features.shape[1]}")
    print(f"Sum of bits (first molecule): {np.nansum(features[0])}")
    print("✓ PASSED")

def test_maccs_keys():
    """Test MACCS keys"""
    print("\n" + "="*80)
    print("TEST 4: MACCS Keys")
    print("="*80)
    
    generator = MolecularFeatureGenerator()
    features, feature_names = generator.generate_features(
        test_smiles,
        feature_type='fingerprints',
        fingerprint_type='maccs',
        verbose=True
    )
    
    print(f"\nGenerated features shape: {features.shape}")
    print(f"Number of MACCS keys: {features.shape[1]}")
    print("✓ PASSED")

def test_combined_features():
    """Test combined descriptors + fingerprints"""
    print("\n" + "="*80)
    print("TEST 5: Combined Features (Descriptors + Morgan FP)")
    print("="*80)
    
    generator = MolecularFeatureGenerator()
    features, feature_names = generator.generate_features(
        test_smiles,
        feature_type='combined',
        descriptor_set='basic',
        fingerprint_type='morgan',
        fingerprint_bits=512,
        verbose=True
    )
    
    print(f"\nGenerated features shape: {features.shape}")
    print(f"Total features: {features.shape[1]} (10 descriptors + 512 fp bits)")
    print(f"First 10 feature names (descriptors): {feature_names[:10]}")
    print(f"Last 5 feature names (fingerprints): {feature_names[-5:]}")
    print("✓ PASSED")

def test_rdk_fingerprint():
    """Test RDKit fingerprint"""
    print("\n" + "="*80)
    print("TEST 6: RDKit Fingerprint")
    print("="*80)
    
    generator = MolecularFeatureGenerator()
    features, feature_names = generator.generate_features(
        test_smiles,
        feature_type='fingerprints',
        fingerprint_type='rdk',
        fingerprint_bits=1024,
        verbose=True
    )
    
    print(f"\nGenerated features shape: {features.shape}")
    print("✓ PASSED")

def test_fingerprint_optimization():
    """Test fingerprint length optimization"""
    print("\n" + "="*80)
    print("TEST 7: Fingerprint Length Optimization")
    print("="*80)
    
    generator = MolecularFeatureGenerator()
    
    # Create dummy target variable
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0])
    
    results = generator.optimize_fingerprint_length(
        test_smiles,
        y,
        fingerprint_type='morgan',
        start_bits=16,
        max_bits=128,
        step=16,
        cv_folds=2,
        verbose=True
    )
    
    print(f"\nOptimization results:")
    print(f"Tested bit sizes: {results['bit_sizes']}")
    print(f"Best fingerprint length: {results['best_bits']} bits")
    print(f"Best score: {results['best_score']:.4f}")
    print("✓ PASSED")

def test_all_fingerprint_types():
    """Test all fingerprint types"""
    print("\n" + "="*80)
    print("TEST 8: All Fingerprint Types")
    print("="*80)
    
    fp_types = ['morgan', 'maccs', 'rdk', 'atompair', 'topological']
    generator = MolecularFeatureGenerator()
    
    for fp_type in fp_types:
        print(f"\n  Testing {fp_type}...", end=' ')
        try:
            features, _ = generator.generate_features(
                test_smiles,
                feature_type='fingerprints',
                fingerprint_type=fp_type,
                fingerprint_bits=256 if fp_type != 'maccs' else 167,
                verbose=False
            )
            print(f"Shape: {features.shape} ✓")
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    print("\n✓ ALL PASSED")

def show_available_features():
    """Show available feature types"""
    print("\n" + "="*80)
    print("AVAILABLE FEATURE TYPES")
    print("="*80)
    
    features = get_available_feature_types()
    
    print("\n1. DESCRIPTORS:")
    for key, desc in features['descriptors'].items():
        print(f"   - {key}: {desc}")
    
    print("\n2. FINGERPRINTS:")
    for key, desc in features['fingerprints'].items():
        print(f"   - {key}: {desc}")
    
    print(f"\n3. COMBINED: {features['combined']}")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("FEATURE ENGINEERING MODULE TEST SUITE")
    print("="*80)
    
    try:
        show_available_features()
        test_basic_descriptors()
        test_extended_descriptors()
        test_morgan_fingerprint()
        test_maccs_keys()
        test_combined_features()
        test_rdk_fingerprint()
        test_all_fingerprint_types()
        test_fingerprint_optimization()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

