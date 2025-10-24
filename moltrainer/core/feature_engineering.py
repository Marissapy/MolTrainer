"""
Feature Engineering Module - Comprehensive molecular descriptors and fingerprints
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


class MolecularFeatureGenerator:
    """Generate molecular features from SMILES using RDKit"""
    
    def __init__(self):
        self.invalid_smiles = []
        self.feature_names = []
        
        # Try to import RDKit
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
            from rdkit import RDLogger
            self.Chem = Chem
            self.Descriptors = Descriptors
            self.AllChem = AllChem
            self.MACCSkeys = MACCSkeys
            self.rdMolDescriptors = rdMolDescriptors
            self.RDLogger = RDLogger
        except ImportError:
            raise ImportError(
                "RDKit is required for molecular feature generation.\n"
                "Install with: conda install -c conda-forge rdkit"
            )
    
    def generate_features(self, smiles_series: pd.Series, 
                        feature_type: str = 'descriptors',
                        descriptor_set: str = 'basic',
                        fingerprint_type: str = 'morgan',
                        fingerprint_bits: int = 2048,
                        fingerprint_radius: int = 2,
                        combine_features: bool = False,
                        feature_spec: str = None,
                        verbose: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Generate molecular features from SMILES
        
        Args:
            smiles_series: Series of SMILES strings
            feature_type: 'descriptors', 'fingerprints', or 'combined'
            descriptor_set: 'basic', 'extended', 'all'
            fingerprint_type: 'morgan', 'maccs', 'rdk', 'atompair', 'topological'
            fingerprint_bits: Number of bits for fingerprint (if applicable)
            fingerprint_radius: Radius for Morgan fingerprint
            combine_features: Combine descriptors + fingerprints
            feature_spec: Custom feature specification (e.g., "desc:basic+fp:morgan:1024+fp:maccs")
            verbose: Print progress
        
        Returns:
            Tuple of (features_array, feature_names)
            
        Note:
            - In 'combined' mode, features are concatenated as: [descriptors, fingerprints]
            - Use feature_spec for custom combinations with specific order
        """
        self.invalid_smiles = []
        
        # Suppress RDKit warnings temporarily
        self.RDLogger.DisableLog('rdApp.*')
        
        # If feature_spec is provided, use custom combination
        if feature_spec:
            features, feature_names = self._generate_custom_features(
                smiles_series, feature_spec, verbose
            )
        elif feature_type == 'descriptors':
            if verbose:
                print(f"   Generating descriptors: {descriptor_set}")
            features, feature_names = self._generate_descriptors(
                smiles_series, descriptor_set, verbose
            )
        elif feature_type == 'fingerprints':
            if verbose:
                print(f"   Generating fingerprint: {fingerprint_type} ({fingerprint_bits} bits)")
            features, feature_names = self._generate_fingerprints(
                smiles_series, fingerprint_type, fingerprint_bits, 
                fingerprint_radius, verbose
            )
        elif feature_type == 'combined':
            if verbose:
                print(f"   Generating combined features:")
                print(f"     1. Descriptors: {descriptor_set}")
                print(f"     2. Fingerprint: {fingerprint_type} ({fingerprint_bits} bits)")
                print(f"     Concatenation order: [descriptors, fingerprints]")
            # Generate both descriptors and fingerprints
            desc_features, desc_names = self._generate_descriptors(
                smiles_series, descriptor_set, verbose=False
            )
            fp_features, fp_names = self._generate_fingerprints(
                smiles_series, fingerprint_type, fingerprint_bits,
                fingerprint_radius, verbose=False
            )
            # Combine: descriptors first, then fingerprints
            features = np.hstack([desc_features, fp_features])
            feature_names = desc_names + fp_names
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
        self.feature_names = feature_names
        
        # Re-enable RDKit logging
        self.RDLogger.EnableLog('rdApp.*')
        
        if verbose and self.invalid_smiles:
            print(f"   Warning: {len(self.invalid_smiles)} invalid SMILES found")
        
        return features, feature_names
    
    def _generate_custom_features(self, smiles_series: pd.Series, 
                                  feature_spec: str, 
                                  verbose: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Generate features based on custom specification
        
        Args:
            smiles_series: Series of SMILES strings
            feature_spec: Feature specification string
                        Format: "desc:basic+fp:morgan:1024+fp:maccs"
                        - desc:<set>: descriptors (basic, extended, all)
                        - fp:<type>:<bits>:<radius>: fingerprint
                        Example: "desc:basic+desc:extended+fp:morgan:1024:2+fp:maccs"
            verbose: Print progress
        
        Returns:
            Tuple of (features_array, feature_names)
        """
        if verbose:
            print(f"   Generating custom features: {feature_spec}")
        
        feature_parts = feature_spec.split('+')
        all_features = []
        all_names = []
        
        for i, part in enumerate(feature_parts, 1):
            part = part.strip()
            
            if part.startswith('desc:'):
                # Descriptor specification: desc:basic, desc:extended, desc:all
                desc_set = part.split(':')[1]
                if verbose:
                    print(f"     {i}. Descriptors: {desc_set}")
                features, names = self._generate_descriptors(smiles_series, desc_set, verbose=False)
                all_features.append(features)
                all_names.extend(names)
                
            elif part.startswith('fp:'):
                # Fingerprint specification: fp:morgan:1024:2 or fp:maccs
                parts = part.split(':')
                fp_type = parts[1]
                fp_bits = int(parts[2]) if len(parts) > 2 else 2048
                fp_radius = int(parts[3]) if len(parts) > 3 else 2
                
                if verbose:
                    print(f"     {i}. Fingerprint: {fp_type} ({fp_bits} bits" + 
                          (f", radius={fp_radius}" if fp_type == 'morgan' else "") + ")")
                features, names = self._generate_fingerprints(
                    smiles_series, fp_type, fp_bits, fp_radius, verbose=False
                )
                all_features.append(features)
                all_names.extend(names)
            else:
                raise ValueError(f"Invalid feature specification: {part}")
        
        # Concatenate all features in the specified order
        combined_features = np.hstack(all_features)
        
        if verbose:
            print(f"   Total features: {len(all_names)} ({' + '.join([str(f.shape[1]) for f in all_features])})")
        
        return combined_features, all_names
    
    def _generate_descriptors(self, smiles_series: pd.Series, 
                              descriptor_set: str,
                              verbose: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Generate physicochemical descriptors"""
        
        # Define descriptor sets
        if descriptor_set == 'basic':
            descriptors = self._get_basic_descriptors()
        elif descriptor_set == 'extended':
            descriptors = self._get_extended_descriptors()
        elif descriptor_set == 'all':
            descriptors = self._get_all_descriptors()
        else:
            raise ValueError(f"Unknown descriptor_set: {descriptor_set}")
        
        descriptor_names = [name for name, _ in descriptors]
        descriptor_funcs = [func for _, func in descriptors]
        
        if verbose:
            print(f"   Computing {len(descriptors)} descriptors...")
        
        features_list = []
        for idx, smiles in enumerate(smiles_series):
            if pd.isna(smiles):
                features_list.append([np.nan] * len(descriptors))
                self.invalid_smiles.append((idx, smiles, 'Missing SMILES'))
                continue
            
            mol = self.Chem.MolFromSmiles(str(smiles))
            if mol is None:
                features_list.append([np.nan] * len(descriptors))
                self.invalid_smiles.append((idx, str(smiles), 'Invalid SMILES'))
                continue
            
            desc_values = []
            for func in descriptor_funcs:
                try:
                    val = func(mol)
                    desc_values.append(val)
                except:
                    desc_values.append(np.nan)
            
            features_list.append(desc_values)
        
        return np.array(features_list), descriptor_names
    
    def _generate_fingerprints(self, smiles_series: pd.Series,
                               fingerprint_type: str,
                               n_bits: int,
                               radius: int,
                               verbose: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Generate molecular fingerprints"""
        
        if verbose:
            print(f"   Computing {fingerprint_type} fingerprints...")
        
        features_list = []
        for idx, smiles in enumerate(smiles_series):
            if pd.isna(smiles):
                features_list.append([np.nan] * n_bits if fingerprint_type != 'maccs' else [np.nan] * 167)
                self.invalid_smiles.append((idx, smiles, 'Missing SMILES'))
                continue
            
            mol = self.Chem.MolFromSmiles(str(smiles))
            if mol is None:
                features_list.append([np.nan] * n_bits if fingerprint_type != 'maccs' else [np.nan] * 167)
                self.invalid_smiles.append((idx, str(smiles), 'Invalid SMILES'))
                continue
            
            try:
                if fingerprint_type == 'morgan':
                    fp = self.AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    features_list.append(list(fp))
                
                elif fingerprint_type == 'maccs':
                    fp = self.MACCSkeys.GenMACCSKeys(mol)
                    features_list.append(list(fp))
                
                elif fingerprint_type == 'rdk':
                    fp = self.Chem.RDKFingerprint(mol, fpSize=n_bits)
                    features_list.append(list(fp))
                
                elif fingerprint_type == 'atompair':
                    fp = self.AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
                    features_list.append(list(fp))
                
                elif fingerprint_type == 'topological':
                    fp = self.AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
                    features_list.append(list(fp))
                
                else:
                    raise ValueError(f"Unknown fingerprint_type: {fingerprint_type}")
            except:
                features_list.append([np.nan] * n_bits if fingerprint_type != 'maccs' else [np.nan] * 167)
                self.invalid_smiles.append((idx, str(smiles), f'Error generating {fingerprint_type} fingerprint'))
        
        # Generate feature names
        if fingerprint_type == 'maccs':
            feature_names = [f'MACCS_{i}' for i in range(167)]
        else:
            feature_names = [f'{fingerprint_type.upper()}_bit_{i}' for i in range(n_bits)]
        
        return np.array(features_list), feature_names
    
    def _get_basic_descriptors(self) -> List[Tuple[str, callable]]:
        """Get basic descriptor set (fast computation)"""
        return [
            ('MolWt', self.Descriptors.MolWt),
            ('LogP', self.Descriptors.MolLogP),
            ('NumHDonors', self.Descriptors.NumHDonors),
            ('NumHAcceptors', self.Descriptors.NumHAcceptors),
            ('TPSA', self.Descriptors.TPSA),
            ('NumRotatableBonds', self.Descriptors.NumRotatableBonds),
            ('NumAromaticRings', self.Descriptors.NumAromaticRings),
            ('NumSaturatedRings', self.Descriptors.NumSaturatedRings),
            ('NumAliphaticRings', self.Descriptors.NumAliphaticRings),
            ('RingCount', self.Descriptors.RingCount),
        ]
    
    def _get_extended_descriptors(self) -> List[Tuple[str, callable]]:
        """Get extended descriptor set (more comprehensive)"""
        basic = self._get_basic_descriptors()
        extended = [
            # Lipophilicity
            ('MolMR', self.Descriptors.MolMR),
            
            # Topology
            ('BertzCT', self.Descriptors.BertzCT),
            ('Chi0', self.Descriptors.Chi0),
            ('Chi1', self.Descriptors.Chi1),
            ('HallKierAlpha', self.Descriptors.HallKierAlpha),
            ('Kappa1', self.Descriptors.Kappa1),
            ('Kappa2', self.Descriptors.Kappa2),
            ('Kappa3', self.Descriptors.Kappa3),
            
            # Electronic
            ('NumValenceElectrons', self.Descriptors.NumValenceElectrons),
            ('NumRadicalElectrons', self.Descriptors.NumRadicalElectrons),
            
            # Structural
            ('FractionCSP3', self.Descriptors.FractionCSP3),
            ('NumHeteroatoms', self.Descriptors.NumHeteroatoms),
            ('NumSaturatedCarbocycles', self.Descriptors.NumSaturatedCarbocycles),
            ('NumSaturatedHeterocycles', self.Descriptors.NumSaturatedHeterocycles),
            ('NumAromaticCarbocycles', self.Descriptors.NumAromaticCarbocycles),
            ('NumAromaticHeterocycles', self.Descriptors.NumAromaticHeterocycles),
            
            # Pharmacophore
            ('LabuteASA', self.Descriptors.LabuteASA),
            ('PEOE_VSA1', self.Descriptors.PEOE_VSA1),
            ('PEOE_VSA2', self.Descriptors.PEOE_VSA2),
            ('SMR_VSA1', self.Descriptors.SMR_VSA1),
            ('SMR_VSA2', self.Descriptors.SMR_VSA2),
        ]
        return basic + extended
    
    def _get_all_descriptors(self) -> List[Tuple[str, callable]]:
        """Get all available 2D descriptors (comprehensive, slower)"""
        # Get all descriptor names from RDKit
        descriptor_names = [name for name, _ in self.Descriptors.descList]
        
        # Filter out 3D descriptors and problematic ones
        exclude = ['Ipc', 'BalabanJ']  # Some may fail on certain molecules
        
        descriptors = []
        for name in descriptor_names:
            if name not in exclude:
                try:
                    func = getattr(self.Descriptors, name)
                    descriptors.append((name, func))
                except:
                    pass
        
        return descriptors
    
    def optimize_fingerprint_length(self, smiles_series: pd.Series,
                                   y: np.ndarray,
                                   fingerprint_type: str = 'morgan',
                                   start_bits: int = 16,
                                   max_bits: int = 2048,
                                   step: int = 16,
                                   model_type: str = 'rf',
                                   task: str = 'classification',
                                   cv_folds: int = 3,
                                   verbose: bool = False) -> Dict:
        """
        Optimize fingerprint length by training models at different bit sizes
        
        Args:
            smiles_series: SMILES strings
            y: Target values
            fingerprint_type: Type of fingerprint
            start_bits: Starting bit size
            max_bits: Maximum bit size
            step: Step size for bit increments
            model_type: Model to use for evaluation
            task: 'classification' or 'regression'
            cv_folds: Cross-validation folds
            verbose: Print progress
        
        Returns:
            Dictionary with optimization results
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"FINGERPRINT LENGTH OPTIMIZATION")
            print(f"{'='*80}")
            print(f"Fingerprint: {fingerprint_type}")
            print(f"Bit range: {start_bits} to {max_bits} (step={step})")
            print(f"Model: {model_type}, Task: {task}, CV folds: {cv_folds}")
            print(f"{'='*80}\n")
        
        results = {
            'bit_sizes': [],
            'scores_mean': [],
            'scores_std': [],
            'best_bits': None,
            'best_score': -np.inf,
            'all_results': []
        }
        
        # Iterate through bit sizes
        bit_sizes = range(start_bits, max_bits + 1, step)
        
        for bits in bit_sizes:
            if verbose:
                print(f"Testing {bits} bits...", end=' ')
            
            # Generate features
            try:
                X, _ = self._generate_fingerprints(
                    smiles_series, fingerprint_type, bits, radius=2, verbose=False
                )
                
                # Remove rows with NaN
                valid_mask = ~np.isnan(X).any(axis=1)
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]
                
                if len(X_valid) < cv_folds:
                    if verbose:
                        print(f"Skipped (insufficient valid samples)")
                    continue
                
                # Train model with cross-validation
                if task == 'classification':
                    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                    scoring = 'accuracy'
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    scoring = 'r2'
                
                scores = cross_val_score(model, X_valid, y_valid, cv=cv_folds, scoring=scoring, n_jobs=-1)
                mean_score = scores.mean()
                std_score = scores.std()
                
                # Store results
                results['bit_sizes'].append(bits)
                results['scores_mean'].append(mean_score)
                results['scores_std'].append(std_score)
                results['all_results'].append({
                    'bits': bits,
                    'mean': mean_score,
                    'std': std_score,
                    'scores': scores.tolist()
                })
                
                # Update best
                if mean_score > results['best_score']:
                    results['best_bits'] = bits
                    results['best_score'] = mean_score
                
                if verbose:
                    print(f"Score: {mean_score:.4f} (+/- {std_score:.4f})")
            
            except Exception as e:
                if verbose:
                    print(f"Error: {str(e)}")
                continue
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"Best fingerprint length: {results['best_bits']} bits")
            print(f"Best score: {results['best_score']:.4f}")
            print(f"{'='*80}\n")
        
        return results


def get_available_feature_types():
    """Get list of available feature types"""
    return {
        'descriptors': {
            'basic': '10 basic physicochemical descriptors (fast)',
            'extended': '~30 extended descriptors (moderate)',
            'all': '200+ comprehensive descriptors (slow)'
        },
        'fingerprints': {
            'morgan': 'Morgan (circular) fingerprint',
            'maccs': 'MACCS keys (167 bits, fixed)',
            'rdk': 'RDKit fingerprint',
            'atompair': 'Atom pair fingerprint',
            'topological': 'Topological torsion fingerprint'
        },
        'combined': 'Descriptors + Fingerprints'
    }

