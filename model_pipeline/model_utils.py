# senn_classifier.py

"""
SENN Classifier Utility Module
Production classifier for Self-Explaining Neural Networks (SENN)
Handles single and batch variant classification with feature importance analysis
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# ==================== MODEL ARCHITECTURE ====================

class IdentityConceptizer(nn.Module):
    """Identity conceptizer - passes features through unchanged"""
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
    
    def encode(self, x):
        return x.unsqueeze(-1)  # (BATCH, FEATURES, 1)
    
    def decode(self, z):
        return z.squeeze(-1)  # (BATCH, FEATURES)


class LinearParameterizer(nn.Module):
    """Parameterizer network for computing relevances"""
    def __init__(self, num_features, num_concepts, num_classes, hidden_sizes=None, dropout=0.3):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        
        if hidden_sizes is None:
            hidden_sizes = [num_features, 128, 64, 32, num_concepts * num_classes]
        else:
            hidden_sizes = [num_features] + list(hidden_sizes) + [num_concepts * num_classes]
        
        layers = []
        for h, h_next in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.Linear(h, h_next))
            if h_next != hidden_sizes[-1]:
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.layers(x)
        return output.view(x.size(0), self.num_concepts, self.num_classes)


class SumAggregator(nn.Module):
    """Aggregates concepts and relevances"""
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, concepts, relevances):
        aggregated = torch.bmm(relevances.permute(0, 2, 1), concepts).squeeze(-1)
        return F.log_softmax(aggregated, dim=1)


class SENN(nn.Module):
    """Self-Explaining Neural Network"""
    def __init__(self, conceptizer, parameterizer, aggregator):
        super().__init__()
        self.conceptizer = conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator
    
    def forward(self, x):
        concepts, recon_x = self.conceptizer(x)
        relevances = self.parameterizer(x)
        predictions = self.aggregator(concepts, relevances)
        explanations = (concepts, relevances)
        return predictions, explanations, recon_x


# ==================== CLASSIFIER CLASS ====================

class SENNClassifier:
    """
    Production classifier for SENN model.
    Handles single and batch classification with feature importance and weight adjustment.
    """
    
    # 33 features in exact order (after chromosome removal)
    FEATURE_NAMES = [
        # 14 MC features
        "has_MC_nonsense",
        "has_MC_non-coding_transcript_variant",
        "has_MC_missense_variant",
        "has_MC_intron_variant",
        "has_MC_5_prime_UTR_variant",
        "has_MC_splice_donor_variant",
        "has_MC_synonymous_variant",
        "has_MC_splice_acceptor_variant",
        "has_MC_initiator_codon_variant",
        "has_MC_3_prime_UTR_variant",
        "has_MC_no_sequence_alteration",
        "has_MC_stop_lost",
        "has_MC_genic_upstream_transcript_variant",
        "has_MC_genic_downstream_transcript_variant",
        
        # 11 Origin features
        "has_Origin_germline",
        "has_Origin_biparental",
        "has_Origin_unknown",
        "has_Origin_maternal",
        "has_Origin_paternal",
        "has_Origin_inherited",
        "has_Origin_de novo",
        "has_Origin_not applicable",
        "has_Origin_tested-inconclusive",
        "has_Origin_uniparental",
        "has_Origin_not-reported",
        
        # 6 VariantGeneRelation features
        "has_VariantGeneRelation_within single gene",
        "has_VariantGeneRelation_within multiple genes by overlap",
        "has_VariantGeneRelation_asserted, but not computed",
        "has_VariantGeneRelation_near gene, upstream",
        "has_VariantGeneRelation_near gene, downstream",
        "has_VariantGeneRelation_not identified",
        
        # 2 location features
        "is_genomic",
        "is_mitochondrial"
    ]
    
    CLASS_NAMES = ["Benign", "Pathogenic"]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize classifier and load model
        
        Args:
            model_path: Path to saved model weights (.pth file)
                       If None, loads from SAVE_BEST_MODEL_PATH environment variable
        """
        # Get model path
        if model_path is None:
            model_path = os.getenv('SAVE_BEST_MODEL_PATH')
            if model_path is None:
                raise ValueError("Model path must be provided or set in SAVE_BEST_MODEL_PATH environment variable")
        
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters (must match training configuration)
        self.num_features = len(self.FEATURE_NAMES)  # 33
        self.num_concepts = self.num_features
        self.num_classes = 2
        
        # Load model
        self.model = self._load_model()
        logger.info(f"SENN Classifier initialized on device: {self.device}")
    
    def _load_model(self) -> SENN:
        """Load SENN model from saved weights"""
        try:
            # Initialize model architecture (same as training)
            conceptizer = IdentityConceptizer()
            parameterizer = LinearParameterizer(
                num_features=self.num_features,
                num_concepts=self.num_concepts,
                num_classes=self.num_classes,
                hidden_sizes=[128, 64, 32],
                dropout=0.3
            )
            aggregator = SumAggregator(num_classes=self.num_classes)
            
            model = SENN(conceptizer, parameterizer, aggregator)
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            model.eval()
            model.to(self.device)
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def _prepare_input(self, encoded_features: List[float]) -> torch.Tensor:
        """
        Convert encoded features list to PyTorch tensor
        
        Args:
            encoded_features: List of 33 floats
            
        Returns:
            torch.Tensor of shape (1, 33)
        """
        if len(encoded_features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {len(encoded_features)}")
        
        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(encoded_features).unsqueeze(0).to(self.device)
        return tensor

    def _get_ranked_features_by_importance(self, 
                                       feature_importance: Dict[str, Dict[str, float]],
                                       top_k: int = 10) -> Dict[str, List[Dict]]:
        """
        Get top K features ranked separately for global, benign, and pathogenic
        
        Returns:
            Dict with three lists: global_ranking, benign_ranking, pathogenic_ranking
        """
        # Sort by global importance (absolute value)
        global_sorted = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]["global"]),
            reverse=True
        )
        
        # Sort by benign importance
        benign_sorted = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]["Benign"]),
            reverse=True
        )
        
        # Sort by pathogenic importance
        pathogenic_sorted = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]["Pathogenic"]),
            reverse=True
        )
        
        # Build ranking lists
        global_ranking = []
        for i, (feature, scores) in enumerate(global_sorted[:top_k], 1):
            global_ranking.append({
                "rank": i,
                "feature": feature,
                "score": scores["global"]
            })
        
        benign_ranking = []
        for i, (feature, scores) in enumerate(benign_sorted[:top_k], 1):
            benign_ranking.append({
                "rank": i,
                "feature": feature,
                "score": scores["Benign"]
            })
        
        pathogenic_ranking = []
        for i, (feature, scores) in enumerate(pathogenic_sorted[:top_k], 1):
            pathogenic_ranking.append({
                "rank": i,
                "feature": feature,
                "score": scores["Pathogenic"]
            })
        
        return {
            "global_ranking": global_ranking,
            "benign_ranking": benign_ranking,
            "pathogenic_ranking": pathogenic_ranking
        }


    def _get_detailed_concept_analysis(self, 
                                    feature_importance: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Get detailed concept-level analysis with all feature contributions
        
        Returns:
            Dict with concept scores and detailed feature breakdowns
        """
        concept_groups = {
            'molecular_consequence': ['has_MC_'],
            'data_source': ['has_Origin_'],
            'gene_context': ['has_VariantGeneRelation_'],
            'genomic_location': ['is_genomic', 'is_mitochondrial'],
            'sequence_change': ['ref_is_', 'alt_is_']  # In case you add these back
        }
        
        concept_scores = {}
        detailed_contributions = {}
        
        for concept_name, prefixes in concept_groups.items():
            concept_features = []
            total_global_importance = 0
            
            for feature_name, scores in feature_importance.items():
                # Check if feature belongs to this concept
                is_match = False
                for prefix in prefixes:
                    if feature_name.startswith(prefix) or feature_name == prefix:
                        is_match = True
                        break
                
                if is_match:
                    concept_features.append({
                        "feature": feature_name,
                        "global": scores["global"],
                        "benign": scores["Benign"],
                        "pathogenic": scores["Pathogenic"]
                    })
                    total_global_importance += abs(scores["global"])
            
            # Sort features within concept by absolute global importance
            concept_features.sort(key=lambda x: abs(x["global"]), reverse=True)
            
            concept_scores[concept_name] = total_global_importance
            detailed_contributions[concept_name] = concept_features
        
        # Sort concepts by total importance
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "concept_scores": dict(sorted_concepts),
            "detailed_contributions": detailed_contributions
        }

    
    def classify_single(self, encoder_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single variant with detailed output for LLM
        
        [Keep existing docstring]
        """
        # Extract encoded features
        encoded_features = encoder_output.get('encoded_features')
        if encoded_features is None:
            raise ValueError("No encoded_features found in encoder output")
        
        # Prepare input
        input_tensor = self._prepare_input(encoded_features)
        
        # Make prediction
        with torch.inference_mode():
            predictions, explanations, _ = self.model(input_tensor)
            
            # Get probabilities (predictions are log_softmax, so exp them)
            probs = torch.exp(predictions).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            
            # Extract explanations
            concepts, relevances = explanations
        
        # Calculate feature importance for this specific input
        feature_importance = self._calculate_feature_importance(
            input_tensor, concepts, relevances
        )
        
        # Get top contributing features (original simple version)
        top_features = self._get_top_features(feature_importance, top_k=10)
        
        # ===== NEW: Get comprehensive rankings =====
        feature_rankings = self._get_ranked_features_by_importance(
            feature_importance, top_k=10
        )
        
        # ===== NEW: Get detailed concept analysis =====
        detailed_concept_analysis = self._get_detailed_concept_analysis(
            feature_importance
        )
        
        # Build result
        result = {
            "prediction": self.CLASS_NAMES[pred_class],
            "prediction_label": pred_class,
            "confidence": confidence,
            "probabilities": {
                self.CLASS_NAMES[0]: float(probs[0]),
                self.CLASS_NAMES[1]: float(probs[1])
            },
            "feature_importance": feature_importance,  # All 33 features with scores
            "top_contributing_features": top_features,  # Simple top 10
            
            # ===== NEW FIELDS =====
            "feature_rankings": feature_rankings,  # Top 10 for global, benign, pathogenic
            "detailed_concept_analysis": detailed_concept_analysis,  # Full concept breakdown
            
            "input_metadata": {
                "allele_id": encoder_output.get('allele_id'),
                "gene_id": encoder_output.get('gene_id'),
                "clinical_significance": encoder_output.get('clinical_significance'),
                "validation_issues": encoder_output.get('validation_issues', [])
            }
        }
        
        return result

    
    def classify_batch(self, 
                      batch_encoder_output: Dict[str, Any],
                      original_csv_path: str,
                      output_csv_path: str,
                      include_confidence: bool = True) -> Dict[str, Any]:
        """
        Classify batch of variants and save results to CSV
        
        Args:
            batch_encoder_output: Output from encode_batch_variants() containing:
                - results: List of dicts with encoded_features for each variant
            original_csv_path: Path to original input CSV file
            output_csv_path: Path where to save results CSV
            include_confidence: Whether to include confidence column
        
        Returns:
            Dictionary with:
                - total_variants: int
                - successful_predictions: int
                - failed_predictions: int
                - output_file: str (path to output CSV)
                - summary: Dict with prediction counts
        """
        try:
            # Load original CSV
            original_df = pd.read_csv(original_csv_path)
            logger.info(f"Loaded original CSV with {len(original_df)} rows")
            
            # Get encoded results
            encoded_results = batch_encoder_output.get('results', [])
            if not encoded_results:
                raise ValueError("No results found in batch encoder output")
            
            # Prepare batch predictions
            predictions = []
            confidences = []
            allele_ids = []
            
            for result in encoded_results:
                allele_id = result.get('allele_id')
                encoded_features = result.get('encoded_features')
                
                if encoded_features is None:
                    # Skip failed encodings
                    predictions.append(None)
                    confidences.append(None)
                    allele_ids.append(allele_id)
                    continue
                
                try:
                    # Prepare input
                    input_tensor = self._prepare_input(encoded_features)
                    
                    # Make prediction
                    with torch.inference_mode():
                        preds, _, _ = self.model(input_tensor)
                        probs = torch.exp(preds).cpu().numpy()[0]
                        pred_class = int(np.argmax(probs))
                        confidence = float(probs[pred_class])
                    
                    predictions.append(self.CLASS_NAMES[pred_class])
                    confidences.append(confidence)
                    allele_ids.append(allele_id)
                    
                except Exception as e:
                    logger.error(f"Failed to classify AlleleID {allele_id}: {e}")
                    predictions.append(None)
                    confidences.append(None)
                    allele_ids.append(allele_id)
            
            # Create predictions DataFrame
            pred_df = pd.DataFrame({
                'AlleleID': allele_ids,
                'Prediction': predictions
            })
            
            if include_confidence:
                pred_df['Confidence'] = confidences
            
            # Merge with original CSV on AlleleID
            result_df = original_df.merge(pred_df, on='AlleleID', how='left')
            
            # Save to output file
            result_df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved predictions to {output_csv_path}")
            
            # Calculate summary statistics
            successful = sum(1 for p in predictions if p is not None)
            failed = len(predictions) - successful
            
            prediction_counts = {
                "Benign": predictions.count("Benign"),
                "Pathogenic": predictions.count("Pathogenic"),
                "Failed": failed
            }
            
            return {
                "total_variants": len(predictions),
                "successful_predictions": successful,
                "failed_predictions": failed,
                "output_file": output_csv_path,
                "summary": prediction_counts
            }
            
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            raise
    
    def classify_with_weight_adjustment(self,
                                       encoder_output: Dict[str, Any],
                                       weight_adjustments: Dict[str, float]) -> Dict[str, Any]:
        """
        Re-classify with adjusted feature weights (for Gradio sliders)
        
        Note:
            Weight adjustments are applied MULTIPLICATIVELY to the input features,
            not to the model's learned weights. This simulates "what if this feature
            had a different value" rather than "what if the model weighted this feature differently".
    
            For interpretation: A multiplier >1.0 amplifies the feature's contribution,
            <1.0 diminishes it.
            
        Args:
            encoder_output: Output from encode_single_variant() containing encoded_features
            weight_adjustments: Dict mapping feature names to multipliers
                               e.g., {"has_MC_nonsense": 1.5, "has_Origin_germline": 0.8}
                               Values typically range from 0.1 to 3.0
        
        Returns:
            Dictionary with:
                - original_prediction: str
                - original_confidence: float
                - original_probabilities: Dict[str, float]
                - adjusted_prediction: str
                - adjusted_confidence: float
                - adjusted_probabilities: Dict[str, float]
                - prediction_changed: bool
                - confidence_change: float (adjusted - original)
                - weight_adjustments_applied: Dict[str, float]
                - adjusted_feature_importance: Dict (feature importance after adjustment)
        """
        # Get original encoded features
        original_features = encoder_output.get('encoded_features')
        if original_features is None:
            raise ValueError("No encoded_features found in encoder output")
        
        # Make original prediction first
        original_tensor = self._prepare_input(original_features)
        
        with torch.inference_mode():
            original_preds, original_explanations, _ = self.model(original_tensor)
            original_probs = torch.exp(original_preds).cpu().numpy()[0]
            original_class = int(np.argmax(original_probs))
            original_confidence = float(original_probs[original_class])
        
        # Apply weight adjustments to features
        adjusted_features = original_features.copy()
        applied_adjustments = {}
        
        for feature_name, multiplier in weight_adjustments.items():
            if feature_name not in self.FEATURE_NAMES:
                logger.warning(f"Unknown feature name: {feature_name}, skipping")
                continue
            
            feature_idx = self.FEATURE_NAMES.index(feature_name)
            adjusted_features[feature_idx] = adjusted_features[feature_idx] * multiplier
            applied_adjustments[feature_name] = multiplier
        
        # Make adjusted prediction
        adjusted_tensor = self._prepare_input(adjusted_features)
        
        with torch.inference_mode():
            adjusted_preds, adjusted_explanations, _ = self.model(adjusted_tensor)
            adjusted_probs = torch.exp(adjusted_preds).cpu().numpy()[0]
            adjusted_class = int(np.argmax(adjusted_probs))
            adjusted_confidence = float(adjusted_probs[adjusted_class])
        
        # Calculate adjusted feature importance
        adjusted_concepts, adjusted_relevances = adjusted_explanations
        adjusted_importance = self._calculate_feature_importance(
            adjusted_tensor, adjusted_concepts, adjusted_relevances
        )
        
        # Build comparison result
        result = {
            "original_prediction": self.CLASS_NAMES[original_class],
            "original_confidence": original_confidence,
            "original_probabilities": {
                self.CLASS_NAMES[0]: float(original_probs[0]),
                self.CLASS_NAMES[1]: float(original_probs[1])
            },
            "adjusted_prediction": self.CLASS_NAMES[adjusted_class],
            "adjusted_confidence": adjusted_confidence,
            "adjusted_probabilities": {
                self.CLASS_NAMES[0]: float(adjusted_probs[0]),
                self.CLASS_NAMES[1]: float(adjusted_probs[1])
            },
            "prediction_changed": (original_class != adjusted_class),
            "confidence_change": adjusted_confidence - original_confidence,
            "weight_adjustments_applied": applied_adjustments,
            "adjusted_feature_importance": adjusted_importance,
            "input_metadata": {
                "allele_id": encoder_output.get('allele_id'),
                "gene_id": encoder_output.get('gene_id'),
                "clinical_significance": encoder_output.get('clinical_significance')
            }
        }
        
        return result
    
    def _calculate_feature_importance(self,
                                     input_tensor: torch.Tensor,
                                     concepts: torch.Tensor,
                                     relevances: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature importance for specific input
        
        Returns:
            Dict mapping feature names to importance scores:
            {
                "feature_name": {
                    "global": float,
                    "Benign": float,
                    "Pathogenic": float
                }
            }
        """
        # relevances shape: [batch=1, num_concepts=33, num_classes=2]
        relevances_np = relevances.cpu().numpy()[0]  # [33, 2]
        
        importance_dict = {}
        for i, feature_name in enumerate(self.FEATURE_NAMES):
            benign_score = float(relevances_np[i, 0])
            pathogenic_score = float(relevances_np[i, 1])
            global_score = float(np.mean(np.abs(relevances_np[i, :])))
            
            importance_dict[feature_name] = {
                "global": global_score,
                "Benign": benign_score,
                "Pathogenic": pathogenic_score
            }
        
        return importance_dict
    
    def _get_top_features(self,
                         feature_importance: Dict[str, Dict[str, float]],
                         top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top K most important features
        
        Returns:
            List of dicts with feature info, sorted by global importance
        """
        # Sort by global importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]["global"]),
            reverse=True
        )
        
        top_features = []
        for feature_name, scores in sorted_features[:top_k]:
            # Determine which class this feature favors
            favored_class = "Pathogenic" if scores["Pathogenic"] > scores["Benign"] else "Benign"
            
            top_features.append({
                "feature": feature_name,
                "global_importance": scores["global"],
                "benign_weight": scores["Benign"],
                "pathogenic_weight": scores["Pathogenic"],
                "favored_class": favored_class
            })
        
        return top_features
    
    def _group_into_concepts(self,
                            feature_importance: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Group features into high-level concepts
        
        Returns:
            Dict with concept-level importance and detailed contributions
        """
        concept_groups = {
            'molecular_consequence': ['has_MC_'],
            'origin': ['has_Origin_'],
            'gene_relation': ['has_VariantGeneRelation_'],
            'genomic_location': ['is_genomic', 'is_mitochondrial']
        }
        
        concept_importance = {}
        detailed_contributions = {}
        
        for concept_name, prefixes in concept_groups.items():
            concept_features = {}
            total_importance = 0
            
            for feature_name, scores in feature_importance.items():
                # Check if feature belongs to this concept
                is_match = False
                if isinstance(prefixes, list):
                    for prefix in prefixes:
                        if feature_name.startswith(prefix) or feature_name == prefix:
                            is_match = True
                            break
                
                if is_match:
                    concept_features[feature_name] = scores
                    total_importance += abs(scores["global"])
            
            concept_importance[concept_name] = total_importance
            detailed_contributions[concept_name] = concept_features
        
        return {
            "concept_scores": concept_importance,
            "detailed_contributions": detailed_contributions
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.FEATURE_NAMES.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info"""
        return {
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "class_names": self.CLASS_NAMES,
            "device": str(self.device),
            "model_path": self.model_path
        }


# ==================== CONVENIENCE FUNCTIONS ====================

def create_classifier(model_path: Optional[str] = None) -> SENNClassifier:
    """
    Factory function to create SENN classifier instance
    
    Args:
        model_path: Path to model weights (optional, uses env var if None)
    
    Returns:
        SENNClassifier instance
    """
    return SENNClassifier(model_path=model_path)


# ==================== EXAMPLE USAGE ====================
def print_detailed_results(result: Dict[str, Any]):
    """
    Pretty print classification results in the original training format
    (Add this as a standalone function at the bottom of the file)
    """
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: Benign={result['probabilities']['Benign']:.4f}, "
          f"Pathogenic={result['probabilities']['Pathogenic']:.4f}")
    
    # Top 10 Most Important Features (Global)
    print("\n" + "="*80)
    print("Top 10 Most Important Features (Global):")
    print("="*80)
    for item in result['feature_rankings']['global_ranking']:
        print(f"{item['rank']:2d}. {item['feature']:<45} | global: {item['score']:>8.4f}")
    
    # Top 10 Most Important Features (Benign)
    print("\nTop 10 Most Important Features (Benign):")
    print("="*80)
    for item in result['feature_rankings']['benign_ranking']:
        print(f"{item['rank']:2d}. {item['feature']:<45} | benign: {item['score']:>8.4f}")
    
    # Top 10 Most Important Features (Pathogenic)
    print("\nTop 10 Most Important Features (Pathogenic):")
    print("="*80)
    for item in result['feature_rankings']['pathogenic_ranking']:
        print(f"{item['rank']:2d}. {item['feature']:<45} | pathogenic: {item['score']:>8.4f}")
    
    # Concept-Level Analysis
    print("\n" + "="*80)
    print("CONCEPT-LEVEL IMPORTANCE ANALYSIS")
    print("="*80)
    
    print("\nConcept-Level Importance (Global):")
    for concept, score in result['detailed_concept_analysis']['concept_scores'].items():
        print(f"{concept:<25}: {score:>8.4f}")
    
    print("\nDetailed Feature Contributions by Concept:")
    for concept_name, features in result['detailed_concept_analysis']['detailed_contributions'].items():
        if features:
            print(f"\n{concept_name.upper().replace('_', ' ')}:")
            for feat in features[:5]:  # Show top 5 per concept
                print(f"  {feat['feature']:<45}: "
                      f"global={feat['global']:>7.3f}, "
                      f"benign={feat['benign']:>7.3f}, "
                      f"pathogenic={feat['pathogenic']:>7.3f}")

if __name__ == "__main__":
    # Example usage
    from input_pipeline import encode_single_variant,encode_batch_variants
    
    # Initialize classifier
    classifier = create_classifier()
    
    print("="*80)
    print("SENN CLASSIFIER - COMPREHENSIVE FEATURE IMPORTANCE")
    print("="*80)
    
    # Single classification
    encoder_output = encode_single_variant({
        'AlleleID': 15040,
        'GeneID': 55572,
        'Origin': 'germline',
        'VariantGeneRelation': 'within single gene',
        'MC': 'nonsense,non-coding_transcript_variant',
        'GenomicLocationData': 'g'
    })
    
    result = classifier.classify_single(encoder_output)
    
    # Use the pretty-print function
    print_detailed_results(result)
    
    # EXAMPLE 2: Weight adjustment
    print("\n=== WEIGHT ADJUSTMENT ===")
    weight_adjustments = {
        "has_MC_nonsense": 2.0,  # Double the importance
        "has_Origin_germline": 0.5  # Halve the importance
    }
    
    adjusted_result = classifier.classify_with_weight_adjustment(
        encoder_output, weight_adjustments
    )
    
    print(f"Original: {adjusted_result['original_prediction']} ({adjusted_result['original_confidence']:.3f})")
    print(f"Adjusted: {adjusted_result['adjusted_prediction']} ({adjusted_result['adjusted_confidence']:.3f})")
    print(f"Prediction changed: {adjusted_result['prediction_changed']}")
    print(f"Confidence change: {adjusted_result['confidence_change']:+.3f}")
    
    # EXAMPLE 3: Batch classification
    print("\n=== BATCH CLASSIFICATION ===")
    batch_output = encode_batch_variants(r"C:\Users\vigne\Desktop\Capstone-Project-allele-Pathogenicty-classification-with-contextual-insights\batch_encoder_test.csv")
    
    batch_result = classifier.classify_batch(
        batch_encoder_output=batch_output,
        original_csv_path=r"C:\Users\vigne\Desktop\Capstone-Project-allele-Pathogenicty-classification-with-contextual-insights\batch_encoder_test.csv",
        output_csv_path="batch_test_predictions.csv",
        include_confidence=True
    )
    
    print(f"Total variants: {batch_result['total_variants']}")
    print(f"Successful: {batch_result['successful_predictions']}")
    print(f"Failed: {batch_result['failed_predictions']}")
    print(f"Output saved to: {batch_result['output_file']}")
    print(f"Predictions summary: {batch_result['summary']}")
    
    print("\n" + "="*80)
