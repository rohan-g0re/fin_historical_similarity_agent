"""
Similarity Calculator for Pattern Matching

Implements high-performance similarity calculations between 7-day market windows using:
- Cosine similarity for feature vector comparison
- Batch similarity calculations for efficiency
- Similarity ranking and scoring
- Robust handling of edge cases and normalization

Optimized for finding similar historical market patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings

from ..core.config_manager import ConfigManager


class SimilarityCalculator:
    """
    High-performance similarity calculator for market pattern matching.
    
    Uses cosine similarity to compare 7-day feature vectors representing
    market conditions, with optimizations for:
    - Batch calculations for efficiency
    - Proper normalization handling
    - Similarity ranking and filtering
    - Edge case robustness
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the similarity calculator.
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance.
                                                     Creates new one if None.
        """
        self.config = config_manager if config_manager else ConfigManager()
        self.similarity_config = self.config.config.get('similarity', {})
        
        # Similarity parameters
        self.method = self.similarity_config.get('method', 'cosine')
        self.min_gap_days = self.similarity_config.get('min_gap_days', 30)
        self.similarity_threshold = self.similarity_config.get('similarity_threshold', 0.65)
        self.max_results = self.similarity_config.get('max_results', 50)
        
        # Normalization settings
        self._scaler = StandardScaler()
        self._is_fitted = False
    
    def calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two feature vectors.
        
        Args:
            vector1 (np.ndarray): First feature vector
            vector2 (np.ndarray): Second feature vector
            
        Returns:
            float: Cosine similarity score (0 to 1, where 1 is identical)
            
        Raises:
            ValueError: If vectors have different dimensions or are invalid
        """
        # Validate inputs
        if len(vector1) != len(vector2):
            raise ValueError(f"Vector dimension mismatch: {len(vector1)} vs {len(vector2)}")
        
        if len(vector1) == 0:
            raise ValueError("Empty vectors provided")
        
        # Ensure vectors are numpy arrays
        v1 = np.array(vector1).flatten()
        v2 = np.array(vector2).flatten()
        
        # Handle edge cases
        v1 = np.nan_to_num(v1, nan=0.0, posinf=1e6, neginf=-1e6)
        v2 = np.nan_to_num(v2, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Calculate cosine similarity
        try:
            # Reshape for sklearn
            v1_2d = v1.reshape(1, -1)
            v2_2d = v2.reshape(1, -1)
            
            similarity = cosine_similarity(v1_2d, v2_2d)[0, 0]
            
            # Ensure result is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            # Fallback: manual cosine similarity calculation
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, float(similarity)))
    
    def calculate_batch_similarity(self, target_vector: np.ndarray, 
                                 comparison_vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between target vector and multiple comparison vectors.
        
        Args:
            target_vector (np.ndarray): Target feature vector
            comparison_vectors (np.ndarray): Array of comparison vectors (2D)
            
        Returns:
            np.ndarray: Array of similarity scores
        """
        if comparison_vectors.shape[0] == 0:
            return np.array([])
        
        # Ensure target is 2D
        if len(target_vector.shape) == 1:
            target_2d = target_vector.reshape(1, -1)
        else:
            target_2d = target_vector
        
        # Ensure comparison vectors are 2D
        if len(comparison_vectors.shape) == 1:
            comparison_2d = comparison_vectors.reshape(1, -1)
        else:
            comparison_2d = comparison_vectors
        
        # Validate dimensions
        if target_2d.shape[1] != comparison_2d.shape[1]:
            raise ValueError(f"Dimension mismatch: target {target_2d.shape[1]} vs comparison {comparison_2d.shape[1]}")
        
        try:
            # Calculate batch similarities using sklearn
            similarities = cosine_similarity(target_2d, comparison_2d)[0]
            
            # Ensure all values are between 0 and 1
            similarities = np.clip(similarities, 0.0, 1.0)
            
            return similarities
            
        except Exception as e:
            # Fallback: calculate similarities one by one
            similarities = []
            target_flat = target_vector.flatten()
            
            for i in range(comparison_2d.shape[0]):
                comp_flat = comparison_2d[i].flatten()
                sim = self.calculate_cosine_similarity(target_flat, comp_flat)
                similarities.append(sim)
            
            return np.array(similarities)
    
    def rank_similarities(self, similarities: np.ndarray, 
                         metadata: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Rank similarity results and combine with metadata.
        
        Args:
            similarities (np.ndarray): Array of similarity scores
            metadata (List[Dict]): Metadata for each comparison (same order as similarities)
            
        Returns:
            List[Dict]: Ranked results with similarity scores and metadata
        """
        if len(similarities) != len(metadata):
            raise ValueError(f"Length mismatch: {len(similarities)} similarities vs {len(metadata)} metadata")
        
        # Combine similarities with metadata
        results = []
        for i, (similarity, meta) in enumerate(zip(similarities, metadata)):
            result = {
                'similarity_score': float(similarity),
                'rank': None,  # Will be set after sorting
                'index': i,
                **meta  # Include all metadata
            }
            results.append(result)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Add rank information
        for rank, result in enumerate(results, 1):
            result['rank'] = rank
        
        # Apply similarity threshold filter
        filtered_results = [r for r in results if r['similarity_score'] >= self.similarity_threshold]
        
        # Limit to max results
        limited_results = filtered_results[:self.max_results]
        
        return limited_results
    
    def find_similar_patterns(self, target_window: Dict[str, any], 
                            historical_windows: List[Dict[str, any]],
                            apply_gap_filter: bool = True) -> List[Dict[str, any]]:
        """
        Find historical windows similar to target window.
        
        Args:
            target_window (Dict): Target window with feature_vector
            historical_windows (List[Dict]): List of historical windows to compare
            apply_gap_filter (bool): Whether to apply minimum gap filtering
            
        Returns:
            List[Dict]: Ranked similar patterns with similarity scores
        """
        if not historical_windows:
            return []
        
        target_vector = target_window.get('feature_vector')
        if target_vector is None:
            raise ValueError("Target window missing feature_vector")
        
        # Filter by gap if requested
        valid_windows = historical_windows
        if apply_gap_filter:
            target_end_idx = target_window.get('window_end_idx')
            if target_end_idx is not None:
                valid_windows = []
                for window in historical_windows:
                    window_end_idx = window.get('window_end_idx')
                    if window_end_idx is not None:
                        gap = abs(target_end_idx - window_end_idx)
                        if gap >= self.min_gap_days:
                            valid_windows.append(window)
        
        if not valid_windows:
            print(f"⚠ No windows meet gap requirement of {self.min_gap_days} days")
            return []
        
        print(f"📊 Comparing target window against {len(valid_windows)} historical windows...")
        
        # Extract feature vectors and metadata
        comparison_vectors = []
        metadata = []
        
        for window in valid_windows:
            vector = window.get('feature_vector')
            if vector is not None and len(vector) > 0:
                comparison_vectors.append(vector)
                
                # Create metadata for this window
                meta = {
                    'window_start_date': window.get('window_start_date'),
                    'window_end_date': window.get('window_end_date'),
                    'window_start_idx': window.get('window_start_idx'),
                    'window_end_idx': window.get('window_end_idx'),
                    'vector_length': len(vector)
                }
                metadata.append(meta)
        
        if not comparison_vectors:
            print("⚠ No valid feature vectors found in historical windows")
            return []
        
        # Convert to numpy array
        comparison_matrix = np.array(comparison_vectors)
        
        # Calculate similarities
        similarities = self.calculate_batch_similarity(target_vector, comparison_matrix)
        
        # Rank and filter results
        ranked_results = self.rank_similarities(similarities, metadata)
        
        print(f"✓ Found {len(ranked_results)} similar patterns above threshold {self.similarity_threshold}")
        
        return ranked_results
    
    def get_similarity_statistics(self, similarities: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a set of similarity scores.
        
        Args:
            similarities (List[float]): List of similarity scores
            
        Returns:
            Dict[str, float]: Statistical summary
        """
        if not similarities:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        similarities_array = np.array(similarities)
        
        return {
            'count': len(similarities),
            'mean': float(np.mean(similarities_array)),
            'std': float(np.std(similarities_array)),
            'min': float(np.min(similarities_array)),
            'max': float(np.max(similarities_array)),
            'median': float(np.median(similarities_array)),
            'q25': float(np.percentile(similarities_array, 25)),
            'q75': float(np.percentile(similarities_array, 75))
        }
    
    def get_similarity_level(self, similarity: float) -> str:
        """Get descriptive similarity level."""
        if similarity >= 0.90:
            return "Very High"
        elif similarity >= 0.80:
            return "High"
        elif similarity >= 0.70:
            return "Medium-High"
        elif similarity >= 0.60:
            return "Medium"
        elif similarity >= 0.50:
            return "Low-Medium"
        else:
            return "Low" 