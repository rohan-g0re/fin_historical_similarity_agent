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
        Rank similarity results and combine with metadata for comprehensive analysis.
        
        This method transforms raw similarity scores into a structured, ranked list
        suitable for business analysis and decision-making. The ranking process
        combines mathematical similarity with contextual metadata to provide
        actionable insights.
        
        **Ranking Process:**
        1. **Score Integration**: Combine similarity scores with metadata
        2. **Descending Sort**: Order by similarity score (highest first)
        3. **Rank Assignment**: Add sequential rank numbers for easy reference
        4. **Threshold Filtering**: Remove patterns below minimum similarity
        5. **Result Limiting**: Cap results to prevent information overload
        
        **Business Value:**
        - Provides clear hierarchy of pattern similarity
        - Enables focus on most relevant historical precedents
        - Supports evidence-based investment decisions
        - Facilitates communication of findings to stakeholders
        
        Args:
            similarities (np.ndarray): Array of similarity scores (0-1 range)
            metadata (List[Dict]): Metadata for each comparison in same order
                                  Contains dates, indices, and other contextual info
            
        Returns:
            List[Dict[str, any]]: Ranked results with similarity scores and metadata
                                 Each entry contains rank, score, and temporal context
            
        Raises:
            ValueError: If similarities and metadata lengths don't match
        """
        # Validate input alignment - critical for accurate results
        if len(similarities) != len(metadata):
            raise ValueError(f"Length mismatch: {len(similarities)} similarities vs {len(metadata)} metadata")
        
        # Combine similarities with metadata for unified processing
        results = []
        for i, (similarity, meta) in enumerate(zip(similarities, metadata)):
            result = {
                'similarity_score': float(similarity),  # Ensure consistent data type
                'rank': None,  # Will be set after sorting
                'index': i,    # Original index for tracking
                **meta  # Include all metadata (dates, indices, quality scores, etc.)
            }
            results.append(result)
        
        # Sort by similarity score in descending order (highest similarity first)
        # This prioritizes the most similar patterns for analysis
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Add rank information for easy reference and communication
        # Rank 1 = most similar, Rank 2 = second most similar, etc.
        for rank, result in enumerate(results, 1):
            result['rank'] = rank
        
        # Apply similarity threshold filter to ensure quality
        # Only return patterns that meet minimum similarity criteria
        # This prevents weak patterns from diluting analysis quality
        filtered_results = [r for r in results if r['similarity_score'] >= self.similarity_threshold]
        
        # Limit to maximum results to prevent information overload
        # Focus on top patterns rather than overwhelming with marginal matches
        limited_results = filtered_results[:self.max_results]
        
        return limited_results
    
    def find_similar_patterns(self, target_window: Dict[str, any], 
                            historical_windows: List[Dict[str, any]],
                            apply_gap_filter: bool = True) -> List[Dict[str, any]]:
        """
        Find historical windows most similar to target window using comprehensive matching.
        
        This is the primary method for pattern discovery that orchestrates the complete
        similarity search workflow. It handles data validation, gap filtering, batch
        similarity calculation, and result ranking to identify the most relevant
        historical precedents for current market conditions.
        
        **Complete Search Workflow:**
        1. **Input Validation**: Ensure target window has valid feature vector
        2. **Gap Filtering**: Remove overlapping windows for statistical independence
        3. **Vector Extraction**: Prepare historical vectors for batch comparison
        4. **Similarity Calculation**: Compute cosine similarity against all candidates
        5. **Result Ranking**: Sort and filter based on similarity thresholds
        6. **Quality Control**: Return only high-confidence pattern matches
        
        **Gap Filtering Logic:**
        Prevents contamination from overlapping time periods that share market data.
        For example, if target window ends on 2023-12-15, exclude historical windows
        ending between 2023-12-08 and 2023-12-22 (assuming 7-day minimum gap).
        This ensures statistical independence between compared patterns.
        
        Args:
            target_window (Dict[str, any]): Target window with feature_vector and metadata
            historical_windows (List[Dict[str, any]]): Historical windows to search
            apply_gap_filter (bool): Whether to enforce minimum gap for independence
            
        Returns:
            List[Dict[str, any]]: Ranked similar patterns with scores and metadata
                                 Empty list if no patterns meet similarity criteria
            
        Raises:
            ValueError: If target window lacks required feature vector
        """
        # Early exit for empty historical data
        if not historical_windows:
            return []
        
        # Validate target window has required feature vector
        target_vector = target_window.get('feature_vector')
        if target_vector is None:
            raise ValueError("Target window missing feature_vector")
        
        # Apply gap filtering to ensure statistical independence
        valid_windows = historical_windows
        if apply_gap_filter:
            # Get target window end index for gap calculation
            target_end_idx = target_window.get('window_end_index') or target_window.get('window_end_idx')
            
            if target_end_idx is not None:
                valid_windows = []
                for window in historical_windows:
                    # Check multiple possible keys for window end index
                    window_end_idx = window.get('window_end_index') or window.get('window_end_idx')
                    
                    if window_end_idx is not None:
                        # Calculate temporal gap between windows
                        gap = abs(target_end_idx - window_end_idx)
                        
                        # Only include windows with sufficient gap for independence
                        if gap >= self.min_gap_days:
                            valid_windows.append(window)
        
        # Check if gap filtering eliminated all candidates
        if not valid_windows:
            print(f"⚠ No windows meet gap requirement of {self.min_gap_days} days")
            return []
        
        print(f"📊 Comparing target window against {len(valid_windows)} historical windows...")
        
        # Extract feature vectors and metadata for batch processing
        comparison_vectors = []
        metadata = []
        
        for window in valid_windows:
            vector = window.get('feature_vector')
            
            # Only process windows with valid feature vectors
            if vector is not None and len(vector) > 0:
                comparison_vectors.append(vector)
                
                # Create comprehensive metadata for result interpretation
                meta = {
                    'window_start_date': window.get('window_start_date'),
                    'window_end_date': window.get('window_end_date'),
                    'window_start_idx': window.get('window_start_idx'),
                    'window_end_idx': window.get('window_end_idx'),
                    'window_end_index': window.get('window_end_index'),  # Alternative key
                    'vector_length': len(vector),
                    'data_quality_score': window.get('data_quality_score', 1.0)
                }
                metadata.append(meta)
        
        # Validate we have vectors to compare against
        if not comparison_vectors:
            print("⚠ No valid feature vectors found in historical windows")
            return []
        
        # Convert to numpy array for efficient batch processing
        comparison_matrix = np.array(comparison_vectors)
        
        # Perform batch similarity calculation for performance
        similarities = self.calculate_batch_similarity(target_vector, comparison_matrix)
        
        # Rank and filter results based on similarity scores
        ranked_results = self.rank_similarities(similarities, metadata)
        
        print(f"✓ Found {len(ranked_results)} similar patterns above threshold {self.similarity_threshold}")
        
        return ranked_results
    
    def get_similarity_statistics(self, similarities: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for similarity score analysis.
        
        Provides detailed statistical analysis of similarity scores to support
        decision-making and result interpretation. These statistics help assess
        the quality and confidence of pattern matching results.
        
        **Statistical Measures:**
        - **Central Tendency**: Mean and median for typical similarity levels
        - **Variability**: Standard deviation for score consistency assessment
        - **Range**: Min/max for understanding score distribution
        - **Percentiles**: Quartiles for detailed distribution analysis
        
        **Business Applications:**
        - Quality assessment: High mean/median suggests good pattern matching
        - Confidence evaluation: Low std deviation suggests consistent results
        - Outlier detection: Large range may indicate mixed pattern quality
        - Threshold optimization: Percentiles help set appropriate cutoffs
        
        Args:
            similarities (List[float]): List of similarity scores (0-1 range)
            
        Returns:
            Dict[str, float]: Comprehensive statistical summary with all key measures
        """
        # Handle empty input gracefully
        if not similarities:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'q25': 0.0,
                'q75': 0.0
            }
        
        # Convert to numpy for efficient statistical calculations
        similarities_array = np.array(similarities)
        
        return {
            # Basic counts and central tendency
            'count': len(similarities),                                   # Number of patterns analyzed
            'mean': float(np.mean(similarities_array)),                  # Average similarity level
            'median': float(np.median(similarities_array)),              # Middle similarity value
            
            # Variability measures
            'std': float(np.std(similarities_array)),                    # Score consistency indicator
            'min': float(np.min(similarities_array)),                    # Lowest similarity found
            'max': float(np.max(similarities_array)),                    # Highest similarity found
            
            # Distribution analysis
            'q25': float(np.percentile(similarities_array, 25)),         # 25th percentile (lower quartile)
            'q75': float(np.percentile(similarities_array, 75))          # 75th percentile (upper quartile)
        }
    
    def get_similarity_level(self, similarity: float) -> str:
        """
        Convert numerical similarity score to business-friendly descriptive level.
        
        Transforms mathematical similarity values into intuitive categories that
        business users can easily understand and act upon. These categories help
        communicate confidence levels and support decision-making processes.
        
        **Similarity Level Mapping:**
        - Very High (≥90%): Exceptional similarity, high confidence patterns
        - High (80-89%): Strong similarity, reliable patterns for analysis
        - Medium-High (70-79%): Good similarity, useful for context
        - Medium (60-69%): Moderate similarity, use with caution
        - Low-Medium (50-59%): Weak similarity, limited relevance
        - Low (<50%): Poor similarity, likely not meaningful
        
        **Business Usage:**
        - Investment decisions: Focus on "High" and "Very High" patterns
        - Risk assessment: Consider "Medium-High" for additional context
        - Research analysis: Include "Medium" patterns for comprehensive study
        - Communication: Use descriptive levels in reports and presentations
        
        Args:
            similarity (float): Numerical similarity score (0-1 range)
            
        Returns:
            str: Business-friendly descriptive similarity level
        """
        # Define similarity thresholds based on practical experience
        # These thresholds balance sensitivity with specificity for business use
        
        if similarity >= 0.90:
            return "Very High"      # Exceptional patterns, act with confidence
        elif similarity >= 0.80:
            return "High"           # Strong patterns, reliable for decisions
        elif similarity >= 0.70:
            return "Medium-High"    # Good patterns, useful for context
        elif similarity >= 0.60:
            return "Medium"         # Moderate patterns, use with caution
        elif similarity >= 0.50:
            return "Low-Medium"     # Weak patterns, limited value
        else:
            return "Low"            # Poor patterns, likely not meaningful 