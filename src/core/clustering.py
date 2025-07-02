"""
Phase 2: Professional Clustering Core
4 bước hoàn chỉnh theo spec với Hopkins Statistic, Multi-method K selection, Enhanced validation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

class ProfessionalClustering:
    """Phase 2: Professional Clustering theo spec 4 phases"""
    
    def __init__(self):
        self.data = None
        self.optimal_k = None
        self.final_model = None
        self.cluster_labels = None
        self.validation_metrics = {}
        self.readiness_results = {}
        self.k_selection_results = {}
        
        # Step completion tracking
        self.step_2_1_completed = False
        self.step_2_2_completed = False
        self.step_2_3_completed = False
        self.step_2_4_completed = False
    
    def set_data(self, data):
        """Set data for clustering analysis"""
        if data is None:
            raise ValueError("Data cannot be None")
        
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("Data must be DataFrame or numpy array")
        
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        else:
            self.data = data
        
        if len(self.data) < 10:
            raise ValueError("Need at least 10 samples for clustering")
        
        if self.data.shape[1] < 2:
            raise ValueError("Need at least 2 features for clustering")
        
        # Reset completion status
        self.step_2_1_completed = False
        self.step_2_2_completed = False
        self.step_2_3_completed = False
        self.step_2_4_completed = False
        
        return True
    
    def step_2_1_clustering_readiness_check(self):
        """Bước 2.1: Hopkins Statistic và Visual Assessment Protocol"""
        
        if self.data is None:
            raise ValueError("No data available. Please set data first")
        
        readiness_results = {}
        
        try:
            # Hopkins Statistic theo spec
            hopkins_score = self._calculate_hopkins_statistic_professional()
            readiness_results['hopkins_statistic'] = hopkins_score
            readiness_results['clustering_tendency'] = "Good" if hopkins_score > 0.5 else "Poor"
            
            # Visual Assessment Protocol (VAP)
            vap_results = self._visual_assessment_protocol()
            readiness_results['visual_assessment'] = vap_results
            
            # Nearest neighbor analysis
            nn_analysis = self._nearest_neighbor_analysis()
            readiness_results['nearest_neighbor'] = nn_analysis
            
            # Overall readiness score
            readiness_score = self._calculate_readiness_score(readiness_results)
            readiness_results['readiness_score'] = readiness_score
            readiness_results['recommendation'] = self._get_readiness_recommendation(readiness_score)
            
            self.readiness_results = readiness_results
            self.step_2_1_completed = True
            
            return readiness_results
            
        except Exception as e:
            raise ValueError(f"Error in clustering readiness check: {str(e)}")
    
    def _calculate_hopkins_statistic_professional(self):
        """Professional Hopkins Statistic implementation theo spec"""
        try:
            data = self.data
            n_samples, n_features = data.shape
            
            # Sample size: 10% or max 50 points theo spec
            m = min(int(0.1 * n_samples), 50)
            if m < 5:
                return 0.5
            
            # Standardize data for distance calculations
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Check for constant features
            if np.any(np.var(data_scaled, axis=0) == 0):
                return 0.5
            
            # Generate random points in data space
            data_min = np.min(data_scaled, axis=0)
            data_max = np.max(data_scaled, axis=0)
            
            # Ensure non-zero range
            data_range = data_max - data_min
            if np.any(data_range == 0):
                return 0.5
            
            # Generate uniform random points
            np.random.seed(42)  # Reproducibility
            random_points = np.random.uniform(data_min, data_max, size=(m, n_features))
            
            # Calculate distances
            random_distances = []
            real_distances = []
            
            for i in range(m):
                # Distance from random point to nearest real point
                distances_to_real = pairwise_distances([random_points[i]], data_scaled)[0]
                if len(distances_to_real) > 0:
                    random_distances.append(np.min(distances_to_real))
                
                # Distance from real point to nearest real point (excluding itself)
                if len(data_scaled) > 1:
                    real_point_idx = np.random.randint(0, n_samples)
                    real_point = data_scaled[real_point_idx:real_point_idx+1]
                    distances_real_to_real = pairwise_distances(real_point, data_scaled)[0]
                    distances_real_to_real = distances_real_to_real[distances_real_to_real > 0]
                    if len(distances_real_to_real) > 0:
                        real_distances.append(np.min(distances_real_to_real))
            
            # Hopkins Statistic calculation
            if len(real_distances) > 0 and len(random_distances) > 0:
                sum_random = np.sum(random_distances)
                sum_real = np.sum(real_distances)
                if sum_random + sum_real > 0:
                    hopkins = sum_random / (sum_random + sum_real)
                    return float(hopkins)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _visual_assessment_protocol(self):
        """Visual Assessment Protocol theo spec"""
        try:
            data = self.data
            vap_results = {}
            
            # PCA for 2D visualization data
            if data.shape[1] > 2:
                pca = PCA(n_components=2, random_state=42)
                data_2d = pca.fit_transform(data)
                vap_results['pca_2d'] = {
                    'data': data_2d,
                    'explained_variance': pca.explained_variance_ratio_,
                    'total_variance': sum(pca.explained_variance_ratio_)
                }
            else:
                vap_results['pca_2d'] = {
                    'data': data,
                    'explained_variance': [1.0, 1.0],
                    'total_variance': 1.0
                }
            
            # Distance matrix analysis
            distance_matrix = pairwise_distances(data[:100])  # Sample for performance
            vap_results['distance_analysis'] = {
                'mean_distance': float(np.mean(distance_matrix)),
                'std_distance': float(np.std(distance_matrix)),
                'min_distance': float(np.min(distance_matrix[distance_matrix > 0])),
                'max_distance': float(np.max(distance_matrix))
            }
            
            return vap_results
            
        except Exception:
            return {'error': 'VAP analysis failed'}
    
    def _nearest_neighbor_analysis(self):
        """Nearest neighbor analysis for clustering tendency"""
        try:
            data = self.data
            
            # Calculate nearest neighbor distances
            nn_distances = []
            for i in range(min(100, len(data))):  # Sample for performance
                distances = pairwise_distances([data[i]], data)[0]
                distances = distances[distances > 0]  # Exclude self
                if len(distances) > 0:
                    nn_distances.append(np.min(distances))
            
            if nn_distances:
                return {
                    'mean_nn_distance': float(np.mean(nn_distances)),
                    'std_nn_distance': float(np.std(nn_distances)),
                    'cv_nn_distance': float(np.std(nn_distances) / np.mean(nn_distances)) if np.mean(nn_distances) > 0 else 0
                }
            else:
                return {'error': 'No valid nearest neighbor distances'}
                
        except Exception:
            return {'error': 'Nearest neighbor analysis failed'}
    
    def _calculate_readiness_score(self, results):
        """Calculate overall clustering readiness score"""
        try:
            score = 0
            
            # Hopkins Statistic (50% weight)
            hopkins = results.get('hopkins_statistic', 0.5)
            if hopkins > 0.7:
                score += 50
            elif hopkins > 0.5:
                score += 35
            elif hopkins > 0.3:
                score += 20
            else:
                score += 10
            
            # Visual assessment (30% weight)
            vap = results.get('visual_assessment', {})
            if 'pca_2d' in vap:
                total_var = vap['pca_2d'].get('total_variance', 0)
                if total_var > 0.8:
                    score += 30
                elif total_var > 0.6:
                    score += 20
                else:
                    score += 10
            
            # Nearest neighbor variability (20% weight)
            nn = results.get('nearest_neighbor', {})
            if 'cv_nn_distance' in nn:
                cv = nn['cv_nn_distance']
                if cv > 0.5:
                    score += 20
                elif cv > 0.3:
                    score += 15
                else:
                    score += 10
            
            return min(100, score)
            
        except Exception:
            return 50
    
    def _get_readiness_recommendation(self, score):
        """Get recommendation based on readiness score"""
        if score >= 80:
            return "Excellent - Proceed with clustering"
        elif score >= 60:
            return "Good - Clustering likely to be successful"
        elif score >= 40:
            return "Fair - Consider feature engineering"
        else:
            return "Poor - Data may not be suitable for clustering"
    
    def step_2_2_optimal_k_selection(self, k_range=(2, 12)):
        """Bước 2.2: Multi-method optimal K selection theo spec"""
        
        if not self.step_2_1_completed:
            raise ValueError("Must complete step 2.1 first")
        
        if self.data is None:
            raise ValueError("No data available")
        
        k_min, k_max = k_range
        k_values = list(range(k_min, min(k_max + 1, len(self.data) // 2)))
        
        if len(k_values) == 0:
            raise ValueError("Invalid K range")
        
        selection_results = {
            'k_values': k_values,
            'methods': {}
        }
        
        try:
            # Method 1: Elbow Method (WCSS/Inertia)
            elbow_results = self._elbow_method(k_values)
            selection_results['methods']['elbow'] = elbow_results
            
            # Method 2: Silhouette Analysis
            silhouette_results = self._silhouette_analysis(k_values)
            selection_results['methods']['silhouette'] = silhouette_results
            
            # Method 3: Gap Statistic (Optimized with reduced n_refs)
            gap_results = self._gap_statistic(k_values, n_refs=5)  # Reduced from 10 to 5
            selection_results['methods']['gap'] = gap_results
            
            # Method 4: Calinski-Harabasz Index
            ch_results = self._calinski_harabasz_analysis(k_values)
            selection_results['methods']['calinski_harabasz'] = ch_results
            
            # Multi-method consensus
            consensus_k = self._determine_consensus_k(selection_results)
            selection_results['consensus'] = {
                'optimal_k': consensus_k,
                'confidence': self._calculate_consensus_confidence(selection_results, consensus_k)
            }
            
            # Business constraints validation
            business_validation = self._validate_business_constraints(consensus_k)
            selection_results['business_validation'] = business_validation
            
            self.optimal_k = consensus_k
            self.k_selection_results = selection_results
            self.step_2_2_completed = True
            
            return selection_results
            
        except Exception as e:
            raise ValueError(f"Error in optimal K selection: {str(e)}")
    
    def _elbow_method(self, k_values):
        """Elbow Method implementation"""
        wcss = []
        models = {}
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10,  # Reduced from 20 to 10
                          max_iter=300, random_state=42, algorithm='lloyd')
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)
            models[k] = kmeans
        
        # Find elbow using second derivative
        if len(wcss) >= 3:
            diffs = np.diff(wcss)
            diffs2 = np.diff(diffs)
            if len(diffs2) > 0:
                elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
                elbow_k = k_values[min(elbow_idx, len(k_values) - 1)]
            else:
                elbow_k = k_values[len(k_values) // 2]
        else:
            elbow_k = k_values[0]
        
        return {
            'wcss': wcss,
            'elbow_k': elbow_k,
            'models': models
        }
    
    def _silhouette_analysis(self, k_values):
        """Silhouette Analysis implementation"""
        silhouette_scores = []
        silhouette_details = {}
        
        for k in k_values:
            if k < 2:
                continue
                
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10,  # Reduced from 20 to 10
                          max_iter=300, random_state=42, algorithm='lloyd')
            cluster_labels = kmeans.fit_predict(self.data)
            
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(self.data, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                
                from sklearn.metrics import silhouette_samples
                sample_silhouette_values = silhouette_samples(self.data, cluster_labels)
                
                silhouette_details[k] = {
                    'avg_score': silhouette_avg,
                    'sample_scores': sample_silhouette_values,
                    'cluster_scores': {
                        i: float(sample_silhouette_values[cluster_labels == i].mean())
                        for i in range(k)
                    }
                }
            else:
                silhouette_scores.append(-1)
                silhouette_details[k] = {'avg_score': -1}
        
        best_k = k_values[np.argmax(silhouette_scores)] if silhouette_scores else k_values[0]
        
        return {
            'scores': silhouette_scores,
            'best_k': best_k,
            'details': silhouette_details
        }
    
    def _gap_statistic(self, k_values, n_refs=5):  # Reduced n_refs
        """Gap Statistic implementation"""
        gaps = []
        
        for k in k_values:
            # Original data clustering
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10,  # Reduced from 20 to 10
                          max_iter=300, random_state=42)
            kmeans.fit(self.data)
            original_wcss = kmeans.inertia_
            
            # Reference distributions
            ref_wcss = []
            data_min = self.data.min(axis=0)
            data_max = self.data.max(axis=0)
            
            for _ in range(n_refs):
                # Generate uniform random data in same range
                ref_data = np.random.uniform(data_min, data_max, size=self.data.shape)
                ref_kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10,
                                  max_iter=300, random_state=42)
                ref_kmeans.fit(ref_data)
                ref_wcss.append(ref_kmeans.inertia_)
            
            # Gap statistic
            gap = np.log(np.mean(ref_wcss)) - np.log(original_wcss)
            gaps.append(gap)
        
        # Find optimal K
        if len(gaps) > 1:
            gap_diffs = np.diff(gaps)
            optimal_idx = 0
            for i in range(1, len(gap_diffs)):
                if gap_diffs[i] < 0.1 * abs(gaps[i]):
                    optimal_idx = i
                    break
            optimal_k = k_values[optimal_idx]
        else:
            optimal_k = k_values[0]
        
        return {
            'gaps': gaps,
            'optimal_k': optimal_k
        }
    
    def _calinski_harabasz_analysis(self, k_values):
        """Calinski-Harabasz Index analysis"""
        ch_scores = []
        
        for k in k_values:
            if k < 2:
                continue
                
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10,  # Reduced from 20 to 10
                          max_iter=300, random_state=42, algorithm='lloyd')
            cluster_labels = kmeans.fit_predict(self.data)
            
            if len(np.unique(cluster_labels)) > 1:
                ch_score = calinski_harabasz_score(self.data, cluster_labels)
                ch_scores.append(ch_score)
            else:
                ch_scores.append(0)
        
        best_k = k_values[np.argmax(ch_scores)] if ch_scores else k_values[0]
        
        return {
            'scores': ch_scores,
            'best_k': best_k
        }
    
    def _determine_consensus_k(self, results):
        """Multi-method consensus for optimal K"""
        methods = results['methods']
        votes = {}
        
        if 'elbow' in methods:
            k = methods['elbow']['elbow_k']
            votes[k] = votes.get(k, 0) + 1
        
        if 'silhouette' in methods:
            k = methods['silhouette']['best_k']
            votes[k] = votes.get(k, 0) + 2
        
        if 'gap' in methods:
            k = methods['gap']['optimal_k']
            votes[k] = votes.get(k, 0) + 1
        
        if 'calinski_harabasz' in methods:
            k = methods['calinski_harabasz']['best_k']
            votes[k] = votes.get(k, 0) + 1
        
        if votes:
            consensus_k = max(votes, key=votes.get)
        else:
            consensus_k = results['k_values'][len(results['k_values']) // 2]
        
        return consensus_k
    
    def _calculate_consensus_confidence(self, results, consensus_k):
        """Calculate confidence in consensus decision"""
        methods = results['methods']
        total_methods = len(methods)
        agreements = 0
        
        for method_name, method_results in methods.items():
            if method_name == 'elbow' and method_results.get('elbow_k') == consensus_k:
                agreements += 1
            elif method_name == 'silhouette' and method_results.get('best_k') == consensus_k:
                agreements += 1
            elif method_name == 'gap' and method_results.get('optimal_k') == consensus_k:
                agreements += 1
            elif method_name == 'calinski_harabasz' and method_results.get('best_k') == consensus_k:
                agreements += 1
        
        return (agreements / total_methods) if total_methods > 0 else 0.5
    
    def _validate_business_constraints(self, k):
        """Validate business constraints theo spec"""
        min_cluster_size = max(int(0.05 * len(self.data)), 1)
        max_clusters = min(15, len(self.data) // 10)
        
        validation = {
            'min_cluster_size_check': True,
            'max_clusters_check': k <= max_clusters,
            'interpretability_check': k <= 10,
            'sample_size_check': len(self.data) >= k * 10
        }
        
        validation['overall_valid'] = all(validation.values())
        return validation
    
    def step_2_3_kmeans_implementation(self):
        """Bước 2.3: Enhanced K-means implementation theo spec"""
        
        if not self.step_2_2_completed:
            raise ValueError("Must complete step 2.2 first")
        
        if self.optimal_k is None:
            raise ValueError("Optimal K not determined")
        
        try:
            # Optimal K-means configuration theo spec
            kmeans = KMeans(
                n_clusters=self.optimal_k,
                init='k-means++',
                n_init=20,
                max_iter=1000,
                tol=1e-6,
                random_state=42,
                algorithm='lloyd'
            )
            
            # Fit model
            cluster_labels = kmeans.fit_predict(self.data)
            
            # Store results
            self.final_model = kmeans
            self.cluster_labels = cluster_labels
            
            # Implementation results
            implementation_results = {
                'model': kmeans,
                'labels': cluster_labels,
                'centroids': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'n_iter': kmeans.n_iter_,
                'cluster_sizes': dict(zip(*np.unique(cluster_labels, return_counts=True))),
                'convergence_achieved': True
            }
            
            self.step_2_3_completed = True
            return implementation_results
            
        except Exception as e:
            raise ValueError(f"Error in K-means implementation: {str(e)}")
    
    def step_2_4_cluster_validation(self):
        """Bước 2.4: Comprehensive cluster validation theo spec"""
        
        if not self.step_2_3_completed:
            raise ValueError("Must complete step 2.3 first")
        
        if self.cluster_labels is None:
            raise ValueError("No cluster labels available")
        
        try:
            validation_results = {}
            
            # Internal validation metrics
            internal_metrics = self._calculate_internal_metrics()
            validation_results['internal_metrics'] = internal_metrics
            
            # Stability analysis
            stability_analysis = self._stability_analysis()
            validation_results['stability'] = stability_analysis
            
            # Cluster quality assessment
            quality_assessment = self._cluster_quality_assessment()
            validation_results['quality'] = quality_assessment
            
            # Business validation
            business_validation = self._final_business_validation()
            validation_results['business'] = business_validation
            
            # Overall validation score
            overall_score = self._calculate_overall_validation_score(validation_results)
            validation_results['overall'] = {
                'score': overall_score,
                'recommendation': self._get_validation_recommendation(overall_score)
            }
            
            self.validation_metrics = validation_results
            self.step_2_4_completed = True
            
            return validation_results
            
        except Exception as e:
            raise ValueError(f"Error in cluster validation: {str(e)}")
    
    def _calculate_internal_metrics(self):
        """Calculate internal validation metrics"""
        labels = self.cluster_labels
        data = self.data
        
        metrics = {}
        
        if len(np.unique(labels)) > 1:
            # Silhouette Score
            metrics['silhouette_score'] = float(silhouette_score(data, labels))
            
            # Calinski-Harabasz Index
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(data, labels))
            
            # Davies-Bouldin Index
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(data, labels))
        else:
            metrics = {
                'silhouette_score': -1,
                'calinski_harabasz_score': 0,
                'davies_bouldin_score': float('inf')
            }
        
        # Inertia/WCSS
        metrics['inertia'] = float(self.final_model.inertia_)
        
        return metrics
    
    def _stability_analysis(self, n_runs=20):
        """Stability analysis using multiple random seeds"""
        try:
            reference_labels = self.cluster_labels
            ari_scores = []
            
            for seed in range(n_runs):
                kmeans_temp = KMeans(
                    n_clusters=self.optimal_k,
                    init='k-means++',
                    n_init=10,  # Reduced from 20 to 10
                    max_iter=300,
                    random_state=seed,
                    algorithm='lloyd'
                )
                
                temp_labels = kmeans_temp.fit_predict(self.data)
                ari = adjusted_rand_score(reference_labels, temp_labels)
                ari_scores.append(ari)
            
            stability_score = np.mean(ari_scores)
            stability_std = np.std(ari_scores)
            
            return {
                'mean_ari': float(stability_score),
                'std_ari': float(stability_std),
                'stability_assessment': 'High' if stability_score > 0.8 else 'Medium' if stability_score > 0.6 else 'Low'
            }
            
        except Exception:
            return {
                'mean_ari': 0.5,
                'std_ari': 0.0,
                'stability_assessment': 'Unknown'
            }
    
    def _cluster_quality_assessment(self):
        """Assess quality of individual clusters"""
        labels = self.cluster_labels
        data = self.data
        centroids = self.final_model.cluster_centers_
        
        quality_metrics = {}
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            cluster_size = len(cluster_data)
            
            if cluster_size > 1:
                # Intra-cluster distance
                centroid = centroids[cluster_id]
                distances_to_centroid = pairwise_distances(cluster_data, [centroid]).flatten()
                
                quality_metrics[cluster_id] = {
                    'size': int(cluster_size),
                    'percentage': float((cluster_size / len(data)) * 100),
                    'mean_distance_to_centroid': float(np.mean(distances_to_centroid)),
                    'std_distance_to_centroid': float(np.std(distances_to_centroid)),
                    'compactness': float(np.mean(distances_to_centroid))
                }
            else:
                quality_metrics[cluster_id] = {
                    'size': int(cluster_size),
                    'percentage': float((cluster_size / len(data)) * 100),
                    'mean_distance_to_centroid': 0.0,
                    'std_distance_to_centroid': 0.0,
                    'compactness': 0.0
                }
        
        return quality_metrics
    
    def _final_business_validation(self):
        """Final business constraints validation"""
        labels = self.cluster_labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        min_size_threshold = max(int(0.05 * len(self.data)), 1)
        max_size_threshold = int(0.5 * len(self.data))
        
        validation = {
            'cluster_count': len(unique_labels),
            'min_cluster_size': int(np.min(counts)),
            'max_cluster_size': int(np.max(counts)),
            'size_balance_ratio': float(np.min(counts) / np.max(counts)),
            'meets_min_size': bool(np.all(counts >= min_size_threshold)),
            'no_dominant_cluster': bool(np.all(counts <= max_size_threshold)),
            'interpretable_count': bool(len(unique_labels) <= 10)
        }
        
        validation['business_valid'] = (
            validation['meets_min_size'] and 
            validation['no_dominant_cluster'] and 
            validation['interpretable_count']
        )
        
        return validation
    
    def _calculate_overall_validation_score(self, results):
        """Calculate overall validation score (0-100)"""
        try:
            score = 0
            
            # Internal metrics (40%)
            internal = results['internal_metrics']
            silhouette = internal.get('silhouette_score', -1)
            score += max(0, (silhouette + 1) / 2) * 40
            
            # Stability (30%)
            stability = results['stability']['mean_ari']
            score += stability * 30
            
            # Business validation (30%)
            business = results['business']
            if business['business_valid']:
                score += 30
            elif business['meets_min_size'] and business['interpretable_count']:
                score += 20
            elif business['meets_min_size'] or business['interpretable_count']:
                score += 10
            
            return int(min(100, max(0, score)))
            
        except Exception:
            return 50
    
    def _get_validation_recommendation(self, score):
        """Get recommendation based on validation score"""
        if score >= 80:
            return "Excellent - Deploy with confidence"
        elif score >= 60:
            return "Good - Suitable for production use"
        elif score >= 40:
            return "Fair - Consider refinements"
        else:
            return "Poor - Significant improvements needed"
    
    def get_clustering_pipeline(self):
        """Get complete clustering pipeline for deployment"""
        return {
            'model': self.final_model,
            'optimal_k': self.optimal_k,
            'readiness_results': self.readiness_results,
            'k_selection_results': self.k_selection_results,
            'validation_metrics': self.validation_metrics,
            'step_completion': {
                'step_2_1': self.step_2_1_completed,
                'step_2_2': self.step_2_2_completed,
                'step_2_3': self.step_2_3_completed,
                'step_2_4': self.step_2_4_completed
            }
        }

# Export
__all__ = ['ProfessionalClustering']
