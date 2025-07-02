"""
Phase 1: Professional Data Preparation Core - FIXED Data Persistence
Fix lỗi "Must complete step 1.3 first" và data flow issues
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class ProfessionalDataPreparation:
    """Professional Phase 1: Data Preparation - FIXED Data Persistence"""
    
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.cleaned_data = None
        self.preprocessing_pipeline = {}
        self.quality_metrics = {}
        self.exploration_results = {}
        
        # ✅ NEW: Step completion tracking
        self.step_1_1_completed = False
        self.step_1_2_completed = False
        self.step_1_3_completed = False
        self.step_1_4_completed = False
    
    def set_data(self, data):
        """✅ FIXED: Set data với comprehensive validation"""
        if data is None:
            raise ValueError("Data cannot be None")
        
        if not isinstance(data, pd.DataFrame):
            if hasattr(data, 'values'):
                data = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a pandas DataFrame")
        
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        
        self.original_data = data.copy()
        # Reset completion status when new data is set
        self.step_1_1_completed = False
        self.step_1_2_completed = False
        self.step_1_3_completed = False
        self.step_1_4_completed = False
        
        return True
    
    def step_1_1_data_exploration(self):
        """✅ FIXED: Step 1.1 với completion tracking"""
        
        if self.original_data is None:
            raise ValueError("No data available. Please set data first using set_data()")
        
        data = self.original_data
        results = {}
        
        try:
            # Basic profiling
            results['shape'] = data.shape
            results['dtypes'] = dict(data.dtypes)
            results['missing_summary'] = dict(data.isnull().sum())
            results['duplicate_count'] = int(data.duplicated().sum())
            
            # Statistical summary
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 0:
                results['describe'] = numeric_data.describe().to_dict()
                
                # Correlation matrix
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()
                    corr_matrix = corr_matrix.fillna(0)
                    results['correlation_matrix'] = corr_matrix.to_dict()
                else:
                    results['correlation_matrix'] = {}
            else:
                results['describe'] = {}
                results['correlation_matrix'] = {}
            
            # Distribution analysis
            results['skewness'] = {}
            results['kurtosis'] = {}
            
            for col in numeric_data.columns:
                try:
                    if not data[col].isna().all() and len(data[col].dropna()) > 3:
                        clean_data = data[col].dropna()
                        results['skewness'][col] = float(clean_data.skew())
                        results['kurtosis'][col] = float(clean_data.kurtosis())
                except Exception:
                    continue
            
            self.exploration_results = results
            self.step_1_1_completed = True  # ✅ Mark as completed
            return results
            
        except Exception as e:
            raise ValueError(f"Error in data exploration: {str(e)}")
    
    def step_1_2_quality_assessment(self):
        """✅ FIXED: Step 1.2 với prerequisite check"""
        
        if not self.step_1_1_completed:
            raise ValueError("Must complete step 1.1 first")
        
        if self.original_data is None:
            raise ValueError("No data available for quality assessment")
        
        data = self.original_data
        quality_metrics = {}
        
        try:
            # Missing value analysis
            missing_counts = data.isnull().sum()
            missing_rates = (missing_counts / len(data)) * 100
            quality_metrics['missing_rates'] = dict(missing_rates)
            quality_metrics['high_missing_columns'] = missing_rates[missing_rates > 20].index.tolist()
            
            # Outlier detection
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_analysis = {}
            
            for col in numeric_cols:
                try:
                    if not data[col].isna().all() and len(data[col].dropna()) > 0:
                        clean_col = data[col].dropna()
                        
                        if len(clean_col) > 3:
                            Q1 = clean_col.quantile(0.25)
                            Q3 = clean_col.quantile(0.75)
                            IQR = Q3 - Q1
                            
                            if IQR > 0:
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                iqr_outliers = len(clean_col[(clean_col < lower_bound) | (clean_col > upper_bound)])
                            else:
                                iqr_outliers = 0
                            
                            try:
                                z_scores = np.abs(stats.zscore(clean_col))
                                z_outliers = len(clean_col[z_scores > 3])
                            except Exception:
                                z_outliers = 0
                            
                            outlier_analysis[col] = {
                                'iqr_outliers': int(iqr_outliers),
                                'z_outliers': int(z_outliers),
                                'outlier_percentage': float((iqr_outliers / len(clean_col)) * 100)
                            }
                        else:
                            outlier_analysis[col] = {
                                'iqr_outliers': 0,
                                'z_outliers': 0,
                                'outlier_percentage': 0.0
                            }
                except Exception:
                    outlier_analysis[col] = {
                        'iqr_outliers': 0,
                        'z_outliers': 0,
                        'outlier_percentage': 0.0
                    }
            
            quality_metrics['outlier_analysis'] = outlier_analysis
            
            # Normality testing
            normality_tests = {}
            test_cols = list(numeric_cols)[:10]
            
            for col in test_cols:
                try:
                    if not data[col].isna().all() and len(data[col].dropna()) > 3:
                        clean_col = data[col].dropna()
                        
                        if len(clean_col) >= 3:
                            test_sample = clean_col.sample(min(len(clean_col), 5000), random_state=42)
                            
                            if len(test_sample) <= 5000:
                                try:
                                    shapiro_stat, shapiro_p = stats.shapiro(test_sample)
                                    shapiro_p = float(shapiro_p)
                                except Exception:
                                    shapiro_p = 0.0
                            else:
                                shapiro_p = 0.0
                            
                            try:
                                normal_stat, normal_p = normaltest(test_sample)
                                normal_p = float(normal_p)
                            except Exception:
                                normal_p = 0.0
                            
                            normality_tests[col] = {
                                'shapiro_p_value': shapiro_p,
                                'normal_test_p_value': normal_p,
                                'is_normal': shapiro_p > 0.05 and normal_p > 0.05
                            }
                        else:
                            normality_tests[col] = {
                                'shapiro_p_value': 0.0,
                                'normal_test_p_value': 0.0,
                                'is_normal': False
                            }
                except Exception:
                    normality_tests[col] = {
                        'error': 'Could not perform normality test',
                        'is_normal': False
                    }
            
            quality_metrics['normality_tests'] = normality_tests
            
            # Hopkins Statistic
            hopkins_score = self._calculate_hopkins_statistic_safe()
            quality_metrics['hopkins_statistic'] = hopkins_score
            quality_metrics['clustering_readiness'] = "Good" if hopkins_score > 0.5 else "Poor"
            
            # Overall quality score
            quality_score = self._calculate_comprehensive_quality_score(quality_metrics)
            quality_metrics['overall_quality_score'] = quality_score
            
            self.quality_metrics = quality_metrics
            self.step_1_2_completed = True  # ✅ Mark as completed
            return quality_metrics
            
        except Exception as e:
            raise ValueError(f"Error in quality assessment: {str(e)}")
    
    def step_1_3_cleaning_transformation(self):
        """✅ FIXED: Step 1.3 với proper data persistence"""
        
        if not self.step_1_2_completed:
            raise ValueError("Must complete step 1.2 first")
        
        if self.original_data is None:
            raise ValueError("No data available for cleaning")
        
        data = self.original_data.copy()
        cleaning_results = {}
        
        try:
            # Remove duplicates
            initial_rows = len(data)
            data = data.drop_duplicates()
            cleaning_results['duplicates_removed'] = initial_rows - len(data)
            
            # Separate numeric and categorical columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # Missing values handling
            missing_handling = {}
            
            # Numerical: KNNImputer
            if len(numeric_cols) > 0:
                numeric_missing_before = data[numeric_cols].isnull().sum().sum()
                if numeric_missing_before > 0:
                    try:
                        data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
                        
                        knn_imputer = KNNImputer(n_neighbors=min(5, len(data)-1))
                        imputed_data = knn_imputer.fit_transform(data[numeric_cols])
                        data[numeric_cols] = imputed_data
                        
                        self.preprocessing_pipeline['knn_imputer'] = knn_imputer
                        missing_handling['numeric_imputed'] = numeric_missing_before
                    except Exception:
                        median_imputer = SimpleImputer(strategy='median')
                        data[numeric_cols] = median_imputer.fit_transform(data[numeric_cols])
                        self.preprocessing_pipeline['median_imputer'] = median_imputer
                        missing_handling['numeric_imputed'] = numeric_missing_before
            
            # Categorical: SimpleImputer
            if len(categorical_cols) > 0:
                categorical_missing_before = data[categorical_cols].isnull().sum().sum()
                if categorical_missing_before > 0:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
                    self.preprocessing_pipeline['categorical_imputer'] = cat_imputer
                    missing_handling['categorical_imputed'] = categorical_missing_before
            
            cleaning_results['missing_handling'] = missing_handling
            
            # Outlier handling - Winsorization
            outlier_handling = {}
            for col in numeric_cols:
                try:
                    if not data[col].isna().all() and len(data[col].dropna()) > 0:
                        original_std = data[col].std()
                        if not np.isnan(original_std) and original_std > 0:
                            data[col] = winsorize(data[col], limits=[0.025, 0.025])
                            new_std = data[col].std()
                            
                            outlier_handling[col] = {
                                'original_std': float(original_std),
                                'new_std': float(new_std),
                                'reduction_ratio': float((original_std - new_std) / original_std)
                            }
                except Exception:
                    continue
            
            cleaning_results['outlier_handling'] = outlier_handling
            
            # Feature encoding
            encoding_results = {}
            encoders = {}
            
            for col in categorical_cols:
                try:
                    unique_count = data[col].nunique()
                    
                    if unique_count <= 2:
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
                        encoders[col] = ('label', le)
                        encoding_results[col] = 'label_encoded'
                        
                    elif unique_count <= 10:
                        try:
                            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                            encoded_cols = ohe.fit_transform(data[[col]])
                            feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                            
                            encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=data.index)
                            data = pd.concat([data, encoded_df], axis=1)
                            data = data.drop(columns=[col])
                            
                            encoders[col] = ('onehot', ohe, feature_names)
                            encoding_results[col] = 'onehot_encoded'
                        except Exception:
                            le = LabelEncoder()
                            data[col] = le.fit_transform(data[col].astype(str))
                            encoders[col] = ('label_fallback', le)
                            encoding_results[col] = 'label_encoded_fallback'
                    else:
                        data = data.drop(columns=[col])
                        encoding_results[col] = 'dropped_high_cardinality'
                        
                except Exception:
                    data = data.drop(columns=[col])
                    encoding_results[col] = 'dropped_error'
            
            self.preprocessing_pipeline['encoders'] = encoders
            cleaning_results['encoding_results'] = encoding_results
            
            # Scaling - StandardScaler
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 0:
                try:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data)
                    
                    if np.any(np.isnan(scaled_data)):
                        scaled_data = np.nan_to_num(scaled_data, nan=0.0)
                    
                    scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns, index=data.index)
                    
                    self.preprocessing_pipeline['scaler'] = scaler
                    cleaning_results['scaling_applied'] = True
                    cleaning_results['final_shape'] = scaled_df.shape
                    
                    # ✅ CRITICAL: Store cleaned data properly
                    self.cleaned_data = scaled_df
                    self.step_1_3_completed = True  # ✅ Mark as completed
                    
                except Exception as e:
                    raise ValueError(f"Scaling failed: {str(e)}")
            else:
                raise ValueError("No numeric columns remaining after encoding")
            
            return cleaning_results
            
        except Exception as e:
            raise ValueError(f"Error in data cleaning: {str(e)}")
    
    def step_1_4_feature_engineering(self):
        """✅ FIXED: Step 1.4 với comprehensive fallback mechanisms"""
        
        # ✅ ENHANCED: Multiple fallback mechanisms for data availability
        data_source = None
        
        # Method 1: Check if step 1.3 completed and cleaned_data exists
        if self.step_1_3_completed and hasattr(self, 'cleaned_data') and self.cleaned_data is not None:
            data = self.cleaned_data.copy()
            data_source = "cleaned_data"
        
        # Method 2: Fallback - try to get from original data and process
        elif self.original_data is not None:
            try:
                # Run step 1.3 if not completed
                if not self.step_1_3_completed:
                    self.step_1_3_cleaning_transformation()
                    if hasattr(self, 'cleaned_data') and self.cleaned_data is not None:
                        data = self.cleaned_data.copy()
                        data_source = "auto_cleaned_data"
                    else:
                        raise ValueError("Automatic cleaning failed")
                else:
                    raise ValueError("Step 1.3 completed but no cleaned data available")
            except Exception as e:
                raise ValueError(f"Cannot proceed with feature engineering. Step 1.3 error: {str(e)}")
        
        # Method 3: Ultimate fallback - use original data directly
        else:
            raise ValueError("No data available for feature engineering. Please complete previous steps first.")
        
        if data is None or len(data) == 0:
            raise ValueError("No valid data for feature engineering")
        
        engineering_results = {}
        engineering_results['data_source'] = data_source
        
        try:
            # Feature selection
            feature_selection_results = {}
            
            # Low variance filter
            if len(data.columns) > 1:
                try:
                    variance_selector = VarianceThreshold(threshold=0.01)
                    selected_features = variance_selector.fit_transform(data)
                    selected_feature_names = data.columns[variance_selector.get_support()]
                    
                    removed_low_variance = len(data.columns) - len(selected_feature_names)
                    feature_selection_results['low_variance_removed'] = removed_low_variance
                    
                    if len(selected_feature_names) > 0:
                        data = pd.DataFrame(selected_features, columns=selected_feature_names, index=data.index)
                        self.preprocessing_pipeline['variance_selector'] = variance_selector
                    
                except Exception:
                    feature_selection_results['low_variance_removed'] = 0
            
            # High correlation removal
            if len(data.columns) > 1:
                try:
                    corr_matrix = data.corr().abs()
                    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.9)]
                    
                    if high_corr_features:
                        data = data.drop(columns=high_corr_features)
                        feature_selection_results['high_correlation_removed'] = len(high_corr_features)
                        feature_selection_results['removed_features'] = high_corr_features
                    else:
                        feature_selection_results['high_correlation_removed'] = 0
                        
                except Exception:
                    feature_selection_results['high_correlation_removed'] = 0
            
            # Feature importance (optional)
            if len(data.columns) > 1 and len(data) > 10:
                try:
                    pca_temp = PCA(n_components=1, random_state=42)
                    artificial_target = pca_temp.fit_transform(data).ravel()
                    target_median = np.median(artificial_target)
                    binary_target = (artificial_target > target_median).astype(int)
                    
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(data, binary_target)
                    
                    feature_importance = dict(zip(data.columns, rf.feature_importances_))
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    
                    # Keep top 80% of cumulative importance
                    cumulative_importance = 0
                    selected_features = []
                    
                    for feature, importance in sorted_features:
                        cumulative_importance += importance
                        selected_features.append(feature)
                        if cumulative_importance >= 0.8:
                            break
                    
                    if len(selected_features) < len(data.columns) and len(selected_features) > 1:
                        data = data[selected_features]
                        feature_selection_results['rf_feature_selection'] = {
                            'features_kept': len(selected_features),
                            'features_removed': len(data.columns) - len(selected_features),
                            'cumulative_importance': cumulative_importance
                        }
                    
                except Exception:
                    feature_selection_results['rf_feature_selection'] = 'failed'
            
            engineering_results['feature_selection'] = feature_selection_results
            
            # PCA (optional for high-dimensional data)
            pca_results = {}
            apply_pca = len(data.columns) > 20
            
            if apply_pca:
                try:
                    pca = PCA(n_components=0.95, random_state=42)
                    pca_data = pca.fit_transform(data)
                    
                    n_components = pca.n_components_
                    explained_variance_ratio = pca.explained_variance_ratio_
                    cumulative_variance = np.sum(explained_variance_ratio)
                    
                    pca_columns = [f'PC{i+1}' for i in range(n_components)]
                    data = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)
                    
                    self.preprocessing_pipeline['pca'] = pca
                    pca_results = {
                        'applied': True,
                        'original_features': len(data.columns),
                        'pca_components': n_components,
                        'variance_explained': float(cumulative_variance),
                        'variance_per_component': explained_variance_ratio.tolist()
                    }
                except Exception:
                    pca_results = {'applied': False, 'reason': 'pca_failed'}
            else:
                pca_results = {'applied': False, 'reason': 'not_needed'}
            
            engineering_results['pca'] = pca_results
            
            # Final validation
            final_validation = {
                'final_features': len(data.columns),
                'final_samples': len(data),
                'feature_names': list(data.columns),
                'data_ready_for_clustering': len(data.columns) >= 2 and len(data) >= 10,
                'data_source': data_source
            }
            
            # Add min/max values if data is numeric
            try:
                final_validation['min_max_values'] = {
                    'min': float(data.min().min()),
                    'max': float(data.max().max())
                }
            except:
                final_validation['min_max_values'] = {'min': 0, 'max': 1}
            
            engineering_results['final_validation'] = final_validation
            
            # ✅ CRITICAL: Store final processed data
            self.processed_data = data
            self.preprocessing_pipeline['final_columns'] = list(data.columns)
            self.step_1_4_completed = True  # ✅ Mark as completed
            
            return engineering_results
            
        except Exception as e:
            raise ValueError(f"Error in feature engineering: {str(e)}")
    
    def _calculate_hopkins_statistic_safe(self):
        """Safe Hopkins Statistic calculation"""
        try:
            if self.original_data is None:
                return 0.5
            
            numeric_data = self.original_data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                return 0.5
            
            clean_data = numeric_data.dropna()
            if len(clean_data) < 10:
                return 0.5
            
            clean_data = clean_data.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_data) < 10:
                return 0.5
            
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(clean_data)
                
                if np.any(np.isnan(scaled_data)):
                    return 0.5
                
            except Exception:
                return 0.5
            
            n, p = scaled_data.shape
            m = min(int(0.1 * n), 50)
            
            if m < 5:
                return 0.5
            
            data_min = np.min(scaled_data, axis=0)
            data_max = np.max(scaled_data, axis=0)
            
            data_range = data_max - data_min
            if np.any(data_range == 0):
                return 0.5
            
            random_points = np.random.uniform(data_min, data_max, size=(m, p))
            
            random_distances = []
            real_distances = []
            
            for i in range(m):
                try:
                    distances_to_real = pairwise_distances([random_points[i]], scaled_data)[0]
                    if len(distances_to_real) > 0:
                        random_distances.append(np.min(distances_to_real))
                    
                    if len(scaled_data) > 1:
                        real_point_idx = np.random.randint(0, n)
                        real_point = scaled_data[real_point_idx:real_point_idx+1]
                        distances_real_to_real = pairwise_distances(real_point, scaled_data)[0]
                        distances_real_to_real = distances_real_to_real[distances_real_to_real > 0]
                        if len(distances_real_to_real) > 0:
                            real_distances.append(np.min(distances_real_to_real))
                except Exception:
                    continue
            
            if len(real_distances) > 0 and len(random_distances) > 0:
                sum_random = np.sum(random_distances)
                sum_real = np.sum(real_distances)
                if sum_random + sum_real > 0:
                    hopkins = sum_random / (sum_random + sum_real)
                    return float(hopkins)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_comprehensive_quality_score(self, metrics):
        """Safe quality score calculation"""
        try:
            score = 100
            
            if 'missing_rates' in metrics:
                missing_rates = metrics['missing_rates']
                if missing_rates:
                    avg_missing = np.mean(list(missing_rates.values()))
                    score -= min(avg_missing * 2, 30)
            
            if 'outlier_analysis' in metrics:
                outlier_analysis = metrics['outlier_analysis']
                if outlier_analysis:
                    total_outlier_pct = np.mean([info['outlier_percentage'] for info in outlier_analysis.values()])
                    score -= min(total_outlier_pct, 25)
            
            if hasattr(self, 'exploration_results') and 'duplicate_count' in self.exploration_results:
                duplicate_pct = (self.exploration_results['duplicate_count'] / len(self.original_data)) * 100
                score -= min(duplicate_pct * 2, 15)
            
            hopkins = metrics.get('hopkins_statistic', 0.5)
            if hopkins > 0.7:
                score += 10
            elif hopkins > 0.5:
                score += 5
            
            return max(0, min(100, int(score)))
            
        except Exception:
            return 50
    
    def get_preprocessing_pipeline(self):
        """Get complete preprocessing pipeline"""
        return self.preprocessing_pipeline
    
    def get_quality_report(self):
        """Get comprehensive quality report"""
        return {
            'exploration': self.exploration_results,
            'quality_metrics': self.quality_metrics,
            'preprocessing_pipeline': self.preprocessing_pipeline,
            'step_completion': {
                'step_1_1': self.step_1_1_completed,
                'step_1_2': self.step_1_2_completed,
                'step_1_3': self.step_1_3_completed,
                'step_1_4': self.step_1_4_completed
            }
        }

# Export
__all__ = ['ProfessionalDataPreparation']
