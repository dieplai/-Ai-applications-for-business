"""
Dashboard Analytics Core Logic
Performance metrics & KPI calculation
"""
import pandas as pd
from typing import Dict, Any, Optional

class DashboardAnalyzer:
    """Core dashboard analytics engine"""
    
    def __init__(self):
        self.session_data = {}
        self.workflow_status = {
            'data_prep': 0,
            'clustering': 0, 
            'insights': 0,
            'reports': 0
        }
    
    def calculate_workflow_progress(self, session_state: Dict) -> Dict[str, Any]:
        """Calculate overall workflow completion"""
        progress = {
            'data_prep': self._calc_data_prep_progress(session_state),
            'clustering': self._calc_clustering_progress(session_state),
            'insights': self._calc_insights_progress(session_state),
            'reports': self._calc_reports_progress(session_state),
            'overall': 0
        }
        
        progress['overall'] = sum(progress.values()) / 4
        return progress
    
    def generate_performance_metrics(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate key performance indicators"""
        metrics = {
            'data_quality_score': 0,
            'clustering_readiness': 0,
            'processing_speed': 'N/A',
            'memory_usage': 'N/A',
            'accuracy_estimate': 'N/A'
        }
        
        if data is not None:
            metrics['data_quality_score'] = self._calculate_data_quality(data)
            metrics['clustering_readiness'] = self._assess_clustering_readiness(data)
            
        return metrics
    
    def track_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate comprehensive data quality score (0-100)"""
        score = 0
        
        # Size factor (25 points)
        if len(data) >= 1000:
            score += 25
        elif len(data) >= 500:
            score += 20
        elif len(data) >= 100:
            score += 15
        
        # Completeness (25 points)
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score += max(0, 25 - (missing_pct * 100))
        
        # Feature diversity (25 points)
        numeric_cols = len(data.select_dtypes(include=['number']).columns)
        if numeric_cols >= 5:
            score += 25
        elif numeric_cols >= 3:
            score += 20
        elif numeric_cols >= 2:
            score += 15
        
        # Variance (25 points)
        if numeric_cols > 0:
            variance_score = 0
            for col in data.select_dtypes(include=['number']).columns:
                if data[col].std() > 0:
                    variance_score += 25 / numeric_cols
            score += variance_score
        
        return min(score, 100)
    
    def get_quick_insights(self, data: Optional[pd.DataFrame] = None) -> list:
        """Generate quick data insights"""
        insights = []
        
        if data is not None:
            insights.append(f"Dataset contains {len(data):,} records with {len(data.columns)} features")
            
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                insights.append(f"Found {len(numeric_cols)} numeric columns for clustering")
            
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            if missing_pct > 10:
                insights.append(f"Missing data: {missing_pct:.1f}% - consider cleaning")
            else:
                insights.append("Data quality looks good for analysis")
        else:
            insights.append("Upload your customer data to begin analysis")
            insights.append("Platform supports CSV and Excel formats")
            insights.append("Minimum 100 records recommended")
        
        return insights
    
    def _calc_data_prep_progress(self, session_state: Dict) -> float:
        """Calculate data preparation progress (0-100)"""
        progress = 0
        if session_state.get('uploaded_data') is not None:
            progress += 25
        if session_state.get('cleaned_data') is not None:
            progress += 25
        if session_state.get('engineered_features') is not None:
            progress += 25
        if session_state.get('reduced_features') is not None:
            progress += 25
        return progress
    
    def _calc_clustering_progress(self, session_state: Dict) -> float:
        """Calculate clustering progress (0-100)"""
        progress = 0
        if session_state.get('clustering_initialized') is not None:
            progress += 25
        if session_state.get('optimal_k_determined') is not None:
            progress += 25
        if session_state.get('kmeans_model_trained') is not None:
            progress += 25
        if session_state.get('clusters_validated') is not None:
            progress += 25
        return progress
    
    def _calc_insights_progress(self, session_state: Dict) -> float:
        """Calculate insights progress (0-100)"""
        progress = 0
        if session_state.get('cluster_profiles_generated') is not None:
            progress += 33
        if session_state.get('personas_created') is not None:
            progress += 33
        if session_state.get('campaigns_proposed') is not None:
            progress += 34
        return progress
    
    def _calc_reports_progress(self, session_state: Dict) -> float:
        """Calculate reports progress (0-100)"""
        progress = 0
        if session_state.get('report_content_prepared') is not None:
            progress += 50
        if session_state.get('report_exported') is not None:
            progress += 50
        return progress
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Internal data quality calculation"""
        return self.track_data_quality_score(data)
    
    def _assess_clustering_readiness(self, data: pd.DataFrame) -> float:
        """Assess how ready data is for clustering"""
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return 0
        
        readiness = 50  # Base score
        
        # Add points for good numeric features
        if len(numeric_cols) >= 3:
            readiness += 25
        
        # Add points for low missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct < 0.1:
            readiness += 25
        
        return min(readiness, 100)