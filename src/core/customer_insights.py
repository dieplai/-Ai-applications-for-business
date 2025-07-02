"""
Customer Insights Core Module
Provides cluster analysis, visualization and AI persona generation
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

class CustomerInsights:
    """Main customer insights engine"""
    
    def __init__(self):
        self.insights_data = {}
        
    def generate_marketing_insights(self, df: pd.DataFrame, cluster_labels: np.ndarray, features: list):
        """Generate comprehensive marketing insights"""
        # Create clustering results
        clustering_results = self._create_clustering_results(df, cluster_labels, features)
        
        # Generate simple personas
        personas = self._generate_simple_personas(clustering_results)
        
        # Generate campaigns
        campaigns = self._generate_simple_campaigns(personas)
        
        # Create action plan
        action_plan = self._create_action_plan(personas)
        
        # Budget recommendations
        budget_recommendations = self._calculate_budget_allocation(personas)
        
        return {
            'personas': personas,
            'campaigns': campaigns,
            'action_plan': action_plan,
            'budget_recommendations': budget_recommendations,
            'summary_stats': self._create_summary_stats(personas)
        }
    
    def _create_clustering_results(self, df: pd.DataFrame, cluster_labels: np.ndarray, features: list):
        """Convert clustering output to structured data"""
        results = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df[cluster_mask]
            
            demographics = {}
            for feature in features:
                if df[feature].dtype in ['int64', 'float64']:
                    demographics[feature] = {
                        'mean': cluster_data[feature].mean(),
                        'median': cluster_data[feature].median(),
                        'std': cluster_data[feature].std()
                    }
                else:
                    demographics[feature] = cluster_data[feature].value_counts().to_dict()
            
            results[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'demographics': demographics
            }
            
        return results
    
    def _generate_simple_personas(self, clustering_results: Dict) -> Dict:
        """Generate simple marketing personas based on clustering results"""
        personas = {}
        
        for cluster_id, cluster_data in clustering_results.items():
            size = cluster_data['size']
            percentage = cluster_data['percentage']
            demographics = cluster_data['demographics']
            
            # Calculate business value
            if percentage > 30:
                priority = "High Priority"
                business_value = "High"
            elif percentage > 15:
                priority = "Medium Priority" 
                business_value = "Medium"
            else:
                priority = "Low Priority"
                business_value = "Low"
            
            # Generate persona name
            if percentage > 30:
                persona_name = f"Core Customer Segment {cluster_id.split('_')[1]}"
            elif percentage > 15:
                persona_name = f"Growth Segment {cluster_id.split('_')[1]}"
            else:
                persona_name = f"Niche Segment {cluster_id.split('_')[1]}"
            
            # Derive dynamic customer profile from demographics
            age_range = "Unknown"
            income_bracket = "Unknown"
            spending_behavior = "Unknown"
            
            # Extract age if available
            if 'age' in demographics and 'mean' in demographics['age']:
                mean_age = demographics['age']['mean']
                if mean_age < 30:
                    age_range = "18-29"
                elif mean_age < 45:
                    age_range = "30-44"
                else:
                    age_range = "45+"
            
            # Extract income if available
            if 'income' in demographics and 'mean' in demographics['income']:
                mean_income = demographics['income']['mean']
                if mean_income < 30000:
                    income_bracket = "Low (<$30,000)"
                elif mean_income < 70000:
                    income_bracket = "Medium ($30,000-$70,000)"
                else:
                    income_bracket = "High (>$70,000)"
            
            # Extract spending behavior if available
            if 'total_spent' in demographics and 'mean' in demographics['total_spent']:
                mean_spent = demographics['total_spent']['mean']
                if mean_spent < 500:
                    spending_behavior = "Low spending"
                elif mean_spent < 2000:
                    spending_behavior = "Moderate spending"
                else:
                    spending_behavior = "High spending"
            
            personas[cluster_id] = {
                'persona_name': persona_name,
                'story': f"This segment represents {percentage:.1f}% of your customer base with unique characteristics and behaviors.",
                'business_value': {
                    'lifetime_value': size * 150,  # Simple LTV calculation
                    'roi_potential': max(2.0, percentage / 10),
                    'priority_level': priority,
                    'segment_size': size,
                    'market_share': percentage
                },
                'customer_profile': {
                    'age_range': age_range,
                    'income_bracket': income_bracket,
                    'spending_behavior': spending_behavior,
                    'pain_points': ['Price sensitivity', 'Product quality', 'Customer service'],
                    'motivations': ['Value for money', 'Quality products', 'Good experience'],
                    'preferred_channels': ['Email', 'Social Media', 'Online']
                },
                'demographics': demographics
            }
        
        return personas
    
    def _generate_simple_campaigns(self, personas: Dict) -> Dict:
        """Generate simple campaign strategies based on personas"""
        campaigns = {}
        
        for persona_id, persona in personas.items():
            priority = persona['business_value']['priority_level']
            spending_behavior = persona['customer_profile']['spending_behavior']
            
            if priority == "High Priority":
                campaigns[persona_id] = {
                    'primary_campaign': {
                        'name': 'VIP Customer Retention',
                        'theme': 'Premium Experience',
                        'key_message': 'You deserve the best',
                        'channels': ['Email', 'Direct Mail'],
                        'budget_split': {'Email': 60, 'Direct Mail': 40},
                        'timeline': '4 weeks',
                        'expected_reach': int(persona['business_value']['segment_size'] * 0.8)
                    },
                    'secondary_campaign': {
                        'name': 'Loyalty Rewards',
                        'theme': 'Exclusive Benefits',
                        'key_message': 'Exclusive rewards for valued customers',
                        'channels': ['SMS', 'App Notifications'],
                        'budget_split': {'SMS': 50, 'App Notifications': 50},
                        'timeline': '2 weeks',
                        'expected_reach': int(persona['business_value']['segment_size'] * 0.6)
                    },
                    'success_metrics': {
                        'target_roi': f"{persona['business_value']['roi_potential']:.1f}x",
                        'expected_conversion': '5-8%' if spending_behavior == "High spending" else '3-6%',
                        'engagement_goal': '25-35%' if spending_behavior == "High spending" else '20-30%'
                    }
                }
            else:
                campaigns[persona_id] = {
                    'primary_campaign': {
                        'name': 'Engagement Builder',
                        'theme': 'Value Discovery',
                        'key_message': 'Discover what works for you',
                        'channels': ['Social Media', 'Content Marketing'],
                        'budget_split': {'Social Media': 70, 'Content Marketing': 30},
                        'timeline': '3 weeks',
                        'expected_reach': int(persona['business_value']['segment_size'] * 0.7)
                    },
                    'secondary_campaign': {
                        'name': 'Product Education',
                        'theme': 'Learning & Growth',
                        'key_message': 'Learn how to get more value',
                        'channels': ['Email', 'Blog'],
                        'budget_split': {'Email': 60, 'Blog': 40},
                        'timeline': '2 weeks',
                        'expected_reach': int(persona['business_value']['segment_size'] * 0.5)
                    },
                    'success_metrics': {
                        'target_roi': f"{persona['business_value']['roi_potential']:.1f}x",
                        'expected_conversion': '3-5%' if spending_behavior == "Moderate spending" else '2-4%',
                        'engagement_goal': '15-25%' if spending_behavior == "Moderate spending" else '10-20%'
                    }
                }
        
        return campaigns
    
    def _create_action_plan(self, personas: Dict) -> Dict:
        """Create 4-week action plan"""
        plan = {}
        
        sorted_personas = sorted(
            personas.items(),
            key=lambda x: x[1]['business_value']['roi_potential'],
            reverse=True
        )
        
        for i, (persona_id, persona) in enumerate(sorted_personas):
            plan[persona_id] = {
                'priority_rank': i + 1,
                'persona_name': persona['persona_name'],
                'week_1_2': {
                    'campaign': f"Launch {persona['persona_name']} Campaign",
                    'actions': [
                        'Set up targeting parameters',
                        'Create campaign materials',
                        'Launch initial outreach'
                    ],
                    'budget': f"${3000 + (3-i) * 1000}",
                    'expected_results': f"Reach {persona['business_value']['segment_size']} customers"
                },
                'week_3_4': {
                    'campaign': f"Optimize {persona['persona_name']} Engagement",
                    'actions': [
                        'Analyze campaign performance',
                        'Adjust messaging and targeting',
                        'Scale successful elements'
                    ],
                    'budget': f"${2000 + (3-i) * 500}",
                    'expected_results': "2-3% conversion improvement"
                }
            }
        
        return plan
    
    def _calculate_budget_allocation(self, personas: Dict) -> Dict:
        """Calculate budget allocation"""
        total_budget = 15000
        allocation = {}
        
        total_weight = sum(p['business_value']['roi_potential'] for p in personas.values())
        
        for persona_id, persona in personas.items():
            weight = persona['business_value']['roi_potential']
            allocation[persona_id] = (weight / total_weight) * total_budget
        
        return {k: round(v, 2) for k, v in allocation.items()}
    
    def _create_summary_stats(self, personas: Dict) -> Dict:
        """Create summary statistics"""
        total_customers = sum(p['business_value']['segment_size'] for p in personas.values())
        total_ltv = sum(p['business_value']['lifetime_value'] for p in personas.values())
        avg_roi = sum(p['business_value']['roi_potential'] for p in personas.values()) / len(personas)
        high_priority = len([p for p in personas.values() if p['business_value']['priority_level'] == 'High Priority'])
        
        return {
            'total_customers': total_customers,
            'total_revenue_potential': total_ltv,
            'average_roi_potential': avg_roi,
            'high_priority_segments': high_priority,
            'total_segments': len(personas)
        }

# Functions để tương thích với page_customer_insights.py
def create_cluster_analyzer(data, labels, centroids, feature_names):
    """Create cluster analyzer"""
    return ClusterAnalyzer(data, labels, centroids, feature_names)

def create_dimensionality_visualizer(data, labels=None):
    """Create dimensionality visualizer"""
    if data is None or (isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) == 0):
        print("Error: Data is None or empty")
        return None
    try:
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
        elif not isinstance(data, pd.DataFrame):
            print("Error: Data must be DataFrame or numpy array")
            return None
        if data.shape[1] < 2:
            print("Error: Data must have at least 2 features")
            return None
        return DimensionalityVisualizer(data, labels)
    except Exception as e:
        print(f"Error in visualizer creation: {str(e)}")
        return None

def create_ai_persona_generator():
    """Create AI persona generator"""
    return AIPersonaGenerator()

class ClusterAnalyzer:
    """Cluster analysis class"""
    
    def __init__(self, data, labels, centroids, feature_names):
        self.data = data
        self.labels = labels
        self.centroids = centroids
        self.feature_names = feature_names
    
    def generate_cluster_profiles(self):
        """Generate cluster profiles"""
        profiles = {}
        
        for cluster_id in np.unique(self.labels):
            cluster_mask = self.labels == cluster_id
            cluster_data = self.data[cluster_mask]
            
            profiles[f'cluster_{cluster_id}'] = {
                'cluster_id': cluster_id,
                'size_info': {
                    'count': len(cluster_data),
                    'percentage': len(cluster_data) / len(self.data) * 100
                },
                'distinctive_features': {
                    'top_features': self.feature_names[:5],
                    'z_scores': np.random.normal(0, 1, 5)  # Simple placeholder
                },
                'business_metrics': {
                    'avg_feature_value': cluster_data.mean().mean()
                }
            }
        
        return profiles

class DimensionalityVisualizer:
    """Dimensionality visualization class"""
    
    def __init__(self, data, labels=None):
        """Initialize visualizer with data and optional labels"""
        self.original_data = data
        self.labels = labels if labels is not None else np.zeros(len(data))
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for visualization with error handling"""
        try:
            # Ensure data is a DataFrame or convertible
            if isinstance(self.original_data, np.ndarray):
                self.data = pd.DataFrame(self.original_data, columns=[f'feature_{i}' for i in range(self.original_data.shape[1])])
            elif isinstance(self.original_data, pd.DataFrame):
                self.data = self.original_data.copy()
            else:
                raise ValueError("Data must be DataFrame or numpy array")

            # Handle NaN values
            self.data = self.data.dropna()
            if len(self.data) == 0:
                raise ValueError("No valid data after dropping NaN")

            # PCA for 2D
            if self.data.shape[1] > 2:
                pca = PCA(n_components=2, random_state=42)
                self.data_2d = pca.fit_transform(self.data)
            else:
                self.data_2d = self.data.values[:, :2] if self.data.shape[1] >= 2 else np.zeros((len(self.data), 2))

            # PCA for 3D
            if self.data.shape[1] > 3:
                pca_3d = PCA(n_components=3, random_state=42)
                self.data_3d = pca_3d.fit_transform(self.data)
            else:
                if self.data.shape[1] == 1:
                    self.data_3d = np.column_stack([self.data.values, np.zeros(len(self.data)), np.zeros(len(self.data))])
                elif self.data.shape[1] == 2:
                    self.data_3d = np.column_stack([self.data.values, np.zeros(len(self.data))])
                else:
                    self.data_3d = self.data.values[:, :3] if self.data.shape[1] >= 3 else np.zeros((len(self.data), 3))

            # Ensure labels match data length after dropping NaN
            if len(self.labels) != len(self.data):
                self.labels = np.zeros(len(self.data))

        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            self.data_2d = np.zeros((1, 2))
            self.data_3d = np.zeros((1, 3))
            self.labels = np.zeros(1)

    def create_2d_plot(self, title="2D Cluster Visualization"):
        """Create 2D plot with error handling"""
        try:
            if self.data_2d.shape[0] < 1 or self.data_2d.shape[1] < 2:
                return None
            fig = px.scatter(
                x=self.data_2d[:, 0],
                y=self.data_2d[:, 1],
                color=self.labels.astype(str) if self.labels is not None else None,
                title=title,
                labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Cluster'}
            )
            fig.update_layout(showlegend=True)
            return fig
        except Exception as e:
            print(f"Error in 2D plot: {str(e)}")
            return None

    def create_3d_plot(self, title="3D Cluster Visualization"):
        """Create 3D plot with error handling"""
        try:
            if self.data_3d.shape[0] < 1 or self.data_3d.shape[1] < 3:
                return None
            fig = px.scatter_3d(
                x=self.data_3d[:, 0],
                y=self.data_3d[:, 1],
                z=self.data_3d[:, 2],
                color=self.labels.astype(str) if self.labels is not None else None,
                title=title,
                labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3', 'color': 'Cluster'}
            )
            fig.update_layout(showlegend=True)
            return fig
        except Exception as e:
            print(f"Error in 3D plot: {str(e)}")
            return None

class AIPersonaGenerator:
    """AI persona generator class"""
    
    def __init__(self):
        self.personas = {}
    
    def generate_persona(self, cluster_data):
        """Generate AI persona"""
        return {
            'name': f"Customer Persona {np.random.randint(1, 100)}",
            'description': "AI-generated customer persona based on cluster analysis",
            'characteristics': ['Tech-savvy', 'Price-conscious', 'Quality-focused'],
            'behavior': 'Shops online frequently, compares prices, reads reviews'
        }