import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.core.customer_insights import create_cluster_analyzer, create_ai_persona_generator
from utils.marketing_prompts import MarketingAIAssistant
import logging
import time
import json
from src.shared.ui_utils import load_shared_styles
load_shared_styles()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data
def cache_gemini_response(prompt):
    ai_assistant = MarketingAIAssistant()
    return ai_assistant.generate_content(prompt)

@st.cache_data
def cache_create_persona(_ai_assistant, cluster_info):
    return _ai_assistant.create_cluster_buyer_persona(cluster_info)

@st.cache_data
def cache_create_campaign(_ai_assistant, campaign_info, service_type, budget, duration):
    return _ai_assistant.create_cluster_campaign_plan(campaign_info, service_type, budget, duration)

def normalize_profile(profile):
    if isinstance(profile, dict):
        normalized = {}
        for key, value in profile.items():
            if isinstance(value, np.integer):
                normalized[key] = int(value)
            elif isinstance(value, np.floating):
                normalized[key] = float(value)
            elif isinstance(value, np.ndarray):
                normalized[key] = value.tolist() if value.size > 0 else []
            elif isinstance(value, dict):
                normalized[key] = normalize_profile(value)
            elif isinstance(value, list):
                normalized[key] = [normalize_profile(item) if isinstance(item, dict) else item for item in value]
            else:
                normalized[key] = value
        return normalized
    return profile

def safe_get_nested_value(data, keys, default=None):
    try:
        result = data
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result
    except (TypeError, KeyError):
        return default

class CustomerSegmentationInsights:
    def __init__(self):
        self.setup_session_state()
        api_key = st.sidebar.text_input("Khóa API Gemini (tùy chọn)", type="password", value="", help="Nhập API key để ghi đè key cố định trong mã")
        logger.info(f"API key from sidebar: {api_key}")
        self.ai_assistant = MarketingAIAssistant(api_key=api_key)
        logger.info(f"AI Assistant availability: {self.ai_assistant.available}, Error: {self.ai_assistant.error_message}")

    def setup_session_state(self):
        if 'phase_3_completed' not in st.session_state:
            st.session_state.phase_3_completed = False
        if 'current_insights_step' not in st.session_state:
            st.session_state.current_insights_step = 9
        if 'cluster_profiles' not in st.session_state:
            st.session_state.cluster_profiles = {}
        if 'ai_personas' not in st.session_state:
            st.session_state.ai_personas = {}
        if 'ai_campaigns' not in st.session_state:
            st.session_state.ai_campaigns = {}
        if 'progress_reset' not in st.session_state:
            st.session_state.progress_reset = True

    def render(self):
        st.markdown(MODERN_CSS, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="page-header">
            <h1>🎯 Thông Tin Chi Tiết Khách Hàng</h1>
            <p>Phân tích đặc điểm và tạo chiến lược tiếp thị cho từng nhóm khách hàng</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not self._check_prerequisites():
            return
        
        steps = {
            9: "Phân tích đặc điểm nhóm",
            10: "Tạo chiến lược tiếp thị", 
            11: "Trực quan hóa kết quả"
        }
        
        st.markdown(f"""
        <div class="step-indicator">
            <span class="step-label">Bước hiện tại</span>
            <span class="step-title">{steps[st.session_state.current_insights_step]}</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.current_insights_step == 9:
            self._render_step_9()
        elif st.session_state.current_insights_step == 10:
            self._render_step_10()
        elif st.session_state.current_insights_step == 11:
            self._render_step_11()

    def _check_prerequisites(self):
        if not self.ai_assistant.available:
            st.sidebar.markdown(f"""
            <div class="alert-card error">
                ⚠️ Không thể kết nối AI. {self.ai_assistant.error_message}
            </div>
            """, unsafe_allow_html=True)
            return False
        if not st.session_state.get('phase_2_completed', False) or not st.session_state.get('final_model_results'):
            st.markdown("""
            <div class="alert-card warning">
                📋 Vui lòng hoàn thành bước Phân cụm khách hàng trước.
            </div>
            """, unsafe_allow_html=True)
            return False
        return True

    def _get_valid_data(self, for_visualization=False, sample_size=1000):
        data = None
        for key in ['reduced_features', 'final_data', 'uploaded_data']:
            data = st.session_state.get(key)
            if data is not None:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    if for_visualization and len(data) > sample_size:
                        sample_indices = np.random.choice(len(data), sample_size, replace=False)
                        sampled_data = data.iloc[sample_indices]
                        st.session_state.sample_indices = sample_indices
                        return sampled_data.values
                    return data.values if for_visualization else data
                elif isinstance(data, np.ndarray) and data.size > 0:
                    if for_visualization and len(data) > sample_size:
                        sample_indices = np.random.choice(len(data), sample_size, replace=False)
                        sampled_data = data[sample_indices]
                        st.session_state.sample_indices = sample_indices
                        return sampled_data
                    return data if for_visualization else pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
        st.markdown("""
        <div class="alert-card error">
            ❌ Không tìm thấy dữ liệu hợp lệ để xử lý.
        </div>
        """, unsafe_allow_html=True)
        return None

    def _get_valid_labels(self, for_visualization=False):
        labels = st.session_state.final_model_results.get('labels')
        if labels is None:
            return None
        
        if for_visualization and hasattr(st.session_state, 'sample_indices'):
            return labels[st.session_state.sample_indices]
        
        return labels

    def _render_step_9(self):
        st.markdown("""
        <div class="section-header">
            <h2>📊 Phân Tích Đặc Điểm Nhóm</h2>
            <p>Xem xét đặc điểm nổi bật của từng nhóm khách hàng.</p>
        </div>
        """, unsafe_allow_html=True)
        
        data = self._get_valid_data()
        if data is None:
            return
        
        labels = self._get_valid_labels()
        if labels is None or len(labels) != len(data):
            st.markdown("""
            <div class="alert-card error">
                ❌ Nhãn phân cụm không hợp lệ hoặc không khớp với dữ liệu.
            </div>
            """, unsafe_allow_html=True)
            return
        
        try:
            analyzer = create_cluster_analyzer(data, labels, np.zeros((len(np.unique(labels)), data.shape[1])), data.columns.tolist())
            profiles = analyzer.generate_cluster_profiles()
            
            normalized_profiles = {}
            for cluster_id, profile in profiles.items():
                normalized_profiles[cluster_id] = normalize_profile(profile)
            
            st.session_state.cluster_profiles = normalized_profiles
            
            for cluster_id, profile in normalized_profiles.items():
                try:
                    count = safe_get_nested_value(profile, ['size_info', 'count'], 0)
                    percentage = safe_get_nested_value(profile, ['size_info', 'percentage'], 0.0)
                    top_features = safe_get_nested_value(profile, ['distinctive_features', 'top_features'], [])
                    
                    if not isinstance(top_features, list):
                        top_features = [str(top_features)] if top_features else []
                    
                    cluster_info = f"Nhóm {cluster_id} của doanh nghiệp phân tích dữ liệu khách hàng: Số khách hàng: {count}, Tỷ lệ: {percentage:.1f}%, Đặc điểm: {', '.join(top_features)}"
                    
                    if cluster_id not in st.session_state.ai_personas:
                        try:
                            persona = cache_create_persona(self.ai_assistant, cluster_info)
                            st.session_state.ai_personas[cluster_id] = {
                                'content': persona or "Không thể tạo hồ sơ khách hàng",
                                'cluster_info': cluster_info
                            }
                        except Exception as e:
                            st.session_state.ai_personas[cluster_id] = {
                                'content': f"Lỗi khi tạo hồ sơ: {str(e)}",
                                'cluster_info': cluster_info
                            }
                    
                    persona_desc = safe_get_nested_value(st.session_state.ai_personas, [cluster_id, 'content'], "Không có thông tin")
                    
                    st.markdown(f"""
                    <div class="cluster-card">
                        <div class="cluster-header">
                            <span class="cluster-id">Nhóm {cluster_id}</span>
                            <span class="cluster-stats">{count} khách hàng • {percentage:.1f}%</span>
                        </div>
                        <div class="cluster-content">
                            <div class="feature-tags">
                                {' '.join([f'<span class="tag">{feature}</span>' for feature in top_features[:3]])}
                            </div>
                            <div class="ai-insight">
                                <strong>💡 Phân tích từ AI:</strong>
                                <p>{persona_desc}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="alert-card error">
                        ❌ Lỗi khi xử lý nhóm {cluster_id}: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Quay lại Phân cụm khách hàng", key="step_9_back"):
                    st.session_state.current_page = "Clustering"
                    st.rerun()
            with col2:
                if st.button("Tiếp tục đến Tạo chiến lược tiếp thị →", type="primary", key="step_9_next"):
                    st.session_state.current_insights_step = 10
                    st.rerun()
                    
        except Exception as e:
            st.markdown(f"""
            <div class="alert-card error">
                ❌ Lỗi khi phân tích: {str(e)}
            </div>
            """, unsafe_allow_html=True)
            logger.error(f"Error in step 9: {str(e)}")

    def _render_step_10(self):
        st.markdown("""
        <div class="section-header">
            <h2>🚀 Tạo Chiến Lược Tiếp Thị</h2>
            <p>Tạo chiến lược tiếp thị thực tế cho từng nhóm khách hàng.</p>
        </div>
        """, unsafe_allow_html=True)
        
        data = self._get_valid_data()
        if data is None:
            return
        
        try:
            profiles = st.session_state.cluster_profiles
            
            if not isinstance(profiles, dict) or not profiles:
                st.markdown("""
                <div class="alert-card error">
                    ❌ Dữ liệu cluster_profiles không hợp lệ.
                </div>
                """, unsafe_allow_html=True)
                return
            
            if not isinstance(st.session_state.ai_personas, dict):
                st.session_state.ai_personas = {}
            
            if not isinstance(st.session_state.ai_campaigns, dict):
                st.session_state.ai_campaigns = {}
            
            total_clusters = len(profiles)
            
            if st.session_state.progress_reset:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (cluster_id, profile) in enumerate(profiles.items()):
                    try:
                        profile = normalize_profile(profile)
                        
                        size_info = safe_get_nested_value(profile, ['size_info'], {})
                        distinctive_features = safe_get_nested_value(profile, ['distinctive_features'], {})
                        count = safe_get_nested_value(size_info, ['count'], 0)
                        percentage = safe_get_nested_value(size_info, ['percentage'], 0.0)
                        top_features = safe_get_nested_value(distinctive_features, ['top_features'], [])
                        
                        if isinstance(top_features, str):
                            top_features = [top_features]
                        elif not isinstance(top_features, list):
                            top_features = []
                        
                        status_text.markdown(f"""
                        <div class="progress-status">
                            🎯 Đang tạo chiến lược cho Nhóm {cluster_id}...
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.5)
                        
                        if cluster_id not in st.session_state.ai_campaigns:
                            cluster_info = f"Nhóm {cluster_id} của doanh nghiệp phân tích dữ liệu khách hàng: Số khách hàng: {count}, Tỷ lệ: {percentage:.1f}%, Đặc điểm: {', '.join(top_features)}"
                            
                            try:
                                campaign = cache_create_campaign(self.ai_assistant, cluster_info, "Dịch vụ tiếp thị cá nhân hóa", "50 triệu VND", "1 tháng")
                                st.session_state.ai_campaigns[cluster_id] = {
                                    'content': campaign or "Không thể tạo chiến lược tiếp thị",
                                    'cluster_info': cluster_info
                                }
                            except Exception as e:
                                st.session_state.ai_campaigns[cluster_id] = {
                                    'content': f"Lỗi khi tạo chiến lược: {str(e)}",
                                    'cluster_info': cluster_info
                                }
                        
                        progress = (idx + 1) / total_clusters
                        progress_bar.progress(progress)
                        time.sleep(0.5)
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="alert-card error">
                            ❌ Lỗi khi xử lý nhóm {cluster_id}: {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
                        continue
                
                st.session_state.progress_reset = False
                status_text.markdown("""
                <div class="progress-status success">
                    ✅ Hoàn thành tạo chiến lược!
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
            
            for cluster_id in profiles.keys():
                try:
                    persona_desc = safe_get_nested_value(st.session_state.ai_personas, [cluster_id, 'content'], "Không có thông tin")
                    campaign_summary = safe_get_nested_value(st.session_state.ai_campaigns, [cluster_id, 'content'], "Không có chiến lược")
                    
                    st.markdown(f"""
                    <div class="strategy-card">
                        <div class="strategy-header">
                            <span class="strategy-title">📊 Nhóm {cluster_id}</span>
                        </div>
                        <div class="strategy-content">
                            <div class="persona-section">
                                <h4>👤 Hồ sơ khách hàng</h4>
                                <p>{persona_desc}</p>
                            </div>
                            <div class="campaign-section">
                                <h4>🎯 Chiến lược tiếp thị</h4>
                                <p>{campaign_summary}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="alert-card error">
                        ❌ Lỗi hiển thị nhóm {cluster_id}: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.session_state.phase_3_completed = True
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Quay lại Phân tích đặc điểm", key="step_10_back"):
                    st.session_state.current_insights_step = 9
                    st.session_state.progress_reset = True
                    st.rerun()
            with col2:
                if st.button("Tiếp tục đến Trực quan hóa kết quả →", type="primary", key="step_10_next"):
                    st.session_state.current_insights_step = 11
                    st.session_state.progress_reset = True
                    st.rerun()
                    
        except Exception as e:
            st.markdown(f"""
            <div class="alert-card error">
                ❌ Lỗi khi tạo chiến lược: {str(e)}
            </div>
            """, unsafe_allow_html=True)
            logger.error(f"Error in step 10: {str(e)}")

    def _render_step_11(self):
        st.markdown("""
        <div class="section-header">
            <h2>📈 Trực Quan Hóa Kết Quả</h2>
            <p>Hiển thị phân phối và tương quan của các nhóm khách hàng.</p>
        </div>
        """, unsafe_allow_html=True)
        
        data = self._get_valid_data(for_visualization=True)
        if data is None:
            return
        
        labels = self._get_valid_labels(for_visualization=True)
        if labels is None:
            st.markdown("""
            <div class="alert-card error">
                ❌ Nhãn phân cụm không được tìm thấy.
            </div>
            """, unsafe_allow_html=True)
            return
            
        if len(labels) != len(data):
            st.markdown(f"""
            <div class="alert-card error">
                ❌ Nhãn phân cụm không khớp với dữ liệu.
            </div>
            """, unsafe_allow_html=True)
            return
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="chart-card">
                    <h3>🍰 Phân phối nhóm khách hàng</h3>
                </div>
                """, unsafe_allow_html=True)
                label_counts = pd.Series(labels).value_counts()
                fig_pie = px.pie(names=label_counts.index, values=label_counts.values, title="")
                fig_pie.update_layout(showlegend=True, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                st.markdown("""
                <div class="chart-card">
                    <h3>🔗 Tương quan đặc điểm</h3>
                </div>
                """, unsafe_allow_html=True)
                if isinstance(data, np.ndarray):
                    data_df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
                else:
                    data_df = data
                    
                try:
                    corr_matrix = data_df.corr()
                    if not corr_matrix.empty:
                        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="")
                        fig_heatmap.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.markdown("""
                        <div class="alert-card warning">
                            ⚠️ Không thể tính toán ma trận tương quan
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="alert-card error">
                        ❌ Lỗi khi tạo heatmap: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Quay lại Tạo chiến lược tiếp thị", key="step_11_back"):
                    st.session_state.current_insights_step = 10
                    st.session_state.progress_reset = True
                    st.rerun()
            with col2:
                if st.button("Tiếp tục đến Báo cáo và xuất →", type="primary", key="step_11_next"):
                    st.session_state.current_page = "Reports"
                    st.rerun()
                    
        except Exception as e:
            st.markdown(f"""
            <div class="alert-card error">
                ❌ Lỗi khi trực quan hóa: {str(e)}
            </div>
            """, unsafe_allow_html=True)
            logger.error(f"Error in step 11: {str(e)}")

MODERN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.page-header {
    background: white;
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #f1f3f4;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.page-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 0;
    letter-spacing: -0.02em;
}

.page-header p {
    font-size: 1.1rem;
    color: #6b7280;
    margin: 0.5rem 0 0;
    font-weight: 400;
}

.step-indicator {
    background: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    border: 1px solid #f1f3f4;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.step-label {
    font-size: 0.9rem;
    color: #6b7280;
    font-weight: 500;
}

.step-title {
    font-size: 1.2rem;
    color: #1a1a1a;
    font-weight: 600;
}

.section-header {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #f1f3f4;
}

.section-header h2 {
    font-size: 1.8rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0;
}

.section-header p {
    color: #6b7280;
    margin: 0.5rem 0 0;
    font-size: 1rem;
}

.cluster-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #f1f3f4;
    margin-bottom: 1.5rem;
    overflow: hidden;
    transition: all 0.2s ease;
}

.cluster-header {
    background: #f8f9fa;
    padding: 1.25rem 1.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #f1f3f4;
}

.cluster-id {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1a1a1a;
}

.cluster-stats {
    color: #6b7280;
    font-size: 0.95rem;
    font-weight: 500;
}

.cluster-content {
    padding: 1.75rem;
}

.feature-tags {
    margin-bottom: 1.5rem;
}

.tag {
    display: inline-block;
    background: #f3f4f6;
    color: #374151;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0 0.5rem 0.5rem 0;
}

.ai-insight strong {
    color: #1a1a1a;
    font-weight: 600;
}

.ai-insight p {
    color: #4b5563;
    margin: 0.75rem 0 0;
    line-height: 1.6;
}

.strategy-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #f1f3f4;
    margin-bottom: 1.5rem;
    overflow: hidden;
}

.strategy-header {
    background: #f8f9fa;
    padding: 1.25rem 1.75rem;
    border-bottom: 1px solid #f1f3f4;
}

.strategy-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1a1a1a;
}

.strategy-content {
    padding: 1.75rem;
}

.persona-section, .campaign-section {
    margin-bottom: 1.5rem;
}

.persona-section:last-child, .campaign-section:last-child {
    margin-bottom: 0;
}

.strategy-content h4 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0 0 0.75rem;
}

.strategy-content p {
    color: #4b5563;
    line-height: 1.6;
    margin: 0;
}

.chart-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #f1f3f4;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.chart-card h3 {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0 0 1rem;
}

.alert-card {
    padding: 1.25rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    font-weight: 500;
}

.alert-card.error {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
}

.alert-card.warning {
    background: #fffbeb;
    color: #d97706;
    border: 1px solid #fed7aa;
}

.progress-status {
    background: white;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    border: 1px solid #f1f3f4;
    color: #4b5563;
    font-weight: 500;
    text-align: center;
}

.progress-status.success {
    background: #f0fdf4;
    color: #16a34a;
    border-color: #bbf7d0;
}

.stButton > button {
    background: #1a1a1a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}

.stButton > button:hover {
    background: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #9CCEF5 0%, #F4E2D8 100%) !important;
}

.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
}

.stProgress > div > div > div {
    background: #2563eb !important;
}

.stPlotlyChart {
    border-radius: 8px;
    overflow: hidden;
}

div[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #f1f3f4;
}

div[data-testid="stSidebar"] .stTextInput input {
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}
</style>
"""