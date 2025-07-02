import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import io
from src.core.clustering import ProfessionalClustering
from src.core.customer_insights import create_dimensionality_visualizer
import traceback
from src.shared.ui_utils import load_shared_styles
load_shared_styles()


# Modern glassmorphism CSS styling
st.markdown("""
<style>
:root {
    --glass-bg: rgba(255, 255, 255, 0.85);
    --glass-border: rgba(255, 255, 255, 0.3);
    --glass-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
    --primary-color: #2c3e50;
    --secondary-color: #7f8c8d;
    --gradient-start: #B0E1FA;
    --gradient-end: #FBF0EA;
    --gradient-hover-start: #9CCEF5;
    --gradient-hover-end: #F4E2D8;
    --error-color: #ef476f;
    --warning-color: #ffd166;
    --success-color: #06d6a0;
}

body {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    color: var(--primary-color);
}

.glass-container {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 20px !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: var(--glass-shadow) !important;
    padding: 2rem !important;
    margin: 1rem 0 !important;
    width: 100% !important;
}

.step-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 0.5rem 0 !important;
    border: 1px solid var(--glass-border) !important;
    transition: all 0.3s ease !important;
}

.step-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 28px rgba(0,0,0,0.08) !important;
}

.metric-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 14px !important;
    padding: 1.5rem 1rem !important;
    text-align: center !important;
    border: 1px solid var(--glass-border) !important;
    margin: 0.5rem 0 !important;
}

.progress-container {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    margin: 1.5rem 0 !important;
    border: 1px solid var(--glass-border) !important;
}

.status-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    border: 1px solid var(--glass-border) !important;
}

.error-card {
    border-left: 4px solid var(--error-color) !important;
}

.warning-card {
    border-left: 4px solid var(--warning-color) !important;
}

.success-card {
    border-left: 4px solid var(--success-color) !important;
}

.chart-container {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    border: 1px solid var(--glass-border) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%) !important;
    color: var(--primary-color) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--gradient-hover-start) 0%, var(--gradient-hover-end) 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.1) !important;
}

h1, h2, h3, h4 {
    color: var(--primary-color);
    font-weight: 700;
    margin-top: 0;
}

p {
    color: var(--primary-color);
    opacity: 0.9;
    line-height: 1.6;
    margin-bottom: 0;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--primary-color);
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 1rem;
    color: var(--secondary-color);
    margin-bottom: 0.5rem;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--gradient-start) 0%, var(--gradient-end) 100%) !important;
}

.stPlotlyChart {
    border-radius: 12px !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

class CustomerSegmentationClustering:
    def __init__(self):
        self.setup_session_state()
    
    def setup_session_state(self):
        defaults = {
            'phase_2_completed': False,
            'current_step': 2.1,
            'cluster_labels': None,
            'final_model_results': None,
            'optimal_k': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render(self):
        st.markdown("""
        <div class="glass-container">
            <div class="step-card">
                <h1>Phân Cụm Khách Hàng</h1>
                <p>Phân tích dữ liệu để xác định các nhóm khách hàng tương đồng</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not self._check_prerequisites():
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        steps = {
            2.1: "Kiểm tra dữ liệu",
            2.2: "Xác định số nhóm tối ưu", 
            2.3: "Thực hiện phân cụm",
            2.4: "Xác nhận kết quả"
        }
        
        current = st.session_state.current_step
        progress = ((current - 2.1) / 0.3)
        
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.progress(min(progress, 1.0))
        
        cols = st.columns(4)
        for col, (step_num, title) in zip(cols, steps.items()):
            with col:
                completed = step_num < current or (step_num == current and st.session_state.get(f'step_{step_num}_completed', False))
                status = "✓" if completed else "●" if step_num == current else "○"
                st.markdown(f'''
                <div class="step-card">
                    <h4>{status} {title}</h4>
                    <p style="color: var(--secondary-color);">Bước {step_num}</p>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        try:
            if st.session_state.current_step == 2.1:
                self._render_step_2_1()
            elif st.session_state.current_step == 2.2:
                self._render_step_2_2()
            elif st.session_state.current_step == 2.3:
                self._render_step_2_3()
            elif st.session_state.current_step == 2.4:
                self._render_step_2_4()
        except Exception as e:
            st.markdown(f"""
            <div class="glass-container">
                <div class="status-card error-card">
                    <h3>Lỗi trong quá trình phân cụm</h3>
                    <p>{str(e)}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _check_prerequisites(self):
        data_keys = ['reduced_features', 'final_data', 'uploaded_data']
        data_available = any(key in st.session_state and st.session_state[key] is not None and isinstance(st.session_state[key], (pd.DataFrame, np.ndarray)) for key in data_keys)
        
        if not st.session_state.get('phase_1_completed', False) or not data_available:
            st.markdown("""
            <div class="glass-container">
                <div class="status-card error-card">
                    <h3>Yêu cầu</h3>
                    <p>Vui lòng hoàn thành bước Chuẩn bị dữ liệu trước và đảm bảo dữ liệu hợp lệ.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Quay lại Chuẩn bị dữ liệu", key="back_to_data_prep"):
                st.session_state.current_page = "DataPrep"
                st.rerun()
            return False
        return True
    
    def _render_step_2_1(self):
        st.markdown("""
        <div class="glass-container">
            <div class="step-card">
                <h2>Kiểm Tra Dữ Liệu</h2>
                <p>Đánh giá xem dữ liệu có phù hợp để phân cụm hay không.</p>
            </div>
        """, unsafe_allow_html=True)
        
        data = self._get_data()
        if data is None:
            st.markdown("""
            <div class="status-card error-card">
                <p>Không tìm thấy dữ liệu hợp lệ để phân cụm.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        try:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
            
            clustering = ProfessionalClustering()
            clustering.set_data(data)
            readiness_results = clustering.step_2_1_clustering_readiness_check()
            
            hopkins_stat = readiness_results['hopkins_statistic']
            readiness_score = readiness_results['readiness_score']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Điểm sẵn sàng</div>
                    <div class="metric-value">{readiness_score:.1f}/100</div>
                    <p>Trên 70 là lý tưởng</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Hopkins Statistic</div>
                    <div class="metric-value">{hopkins_stat:.3f}</div>
                    <p>>0.5 cho thấy dữ liệu có xu hướng phân cụm</p>
                </div>
                """, unsafe_allow_html=True)
            
            visualizer = create_dimensionality_visualizer(data, None)
            if visualizer is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="chart-container">
                        <h3>Biểu đồ 2D</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        fig_2d = visualizer.create_2d_plot()
                        if fig_2d:
                            st.plotly_chart(fig_2d, use_container_width=True)
                    except:
                        pass
                
                with col2:
                    st.markdown("""
                    <div class="chart-container">
                        <h3>Biểu đồ 3D</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        fig_3d = visualizer.create_3d_plot()
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                    except:
                        pass
            
            if readiness_score < 50:
                st.markdown("""
                <div class="status-card warning-card">
                    <p>Dữ liệu có thể không phù hợp để phân cụm. Cân nhắc quay lại bước Chuẩn bị dữ liệu.</p>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("Tiếp tục đến Xác định số nhóm", key="step_2_1_next"):
                st.session_state.current_step = 2.2
                st.session_state.clustering_instance = clustering
                st.rerun()
                
        except Exception as e:
            st.markdown(f"""
            <div class="status-card error-card">
                <p>Lỗi khi kiểm tra dữ liệu: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_step_2_2(self):
        st.markdown("""
        <div class="glass-container">
            <div class="step-card">
                <h2>Xác Định Số Nhóm Tối Ưu</h2>
                <p>Đánh giá số lượng nhóm khách hàng phù hợp nhất.</p>
            </div>
        """, unsafe_allow_html=True)
        
        data = self._get_data()
        if data is None:
            st.markdown("""
            <div class="status-card error-card">
                <p>Không tìm thấy dữ liệu hợp lệ để phân cụm.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        try:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            
            clustering = st.session_state.get('clustering_instance')
            if clustering is None:
                clustering = ProfessionalClustering()
                clustering.set_data(data)
                st.session_state.clustering_instance = clustering
            
            if not clustering.step_2_1_completed:
                clustering.step_2_1_clustering_readiness_check()
            
            with st.spinner("Đang tính toán số nhóm tối ưu..."):
                k_selection_results = clustering.step_2_2_optimal_k_selection()
            
            st.markdown("""
            <div class="chart-container">
                <h3>Biểu đồ đánh giá số nhóm tối ưu</h3>
            </div>
            """, unsafe_allow_html=True)
            
            fig = go.Figure()
            for method, result in k_selection_results['methods'].items():
                if method == 'elbow':
                    scores = result['wcss']
                elif method in ['silhouette', 'calinski_harabasz']:
                    scores = result['scores']
                elif method == 'gap':
                    scores = result['gaps']
                fig.add_trace(go.Scatter(x=k_selection_results['k_values'], y=scores, mode='lines+markers', name=method))
            
            fig.update_layout(
                title="Đánh giá số nhóm tối ưu",
                xaxis_title="Số nhóm",
                yaxis_title="Điểm số",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            optimal_k = k_selection_results['consensus']['optimal_k']
            confidence = k_selection_results['consensus']['confidence'] * 100
            
            st.markdown(f"""
            <div class="status-card success-card">
                <div class="metric-label">Số nhóm đề xuất</div>
                <div class="metric-value">{optimal_k}</div>
                <p>Độ tin cậy: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Quay lại Kiểm tra dữ liệu", key="step_2_2_back"):
                    st.session_state.current_step = 2.1
                    st.rerun()
            with col2:
                if st.button("Tiếp tục phân cụm", key="step_2_2_next"):
                    st.session_state.optimal_k = optimal_k
                    st.session_state.clustering_instance = clustering
                    st.session_state.current_step = 2.3
                    st.rerun()
                    
        except Exception as e:
            st.markdown(f"""
            <div class="status-card error-card">
                <p>Lỗi khi xác định số nhóm: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_step_2_3(self):
        st.markdown("""
        <div class="glass-container">
            <div class="step-card">
                <h2>Thực Hiện Phân Cụm</h2>
                <p>Áp dụng thuật toán K-means để phân cụm khách hàng.</p>
            </div>
        """, unsafe_allow_html=True)
        
        data = self._get_data()
        if data is None:
            st.markdown("""
            <div class="status-card error-card">
                <p>Không tìm thấy dữ liệu hợp lệ để phân cụm.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        try:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
            
            clustering = st.session_state.get('clustering_instance')
            if clustering is None:
                clustering = ProfessionalClustering()
                clustering.set_data(data)
                st.session_state.clustering_instance = clustering
            
            if not clustering.step_2_1_completed:
                clustering.step_2_1_clustering_readiness_check()
            if not clustering.step_2_2_completed:
                clustering.step_2_2_optimal_k_selection()
            
            if st.session_state.optimal_k is None:
                clustering.step_2_2_optimal_k_selection()
                st.session_state.optimal_k = clustering.optimal_k
            
            with st.spinner("Đang thực hiện phân cụm..."):
                kmeans_results = clustering.step_2_3_kmeans_implementation()
            
            labels = kmeans_results['labels']
            st.session_state.cluster_labels = labels
            st.session_state.final_model_results = kmeans_results
            
            st.markdown("""
            <div class="status-card success-card">
                <h3>Kết quả phân cụm</h3>
                <p>Phân cụm đã hoàn thành thành công</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="chart-container">
                    <h3>Biểu đồ 2D</h3>
                </div>
                """, unsafe_allow_html=True)
                fig_2d = px.scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], color=labels, title="Phân cụm 2D")
                fig_2d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_2d, use_container_width=True)
                
            with col2:
                st.markdown("""
                <div class="chart-container">
                    <h3>Biểu đồ 3D</h3>
                </div>
                """, unsafe_allow_html=True)
                z_data = data.iloc[:, 2] if data.shape[1] > 2 else np.zeros(len(data))
                fig_3d = px.scatter_3d(x=data.iloc[:, 0], y=data.iloc[:, 1], z=z_data, color=labels, title="Phân cụm 3D")
                fig_3d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_3d, use_container_width=True)
            
            buffer = io.BytesIO()
            joblib.dump(kmeans_results['model'], buffer)
            buffer.seek(0)
            st.download_button("Tải xuống mô hình phân cụm", buffer, "kmeans_model.pkl", mime="application/octet-stream")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Quay lại Xác định số nhóm", key="step_2_3_back"):
                    st.session_state.current_step = 2.2
                    st.rerun()
            with col2:
                if st.button("Tiếp tục xác nhận kết quả", key="step_2_3_next"):
                    st.session_state.current_step = 2.4
                    st.session_state.clustering_instance = clustering
                    st.rerun()
                    
        except Exception as e:
            st.markdown(f"""
            <div class="status-card error-card">
                <p>Lỗi khi phân cụm: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_step_2_4(self):
        st.markdown("""
        <div class="glass-container">
            <div class="step-card">
                <h2>Xác Nhận Kết Quả Phân Cụm</h2>
                <p>Đánh giá chất lượng các nhóm khách hàng đã phân cụm.</p>
            </div>
        """, unsafe_allow_html=True)
        
        data = self._get_data()
        labels = st.session_state.get('cluster_labels')
        
        if data is None or labels is None:
            st.markdown("""
            <div class="status-card error-card">
                <p>Không tìm thấy dữ liệu hoặc nhãn phân cụm hợp lệ.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        try:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            
            clustering = st.session_state.get('clustering_instance')
            if clustering is None:
                clustering = ProfessionalClustering()
                clustering.set_data(data)
                st.session_state.clustering_instance = clustering
            
            if not clustering.step_2_1_completed:
                clustering.step_2_1_clustering_readiness_check()
            if not clustering.step_2_2_completed:
                clustering.step_2_2_optimal_k_selection()
            if not clustering.step_2_3_completed:
                clustering.step_2_3_kmeans_implementation()
            
            clustering.cluster_labels = labels
            validation_results = clustering.step_2_4_cluster_validation()
            
            internal_metrics = validation_results['internal_metrics']
            silhouette = internal_metrics.get('silhouette_score', -1)
            calinski = internal_metrics.get('calinski_harabasz_score', 0)
            davies = internal_metrics.get('davies_bouldin_score', float('inf'))
            overall_score = validation_results['overall']['score']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Silhouette Score</div>
                    <div class="metric-value">{silhouette:.3f}</div>
                    <p>Gần 1 là tốt</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Calinski-Harabasz</div>
                    <div class="metric-value">{calinski:.0f}</div>
                    <p>Càng cao càng tốt</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Davies-Bouldin</div>
                    <div class="metric-value">{davies:.3f}</div>
                    <p>Càng thấp càng tốt</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="status-card success-card">
                <div class="metric-label">Điểm tổng thể</div>
                <div class="metric-value">{overall_score:.0f}/100</div>
                <p>Chất lượng phân cụm tổng thể</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.final_model_results = {
                'model': clustering.final_model,
                'labels': labels,
                'metrics': {
                    'silhouette': silhouette,
                    'calinski': calinski,
                    'davies': davies,
                    'overall_score': overall_score
                }
            }
            st.session_state.phase_2_completed = True
            
            if overall_score < 50:
                st.markdown("""
                <div class="status-card warning-card">
                    <p>Chất lượng phân cụm thấp. Cân nhắc quay lại bước Chuẩn bị dữ liệu hoặc điều chỉnh số nhóm.</p>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Quay lại Thực hiện phân cụm", key="step_2_4_back"):
                    st.session_state.current_step = 2.3
                    st.rerun()
            with col2:
                if st.button("Tiếp tục đến Thông tin chi tiết", key="step_2_4_next"):
                    st.session_state.current_page = "Insights"
                    st.rerun()
                    
        except Exception as e:
            st.markdown(f"""
            <div class="status-card error-card">
                <p>Lỗi khi xác nhận kết quả: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _get_data(self):
        for key in ['reduced_features', 'final_data', 'uploaded_data']:
            if key in st.session_state and st.session_state[key] is not None and isinstance(st.session_state[key], (pd.DataFrame, np.ndarray)):
                return st.session_state[key]
        return None