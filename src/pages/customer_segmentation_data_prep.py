import streamlit as st
import pandas as pd
import numpy as np
from src.core.data_prep import ProfessionalDataPreparation
from src.shared.ui_utils import load_shared_styles
load_shared_styles()


class CustomerSegmentationDataPrep:
    def __init__(self):
        self.setup_session_state()
        if 'data_prep_processor' not in st.session_state or st.session_state.data_prep_processor is None:
            st.session_state.data_prep_processor = ProfessionalDataPreparation()
        self.data_prep = st.session_state.data_prep_processor

    def setup_session_state(self):
        defaults = {
            'data_prep_step': 1,
            'uploaded_data': None,
            'step_1_1_results': None,
            'step_1_2_results': None,
            'step_1_3_results': None,
            'step_1_4_results': None,
            'phase_1_completed': False,
            'final_data': None,
            'reduced_features': None,
            'data_prep_processor': None,
            'step_1_1_completed': False,
            'step_1_2_completed': False,
            'step_1_3_completed': False,
            'step_1_4_completed': False,
            'step_1_1_processing': False,
            'step_1_2_processing': False,
            'step_1_3_processing': False,
            'step_1_4_processing': False,
            'results_displayed': {
                'step_1_1': False,
                'step_1_2': False,
                'step_1_3': False,
                'step_1_4': False
            },
            'render_lock': False
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render(self):
        if st.session_state.get('render_lock', False):
            return
        
        # Updated CSS with button gradient #B0E1FA to #FBF0EA and no shadow
        st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Target Streamlit's container to ensure glass effect applies */
        .glass-container, .stApp [data-testid="stElementContainer"] .glass-container {
            background: rgba(255, 255, 255, 0.25) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            padding: 2rem !important;
            margin: 1rem 0 !important;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
            display: block;
            width: 100% !important;
        }
        
        .step-card {
            box-shadow: 0 1px 10px 0 rgba(31, 38, 135, 0.37) !important;
        }

        .progress-container {
            display: none !important;
        }
        .glass-container {
            display: none !important;
        }
                    
        /* Ensure content inside glass-container is not affected by Streamlit's default styling */
        .glass-container > * {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .step-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(5px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }
        
        .step-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
            margin: 0.2rem;
        }
        
        .progress-container {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(8px);
            border-radius: 25px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #B0E1FA 0%, #FBF0EA 100%) !important;
            color: #2c3e50 !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 0.5rem 2rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #9CCEF5 0%, #F4E2D8 100%) !important;
            transform: translateY(-2px) !important;
        }
        
        .success-btn > button {
            background: linear-gradient(135deg, #B0E1FA 0%, #FBF0EA 100%) !important;
            color: #2c3e50 !important;
        }
        
        .secondary-btn > button {
            background: linear-gradient(135deg, #B0E1FA 0%, #FBF0EA 100%) !important;
            color: #2c3e50 !important;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 600;
        }
        
        .stMetric {
            background: rgba(255,255,255,0.8);
            padding: 1rem;
            border-radius: 10px;
            backdrop-filter: blur(5px);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Wrap the main content in glass-container
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.title("Chuẩn Bị Dữ Liệu")
        st.markdown("Chuẩn hóa dữ liệu khách hàng để phân tích", unsafe_allow_html=True)
        self._render_progress()
        st.markdown('</div>', unsafe_allow_html=True)
        
        current_step = st.session_state.data_prep_step
        if current_step == 1:
            self._render_step_1_1_exploration()
        elif current_step == 2:
            self._render_step_1_2_quality()
        elif current_step == 3:
            self._render_step_1_3_cleaning()
        elif current_step == 4:
            self._render_step_1_4_engineering()
        else:
            st.session_state.data_prep_step = 1
            st.rerun()
    
    def _render_progress(self):
        current = st.session_state.data_prep_step
        progress = ((current - 1) / 3)
        
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.progress(progress)
        
        cols = st.columns(4)
        steps = [
            (1, "Khám phá dữ liệu", "Phân tích cấu trúc"),
            (2, "Đánh giá chất lượng", "Kiểm tra độ tin cậy"),
            (3, "Làm sạch dữ liệu", "Chuẩn hóa dữ liệu"),
            (4, "Kỹ thuật hóa", "Tối ưu hóa tính năng")
        ]
        
        for col, (step_num, title, desc) in zip(cols, steps):
            with col:
                completed = st.session_state.get(f'step_1_{step_num}_completed', False)
                status = "✓" if completed else "●" if step_num == current else "○"
                
                st.markdown(f'''
                <div class="step-card">
                    <h4>{status} {title}</h4>
                    <p style="color: #7f8c8d; font-size: 0.9em;">{desc}</p>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_step_1_1_exploration(self):
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("Khám Phá Dữ Liệu")
        
        uploaded_file = st.file_uploader("Tải lên dữ liệu", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            with st.spinner("Đang xử lý..."):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    if df is None or len(df) == 0:
                        st.error("Dữ liệu không hợp lệ")
                        st.markdown('</div>', unsafe_allow_html=True)
                        return
                    
                    if self.data_prep is None:
                        self.data_prep = ProfessionalDataPreparation()
                        st.session_state.data_prep_processor = self.data_prep
                    
                    st.session_state.uploaded_data = df
                    self.data_prep.set_data(df)
                    step_1_1_results = self.data_prep.step_1_1_data_exploration()
                    st.session_state.step_1_1_results = step_1_1_results
                    st.session_state.step_1_1_completed = True
                    
                    self._display_exploration_results(df, step_1_1_results)
                    
                    st.markdown('<div class="success-btn">', unsafe_allow_html=True)
                    if st.button("Tiếp tục đến Đánh giá chất lượng"):
                        st.session_state.data_prep_step = 2
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_step_1_2_quality(self):
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("Đánh Giá Chất Lượng")
        
        if not st.session_state.step_1_1_completed:
            st.error("Hoàn thành bước khám phá dữ liệu trước")
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            if st.button("Quay lại"):
                st.session_state.data_prep_step = 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        if st.session_state.get('step_1_2_processing', False):
            with st.spinner("Đang đánh giá..."):
                try:
                    if self.data_prep.original_data is None and st.session_state.uploaded_data is not None:
                        self.data_prep.set_data(st.session_state.uploaded_data)
                    
                    step_1_2_results = self.data_prep.step_1_2_quality_assessment()
                    st.session_state.step_1_2_results = step_1_2_results
                    st.session_state.step_1_2_completed = True
                    st.session_state.step_1_2_processing = False
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.step_1_2_processing = False
                    st.error(f"Lỗi: {str(e)}")
        
        elif not st.session_state.get('step_1_2_completed', False):
            if st.button("Bắt đầu đánh giá"):
                st.session_state.step_1_2_processing = True
                st.rerun()
        
        if st.session_state.step_1_2_results and st.session_state.step_1_2_completed:
            self._display_quality_results(st.session_state.step_1_2_results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
                if st.button("Quay lại"):
                    st.session_state.data_prep_step = 1
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="success-btn">', unsafe_allow_html=True)
                if st.button("Tiếp tục"):
                    st.session_state.data_prep_step = 3
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_step_1_3_cleaning(self):
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("Làm Sạch Dữ Liệu")
        
        if not st.session_state.step_1_2_completed:
            st.error("Hoàn thành bước đánh giá chất lượng trước")
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            if st.button("Quay lại"):
                st.session_state.data_prep_step = 2
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        if st.session_state.get('step_1_3_processing', False):
            with st.spinner("Đang làm sạch..."):
                try:
                    if self.data_prep.original_data is None:
                        self.data_prep.set_data(st.session_state.uploaded_data)
                    
                    step_1_3_results = self.data_prep.step_1_3_cleaning_transformation()
                    st.session_state.step_1_3_results = step_1_3_results
                    st.session_state.step_1_3_completed = True
                    st.session_state.step_1_3_processing = False
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.step_1_3_processing = False
                    st.error(f"Lỗi: {str(e)}")
        
        elif not st.session_state.get('step_1_3_completed', False):
            if st.button("Bắt đầu làm sạch"):
                st.session_state.step_1_3_processing = True
                st.rerun()
        
        if st.session_state.step_1_3_results and st.session_state.step_1_3_completed:
            self._display_cleaning_results(st.session_state.step_1_3_results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
                if st.button("Qu Knay lại"):
                    st.session_state.data_prep_step = 2
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="success-btn">', unsafe_allow_html=True)
                if st.button("Tiếp tục"):
                    st.session_state.data_prep_step = 4
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_step_1_4_engineering(self):
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("Kỹ Thuật Hóa Tính Năng")
        
        if not st.session_state.step_1_3_completed:
            st.error("Hoàn thành bước làm sạch dữ liệu trước")
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            if st.button("Quay lại"):
                st.session_state.data_prep_step = 3
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        if st.session_state.get('step_1_4_processing', False):
            with st.spinner("Đang kỹ thuật hóa..."):
                try:
                    step_1_4_results = self.data_prep.step_1_4_feature_engineering()
                    st.session_state.step_1_4_results = step_1_4_results
                    
                    if hasattr(self.data_prep, 'processed_data') and self.data_prep.processed_data is not None:
                        st.session_state.final_data = self.data_prep.processed_data
                        st.session_state.reduced_features = self.data_prep.processed_data
                        st.session_state.phase_1_completed = True
                        st.session_state.step_1_4_completed = True
                        st.session_state.workflow_progress = st.session_state.get('workflow_progress', {})
                        st.session_state.workflow_progress['data_prep'] = 100
                        st.session_state.data_prep_processor = self.data_prep
                    
                    st.session_state.step_1_4_processing = False
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.step_1_4_processing = False
                    st.error(f"Lỗi: {str(e)}")
        
        elif not st.session_state.get('step_1_4_completed', False):
            if st.button("Bắt đầu kỹ thuật hóa"):
                st.session_state.step_1_4_processing = True
                st.rerun()
        
        if st.session_state.step_1_4_results and st.session_state.step_1_4_completed:
            self._display_engineering_results(st.session_state.step_1_4_results)
            if st.session_state.phase_1_completed:
                self._show_phase_1_completion()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _display_exploration_results(self, df, results):
        st.markdown("### Kết Quả Khám Phá")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{results['shape'][0]:,}</h3>
                <p>Tổng Mẫu</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{results['shape'][1]}</h3>
                <p>Tổng Tính Năng</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            numeric_count = len(df.select_dtypes(include=[np.number]).columns)
            st.markdown(f'''
            <div class="metric-card">
                <h3>{numeric_count}</h3>
                <p>Tính Năng Số</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            missing_pct = sum(results['missing_summary'].values()) / (results['shape'][0] * results['shape'][1]) * 100
            st.markdown(f'''
            <div class="metric-card">
                <h3>{missing_pct:.1f}%</h3>
                <p>Tỷ Lệ Thiếu</p>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("### Xem Trước Dữ Liệu")
        st.dataframe(df.head(10), use_container_width=True)
    
    def _display_quality_results(self, results):
        st.markdown("### Kết Quả Đánh Giá")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            quality_score = results.get('overall_quality_score', 50)
            status = "Tuyệt vời" if quality_score >= 80 else "Tốt" if quality_score >= 60 else "Khá" if quality_score >= 40 else "Kém"
            st.markdown(f'''
            <div class="metric-card">
                <h3>{quality_score}/100</h3>
                <p>Điểm Chất Lượng</p>
                <small>{status}</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            hopkins = results.get('hopkins_statistic', 0.5)
            clustering_ready = results.get('clustering_readiness', 'Không xác định')
            st.markdown(f'''
            <div class="metric-card">
                <h3>{hopkins:.3f}</h3>
                <p>Hopkins Statistic</p>
                <small>{clustering_ready}</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            high_missing = len(results.get('high_missing_columns', []))
            st.markdown(f'''
            <div class="metric-card">
                <h3>{high_missing}</h3>
                <p>Cột Thiếu Nhiều</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            outlier_analysis = results.get('outlier_analysis', {})
            total_outliers = sum(info.get('iqr_outliers', 0) for info in outlier_analysis.values())
            st.markdown(f'''
            <div class="metric-card">
                <h3>{total_outliers}</h3>
                <p>Tổng Ngoại Lệ</p>
            </div>
            ''', unsafe_allow_html=True)
    
    def _display_cleaning_results(self, results):
        st.markdown("### Kết Quả Làm Sạch")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            duplicates = results.get('duplicates_removed', 0)
            st.markdown(f'''
            <div class="metric-card">
                <h3>{duplicates}</h3>
                <p>Trùng Lặp Xóa</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            numeric_imputed = results.get('missing_handling', {}).get('numeric_imputed', 0)
            st.markdown(f'''
            <div class="metric-card">
                <h3>{numeric_imputed}</h3>
                <p>Số Điền</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            categorical_imputed = results.get('missing_handling', {}).get('categorical_imputed', 0)
            st.markdown(f'''
            <div class="metric-card">
                <h3>{categorical_imputed}</h3>
                <p>Danh Mục Điền</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            scaling_applied = results.get('scaling_applied', False)
            st.markdown(f'''
            <div class="metric-card">
                <h3>{"Hoàn tất" if scaling_applied else "Chưa"}</h3>
                <p>Chuẩn Hóa</p>
            </div>
            ''', unsafe_allow_html=True)
        
        final_shape = results.get('final_shape', (0, 0))
        st.success(f"Kích thước cuối: {final_shape[0]:,} hàng × {final_shape[1]} tính năng")
    
    def _display_engineering_results(self, results):
        st.markdown("### Kết Quả Kỹ Thuật Hóa")
        
        if 'feature_selection' in results:
            fs_results = results['feature_selection']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                low_var_removed = fs_results.get('low_variance_removed', 0)
                st.markdown(f'''
                <div class="metric-card">
                    <h3>{low_var_removed}</h3>
                    <p>Phương Sai Thấp Xóa</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                high_corr_removed = fs_results.get('high_correlation_removed', 0)
                st.markdown(f'''
                <div class="metric-card">
                    <h3>{high_corr_removed}</h3>
                    <p>Tương Quan Cao Xóa</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                features_kept = fs_results.get('rf_feature_selection', {}).get('features_kept', 'N/A')
                st.markdown(f'''
                <div class="metric-card">
                    <h3>{features_kept}</h3>
                    <p>Tính Năng Giữ</p>
                </div>
                ''', unsafe_allow_html=True)
        
        if 'final_validation' in results:
            final_val = results['final_validation']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>{final_val.get('final_features', 0)}</h3>
                    <p>Tính Năng Cuối</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>{final_val.get('final_samples', 0):,}</h3>
                    <p>Mẫu Cuối</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                clustering_ready = final_val.get('data_ready_for_clustering', False)
                st.markdown(f'''
                <div class="metric-card">
                    <h3>{"Hoàn tất" if clustering_ready else "Chưa"}</h3>
                    <p>Sẵn Sàng Phân Cụm</p>
                </div>
                ''', unsafe_allow_html=True)
    
    def _show_phase_1_completion(self):
        st.success("Giai đoạn 1 hoàn tất! Dữ liệu sẵn sàng để phân cụm")
        
        st.markdown('<div class="success-btn">', unsafe_allow_html=True)
        if st.button("Tiếp tục đến Phân cụm"):
            st.session_state.current_page = "Clustering"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)