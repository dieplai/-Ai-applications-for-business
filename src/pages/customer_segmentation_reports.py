"""
Reports & Export Page - Phase 4 Final Step
UI cho việc tạo và export báo cáo PDF với đầy đủ thông tin
"""

import streamlit as st
import time
from datetime import datetime
import io
import base64
from ..core.reports_export import create_pdf_report
import pandas as pd
import plotly.express as px
from src.shared.ui_utils import load_shared_styles
load_shared_styles()



class CustomerSegmentationReports:
    """Page cuối cùng - Tạo và export báo cáo hoàn chỉnh"""
    
    def __init__(self):
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state for reports"""
        if 'report_generated' not in st.session_state:
            st.session_state.report_generated = False
        if 'report_buffer' not in st.session_state:
            st.session_state.report_buffer = None
        if 'report_filename' not in st.session_state:
            st.session_state.report_filename = None

    def render(self):
        """Render reports & export page"""
        
        if not self._check_prerequisites():
            return
        
        # Modern Glass CSS styling
        st.markdown("""
        <style>
        .glass-container {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        .glass-header {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.1));
            backdrop-filter: blur(15px);
            border-radius: 25px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        }
        
        .glass-success {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.3), rgba(139, 195, 74, 0.2));
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(76, 175, 80, 0.3);
            padding: 2rem;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(76, 175, 80, 0.1);
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(8px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.15);
        }
        
        .glass-metric {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(5px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 1rem;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .progress-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(8px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(31, 38, 135, 0.3);
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .step-item {
            display: flex;
            align-items: center;
            margin: 0.8rem 0;
            padding: 0.8rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 700;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)

        self._render_completion_header()
        self._render_workflow_summary()
        self._render_report_preview()
        self._render_report_generation()

    def _check_prerequisites(self):
        """Check if all previous phases are completed"""
        
        required_data = {
            'final_model_results': 'Kết quả phân cụm',
            'cluster_profiles': 'Phân tích cluster',
            'ai_personas': 'AI Personas'
        }
        
        missing_data = []
        for key, description in required_data.items():
            if not st.session_state.get(key):
                missing_data.append(description)
        
        if missing_data:
            st.markdown('<div class="glass-header"><h1>Báo Cáo & Export</h1><p>Tạo báo cáo PDF hoàn chỉnh với tất cả insights</p></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.warning("Cần hoàn thành các bước trước để tạo báo cáo")
            st.markdown("**Dữ liệu còn thiếu:**")
            for item in missing_data:
                st.write(f"• {item}")
            st.info("**Hướng dẫn:** Vui lòng hoàn thành tất cả các bước từ Data Preparation đến Customer Insights")
            st.markdown('</div>', unsafe_allow_html=True)
            
            return False
        
        return True

    def _render_completion_header(self):
        """Render completion celebration header"""
        
        st.markdown('''
        <div class="glass-success">
            <h1>HOÀN THÀNH PHÂN TÍCH KHÁCH HÀNG</h1>
            <h2>Customer Insights Analysis Complete</h2>
            <p style="font-size: 1.2rem; margin-top: 1rem;">Bạn đã thành công phân tích và tạo insights cho dữ liệu khách hàng</p>
        </div>
        ''', unsafe_allow_html=True)

    def _render_workflow_summary(self):
        """Render summary of completed workflow"""
        
        st.subheader("Tổng Kết Quy Trình")
        
        workflow_steps = [
            ("Data Preparation", "Hoàn thành"),
            ("Feature Engineering", "Hoàn thành"), 
            ("Clustering Analysis", "Hoàn thành"),
            ("AI Customer Insights", "Hoàn thành"),
            ("Reports & Export", "Đang thực hiện")
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Các bước đã hoàn thành:**")
            for step, status in workflow_steps:
                status_icon = "✅" if status == "Hoàn thành" else "🔄"
                st.markdown(f'<div class="step-item">{status_icon} {step}: {status}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if st.session_state.get('cluster_profiles'):
                cluster_profiles = st.session_state.cluster_profiles
                total_customers = sum(profile['size_info']['count'] for profile in cluster_profiles.values())
                num_clusters = len(cluster_profiles)
                
                st.markdown("**Kết quả chính:**")
                
                st.markdown(f'''
                <div class="glass-metric">
                    <div class="metric-value">{total_customers:,}</div>
                    <div class="metric-label">Tổng khách hàng phân tích</div>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="glass-metric">
                    <div class="metric-value">{num_clusters}</div>
                    <div class="metric-label">Số phân khúc xác định</div>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="glass-metric">
                    <div class="metric-value">{len(st.session_state.get("ai_personas", {}))}</div>
                    <div class="metric-label">AI Personas được tạo</div>
                </div>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    def _render_report_preview(self):
        """Render preview of what will be in the report"""
        
        st.subheader("Nội Dung Báo Cáo PDF")
        
        preview_content = [
            "Tóm tắt điều hành - Key metrics và insights chính",
            "Tổng quan dữ liệu - Thông tin dataset và phương pháp phân tích",
            "Phân tích chi tiết các phân khúc - Đặc điểm từng cluster",
            "Biểu đồ trực quan - Visual clusters và distributions",
            "AI Marketing Personas - Personas được tạo bởi AI",
            "Chiến lược Marketing Campaigns - Campaign plans cụ thể",
            "Khuyến nghị chiến lược - Actionable recommendations"
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Báo cáo sẽ bao gồm:**")
            for item in preview_content[:4]:
                st.markdown(f"• {item}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Và thêm:**")
            for item in preview_content[4:]:
                st.markdown(f"• {item}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="glass-container">
            <h4>Thông số kỹ thuật báo cáo:</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div>✅ Format: PDF Professional</div>
                <div>✅ Font: Hỗ trợ đầy đủ tiếng Việt</div>
                <div>✅ Charts: High-resolution visualizations</div>
                <div>✅ Layout: Business-ready formatting</div>
                <div>✅ Size: Optimized for sharing</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    def _render_report_generation(self):
        """Render report generation section"""
        
        st.subheader("Tạo Báo Cáo PDF")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if not st.session_state.report_generated:
                if st.button("Tạo Báo Cáo PDF Hoàn Chỉnh", 
                           type="primary", 
                           use_container_width=True,
                           key="generate_report"):
                    self._generate_pdf_report()
            
            else:
                if st.session_state.report_buffer:
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"customer_insights_report_{current_time}.pdf"
                    
                    st.download_button(
                        label="Tải Xuống Báo Cáo PDF",
                        data=st.session_state.report_buffer.getvalue(),
                        file_name=filename,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                    
                    st.success("Báo cáo đã được tạo thành công!")
                    
                    if st.button("Tạo Báo Cáo Mới", 
                               type="secondary", 
                               use_container_width=True):
                        st.session_state.report_generated = False
                        st.session_state.report_buffer = None
                        st.rerun()
        
        if st.session_state.report_generated:
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('''
                <div class="glass-card">
                    <h4>Chia sẻ báo cáo</h4>
                    <p>Báo cáo đã được tạo thành công và có thể:</p>
                    <ul>
                        <li>Chia sẻ với team marketing</li>
                        <li>Trình bày với management</li>
                        <li>Sử dụng cho planning meetings</li>
                        <li>Archive cho tham khảo sau này</li>
                    </ul>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown('''
                <div class="glass-card">
                    <h4>Cập nhật báo cáo</h4>
                    <p>Để có báo cáo mới nhất:</p>
                    <ul>
                        <li>Upload dữ liệu mới</li>
                        <li>Chạy lại toàn bộ workflow</li>
                        <li>Tạo báo cáo mới với insights cập nhật</li>
                        <li>So sánh với báo cáo trước đó</li>
                    </ul>
                </div>
                ''', unsafe_allow_html=True)
        
        if not st.session_state.report_generated:
            self._show_quick_insights()

    def _generate_pdf_report(self):
        """Generate the actual PDF report"""
        
        try:
            st.markdown('<div class="progress-section">', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Thu thập dữ liệu...")
            progress_bar.progress(20)
            time.sleep(1)
            
            final_model_results = st.session_state.get('final_model_results')
            cluster_profiles = st.session_state.get('cluster_profiles')
            ai_personas = st.session_state.get('ai_personas')
            ai_campaigns = st.session_state.get('ai_campaigns')
            
            status_text.text("Tạo biểu đồ...")
            progress_bar.progress(40)
            time.sleep(1)
            
            status_text.text("Tạo file PDF...")
            progress_bar.progress(60)
            time.sleep(1)
            
            pdf_buffer = create_pdf_report(
                final_model_results,
                cluster_profiles, 
                ai_personas,
                ai_campaigns
            )
            
            status_text.text("Hoàn tất báo cáo...")
            progress_bar.progress(100)
            time.sleep(1)
            
            st.session_state.report_buffer = pdf_buffer
            st.session_state.report_generated = True
            
            progress_bar.empty()
            status_text.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.success("Báo cáo PDF đã được tạo thành công!")
            st.balloons()
            
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"Lỗi khi tạo báo cáo: {str(e)}")
            st.info("Vui lòng thử lại hoặc kiểm tra dữ liệu đầu vào")

    def _show_quick_insights(self):
        """Show quick insights while user waits"""
        
        st.markdown("---")
        st.subheader("Quick Insights")
        
        if st.session_state.get('cluster_profiles'):
            cluster_profiles = st.session_state.cluster_profiles
            
            cluster_data = []
            for cluster_key, profile in cluster_profiles.items():
                cluster_data.append({
                    'Cluster': f"Phân khúc {profile['cluster_id']}",
                    'Customers': profile['size_info']['count'],
                    'Percentage': profile['size_info']['percentage']
                })
            
            df = pd.DataFrame(cluster_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig_pie = px.pie(
                    df, 
                    values='Customers', 
                    names='Cluster',
                    title="Phân bố khách hàng theo phân khúc"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig_bar = px.bar(
                    df,
                    x='Cluster', 
                    y='Customers',
                    title="Số lượng khách hàng mỗi phân khúc",
                    color='Customers',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


def create_reports_export_page():
    """Factory function to create reports export page"""
    return CustomerSegmentationReports()