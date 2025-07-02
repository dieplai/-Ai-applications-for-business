import streamlit as st
import plotly.graph_objects as go
from src.core.dashboard import DashboardAnalyzer
from src.shared.ui_utils import load_shared_styles
load_shared_styles()


class CustomerSegmentationDashboard:
    """Dashboard for customer segmentation insights"""
    
    def __init__(self):
        try:
            self.analyzer = DashboardAnalyzer()
        except Exception:
            self.analyzer = None  # Fallback if core module is missing
    
    def render(self):
        """Render the main dashboard"""
        # Add custom CSS for modern glass design
        st.markdown(self._get_custom_css(), unsafe_allow_html=True)
        
        st.markdown(
            """
            <div class="header">
                <h1>Bảng Điều Khiển Phân Tích Khách Hàng</h1>
                <p>Hiển thị các chỉ số chính và xu hướng khách hàng</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        self._render_kpis()
        self._render_main_content()
        self._render_additional_insights()
    
    def _get_custom_css(self):
        """Return custom CSS for modern glass design"""
        return """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        [data-testid="stPlotlyChart"] > div {
            /* Style như .chart-container */
            background: rgba(255,255,255,0.4);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        }

        .chart-container{
        display: none
        }

        
        .header {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .header p {
            color: #5a6c7d;
            font-size: 1.2rem;
            margin: 0;
            font-weight: 400;
        }
        
        .subheader {
            color: #2c3e50;
            font-size: 1.4rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            padding-left: 1rem;
            border-left: 4px solid #3498db;
        }
        
        .metric {
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .metric:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        }
        
        .metric h3 {
            color: #34495e;
            font-size: 0.9rem;
            font-weight: 500;
            margin: 0 0 0.5rem 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric p {
            color: #2c3e50;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }
        
        .metric small {
            color: #27ae60;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.35);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 14px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        }
        
        .card h3 {
            color: #2c3e50;
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
        }
        
        .card p {
            color: #5a6c7d;
            font-size: 0.9rem;
            margin: 0.3rem 0;
            line-height: 1.4;
        }
        
        .insight-box {
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 18px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .stat-item {
            background: rgba(255, 255, 255, 0.35);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        }
        
        .stat-item h4 {
            color: #2c3e50;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 0 0 0.5rem 0;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .stat-item .value {
            color: #3498db;
            font-size: 1.4rem;
            font-weight: 700;
        }
        
        .trend-positive {
            color: #27ae60 !important;
        }
        
        .trend-negative {
            color: #e74c3c !important;
        }
        </style>
        """
    
    def _render_kpis(self):
        """Render key performance indicators"""
        data = st.session_state.get('uploaded_data')
        total_customers = f"{len(data):,}" if data is not None else "12,847"
        
        st.markdown("<div class='subheader'>Chỉ Số Chính</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric">
                    <h3>Doanh Thu Hàng Tháng</h3>
                    <p>$487.2K</p>
                    <small class="trend-positive">↗ Tăng 15.3%</small>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric">
                    <h3>Tổng Số Khách Hàng</h3>
                    <p>{total_customers}</p>
                    <small class="trend-positive">↗ Tăng 8.7%</small>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class="metric">
                    <h3>Giá Trị Đơn Hàng Trung Bình</h3>
                    <p>$156.80</p>
                    <small class="trend-positive">↗ Tăng 4.2%</small>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f"""
                <div class="metric">
                    <h3>Phân Cụm AI</h3>
                    <p>Hoạt động</p>
                    <small>5 phân đoạn được xác định</small>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def _render_main_content(self):
        """Render main content with revenue chart"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='subheader'>Xu Hướng Doanh Thu (30 ngày qua)</div>", unsafe_allow_html=True)
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            
            days = ['Tuần 1', 'Tuần 2', 'Tuần 3', 'Tuần 4']
            revenue = [95000, 118000, 142000, 132000]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days, y=revenue,
                mode='lines+markers',
                line=dict(color='#3498db', width=3, shape='spline'),
                marker=dict(size=10, color='#3498db'),
                fill='tonexty',
                fillcolor='rgba(52, 152, 219, 0.1)',
                name='Doanh thu'
            ))
            fig.update_layout(
                height=280,
                margin=dict(t=10, b=30, l=40, r=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#5a6c7d', size=11)),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(90, 108, 125, 0.1)',
                    zeroline=False,
                    tickfont=dict(color='#5a6c7d', size=11),
                    title=dict(text="Doanh thu ($)", font=dict(color='#2c3e50', size=12))
                ),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='subheader'>Phân Đoạn Khách Hàng Hàng Đầu</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="card">
                    <h3>Champions</h3>
                    <p>Khách hàng giá trị cao nhất</p>
                    <p><strong>Doanh thu:</strong> $247K</p>
                </div>
                <div class="card">
                    <h3>Loyal Customers</h3>
                    <p>Mua hàng lặp lại thường xuyên</p>
                    <p><strong>Doanh thu:</strong> $189K</p>
                </div>
                <div class="card">
                    <h3>At-Risk Customers</h3>
                    <p>Hoạt động mua hàng giảm</p>
                    <p><strong>Doanh thu:</strong> $51K</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def _render_additional_insights(self):
        """Render additional insights to fill the interface"""
        st.markdown("<div class='subheader'>Thống Kê Chi Tiết</div>", unsafe_allow_html=True)
        
        # Performance metrics grid
        st.markdown(
            """
            <div class="stats-grid">
                <div class="stat-item">
                    <h4>Tỷ Lệ Chuyển Đổi</h4>
                    <div class="value trend-positive">12.8%</div>
                </div>
                <div class="stat-item">
                    <h4>Khách Hàng Mới</h4>
                    <div class="value">1,247</div>
                </div>
                <div class="stat-item">
                    <h4>Độ Hài Lòng</h4>
                    <div class="value trend-positive">4.6/5</div>
                </div>
                <div class="stat-item">
                    <h4>Tỷ Lệ Giữ Chân</h4>
                    <div class="value trend-positive">89.2%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Additional insights boxes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                """
                <div class="insight-box">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem;">Xu Hướng Thời Gian</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <h4 style="color: #5a6c7d; font-size: 0.9rem; margin-bottom: 0.5rem;">Giờ Cao Điểm</h4>
                            <p style="color: #2c3e50; font-weight: 600; margin: 0;">14:00 - 18:00</p>
                        </div>
                        <div>
                            <h4 style="color: #5a6c7d; font-size: 0.9rem; margin-bottom: 0.5rem;">Ngày Tốt Nhất</h4>
                            <p style="color: #2c3e50; font-weight: 600; margin: 0;">Thứ 6</p>
                        </div>
                        <div>
                            <h4 style="color: #5a6c7d; font-size: 0.9rem; margin-bottom: 0.5rem;">Mùa Cao Điểm</h4>
                            <p style="color: #2c3e50; font-weight: 600; margin: 0;">Q4</p>
                        </div>
                        <div>
                            <h4 style="color: #5a6c7d; font-size: 0.9rem; margin-bottom: 0.5rem;">Tăng Trưởng YoY</h4>
                            <p style="color: #27ae60; font-weight: 600; margin: 0;">+23.5%</p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div class="insight-box">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem;">Hiệu Suất Kênh</h3>
                    <div style="space-y: 0.8rem;">
                        <div style="margin-bottom: 0.8rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #5a6c7d; font-size: 0.9rem;">Online</span>
                                <span style="color: #2c3e50; font-weight: 600;">68%</span>
                            </div>
                            <div style="background: rgba(52, 152, 219, 0.2); height: 6px; border-radius: 3px; margin-top: 4px;">
                                <div style="background: #3498db; height: 100%; width: 68%; border-radius: 3px;"></div>
                            </div>
                        </div>
                        <div style="margin-bottom: 0.8rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #5a6c7d; font-size: 0.9rem;">Cửa Hàng</span>
                                <span style="color: #2c3e50; font-weight: 600;">32%</span>
                            </div>
                            <div style="background: rgba(46, 204, 113, 0.2); height: 6px; border-radius: 3px; margin-top: 4px;">
                                <div style="background: #2ecc71; height: 100%; width: 32%; border-radius: 3px;"></div>
                            </div>
                        </div>
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(90, 108, 125, 0.1);">
                            <p style="color: #5a6c7d; font-size: 0.85rem; margin: 0;">Tổng đơn hàng tháng này: <strong style="color: #2c3e50;">3,108</strong></p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    dashboard = CustomerSegmentationDashboard()
    dashboard.render()
