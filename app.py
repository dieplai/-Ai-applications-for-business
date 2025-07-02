import streamlit as st
import os
import sys
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/customer_insights.log'
)
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info("Added project root to sys.path: %s", project_root)

logger.info("Python sys.path: %s", sys.path)
logger.info("Current working directory: %s", os.getcwd())
logger.info("Files in current directory: %s", os.listdir('.'))
logger.info("Files in src/pages: %s", os.listdir('src/pages') if os.path.exists('src/pages') else "Directory src/pages not found")
logger.info("Files in utils: %s", os.listdir('utils') if os.path.exists('utils') else "Directory utils not found")

try:
    from src.pages.customer_segmentation_dashboard import CustomerSegmentationDashboard
    from src.pages.customer_segmentation_data_prep import CustomerSegmentationDataPrep
    from src.pages.customer_segmentation_clustering import CustomerSegmentationClustering
    from src.pages.customer_segmentation_insights import CustomerSegmentationInsights
    from src.pages.customer_segmentation_reports import CustomerSegmentationReports
    logger.info("Successfully imported all page classes")
except ImportError as e:
    error_msg = f"Không tìm thấy module hoặc lớp: {str(e)}"
    logger.error("Import error: %s\n%s", str(e), traceback.format_exc())
    st.markdown(f"<p class='error'>{error_msg}</p>", unsafe_allow_html=True)
    st.markdown(
        """
        <p>Vui lòng kiểm tra:</p>
        <ul>
            <li>Thư mục 'src/pages' có tồn tại.</li>
            <li>Các tệp 'customer_segmentation_*.py' trong 'src/pages'.</li>
            <li>Tệp '__init__.py' trong 'src' và 'src/pages'.</li>
            <li>Thư mục 'utils' có tồn tại với tệp 'marketing_prompts.py'.</li>
            <li>Chạy lệnh <code>streamlit run app.py</code> từ thư mục chứa 'src' và 'app.py'.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )
    with st.expander("Chi tiết lỗi"):
        st.code(traceback.format_exc())
    st.stop()

def load_css():
    """Load custom CSS for modern glass effect"""
    st.markdown("""
        <style>
        /* Hide default Streamlit elements */
        .stDeployButton {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255,255,255,0.2);
        }
        
        /* Sidebar header */
        .sidebar-header {
            background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.1));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 16px;
            padding: 24px 20px;
            margin: 20px 0 30px 0;
            text-align: center;
            box-shadow: 0 8px 32px rgba(31,38,135,0.15);
        }
        
        .sidebar-header h2 {
            color: #1f2937;
            font-size: 1.4rem;
            font-weight: 700;
            margin: 0 0 8px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .sidebar-header p {
            color: #6b7280;
            font-size: 0.9rem;
            margin: 0;
            font-weight: 500;
        }
        
        /* Progress container */
        .progress-wrapper {
            margin: 20px 0 30px 0;
        }
        
        .progress-container {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 16px rgba(31,38,135,0.1);
        }
        
        .progress-bar-track {
            background: rgba(255,255,255,0.2);
            height: 8px;
            border-radius: 6px;
            position: relative;
            overflow: hidden;
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s ease;
            box-shadow: 0 2px 8px rgba(102,126,234,0.3);
        }
        
        .progress-text {
            color: #374151;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        /* Steps */
        .steps-container {
            margin: 20px 0;
        }
        
        .step {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 12px 16px;
            margin: 8px 0;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .step:hover {
            background: rgba(255,255,255,0.12);
            border-color: rgba(255,255,255,0.25);
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(31,38,135,0.15);
        }
        
        .step.completed {
            background: rgba(34,197,94,0.1);
            border-color: rgba(34,197,94,0.3);
        }
        
        .step.active {
            background: rgba(99,102,241,0.1);
            border-color: rgba(99,102,241,0.3);
            box-shadow: 0 4px 16px rgba(99,102,241,0.2);
        }
        
        .step-title {
            color: #1f2937;
            font-weight: 600;
            font-size: 0.9rem;
            margin: 0 0 4px 0;
        }
        
        .step-status {
            color: #6b7280;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .step.completed .step-status {
            color: #059669;
        }
        
        .step.active .step-status {
            color: #6366f1;
        }
        
        /* Navigation section */
        .nav-section {
            margin-top: 30px;
        }
        
        .nav-title {
            color: #374151;
            font-size: 1.1rem;
            font-weight: 700;
            margin: 0 0 16px 0;
            text-align: center;
        }
        
        /* Navigation buttons */
        .stButton > button {
            width: 100% !important;
            background: rgba(255,255,255,0.08) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 12px !important;
            padding: 16px !important;
            margin: 8px 0 !important;
            color: #374151 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.3s ease !important;
            box-shadow: none !important;
            height: auto !important;
            min-height: 60px !important;
            white-space: pre-line !important;
            text-align: left !important;
        }
        
        .stButton > button:hover {
            background: rgba(255,255,255,0.15) !important;
            border-color: rgba(255,255,255,0.25) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 16px rgba(31,38,135,0.15) !important;
            color: #1f2937 !important;
        }
        
        .stButton > button:active {
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15)) !important;
            border-color: rgba(99,102,241,0.3) !important;
            color: #6366f1 !important;
            transform: translateY(0) !important;
        }
        
        /* Divider */
        .nav-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            margin: 20px 0;
        }
        
        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
        }
        
        /* Custom scrollbar for sidebar */
        .css-1d391kg::-webkit-scrollbar {
            width: 6px;
        }
        
        .css-1d391kg::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
        }
        
        .css-1d391kg::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.3);
            border-radius: 3px;
        }
        
        .css-1d391kg::-webkit-scrollbar-thumb:hover {
            background: rgba(255,255,255,0.5);
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application for Customer Segmentation Solution"""
    try:
        st.set_page_config(
            page_title="Giải Pháp Phân Cụm Dữ Liệu Khách Hàng Doanh Nghiệp",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        logger.info("Page configuration set")

        load_css()

        if "current_page" not in st.session_state:
            st.session_state.current_page = "Dashboard"
        logger.info("Session state initialized with current_page: %s", st.session_state.current_page)

        with st.sidebar:
            st.markdown(
                """
                <div class="sidebar-header">
                    <h2>Giải Pháp Phân Cụm Khách Hàng</h2>
                    <p>Phân tích và chiến lược tiếp thị thông minh</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            steps = [
                ("1. Chuẩn bị dữ liệu", "DataPrep"),
                ("2. Phân cụm khách hàng", "Clustering"),
                ("3. Thông tin chi tiết", "Insights"),
                ("4. Báo cáo và xuất", "Reports")
            ]
            current_step = {
                "Dashboard": 0,
                "DataPrep": 1,
                "Clustering": 2,
                "Insights": 3,
                "Reports": 4
            }.get(st.session_state.current_page, 0)
            progress = (current_step / len(steps)) * 100

            st.markdown(
                f"""
                <div class="progress-wrapper">
                    <div class="progress-container">
                        <div class="progress-text">Tiến độ: {current_step}/{len(steps)} bước</div>
                        <div class="progress-bar-track">
                            <div class="progress-bar" style="width: {progress}%;"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('<div class="nav-section"><div class="nav-title">Điều Hướng</div></div>', unsafe_allow_html=True)
            
            nav_items = [
                ("Bảng điều khiển", "Dashboard"),
                ("Chuẩn bị dữ liệu", "DataPrep"),
                ("Phân cụm khách hàng", "Clustering"),
                ("Thông tin chi tiết", "Insights"),
                ("Báo cáo và xuất", "Reports")
            ]
            
            st.markdown('<div class="steps-container">', unsafe_allow_html=True)
            for label, page in nav_items:
                # Determine status for steps (skip Dashboard)
                if page != "Dashboard":
                    step_index = ["DataPrep", "Clustering", "Insights", "Reports"].index(page) + 1
                    completed = st.session_state.get(f"phase_{step_index}_completed", False)
                    status = "Hoàn thành" if completed else "Đang thực hiện" if page == st.session_state.current_page else "Chưa bắt đầu"
                    status_class = "completed" if completed else "active" if page == st.session_state.current_page else ""
                else:
                    status = "Tổng quan"
                    status_class = "active" if page == st.session_state.current_page else ""
                
                # Create clickable step
                step_clicked = st.button(
                    f"{label}\n{status}", 
                    key=f"step_{page}",
                    help=f"Chuyển đến {label}",
                    use_container_width=True
                )
                
                if step_clicked and page != st.session_state.current_page:
                    st.session_state.current_page = page
                    logger.info("Navigated to page: %s", page)
                    st.rerun()
                    
            st.markdown('</div>', unsafe_allow_html=True)

        pages = {
            "Dashboard": CustomerSegmentationDashboard,
            "DataPrep": CustomerSegmentationDataPrep,
            "Clustering": CustomerSegmentationClustering,
            "Insights": CustomerSegmentationInsights,
            "Reports": CustomerSegmentationReports
        }
        page_class = pages.get(st.session_state.current_page, CustomerSegmentationDashboard)
        try:
            logger.info("Attempting to render page: %s", st.session_state.current_page)
            page_instance = page_class()
            page_instance.render()
            logger.info("Successfully rendered page: %s", st.session_state.current_page)
        except Exception as e:
            error_msg = f"Lỗi khi tải trang {st.session_state.current_page}: {str(e)}"
            logger.error("%s\n%s", error_msg, traceback.format_exc())
            st.markdown(f"<p class='error'>{error_msg}</p>", unsafe_allow_html=True)
            st.markdown("<p>Vui lòng kiểm tra các tệp trong 'src/pages' và 'src/core'.</p>", unsafe_allow_html=True)
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())
    except Exception as e:
        error_msg = f"Lỗi trong hàm main: {str(e)}"
        logger.error("%s\n%s", error_msg, traceback.format_exc())
        st.markdown(f"<p class='error'>{error_msg}</p>", unsafe_allow_html=True)
        with st.expander("Chi tiết lỗi"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()