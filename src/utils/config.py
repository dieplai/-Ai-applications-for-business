import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st
import logging
from datetime import datetime

class EnterpriseAppConfig:
    APP_METADATA = {
        'name': 'Nền Tảng AI Phân Tích & Phân Cụm Khách Hàng',
        'version': '2.0.0-enterprise',
        'description': 'Nền tảng phân tích khách hàng với giao diện thân thiện cho doanh nghiệp Việt Nam',
        'supported_languages': ['vi'],
        'default_language': 'vi'
    }
    
    COLORS = {
        'primary': '#1E88E5',         # Xanh dương chính
        'secondary': '#E3F2FD',       # Xanh nhạt nền
        'accent': '#43A047',          # Xanh lá nhấn
        'text_primary': '#212121',    # Đen đậm
        'text_secondary': '#757575',  # Xám phụ
        'background': '#FFFFFF',      # Trắng nền
        'border': '#BBDEFB'          # Viền xanh nhạt
    }
    
    CLUSTERING_PARAMS = {
        'min_k': 2,
        'max_k_absolute': 12,
        'default_k_range': (2, 8),
        'min_samples_for_clustering': 100,
        'silhouette_threshold': 0.4,
        'random_state': 42,
        'max_iterations': 500,
        'tolerance': 1e-6
    }
    
    FILE_LIMITS = {
        'max_file_size_mb': 500,
        'supported_formats': ['.csv', '.xlsx', '.xls', '.parquet', '.json'],
        'encoding_options': ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16'],
        'max_columns': 2000,
        'max_rows': 5_000_000,
        'min_rows': 100
    }
    
    AI_CONFIG = {
        'gemini_model': 'gemini-1.5-pro',
        'fallback_model': 'gemini-1.5-flash',
        'max_tokens': 8192,
        'temperature': 0.3,
        'safety_settings': {
            'harassment': 'block_medium_and_above',
            'hate_speech': 'block_medium_and_above',
            'sexually_explicit': 'block_medium_and_above',
            'dangerous_content': 'block_medium_and_above'
        }
    }
    
    EXPORT_CONFIG = {
        'pdf_page_size': 'A4',
        'pdf_margins': {'top': 20, 'bottom': 20, 'left': 20, 'right': 20},
        'chart_dpi': 300,
        'vietnamese_fonts': ['Roboto', 'Arial Unicode MS'],
        'date_format_vietnamese': '%d/%m/%Y',
        'currency_format': 'VND'
    }
    
    DATA_QUALITY_STANDARDS = {
        'max_missing_data_percent': 20,
        'max_duplicate_percent': 10,
        'min_variance_threshold': 0.01,
        'vietnamese_data_patterns': {
            'phone_patterns': [r'^\+84\d{9,10}$', r'^0\d{9,10}$'],
            'email_patterns': [r'^[\w\.-]+@[\w\.-]+\.\w+$'],
            'address_patterns': ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Cần Thơ'],
            'name_patterns': [r'^[A-Za-zÀ-ỹ\s]+$']
        }
    }
    
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        'log_file': 'customer_insights.log',
        'max_log_size_mb': 50,
        'backup_count': 10
    }
    
    BUSINESS_INTELLIGENCE = {
        'kpi_definitions': {
            'cltv': 'Giá trị trọn đời khách hàng',
            'aov': 'Giá trị đơn hàng trung bình',
            'frequency': 'Tần suất mua hàng',
            'recency': 'Thời gian mua hàng gần nhất'
        }
    }
    
    VISUALIZATION_CONFIG = {
        'default_theme': 'enterprise',
        'color_blind_friendly': True,
        'vietnamese_labels': True,
        'supported_chart_types': ['scatter', 'bar', 'pie', '3d_scatter']
    }

    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'streamlit_version': st.__version__,
            'current_time': datetime.now().isoformat(),
            'timezone': 'Asia/Ho_Chi_Minh'
        }

    @classmethod
    def get_enhanced_color_palette(cls, n_colors: int = 10, style: str = 'professional') -> List[str]:
        palettes = {
            'professional': [
                '#1E88E5', '#64B5F6', '#43A047', '#388E3C', '#E3F2FD',
                '#81D4FA', '#4CAF50', '#2196F3', '#1976D2', '#BBDEFB'
            ]
        }
        base_colors = palettes.get(style, palettes['professional'])
        while len(base_colors) < n_colors:
            base_colors.extend(base_colors)
        return base_colors[:n_colors]

def load_environment():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return True
    except ImportError:
        return False

def setup_enterprise_logging():
    config = EnterpriseAppConfig.LOGGING_CONFIG
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger('customer_insights')
    logger.setLevel(getattr(logging, config['level']))
    logger.handlers.clear()
    formatter = logging.Formatter(config['format'])
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / config['log_file'],
        maxBytes=config['max_log_size_mb'] * 1024 * 1024,
        backupCount=config['backup_count']
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def get_vietnamese_ui_text() -> Dict[str, str]:
    return {
        'dashboard': 'Bảng Điều Khiển',
        'data_preparation': 'Chuẩn Bị Dữ Liệu',
        'clustering_analysis': 'Phân Cụm Khách Hàng',
        'customer_insights': 'Thông Tin Chi Tiết',
        'reports_export': 'Báo Cáo & Xuất',
        'upload': 'Tải Lên',
        'analyze': 'Phân Tích',
        'process': 'Xử Lý',
        'generate': 'Tạo',
        'export': 'Xuất',
        'download': 'Tải Xuống',
        'save': 'Lưu',
        'cancel': 'Hủy',
        'continue': 'Tiếp Tục',
        'back': 'Quay Lại',
        'next': 'Tiếp Theo',
        'finish': 'Hoàn Thành',
        'success': 'Thành Công',
        'error': 'Lỗi',
        'warning': 'Cảnh Báo',
        'info': 'Thông Tin',
        'loading': 'Đang Tải',
        'processing': 'Đang Xử Lý',
        'completed': 'Hoàn Thành',
        'failed': 'Thất Bại',
        'customers': 'Khách Hàng',
        'segments': 'Phân Khúc',
        'clusters': 'Nhóm',
        'revenue': 'Doanh Thu',
        'profit': 'Lợi Nhuận',
        'campaign': 'Chiến Dịch',
        'strategy': 'Chiến Lược',
        'insights': 'Thông Tin Chi Tiết',
        'analytics': 'Phân Tích'
    }