import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils.config import EnterpriseAppConfig

class EnterpriseDataValidator:
    """Validator for enterprise customer data with enhanced file and data checks"""
    
    def __init__(self, enable_logging=True):
        self.config = EnterpriseAppConfig()
        self.validation_results = {}
        self.recommendations = []
        self.business_rules = self._load_business_rules()
        if enable_logging:
            logging.basicConfig(level=logging.INFO, format=self.config.LOGGING_CONFIG['format'])
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
    
    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business rules from config"""
        return {
            'minimum_records': self.config.FILE_LIMITS['min_rows'],
            'recommended_records': 2000,
            'maximum_file_size_mb': self.config.FILE_LIMITS['max_file_size_mb'],
            'minimum_features': 3,
            'minimum_numeric_features': 3,
            'maximum_missing_data_percent': self.config.DATA_QUALITY_STANDARDS['max_missing_data_percent'],
            'warning_missing_data_percent': 15,
            'duplicate_threshold_percent': self.config.DATA_QUALITY_STANDARDS['max_duplicate_percent'],
            'clustering_readiness_threshold': 75
        }
    
    def validate_file_format_enterprise(self, file) -> Dict[str, Any]:
        """Validate file format, size, and encoding"""
        start_time = datetime.now()
        results = {
            'is_valid': True,
            'file_size_mb': 0,
            'file_type': 'unknown',
            'file_name': file.name if hasattr(file, 'name') else 'unknown',
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'validation_time_seconds': 0
        }
        
        try:
            # Check file size
            results['file_size_mb'] = len(file.getvalue()) / (1024 * 1024)
            max_size = self.business_rules['maximum_file_size_mb']
            if results['file_size_mb'] > max_size:
                results['errors'].append(f"Kích thước file ({results['file_size_mb']:.1f}MB) vượt quá giới hạn ({max_size}MB)")
                results['is_valid'] = False
            elif results['file_size_mb'] > max_size * 0.8:
                results['warnings'].append(f"File lớn ({results['file_size_mb']:.1f}MB). Có thể mất thời gian xử lý.")
            
            # Check file type
            results['file_type'] = file.name.split('.')[-1].lower() if hasattr(file, 'name') and '.' in file.name else 'unknown'
            supported_formats = [fmt.lstrip('.') for fmt in self.config.FILE_LIMITS['supported_formats']]
            if results['file_type'] not in supported_formats:
                results['errors'].append(f"Định dạng file không hỗ trợ: {results['file_type']}. Hỗ trợ: {', '.join(supported_formats)}")
                results['is_valid'] = False
            
            # Check CSV encoding
            if results['file_type'] == 'csv':
                try:
                    file.seek(0)
                    sample = file.read(1024).decode('utf-8')
                    file.seek(0)
                    results['recommendations'].append("File CSV sử dụng encoding UTF-8.")
                except UnicodeDecodeError:
                    results['warnings'].append("File CSV có thể không sử dụng encoding UTF-8. Khuyến nghị kiểm tra encoding.")
                    for encoding in self.config.FILE_LIMITS['encoding_options']:
                        try:
                            file.seek(0)
                            sample = file.read(1024).decode(encoding)
                            file.seek(0)
                            results['recommendations'].append(f"Phát hiện encoding khả thi: {encoding}")
                            break
                        except UnicodeDecodeError:
                            continue
            
            results['validation_time_seconds'] = (datetime.now() - start_time).total_seconds()
            if self.logger:
                self.logger.info(f"Xác thực file '{results['file_name']}' hoàn thành trong {results['validation_time_seconds']:.3f} giây")
        except Exception as e:
            results['errors'].append(f"Lỗi xác thực file: {str(e)}")
            results['is_valid'] = False
            if self.logger:
                self.logger.error(f"Lỗi xác thực file '{results['file_name']}': {str(e)}")
        
        self.validation_results['file_format'] = results
        return results
    
    def validate_data_structure_enterprise(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data structure, missing values, and duplicates"""
        start_time = datetime.now()
        results = {
            'is_valid': True,
            'rows': 0,
            'columns': 0,
            'numeric_columns': 0,
            'categorical_columns': 0,
            'missing_data_percent': 0.0,
            'duplicate_rows': 0,
            'duplicate_percent': 0.0,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            if df is None or not isinstance(df, pd.DataFrame):
                results['errors'].append("Dữ liệu không hợp lệ: Không phải DataFrame hoặc None")
                results['is_valid'] = False
                return results
            
            results['rows'] = len(df)
            results['columns'] = len(df.columns)
            results['numeric_columns'] = len(df.select_dtypes(include=[np.number]).columns)
            results['categorical_columns'] = len(df.select_dtypes(include=['object', 'category']).columns)
            results['missing_data_percent'] = round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2) if len(df) > 0 and len(df.columns) > 0 else 0.0
            results['duplicate_rows'] = df.duplicated().sum()
            results['duplicate_percent'] = round((results['duplicate_rows'] / len(df)) * 100, 2) if len(df) > 0 else 0.0
            
            # Validate row count
            if len(df) < self.business_rules['minimum_records']:
                results['errors'].append(f"Dataset quá nhỏ ({len(df)} hàng). Cần tối thiểu {self.business_rules['minimum_records']} hàng")
                results['is_valid'] = False
            elif len(df) < self.business_rules['recommended_records']:
                results['warnings'].append(f"Dataset nhỏ ({len(df)} hàng). Khuyến nghị {self.business_rules['recommended_records']}+ hàng")
            
            # Validate column count
            if len(df.columns) < self.business_rules['minimum_features']:
                results['errors'].append(f"Không đủ cột ({len(df.columns)}). Cần tối thiểu {self.business_rules['minimum_features']}")
                results['is_valid'] = False
            
            # Validate numeric columns
            if results['numeric_columns'] < self.business_rules['minimum_numeric_features']:
                results['warnings'].append(f"Ít cột số ({results['numeric_columns']}). Cần {self.business_rules['minimum_numeric_features']}+ cho phân cụm")
            
            # Validate missing data
            if results['missing_data_percent'] > self.business_rules['maximum_missing_data_percent']:
                results['errors'].append(f"Dữ liệu thiếu quá nhiều ({results['missing_data_percent']:.1f}%). Tối đa {self.business_rules['maximum_missing_data_percent']}%")
                results['is_valid'] = False
            elif results['missing_data_percent'] > self.business_rules['warning_missing_data_percent']:
                results['warnings'].append(f"Dữ liệu thiếu cao ({results['missing_data_percent']:.1f}%). Cân nhắc cải thiện dữ liệu")
            
            # Validate duplicates
            if results['duplicate_percent'] > self.business_rules['duplicate_threshold_percent']:
                results['warnings'].append(f"Tỷ lệ trùng lặp cao ({results['duplicate_percent']:.1f}%). Cân nhắc loại bỏ trùng lặp")
            
            results['validation_time_seconds'] = (datetime.now() - start_time).total_seconds()
            if self.logger:
                self.logger.info(f"Xác thực cấu trúc dữ liệu hoàn thành trong {results['validation_time_seconds']:.3f} giây")
        except Exception as e:
            results['errors'].append(f"Lỗi xác thực cấu trúc dữ liệu: {str(e)}")
            results['is_valid'] = False
            if self.logger:
                self.logger.error(f"Lỗi xác thực cấu trúc dữ liệu: {str(e)}")
        
        self.validation_results['data_structure'] = results
        return results
    
    def validate_clustering_readiness_enterprise(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data readiness for clustering"""
        start_time = datetime.now()
        results = {
            'is_ready': True,
            'clustering_confidence': 0,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            if df is None or not isinstance(df, pd.DataFrame):
                results['issues'].append("Dữ liệu không hợp lệ: Không phải DataFrame hoặc None")
                results['is_ready'] = False
                return results
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            confidence = 0
            
            # Sample size contribution
            if len(df) >= 10000:
                confidence += 30
            elif len(df) >= 5000:
                confidence += 25
            elif len(df) >= 2000:
                confidence += 20
            elif len(df) >= 500:
                confidence += 15
            else:
                confidence += 5
                results['issues'].append("Kích thước mẫu quá nhỏ cho phân cụm")
            
            # Feature quality contribution
            quality_features = sum(1 for col in numeric_cols if df[col].std() > self.config.DATA_QUALITY_STANDARDS['min_variance_threshold'] and df[col].isnull().sum() / len(df) < 0.2)
            feature_confidence = (quality_features / len(numeric_cols)) * 40 if len(numeric_cols) > 0 else 0
            confidence += feature_confidence
            
            # Data completeness contribution
            missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 and len(df.columns) > 0 else 0.0
            completeness_confidence = max(0, 20 - missing_percent)
            confidence += completeness_confidence
            
            # Numeric feature count contribution
            if len(numeric_cols) >= 8:
                confidence += 10
            elif len(numeric_cols) >= 5:
                confidence += 8
            elif len(numeric_cols) >= 3:
                confidence += 5
            else:
                confidence += 2
                results['issues'].append("Không đủ cột số cho phân cụm chất lượng cao")
            
            results['clustering_confidence'] = round(confidence, 1)
            results['is_ready'] = confidence >= self.business_rules['clustering_readiness_threshold']
            
            if not results['is_ready']:
                results['warnings'].append("Dữ liệu cần cải thiện để đạt chuẩn phân cụm")
                results['recommendations'].append("Cân nhắc thêm dữ liệu hoặc làm sạch kỹ hơn trước khi phân cụm")
            
            results['validation_time_seconds'] = (datetime.now() - start_time).total_seconds()
            if self.logger:
                self.logger.info(f"Xác thực sẵn sàng phân cụm hoàn thành trong {results['validation_time_seconds']:.3f} giây")
        except Exception as e:
            results['issues'].append(f"Lỗi xác thực sẵn sàng phân cụm: {str(e)}")
            results['is_ready'] = False
            if self.logger:
                self.logger.error(f"Lỗi xác thực sẵn sàng phân cụm: {str(e)}")
        
        self.validation_results['clustering_readiness'] = results
        return results

def create_enterprise_data_validator():
    """Create an instance of EnterpriseDataValidator"""
    return EnterpriseDataValidator(enable_logging=True)

def validate_file_format_enterprise(file):
    """Convenience function to validate file format"""
    validator = create_enterprise_data_validator()
    return validator.validate_file_format_enterprise(file)

def validate_data_structure_enterprise(df):
    """Convenience function to validate data structure"""
    validator = create_enterprise_data_validator()
    return validator.validate_data_structure_enterprise(df)

def validate_clustering_readiness_enterprise(df):
    """Convenience function to validate clustering readiness"""
    validator = create_enterprise_data_validator()
    return validator.validate_clustering_readiness_enterprise(df)
