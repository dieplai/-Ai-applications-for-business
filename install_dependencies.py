"""
Quick dependency installer for Customer Insights Platform
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install all required packages"""
    packages = [
        "streamlit>=1.28.0",
        "pandas>=2.0.0", 
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "google-generativeai>=0.3.0",
        "openpyxl>=3.1.0",
        "psutil>=5.9.0",
        "python-dotenv>=1.0.0"
    ]
    
    print("🚀 Installing Customer Insights Platform dependencies...")
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successful: {success_count}/{len(packages)}")
    
    if success_count == len(packages):
        print("🎉 All dependencies installed successfully!")
        print("You can now run: streamlit run app.py")
    else:
        print("⚠️ Some packages failed to install. Please install manually.")

if __name__ == "__main__":
    main()
