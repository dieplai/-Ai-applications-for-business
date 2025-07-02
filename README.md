# Customer Insights Platform

A powerful Streamlit-based application for data clustering and customer analysis. Upload your dataset, select clustering algorithms, fine-tune parameters, and visualize results with an intuitive web interface.

## Quick Start

**Clone the repository**
```bash
git clone https://github.com/dieplai/-Ai-applications-for-business.git
cd -Ai-applications-for-business
```

**Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the application**
```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── data/                   # Data directory
│   ├── raw/                # Raw datasets (gitignored)
│   └── processed/          # Processed data (gitignored)
├── src/                    # Custom modules and utilities
├── .streamlit/             # Streamlit configuration (gitignored)
└── .gitignore
```

## Features

**Data Management**
- CSV file upload with validation
- Data preprocessing and cleaning options
- Missing value handling and normalization

**Clustering Algorithms**
- K-Means clustering with elbow method
- DBSCAN for density-based clustering
- Agglomerative hierarchical clustering
- Customizable parameters for each algorithm

**Visualization & Analysis**
- Interactive scatter plots and heatmaps
- Cluster centroid visualization
- Silhouette analysis and scoring
- Customer segment profiling

**Export & Sharing**
- Download clustered datasets
- Export visualization charts
- Shareable analysis reports

## Usage Guide

1. **Launch the application** using the streamlit command
2. **Upload your customer dataset** in CSV format
3. **Select preprocessing options** if needed
4. **Choose clustering algorithm** and adjust parameters
5. **Analyze results** through interactive visualizations
6. **Export clustered data** for further analysis

## System Requirements

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Minimum 4GB RAM recommended for large datasets

**Key Dependencies**
- Streamlit for web interface
- Pandas for data manipulation
- Scikit-learn for machine learning
- Plotly for interactive visualizations
- NumPy for numerical computing

## Development

**Contributing**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

**Testing**
Run the application locally and test with sample datasets before contributing.

## Important Notes

- Never upload sensitive or personal data to public repositories
- Raw data files are automatically ignored by git
- Create GitHub issues for bug reports or feature requests
- Check the requirements.txt file if you encounter missing dependencies

## Author

**Lai Diep**  
Repository: https://github.com/dieplai/-Ai-applications-for-business

## License

This project is open source and available under standard terms.

---

*Built with Streamlit for modern data science workflows*
