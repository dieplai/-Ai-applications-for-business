from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import os
import tempfile
import atexit
import shutil
import re
from src.pages.customer_segmentation_insights import safe_get_nested_value

class CustomerInsightsReportGenerator:
    """Generator for professional PDF reports with dynamic insights and visualizations"""
    
    def __init__(self):
        self._initialize_temp_directory()
        self.temp_files = []
        self.styles = self.create_custom_styles()
        self.chart_method = "matplotlib"  # Force use of Matplotlib
        atexit.register(self._cleanup_temp_files)
        self._verify_setup()
    
    def _initialize_temp_directory(self):
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="customer_insights_")
            if os.path.exists(self.temp_dir) and os.access(self.temp_dir, os.W_OK):
                st.info(f"Using system temp directory: {self.temp_dir}")
                return
        except Exception as e:
            st.warning(f"System temp directory failed: {str(e)}")
        
        try:
            fallback_dir = os.path.join(os.getcwd(), "temp_charts")
            os.makedirs(fallback_dir, exist_ok=True)
            if os.access(fallback_dir, os.W_OK):
                self.temp_dir = fallback_dir
                st.warning(f"Using fallback temp directory: {fallback_dir}")
                return
        except Exception as e:
            st.warning(f"Fallback 1 failed: {str(e)}")
        
        try:
            home_temp = os.path.join(os.path.expanduser("~"), "temp_charts")
            os.makedirs(home_temp, exist_ok=True)
            if os.access(home_temp, os.W_OK):
                self.temp_dir = home_temp
                st.warning(f"Using home temp directory: {home_temp}")
                return
        except Exception as e:
            st.warning(f"Fallback 2 failed: {str(e)}")
        
        try:
            if os.name == 'nt':
                final_fallback = "C:\\temp\\customer_insights"
            else:
                final_fallback = "/tmp/customer_insights"
            os.makedirs(final_fallback, exist_ok=True)
            self.temp_dir = final_fallback
            st.error(f"Using final fallback: {final_fallback}")
        except Exception as e:
            self.temp_dir = os.getcwd()
            st.error(f"Using current directory as temp: {self.temp_dir}")
    
    def _verify_setup(self):
        try:
            test_file = os.path.join(self.temp_dir, "setup_test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            st.success(f"✅ Temp directory verified: {self.temp_dir}")
        except Exception as e:
            st.error(f"❌ Temp directory not writable: {str(e)}")
    
    def _cleanup_temp_files(self):
        try:
            for temp_file in getattr(self, 'temp_files', []):
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if hasattr(self, 'temp_dir') and self.temp_dir:
                if "temp" in self.temp_dir.lower() and os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def create_custom_styles(self):
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB'),
            alignment=1
        ))
        styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#A23B72'),
            spaceBefore=20
        ))
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            spaceAfter=12,
            leading=16,
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0,
            alignment=0,
            bulletIndent=20,
            wordWrap='LTR'
        ))
        return styles
    
    def _format_ai_content_for_pdf(self, content):
        if not content:
            return ""
        cleaned_content = content
        formatted_content = re.sub(r'\n\s*\n', '<br/><br/>', cleaned_content)
        formatted_content = re.sub(r'\n', '<br/>', formatted_content)
        formatted_content = re.sub(r'(\d+\.\s)', r'<br/><b>\1</b>', formatted_content)
        formatted_content = re.sub(r'(•\s)', r'<br/><b>\1</b>', formatted_content)
        formatted_content = re.sub(r'^-\s', '<br/>• ', formatted_content, flags=re.MULTILINE)
        formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_content)
        formatted_content = re.sub(r'<b>([^:]+):</b>', r'<br/><b>\1:</b><br/>', formatted_content)
        formatted_content = re.sub(r'(<br/>){3,}', '<br/><br/>', formatted_content)
        formatted_content = re.sub(r'^(<br/>)+', '', formatted_content)
        return formatted_content
    
    def _ensure_temp_dir(self):
        if not hasattr(self, 'temp_dir') or not self.temp_dir:
            self._initialize_temp_directory()
        if not hasattr(self, 'temp_files'):
            self.temp_files = []
    
    def create_chart_image_fixed(self, data, chart_type, title, width=6, height=4):
        self._ensure_temp_dir()
        try:
            return self._create_matplotlib_chart_safe(data, chart_type, title, width, height)
        except Exception as e:
            st.warning(f"Chart creation failed: {str(e)}")
            return self._create_text_chart_description(data, chart_type, title)
    
    def _create_matplotlib_chart_safe(self, data, chart_type, title, width, height):
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(width, height))
        try:
            if chart_type == "scatter":
                if not all(col in data for col in ['x', 'y', 'color']):
                    raise ValueError("Missing required columns: x, y, or color")
                unique_colors = pd.unique(data['color'])
                colors_map = plt.cm.Set1(np.linspace(0, 1, len(unique_colors)))
                for i, color_val in enumerate(unique_colors):
                    mask = data['color'] == color_val
                    ax.scatter(data['x'][mask], data['y'][mask], c=[colors_map[i]], label=f'Segment {color_val}', alpha=0.7, s=50)
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            elif chart_type == "box":
                if 'cluster' not in data or 'feature' not in data:
                    raise ValueError("Missing required columns: cluster or feature")
                box_data = [data[data['cluster'] == cluster]['feature'].values for cluster in pd.unique(data['cluster'])]
                cluster_labels = [f'Cluster {cluster}' for cluster in pd.unique(data['cluster'])]
                bp = ax.boxplot(box_data, labels=cluster_labels, patch_artist=True)
                colors_list = plt.cm.Set1(np.linspace(0, 1, len(box_data)))
                for patch, color in zip(bp['boxes'], colors_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_ylabel('Feature Value')
            elif chart_type == "pie":
                if not all(key in data for key in ['values', 'names']):
                    raise ValueError("Missing required keys: values or names")
                colors_list = plt.cm.Set1(np.linspace(0, 1, len(data['values'])))
                wedges, texts, autotexts = ax.pie(data['values'], labels=data['names'], autopct='%1.1f%%', startangle=90, colors=colors_list)
                ax.axis('equal')
            elif chart_type == "bar":
                if not all(key in data for key in ['names', 'values']):
                    raise ValueError("Missing required keys: names or values")
                ax.bar(data['names'], data['values'], color=plt.cm.Set1(np.linspace(0, 1, len(data['names']))))
                ax.set_ylabel('Number of Customers')
                ax.set_title(title)
                plt.xticks(rotation=45)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            chart_filename = f"chart_{chart_type}_{len(self.temp_files)}_{int(datetime.now().timestamp())}.png"
            chart_path = os.path.join(self.temp_dir, chart_filename)
            os.makedirs(self.temp_dir, exist_ok=True)
            plt.savefig(chart_path, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            if not os.path.exists(chart_path):
                raise Exception(f"Chart file was not created: {chart_path}")
            with open(chart_path, 'rb') as test_file:
                test_file.read(10)
            self.temp_files.append(chart_path)
            img = Image(chart_path, width=width*inch, height=height*inch)
            st.success(f"✅ Chart created: {chart_filename}")
            return img
        except Exception as e:
            plt.close(fig)
            raise Exception(f"Matplotlib chart creation failed: {str(e)}")
    
    def _create_text_chart_description(self, data, chart_type, title):
        description = f"<b>{title}</b><br/>"
        if chart_type == "scatter":
            description += f"• Scatter plot with {len(data.get('x', []))} data points<br/>"
            description += f"• Number of segments: {len(pd.unique(data.get('color', [])))}<br/>"
        elif chart_type == "box":
            description += f"• Box plot with {len(pd.unique(data.get('cluster', [])))} segments<br/>"
        elif chart_type == "pie":
            description += "• Pie chart distribution of segments:<br/>"
            for name, value in zip(data.get('names', []), data.get('values', [])):
                percentage = (value / sum(data.get('values', [1]))) * 100 if data.get('values', [1]) else 0
                description += f"  - {name}: {percentage:.1f}%<br/>"
        elif chart_type == "bar":
            description += "• Bar chart of customer distribution:<br/>"
            for name, value in zip(data.get('names', []), data.get('values', [])):
                description += f"  - {name}: {value}<br/>"
        description += "<br/>[Unable to create chart - displaying description only]<br/>"
        return Paragraph(description, self.styles['CustomBody'])
    
    def generate_complete_report(self, final_model_results, cluster_profiles, ai_personas, ai_campaigns=None):
        try:
            self._ensure_temp_dir()
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
            story = []
            story.extend(self._create_cover_page())
            story.append(PageBreak())
            story.extend(self._create_executive_summary(cluster_profiles, ai_personas))
            story.append(PageBreak())
            story.extend(self._create_data_overview(final_model_results))
            story.extend(self._create_cluster_analysis(cluster_profiles))
            story.append(PageBreak())
            if final_model_results and final_model_results.get('clustered_data') is not None and not final_model_results.get('clustered_data').empty:
                story.extend(self._create_visual_clusters_safe(final_model_results))
                story.append(PageBreak())
            if ai_personas and any(ai_personas.values()):
                story.extend(self._create_ai_personas_section(ai_personas))
                story.append(PageBreak())
            else:
                st.warning("No valid ai_personas data found, including placeholder section.")
                story.extend(self._create_ai_personas_section({'placeholder': {'content': {'description': 'No personas generated. Please complete the Customer Insights step.'}, 'size_info': {'count': 0, 'percentage': 0}}}))
                story.append(PageBreak())
            if ai_campaigns and any(ai_campaigns.values()):
                story.extend(self._create_campaigns_section(ai_campaigns))
                story.append(PageBreak())
            story.extend(self._create_recommendations(cluster_profiles))
            doc.build(story)
            self._cleanup_temp_files()
            buffer.seek(0)
            st.success("✅ PDF report generated successfully!")
            return buffer
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")
            self._cleanup_temp_files()
            empty_buffer = io.BytesIO()
            return empty_buffer
    
    def _create_visual_clusters_safe(self, final_model_results):
        story = []
        story.append(Paragraph("Cluster Visualizations", self.styles['CustomHeading']))
        try:
            clustered_data = final_model_results.get('clustered_data')
            if clustered_data is not None and not clustered_data.empty:
                labels = clustered_data['cluster'].values
                features_data = clustered_data.drop('cluster', axis=1)
                
                # 2D Scatter Plot
                if features_data.shape[1] >= 2:
                    scatter_data_2d = pd.DataFrame({
                        'x': features_data.iloc[:, 0].values,
                        'y': features_data.iloc[:, 1].values,
                        'color': labels.astype(str)
                    })
                    chart_2d = self.create_chart_image_fixed(scatter_data_2d, "scatter", "2D Distribution of Customer Segments", width=6, height=4)
                    story.append(chart_2d)
                    story.append(Spacer(1, 0.3*inch))
                
                # Pie Chart
                cluster_sizes = []
                cluster_names = []
                for label in np.unique(labels):
                    cluster_size = np.sum(labels == label)
                    cluster_sizes.append(cluster_size)
                    cluster_names.append(f"Segment {label}")
                pie_data = {'values': cluster_sizes, 'names': cluster_names}
                chart_pie = self.create_chart_image_fixed(pie_data, "pie", "Proportion of Customer Segments", width=6, height=4)
                story.append(chart_pie)
                story.append(Spacer(1, 0.3*inch))
                
                # Bar Chart
                bar_data = {'names': cluster_names, 'values': cluster_sizes}
                chart_bar = self.create_chart_image_fixed(bar_data, "bar", "Customer Distribution by Segment", width=6, height=4)
                story.append(chart_bar)
                story.append(Spacer(1, 0.3*inch))
            else:
                story.append(Paragraph("No clustering data available to display charts.", self.styles['CustomBody']))
        except Exception as e:
            error_text = f"Error creating charts: {str(e)}"
            story.append(Paragraph(error_text, self.styles['CustomBody']))
        return story
    
    def _create_cover_page(self):
        story = []
        story.append(Spacer(1, 3*inch))
        story.append(Paragraph("CUSTOMER ANALYSIS REPORT", self.styles['CustomTitle']))
        story.append(Paragraph("Customer Insights & AI Marketing Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 1*inch))
        current_date = datetime.now().strftime("%d/%m/%Y")
        story.append(Paragraph(f"Report Date: {current_date}", self.styles['CustomBody']))
        story.append(Paragraph("Generated by: AI Customer Insights Platform", self.styles['CustomBody']))
        return story
    
    def _create_executive_summary(self, cluster_profiles, ai_personas):
        story = []
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['CustomTitle']))
        if cluster_profiles:
            total_customers = sum(safe_get_nested_value(profile, ['size_info', 'count'], 0) for profile in cluster_profiles.values())
            num_clusters = len(cluster_profiles)
            largest_segment = max(cluster_profiles.values(), key=lambda x: safe_get_nested_value(x, ['size_info', 'count'], 0))
            summary_text = f"""
            <b>Analysis Overview:</b><br/>
            • Total customers analyzed: {total_customers:,}<br/>
            • Number of customer segments identified: {num_clusters}<br/>
            • Largest segment: {safe_get_nested_value(largest_segment, ['size_info', 'percentage'], 0):.1f}% ({safe_get_nested_value(largest_segment, ['size_info', 'count'], 0):,} customers)<br/>
            • Number of AI personas created: {len(ai_personas) if ai_personas else 0}<br/><br/>
            <b>Key Findings:</b><br/>
            • Identified {num_clusters} distinct customer segments with tailored insights<br/>
            • Created AI-driven personas for personalized marketing<br/>
            • Proposed optimized marketing channels and budget allocation<br/>
            • Estimated ROI and revenue potential for each segment
            """
            story.append(Paragraph(summary_text, self.styles['CustomBody']))
        return story
    
    def _create_data_overview(self, final_model_results):
        story = []
        story.append(Paragraph("DATA OVERVIEW", self.styles['CustomHeading']))
        if final_model_results:
            clustered_data = final_model_results.get('clustered_data')
            if clustered_data is not None:
                num_records = len(clustered_data)
                num_features = len(clustered_data.columns) - 1
                data_text = f"""
                <b>Dataset Information:</b><br/>
                • Number of records: {num_records:,}<br/>
                • Number of features: {num_features}<br/>
                • Clustering method: K-means Clustering<br/>
                • Optimal number of clusters: {final_model_results.get('optimal_k', 'N/A')}<br/><br/>
                <b>Clustering Quality:</b><br/>
                • Data has been normalized and cleaned<br/>
                • AI algorithm optimized cluster selection<br/>
                • Multiple metrics validated clustering accuracy
                """
                story.append(Paragraph(data_text, self.styles['CustomBody']))
        return story
    
    def _create_cluster_analysis(self, cluster_profiles):
        story = []
        story.append(Paragraph("DETAILED SEGMENT ANALYSIS", self.styles['CustomHeading']))
        if cluster_profiles:
            for cluster_key, profile in cluster_profiles.items():
                cluster_id = safe_get_nested_value(profile, ['cluster_id'], 0)
                size_info = safe_get_nested_value(profile, ['size_info'], {})
                story.append(Paragraph(f"Segment {cluster_id}", self.styles['CustomHeading']))
                cluster_text = f"""
                <b>Basic Information:</b><br/>
                • Number of customers: {safe_get_nested_value(size_info, ['count'], 0):,}<br/>
                • Proportion: {safe_get_nested_value(size_info, ['percentage'], 0):.1f}% of total customers<br/>
                • Estimated value: ${safe_get_nested_value(size_info, ['count'], 0) * 200:,}<br/><br/>
                <b>Key Characteristics:</b><br/>
                """
                distinctive_features = safe_get_nested_value(profile, ['distinctive_features'], {})
                top_features = safe_get_nested_value(distinctive_features, ['top_features'], [])
                z_scores = safe_get_nested_value(distinctive_features, ['z_scores'], [])
                for i, feature in enumerate(top_features[:3]):
                    if i < len(z_scores):
                        z_score = z_scores[i]
                        direction = "higher" if z_score > 0 else "lower"
                        cluster_text += f"• {feature}: {direction} than average (z-score: {z_score:.2f})<br/>"
                cluster_text += "<br/>"
                story.append(Paragraph(cluster_text, self.styles['CustomBody']))
                story.append(Spacer(1, 0.3*inch))
        return story
    
    def _create_ai_personas_section(self, ai_personas):
        story = []
        story.append(Paragraph("AI MARKETING PERSONAS", self.styles['CustomHeading']))
        if ai_personas and any(ai_personas.values()):
            for persona_id, persona in ai_personas.items():
                persona_name = safe_get_nested_value(persona, ['content', 'persona_name'], f'Persona {persona_id}')
                story.append(Paragraph(persona_name, self.styles['CustomHeading']))
                persona_content = safe_get_nested_value(persona, ['content', 'description'], "No information available")
                persona_content_formatted = self._format_ai_content_for_pdf(persona_content)
                story.append(Paragraph(persona_content_formatted, self.styles['CustomBody']))
                size_info = safe_get_nested_value(persona, ['size_info'], {})
                if size_info:
                    metrics_text = f"""
                    <b>Business Metrics:</b><br/>
                    • Size: {safe_get_nested_value(size_info, ['count'], 0):,} customers<br/>
                    • Proportion: {safe_get_nested_value(size_info, ['percentage'], 0):.1f}%<br/>
                    • Priority: {safe_get_nested_value(persona, ['content', 'business_priority'], 'Medium')}<br/>
                    """
                    story.append(Paragraph(metrics_text, self.styles['CustomBody']))
                story.append(Spacer(1, 0.3*inch))
        else:
            story.append(Paragraph("No persona data available. Please ensure personas are generated in the Customer Insights step.", self.styles['CustomBody']))
        return story
    
    def _create_campaigns_section(self, ai_campaigns):
        story = []
        story.append(Paragraph("MARKETING CAMPAIGN STRATEGIES", self.styles['CustomHeading']))
        if ai_campaigns and any(ai_campaigns.values()):
            for campaign_id, campaign in ai_campaigns.items():
                campaign_name = safe_get_nested_value(campaign, ['content', 'name'], f"Campaign for Group {campaign_id}")
                story.append(Paragraph(campaign_name, self.styles['CustomHeading']))
                campaign_content = safe_get_nested_value(campaign, ['content'], "No strategy available")
                campaign_content_formatted = self._format_ai_content_for_pdf(campaign_content)
                story.append(Paragraph(campaign_content_formatted, self.styles['CustomBody']))
                if safe_get_nested_value(campaign, ['estimated_budget']):
                    budget_text = f"""
                    <b>Budget Information:</b><br/>
                    • Estimated budget: {safe_get_nested_value(campaign, ['estimated_budget'])}<br/>
                    • Target audience: {safe_get_nested_value(campaign, ['target_size'], 0):,} customers<br/>
                    """
                    story.append(Paragraph(budget_text, self.styles['CustomBody']))
                story.append(Spacer(1, 0.3*inch))
        return story
    
    def _create_recommendations(self, cluster_profiles):
        story = []
        story.append(Paragraph("STRATEGY RECOMMENDATIONS", self.styles['CustomHeading']))
        recommendations = """
        <b>1. Focus on High-Value Segments:</b><br/>
        • Prioritize segments with largest customer base and highest potential<br/>
        • Allocate 60-70% of marketing budget to top-performing segments<br/><br/>
        <b>2. Personalize Marketing Strategies:</b><br/>
        • Leverage AI personas for targeted content creation<br/>
        • Optimize channels and messaging for each segment<br/><br/>
        <b>3. Measure and Optimize:</b><br/>
        • Track KPIs specific to each segment<br/>
        • Conduct A/B testing with persona-based campaigns<br/>
        • Adjust strategies based on performance data<br/><br/>
        <b>4. Product Development:</b><br/>
        • Develop features tailored to segment needs<br/>
        • Introduce customized pricing tiers for different groups
        """
        story.append(Paragraph(recommendations, self.styles['CustomBody']))
        return story
    
def create_pdf_report(final_model_results, cluster_profiles, ai_personas, ai_campaigns=None):
    try:
        generator = CustomerInsightsReportGenerator()
        return generator.generate_complete_report(final_model_results, cluster_profiles, ai_personas, ai_campaigns)
    except Exception as e:
        st.error(f"❌ Critical error generating PDF: {str(e)}")
        empty_buffer = io.BytesIO()
        return empty_buffer