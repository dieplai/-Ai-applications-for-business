import streamlit as st
import google.generativeai as genai
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set API key (replace with valid key from Google AI Studio)
API_KEY = "AIzaSyBXxq2nq6YMWYcV5cWYExR0IxHLA9ABMSU"

class MarketingAIAssistant:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEY
        self.available = False
        self.error_message = None
        
        logger.info(f"Using API key: {self.api_key[:5] if self.api_key else 'None'} (partial for security)")
        if not self.api_key:
            self.error_message = "API key not found. Please check the source code."
        elif self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                test_response = self.model.generate_content("Hello")
                self.available = True
                self.error_message = None
                logger.info("Successfully connected to Gemini API")
            except Exception as e:
                self.error_message = f"Error connecting to Gemini API: {str(e)}. Please check your API key."
                logger.error(f"Connection error: {str(e)}")

    def generate_content(self, prompt: str, content_type: str = "general") -> str:
        if not self.available:
            return self.error_message or "Cannot connect to AI. Please check API key."
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating content: {str(e)}"

    def create_buyer_persona(self, business_info: str, customer_data: str = "") -> str:
        prompt = f"""
        You are a professional Marketing Director with 15+ years of experience. Create detailed and scientific Buyer Personas for:

        BUSINESS INFORMATION:
        {business_info}

        CUSTOMER DATA:
        {customer_data if customer_data else "No specific data available"}

        PROFESSIONAL ANALYSIS REQUIREMENTS:

        1. DEMOGRAPHIC INFORMATION:
        - Specific age ranges (not broad categories)
        - Gender distribution with percentages
        - Average monthly income (USD/local currency)
        - Education level and professional background
        - Marital status and family composition
        - Geographic location (urban/suburban/rural, major cities)

        2. PSYCHOGRAPHIC ANALYSIS:
        - Core values and beliefs
        - Lifestyle patterns and daily habits
        - Specific interests and hobbies
        - Technology adoption level and digital literacy
        - Attitude toward brands and advertising
        - Social influence factors (influencers, peer recommendations)

        3. BUYING BEHAVIOR PATTERNS:
        - Detailed customer journey mapping
        - Preferred information-seeking channels
        - Purchase timing patterns (daily/weekly/monthly cycles)
        - Purchase frequency and brand loyalty indicators
        - Budget allocation and price sensitivity
        - Decision-making process (individual vs. group decisions)

        4. PAIN POINTS & CHALLENGES:
        - Top 5 pain points with severity rating (1-10 scale)
        - Barriers in the purchasing process
        - Current competitive alternatives being used
        - Unmet needs and market opportunities

        5. DIGITAL FOOTPRINT:
        - Most-used platforms (Facebook, Instagram, TikTok, LinkedIn, etc.)
        - Peak online activity hours
        - Preferred content formats (video, image, text)
        - Social proof requirements
        - Mobile vs. desktop usage patterns

        6. BUSINESS METRICS:
        - Estimated Customer Lifetime Value (CLV)
        - Customer Acquisition Cost (CAC) benchmarks
        - Conversion probability scores
        - Churn risk factors and indicators

        Create 3 PRIMARY PERSONAS with specific names and detailed profiles. Each persona should represent at least 25% of the customer base with distinct targeting strategies.

        FORMAT: Write as a professional report with specific data points, deep insights, and actionable recommendations.
        """
        return self.generate_content(prompt, "buyer_persona")

    def create_campaign_plan(self, product_info: str, target_audience: str, budget: str, duration: str) -> str:
        prompt = f"""
        You are a Marketing Campaign Director for an enterprise company. Create a COMPREHENSIVE CAMPAIGN STRATEGY:

        INPUT INFORMATION:
        - Product/Service: {product_info}
        - Target Audience: {target_audience}
        - Budget: {budget}
        - Duration: {duration}

        CAMPAIGN FRAMEWORK REQUIREMENTS:

        1. SITUATION ANALYSIS:
        - Market size and growth potential assessment
        - Competitive landscape analysis with key players
        - SWOT analysis for the product/service
        - Clear market positioning statement

        2. CAMPAIGN OBJECTIVES (SMART Goals):
        - Primary KPI: Specific revenue/sales targets
        - Secondary KPIs: Brand awareness metrics, lead generation targets
        - Tertiary KPIs: Engagement rates, retention metrics, social metrics
        - Expected ROI and payback period projections

        3. TARGET AUDIENCE SEGMENTATION:
        - Primary segment (60-70% budget allocation)
        - Secondary segment (20-30% budget allocation)
        - Tertiary segment (5-15% budget allocation)
        - Lookalike audiences and expansion opportunities

        4. CREATIVE STRATEGY:
        - Core message architecture and key themes
        - Value proposition hierarchy
        - Creative concepts for each touchpoint
        - Brand voice and tone guidelines
        - Content calendar framework

        5. MEDIA STRATEGY & BUDGET ALLOCATION:
        - Paid Media breakdown by percentage:
          * Google Ads (Search + Display + YouTube)
          * Facebook/Instagram Ads
          * TikTok/LinkedIn (if applicable)
          * Programmatic Display advertising
        - Owned Media optimization strategies
        - Earned Media acquisition tactics
        - Budget allocation by funnel stage (ToFu/MoFu/BoFu)

        6. IMPLEMENTATION ROADMAP:
        - Pre-launch phase (setup, creative production, testing)
        - Launch phase (go-to-market execution)
        - Optimization phase (performance testing and scaling)
        - Post-campaign analysis and learnings
        - Detailed timeline with daily/weekly milestones

        7. MEASUREMENT & ANALYTICS:
        - Primary metrics dashboard setup
        - Attribution modeling approach
        - A/B testing framework and protocols
        - Reporting frequency and stakeholder communication
        - Success criteria and failure threshold definitions

        8. RISK MANAGEMENT:
        - Top 5 identified risks with mitigation strategies
        - Budget contingency planning (15-20% reserve)
        - Crisis communication protocols
        - Performance troubleshooting guidelines

        9. SCALING STRATEGY:
        - Criteria for budget scale-up decisions
        - New channel expansion roadmap
        - Geographic expansion potential assessment
        - Long-term brand building integration plan

        OUTPUT FORMAT: Executive summary followed by detailed strategy document with charts/tables where applicable. All metrics should be realistic and based on industry benchmarks.
        """
        return self.generate_content(prompt, "campaign_plan")

    def create_cluster_buyer_persona(self, cluster_info: str) -> str:
        """Create enterprise-grade buyer persona for specific customer clusters"""
        prompt = f"""
        You are a Senior Customer Intelligence Analyst at a leading consulting firm. Analyze the following CUSTOMER CLUSTER:

        CLUSTER DATA:
        {cluster_info}

        PROFESSIONAL CLUSTER ANALYSIS:

        1. CLUSTER PROFILING:
        - Cluster size and percentage of total customer base
        - Revenue contribution and profitability index
        - Growth trajectory with year-over-year comparisons
        - Cluster stability and retention scores

        2. DEMOGRAPHIC DEEP-DIVE:
        - Age distribution analysis (mean, median, mode)
        - Gender split with purchasing power correlation
        - Income bracket analysis with spending behavior patterns
        - Geographic concentration and regional preferences
        - Life stage analysis (students, professionals, families, retirees)

        3. BEHAVIORAL SEGMENTATION:
        - Purchase frequency patterns and seasonality
        - Price sensitivity analysis and elasticity
        - Brand loyalty indicators and switching behavior
        - Cross-selling and up-selling opportunity assessment
        - Channel preference hierarchy and usage patterns

        4. PSYCHOGRAPHIC INSIGHTS:
        - Core motivations and behavioral triggers
        - Value system alignment with brand messaging
        - Lifestyle characteristics and daily routines
        - Technology adoption curve positioning
        - Social influence susceptibility and peer effects
        - Risk tolerance in purchasing decisions

        5. CUSTOMER JOURNEY ANALYSIS:
        - Awareness stage behavior patterns
        - Consideration phase decision factors
        - Purchase decision criteria (ranked by importance)
        - Post-purchase engagement and satisfaction levels
        - Advocacy potential and referral likelihood

        6. COMMUNICATION PREFERENCES:
        - Preferred touchpoint ranking and effectiveness
        - Message resonance testing insights
        - Content format preferences and engagement rates
        - Communication frequency tolerance levels
        - Personalization expectations and privacy concerns

        7. COMPETITIVE ANALYSIS:
        - Share of wallet analysis vs. competitors
        - Brand switching probability scores
        - Competitive advantages required for retention
        - White space opportunities for growth

        8. ACTIONABLE MARKETING STRATEGY:
        - Top 5 prioritized marketing tactics with rationale
        - Detailed budget allocation recommendations (specific percentages)
        - Campaign timing optimization based on cluster behavior
        - Channel strategy customization guidelines
        - Creative direction and messaging recommendations
        - Personalization opportunities and implementation
        - Retention strategy specific to cluster characteristics

        9. SUCCESS METRICS:
        - Cluster-specific KPIs and measurement framework
        - Benchmark performance targets based on industry standards
        - ROI expectations and profitability projections
        - Growth projections and expansion opportunities

        DELIVERABLE: Comprehensive cluster analysis report with executive summary, detailed findings, strategic recommendations, and implementation roadmap.
        """
        return self.generate_content(prompt, "cluster_buyer_persona")

    def create_cluster_campaign_plan(self, cluster_info: str, product_info: str, budget: str, duration: str) -> str:
        """Create enterprise campaign strategy for specific customer clusters"""
        prompt = f"""
        You are a Marketing Campaign Strategist for a Fortune 500 company. Design a CLUSTER-SPECIFIC CAMPAIGN:

        CAMPAIGN INPUTS:
        - Target Cluster: {cluster_info}
        - Product/Service: {product_info}
        - Budget: {budget}
        - Timeline: {duration}

        CLUSTER-OPTIMIZED CAMPAIGN STRATEGY:

        1. CLUSTER STRATEGY ALIGNMENT:
        - Cluster priority ranking within overall marketing strategy
        - Unique value proposition tailored to this cluster
        - Competitive positioning statement for cluster context
        - Brand messaging adaptation requirements

        2. HYPER-TARGETED OBJECTIVES:
        - Primary Goal: Specific revenue targets with conversion rate assumptions
        - Secondary Goals: Customer acquisition numbers, CLV improvement targets
        - Tertiary Goals: Brand awareness metrics, engagement rate improvements
        - Success definition within cluster behavior context

        3. AUDIENCE ACTIVATION STRATEGY:
        - Lookalike modeling approach and similarity thresholds
        - Custom audience creation guidelines and data sources
        - Behavioral targeting parameters and trigger events
        - Geographic and demographic filter specifications
        - Exclusion audiences to prevent overlap and waste

        4. PERSONALIZED CREATIVE STRATEGY:
        - Core message framework adapted for cluster preferences
        - Creative concept hierarchy (Hero, Hub, Help content strategy)
        - Visual identity guidelines and design principles
        - Copy tone and messaging angle variations
        - Localization requirements and cultural considerations

        5. OPTIMIZED MEDIA MIX:
        - Channel priority ranking based on cluster preferences:
          * Paid Social: Facebook/Instagram/TikTok budget allocation
          * Paid Search: Google Ads strategy and keyword themes
          * Display: Programmatic targeting and placement strategy
          * Native: Content marketing integration opportunities
          * Traditional: TV/Radio/Print if relevant to cluster
        - Budget distribution with strategic rationale
        - Platform-specific bidding strategies and optimization
        - Creative format optimization by channel

        6. TACTICAL EXECUTION PLAN:
        Week 1-2: Campaign setup, creative production, and soft launch testing
        Week 3-4: Full deployment with real-time optimization
        [Continue timeline based on specified duration]
        - Daily budget pacing and spend management
        - Creative rotation schedule and performance monitoring
        - Audience expansion timeline and scaling triggers
        - Optimization checkpoint schedule and decision criteria

        7. MEASUREMENT FRAMEWORK:
        - Cluster-specific attribution modeling approach
        - Primary KPIs with realistic benchmark expectations
        - Secondary metric tracking and correlation analysis
        - Real-time optimization triggers and automated rules
        - Reporting dashboard configuration and stakeholder access
        - Success/failure threshold definitions and escalation protocols

        8. BUDGET OPTIMIZATION:
        - Detailed budget breakdown:
          * Media spend allocation (80-85% of total budget)
          * Creative production costs (8-12% of total budget)
          * Tools and technology expenses (3-5% of total budget)
          * Contingency reserve (5-10% of total budget)
        - Cost-per-acquisition targets by channel and audience
        - Budget reallocation triggers and decision frameworks
        - Scaling criteria and additional budget requirements

        9. ADVANCED TACTICS:
        - A/B testing roadmap (creative variations, audience segments, bidding strategies)
        - Sequential messaging strategy and customer journey optimization
        - Retargeting funnel design with segment-specific messaging
        - Cross-channel synergy optimization and attribution
        - Seasonal adaptation plans and calendar integration

        10. RISK MITIGATION:
        - Performance troubleshooting playbook with common issues
        - Budget protection strategies and automated safeguards
        - Crisis communication plan and stakeholder notification
        - Competitive response preparation and counter-strategies
        - Platform policy compliance checks and approval processes

        11. INTEGRATION STRATEGY:
        - CRM integration requirements and data flow mapping
        - Sales team alignment and lead handoff protocols
        - Customer service team preparation and FAQ development
        - Post-campaign nurturing sequence setup
        - Long-term relationship building and retention planning

        DELIVERABLE: Complete campaign playbook with step-by-step execution guide, templates, decision trees for optimization, and performance benchmarks.
        """
        return self.generate_content(prompt, "cluster_campaign_plan")

    def analyze_cluster_performance(self, cluster_data: str, performance_metrics: str) -> str:
        """Analyze cluster performance and provide optimization recommendations"""
        prompt = f"""
        You are a Performance Marketing Analyst with expertise in customer analytics. Analyze the following CLUSTER PERFORMANCE:

        CLUSTER DATA:
        {cluster_data}

        PERFORMANCE METRICS:
        {performance_metrics}

        PERFORMANCE ANALYSIS FRAMEWORK:

        1. CLUSTER HEALTH ASSESSMENT:
        - Performance vs. industry benchmark comparison
        - Trend analysis over 3-6 month period
        - Cohort behavior evolution and lifecycle patterns
        - Profitability trajectory and margin analysis
        - Risk indicators and early warning signals

        2. KEY PERFORMANCE INSIGHTS:
        - Top performing metrics and primary success drivers
        - Underperforming areas with root cause analysis
        - Statistical anomaly detection and explanations
        - Seasonal pattern identification and impact assessment
        - Competitive impact assessment and market share analysis

        3. OPTIMIZATION OPPORTUNITIES:
        - Quick wins (implementable within 1-2 weeks):
          * Immediate tactical adjustments
          * Budget reallocation opportunities
          * Creative optimization suggestions
        - Medium-term improvements (1-3 months):
          * Audience expansion strategies
          * Channel diversification opportunities
          * Attribution model refinements
        - Strategic initiatives (3-6 months):
          * Customer journey redesign
          * Technology stack optimization
          * Long-term positioning adjustments
        - Innovation opportunities and emerging trend integration
        - Budget reallocation recommendations with expected impact

        4. ACTIONABLE RECOMMENDATIONS:
        - Priority ranking using impact/effort matrix methodology
        - Specific tactics with quantified expected outcomes
        - Resource requirements (budget, personnel, technology)
        - Implementation timeline with key milestones
        - Success measurement criteria and KPI definitions
        - Risk assessment for each recommendation

        5. PERFORMANCE FORECASTING:
        - 3-month performance projections based on current trends
        - Scenario planning (best case, realistic, worst case)
        - Budget impact analysis and ROI projections
        - Market condition sensitivity analysis

        6. COMPETITIVE POSITIONING:
        - Cluster performance vs. competitor benchmarks
        - Market share opportunities and threats
        - Differentiation strategies for improved performance
        - Competitive response recommendations

        DELIVERABLE: Executive performance report with data-driven insights, prioritized action plan, implementation roadmap, and success metrics framework.
        """
        return self.generate_content(prompt, "cluster_performance")

def main():
    st.set_page_config(page_title="Enterprise Marketing Intelligence Platform", layout="wide")
    
    # Enhanced CSS for professional look
    st.markdown("""
    <style>
    body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .header { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; 
        border-radius: 12px; 
        color: white; 
        text-align: center; 
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .enterprise-card { 
        background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem; 
        border-radius: 12px; 
        margin-bottom: 1.5rem; 
        border: 1px solid #cbd5e0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4f46e5;
        margin: 1rem 0;
    }
    .professional-button { 
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white; 
        padding: 0.875rem 2rem; 
        border-radius: 8px; 
        font-size: 1rem; 
        font-weight: 600; 
        border: none; 
        cursor: pointer; 
        width: 100%; 
        text-align: center;
        transition: all 0.3s ease;
    }
    .professional-button:hover { 
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    .insight-text { 
        color: #374151; 
        font-size: 0.95rem; 
        line-height: 1.6;
        margin-bottom: 1rem; 
    }
    .warning-banner {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #92400e;
    }
    </style>
    """, unsafe_allow_html=True)

    # Professional sidebar
    with st.sidebar:
        st.markdown("<div class='header'><h2>🎯 Enterprise Marketing Intelligence</h2></div>", unsafe_allow_html=True)
        api_key = st.text_input("Gemini API Key (Optional Override)", type="password", 
                               help="Enter API key to override default configuration")
        
        st.markdown("### 📊 Platform Features")
        st.markdown("""
        - **Advanced Buyer Personas**: Multi-dimensional customer analysis
        - **Cluster-Based Campaigns**: Optimized campaigns by customer segments
        - **Performance Analytics**: ROI measurement and optimization
        - **Strategic Recommendations**: Professional AI-driven insights
        """)
    
    # Main content
    st.markdown("<div class='header'><h1>🚀 Enterprise Marketing Intelligence Platform</h1><p>Advanced Customer Segmentation & Campaign Optimization for Enterprise Teams</p></div>", unsafe_allow_html=True)
    
    ai_assistant = MarketingAIAssistant(api_key)
    if not ai_assistant.available:
        st.markdown(f"<div class='warning-banner'><strong>⚠️ Connection Issue:</strong> {ai_assistant.error_message}</div>", unsafe_allow_html=True)
        return
    
    tabs = st.tabs(["🎯 Buyer Persona Analysis", "📊 Campaign Strategy", "🔍 Cluster Intelligence", "📈 Performance Analysis"])
    
    with tabs[0]:
        st.markdown("## Advanced Buyer Persona Development")
        st.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            business_info = st.text_area(
                "Business Information & Market Context:", 
                height=120,
                help="Enter detailed information about your business, products, and market"
            )
            customer_data = st.text_area(
                "Existing Customer Data & Insights:", 
                height=120,
                help="Customer data, research insights, analytics data"
            )
        
        with col2:
            st.markdown("### 💡 Best Practices")
            st.markdown("""
            - Include quantitative data when available
            - Describe competitive landscape
            - Include customer feedback
            - Specify business objectives
            """)
        
        if st.button("🎯 Generate Advanced Personas", key="create_persona"):
            if business_info:
                with st.spinner("🔄 Analyzing customer data and generating insights..."):
                    result = ai_assistant.create_buyer_persona(business_info, customer_data)
                    st.markdown("### 📋 Buyer Persona Analysis Results")
                    st.markdown(result)
            else:
                st.error("⚠️ Please provide business information to proceed.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("## Strategic Campaign Planning")
        st.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            product_info = st.text_area("Product/Service Details:", height=100)
            target_audience = st.text_area("Target Audience Specification:", height=100)
        
        with col2:
            budget = st.selectbox("Campaign Budget Range:", [
                "< $10,000", "$10,000-$25,000", "$25,000-$100,000", 
                "$100,000-$250,000", "$250,000-$1M", "> $1M"
            ])
            duration = st.selectbox("Campaign Duration:", [
                "2 weeks", "1 month", "6 weeks", "2 months", "3 months", "6 months", "1 year"
            ])
        
        if st.button("📊 Generate Campaign Strategy", key="create_campaign"):
            if product_info and target_audience:
                with st.spinner("🔄 Developing comprehensive campaign strategy..."):
                    result = ai_assistant.create_campaign_plan(product_info, target_audience, budget, duration)
                    st.markdown("### 📈 Campaign Strategy Document")
                    st.markdown(result)
            else:
                st.error("⚠️ Please provide complete product and audience information.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("## Customer Cluster Intelligence")
        st.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
        
        cluster_analysis_type = st.radio("Analysis Type:", [
            "Cluster Persona Development", 
            "Cluster-Specific Campaign Planning"
        ])
        
        cluster_info = st.text_area(
            "Customer Cluster Data:", 
            height=150,
            help="Enter customer cluster data: demographics, behaviors, purchase patterns, preferences"
        )
        
        if cluster_analysis_type == "Cluster-Specific Campaign Planning":
            col1, col2 = st.columns(2)
            with col1:
                cluster_product_info = st.text_area("Product/Service for This Cluster:", height=80)
                cluster_budget = st.text_input("Cluster-Specific Budget:")
            with col2:
                cluster_duration = st.selectbox("Campaign Timeline:", [
                    "2 weeks", "1 month", "6 weeks", "2 months", "3 months"
                ])
        
        if cluster_analysis_type == "Cluster Persona Development":
            if st.button("🔍 Analyze Customer Cluster", key="analyze_cluster"):
                if cluster_info:
                    with st.spinner("🔄 Performing deep cluster analysis..."):
                        result = ai_assistant.create_cluster_buyer_persona(cluster_info)
                        st.markdown("### 🎯 Cluster Analysis Report")
                        st.markdown(result)
                else:
                    st.error("⚠️ Please provide cluster data for analysis.")
        
        else:  # Campaign Planning
            if st.button("🚀 Create Cluster Campaign", key="cluster_campaign"):
                if cluster_info and cluster_product_info:
                    with st.spinner("🔄 Designing cluster-optimized campaign..."):
                        result = ai_assistant.create_cluster_campaign_plan(
                            cluster_info, cluster_product_info, cluster_budget, cluster_duration
                        )
                        st.markdown("### 📊 Cluster Campaign Strategy")
                        st.markdown(result)
                else:
                    st.error("⚠️ Please provide cluster data and product information.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("## Performance Analytics & Optimization")
        st.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
        
        st.markdown("### 📈 Cluster Performance Analysis")
        perf_cluster_data = st.text_area(
            "Cluster Definition & Characteristics:", 
            height=100,
            help="Describe the cluster for performance analysis"
        )
        
        performance_metrics = st.text_area(
            "Performance Data & Metrics:", 
            height=120,
            help="KPIs, conversion rates, ROI, engagement metrics, trend data"
        )
        
        if st.button("📊 Analyze Performance", key="analyze_performance"):
            if perf_cluster_data and performance_metrics:
                with st.spinner("🔄 Analyzing performance data and generating insights..."):
                    result = ai_assistant.analyze_cluster_performance(perf_cluster_data, performance_metrics)
                    st.markdown("### 📈 Performance Analysis Report")
                    st.markdown(result)
            else:
                st.error("⚠️ Please provide cluster data and performance metrics.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🏢 Enterprise Marketing Intelligence Platform")
    st.markdown("Powered by Advanced AI • Designed for Marketing Professionals • Built for Scale")

if __name__ == "__main__":
    main()