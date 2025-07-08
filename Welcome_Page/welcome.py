import streamlit as st
import os

def main():
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .feature-card {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .navigation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .step-number {
        background-color: #1f77b4;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .highlight-box {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align:left;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with emoji
    st.markdown('<h1 class="main-header">🚀 Flash Insights</h1>', unsafe_allow_html=True)
    
    # Welcome message with better formatting
    st.markdown("---")
    
    # What is this section with enhanced styling
    st.markdown("## 🔍 What is this?")
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Your Complete Data Science Toolkit</h4>
        <p>This comprehensive dashboard empowers you to transform raw data into actionable insights with ease. 
        Whether you're a beginner or an expert, our intuitive interface guides you through every step of the data analysis process.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section with icons
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Data Management**
        - Upload datasets in multiple formats
        - Rename and drop columns effortlessly
        - Real-time data preview and exploration
        """)
        
        st.markdown("""
        **🔧 Data Processing**
        - Advanced smoothing and filtering tools
        - Data interpolation and cleaning
        - Quality enhancement algorithms
        """)
    
    with col2:
        st.markdown("""
        **🤖 Machine Learning**
        - Classification algorithms (Random Forest, SVM, etc.)
        - Regression models with Neural Networks
        - Customizable parameters and hypertuning
        """)
        
        st.markdown("""
        **📈 Visualization & Export**
        - Interactive charts and graphs
        - Downloadable results and reports
        - Professional-quality visualizations
        """)
    
    # Navigation section with enhanced styling
    st.markdown("---")
    st.markdown("## 🧭 Navigation Guide")
    
    # Navigation steps with better visual hierarchy
    navigation_steps = [
        ("Welcome", "This is where you currently are - your starting point", "🏠"),
        ("Data Preview", "Explore your dataset and discover correlations between features", "👀"),
        ("Data Preparation", "Clean and structure your data by dropping/renaming columns", "🔧"),
        ("Smoothing and Filtering", "Enhance data quality with advanced preprocessing tools", "✨"),
        ("Classification", "Apply machine learning classification algorithms", "🎯"),
        ("Regression", "Predict future data points using advanced algorithms", "📈")
    ]
    
    for i, (title, description, icon) in enumerate(navigation_steps):
        st.markdown(f"""
        <div class="highlight-box">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span class="step-number">{i}</span>
                <span style="font-size: 1.2rem;">{icon}</span>
                <strong style="margin-left: 0.5rem; font-size: 1.1rem;">{title}</strong>
            </div>
            <p style="margin-left: 45px; color:white; margin-bottom: 0;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pro tip section
    st.markdown("""
    <div class="highlight-box">
        <h3>💡 Pro Tip</h3>
        <p>For the best experience, navigate through the pages in sequence. Each step builds upon the previous one, 
        ensuring your data analysis journey is smooth and comprehensive!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Success message
    st.success("🎉 Welcome to your data analysis journey! Let's get started by navigating to the Data Preview page.")
    
    # Clean up leftover files from previous runs
    cleanup_files = [
        "Smoothing_and_Filtering//Preprocessing dataset.csv",
        "Smoothing_and_Filtering//Filtered Dataset.csv",
        "Smoothing_and_Filtering//initial.csv"
    ]
    
    for file_path in cleanup_files:
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                st.warning(f"Could not remove {file_path}: {e}")

if __name__ == "__main__":
    main()