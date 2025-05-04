import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import ML models
from ml_models import train_linear_regression, train_logistic_regression, train_kmeans_clustering

# Set page configuration
st.set_page_config(
    page_title="FinFit - Your Financial Health Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'intro'

if 'data' not in st.session_state:
    st.session_state.data = None

if 'model' not in st.session_state:
    st.session_state.model = None
    
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
    
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
    
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
    
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
    
if 'stock_symbol' not in st.session_state:
    st.session_state.stock_symbol = None

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Helper functions
def load_data(file_path):
    """Load financial profile data from a CSV file"""
    try:
        data = pd.read_csv(file_path)
        
        # Basic validation of the dataset structure
        required_columns = ['budget', 'risk_tolerance', 'investment_horizon', 'investment_type', 'fitness']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
        
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(data):
    """Preprocess the financial profile data for ML modeling"""
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Fill missing numeric values with median
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Convert categorical variables to numeric
    # One-hot encode categorical columns except the target variable
    categorical_cols_to_encode = [col for col in categorical_columns if col != 'fitness']
    
    if categorical_cols_to_encode:
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
    
    # Ensure target variable is binary (0 or 1)
    if 'fitness' in df.columns:
        # If fitness is categorical (e.g., 'Fit'/'Unfit'), convert to binary
        if df['fitness'].dtype == 'object':
            fitness_mapping = {'Fit': 1, 'Unfit': 0}
            df['fitness'] = df['fitness'].map(fitness_mapping)
        
        # Ensure fitness is either 0 or 1
        df['fitness'] = df['fitness'].astype(int)
    
    return df

def get_stock_data(symbol, period="1mo", interval="1d"):
    """Get stock data from Yahoo Finance"""
    try:
        # Get stock data from Yahoo Finance
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        # Basic validation
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        return data
    
    except Exception as e:
        raise Exception(f"Error fetching stock data for {symbol}: {str(e)}")

def plot_stock_data(stock_data):
    """Create a candlestick chart for stock price data"""
    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        increasing_line_color='#00cc96',
        decreasing_line_color='#ff4b4b'
    )])
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=stock_data.index,
        y=stock_data['Volume'],
        marker_color='rgba(128, 128, 128, 0.3)',
        name='Volume',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Stock Price Movement',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        )
    )
    
    return fig

# Navigation functions
def go_to_page(page):
    st.session_state.page = page
    st.rerun()

# Helper functions for UI
def display_success(message):
    st.success(message)
    
def display_error(message):
    st.error(message)

def display_info(message):
    st.info(message)

# Pages
def intro_page():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4aa.svg", width=150)
    
    with col2:
        st.title("FinFit – Your Financial Health Dashboard")
        st.subheader("Where Financial Fitness meets Machine Learning")
        st.write("AF3005 – Programming for Finance | FAST-NUCES Islamabad | Spring 2025")

    st.markdown("""
    ### Welcome to Your Financial Gym!
    
    Just like physical fitness, financial fitness requires assessment, training, and monitoring.
    
    With FinFit, you can:
    - Upload your financial profile data
    - Analyze real-time stock information
    - Run a machine learning pipeline to assess your investment fitness
    - Get personalized recommendations for your financial health
    
    Let's start your financial fitness journey today!
    """)
    
    st.button("Start Your Financial Fitness Check 🏋️", on_click=go_to_page, args=('data_loading',), use_container_width=True)

def data_loading_page():
    st.title("Step 1: Financial Health Check-In 📋")
    st.write("Upload your financial profile or use our sample data to get started.")
    
    st.markdown("""
    ### What is a Kragle Financial Profile?
    
    A financial profile includes information such as:
    - Your budget allocation
    - Risk tolerance level
    - Investment time horizon
    - Investment types and preferences
    - Income stability and other financial metrics
    
    Our machine learning model will use this data to evaluate your fitness for specific investments.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload your Kragle financial profile (CSV)", type=["csv"])
        if uploaded_file is not None:
            try:
                data = load_data(uploaded_file)
                st.session_state.data = data
                display_success("Data loaded successfully! Your financial profile is ready for analysis.")
                st.dataframe(data.head(), use_container_width=True)
                st.button("Continue to Stock Selection ➡️", on_click=go_to_page, args=('stock_selection',), use_container_width=True)
            except Exception as e:
                display_error(f"Error loading data: {str(e)}")
        
    with col2:
        st.write("Or use our sample dataset to explore the app")
        if st.button("Use Sample Data", use_container_width=True):
            try:
                # Loading sample data
                data = load_data("kragle_investor_fitness.csv")
                st.session_state.data = data
                display_success("Sample data loaded successfully!")
                st.dataframe(data.head(), use_container_width=True)
                st.button("Continue to Stock Selection ➡️", on_click=go_to_page, args=('stock_selection',), use_container_width=True)
            except Exception as e:
                display_error(f"Error loading sample data: {str(e)}")
                
    st.button("◀️ Back to Intro", on_click=go_to_page, args=('intro',))

def stock_selection_page():
    st.title("Step 2: Choose Your Investment Target 🎯")
    
    if st.session_state.data is None:
        display_error("No data loaded! Please go back and load your financial profile first.")
        st.button("◀️ Go to Data Loading", on_click=go_to_page, args=('data_loading',), use_container_width=True)
        return
        
    st.write("Select a stock to analyze and determine your investment fitness.")
    
    # Popular stock options
    popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Quick Selection")
        stock_symbol = st.selectbox("Choose from popular stocks:", 
                                  popular_stocks,
                                  index=None,
                                  placeholder="Select a stock...")
    
    with col2:
        st.subheader("Custom Search")
        custom_symbol = st.text_input("Enter stock symbol (e.g., AAPL):", "")
        
        if st.button("Use Custom Symbol", use_container_width=True):
            if custom_symbol:
                stock_symbol = custom_symbol.upper()
            else:
                display_error("Please enter a stock symbol")
    
    if stock_symbol:
        st.session_state.stock_symbol = stock_symbol
        with st.spinner(f"Fetching latest data for {stock_symbol}..."):
            try:
                stock_data = get_stock_data(stock_symbol)
                st.session_state.stock_data = stock_data
                
                # Display stock information
                st.subheader(f"{stock_symbol} Stock Overview")
                
                # Get the latest price data
                latest_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2]
                price_change = latest_price - prev_price
                price_change_pct = (price_change / prev_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${latest_price:.2f}", 
                              f"{price_change:.2f} ({price_change_pct:.2f}%)")
                with col2:
                    st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,}")
                with col3:
                    high_low_diff = stock_data['High'].iloc[-1] - stock_data['Low'].iloc[-1]
                    st.metric("Day Range", f"${stock_data['Low'].iloc[-1]:.2f} - ${stock_data['High'].iloc[-1]:.2f}", 
                              f"Spread: ${high_low_diff:.2f}")
                
                # Plot stock data
                st.subheader("Recent Price Movement")
                fig = plot_stock_data(stock_data)
                st.plotly_chart(fig, use_container_width=True)
                
                st.button("Continue to Data Preparation ➡️", on_click=go_to_page, args=('data_prep',), use_container_width=True)
                
            except Exception as e:
                display_error(f"Error fetching stock data: {str(e)}")
    
    st.button("◀️ Back to Data Loading", on_click=go_to_page, args=('data_loading',))

def data_prep_page():
    st.title("Step 3: Financial Data Detox 🧹")
    
    if st.session_state.data is None or st.session_state.stock_data is None:
        display_error("Missing data! Please go back and complete previous steps.")
        st.button("◀️ Go to Data Loading", on_click=go_to_page, args=('data_loading',), use_container_width=True)
        return
    
    st.write("Let's prepare your financial data for the fitness assessment.")
    
    with st.expander("About Data Preparation", expanded=True):
        st.markdown("""
        ### Why Data Preparation Matters
        
        Just like warming up before a workout, proper data preparation is essential for accurate results:
        
        - **Cleaning**: Removing missing or invalid values
        - **Preprocessing**: Scaling and normalizing data for optimal model performance
        - **Feature Engineering**: Creating new insights from existing data
        
        This step ensures our machine learning model can effectively assess your financial fitness.
        """)
    
    # Show original data
    st.subheader("Your Original Financial Profile")
    st.dataframe(st.session_state.data.head(), use_container_width=True)
    
    # Process data when button is clicked
    if st.button("Start Data Preparation", use_container_width=True):
        with st.spinner("Preparing your financial data..."):
            try:
                # Data preprocessing
                processed_data = preprocess_data(st.session_state.data)
                
                # Data splitting
                X = processed_data.drop('fitness', axis=1)
                y = processed_data['fitness']
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Show progress
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Display processed data
                st.subheader("Processed Financial Profile")
                st.dataframe(processed_data.head(), use_container_width=True)
                
                # Show statistics
                st.subheader("Data Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Training data size:** {len(X_train)} profiles")
                    st.write(f"**Testing data size:** {len(X_test)} profiles")
                with col2:
                    st.write(f"**Features used:** {', '.join(X_train.columns)}")
                    st.write(f"**Target variable:** Investment Fitness (1 = Fit, 0 = Unfit)")
                
                display_success("Data preparation complete! Your financial data is now ready for the fitness model.")
                st.button("Continue to Model Training ➡️", on_click=go_to_page, args=('model_training',), use_container_width=True)
                
            except Exception as e:
                display_error(f"Error in data preparation: {str(e)}")
    
    st.button("◀️ Back to Stock Selection", on_click=go_to_page, args=('stock_selection',))

def model_training_page():
    st.title("Step 4: Financial Fitness Training 🏋️‍♀️")
    
    if (st.session_state.X_train is None or st.session_state.y_train is None or 
        st.session_state.X_test is None or st.session_state.y_test is None):
        display_error("Missing processed data! Please go back and complete the data preparation step.")
        st.button("◀️ Go to Data Preparation", on_click=go_to_page, args=('data_prep',), use_container_width=True)
        return
    
    st.write("Now we'll train a machine learning model to assess your investment fitness.")
    
    with st.expander("About the Fitness Model", expanded=True):
        st.markdown("""
        ### Logistic Regression for Financial Fitness
        
        We use Logistic Regression to determine your investment fitness:
        
        - **Binary Classification**: Identifies if you're "Fit" (1) or "Unfit" (0) for the selected investment
        - **Probability Based**: Provides confidence scores for the fitness assessment
        - **Feature Importance**: Reveals which financial factors most impact your fitness level
        
        This is similar to how a personal trainer assesses your physical fitness for different workout types.
        """)
    
    # Show training data overview
    st.subheader("Training Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Number of features: {st.session_state.X_train.shape[1]}")
        st.write(f"Number of training samples: {len(st.session_state.X_train)}")
    with col2:
        fit_percentage = (st.session_state.y_train.sum() / len(st.session_state.y_train)) * 100
        st.write(f"'Fit' investors in training data: {fit_percentage:.1f}%")
        st.write(f"'Unfit' investors in training data: {100-fit_percentage:.1f}%")
    
    # Train model when button is clicked
    if st.button("Train Financial Fitness Model", use_container_width=True):
        with st.spinner("Training your financial fitness model..."):
            try:
                # Model training with progress simulation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate training progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text(f"Initializing model parameters... {i+1}%")
                    elif i < 60:
                        status_text.text(f"Optimizing coefficients... {i+1}%")
                    elif i < 90:
                        status_text.text(f"Calculating probabilities... {i+1}%")
                    else:
                        status_text.text(f"Finalizing model... {i+1}%")
                    time.sleep(0.03)
                
                # Actual model training using our ml_models module
                model, feature_importance = train_logistic_regression(st.session_state.X_train, st.session_state.y_train)
                
                # Store model in session state
                st.session_state.model = model
                
                # Display model information
                status_text.text("Model training complete!")
                st.subheader("Model Information")
                
                # Show coefficients
                st.write("**Feature Importance**")
                
                # Create feature importance plot
                fig = px.bar(
                    feature_importance,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for Financial Fitness'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                display_success("Model training complete! Your financial fitness model is ready for evaluation.")
                st.button("Continue to Model Evaluation ➡️", on_click=go_to_page, args=('model_eval',), use_container_width=True)
                
            except Exception as e:
                display_error(f"Error in model training: {str(e)}")
    
    st.button("◀️ Back to Data Preparation", on_click=go_to_page, args=('data_prep',))

def model_eval_page():
    # Add model evaluation code here
    st.title("Step 5: Fitness Performance Check 📊")
    st.write("Model evaluation page - implementation in progress")
    st.button("◀️ Back to Model Training", on_click=go_to_page, args=('model_training',))

# Main function
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3cb.svg", width=50)
        st.title("FinFit")
        st.write("Your Financial Fitness Dashboard")
        st.divider()
        
        st.subheader("Navigation")
        
        # Simplified navigation buttons
        if st.button("⭐ Intro", use_container_width=True):
            go_to_page('intro')
        if st.button("📋 Data Loading", use_container_width=True):
            go_to_page('data_loading')
        if st.button("🎯 Stock Selection", use_container_width=True):
            go_to_page('stock_selection')
        if st.button("🧹 Data Preparation", use_container_width=True):
            go_to_page('data_prep')
        if st.button("🏋️‍♀️ Model Training", use_container_width=True):
            go_to_page('model_training')
        if st.button("📊 Model Evaluation", use_container_width=True):
            go_to_page('model_eval')
        
        st.divider()
        st.write("AF3005 – Programming for Finance")
        st.write("FAST-NUCES Islamabad | Spring 2025")
    
    # Render the current page
    if st.session_state.page == 'intro':
        intro_page()
    elif st.session_state.page == 'data_loading':
        data_loading_page()
    elif st.session_state.page == 'stock_selection':
        stock_selection_page()
    elif st.session_state.page == 'data_prep':
        data_prep_page()
    elif st.session_state.page == 'model_training':
        model_training_page()
    elif st.session_state.page == 'model_eval':
        model_eval_page()

if __name__ == "__main__":
    main()