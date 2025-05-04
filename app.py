import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

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

def plot_feature_importance(model, feature_names):
    """Create a bar chart of feature importance for a logistic regression model"""
    # Get coefficients from the model
    coefficients = model.coef_[0]
    
    # Create a DataFrame for plotting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coefficients)
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Create the plot
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for Financial Fitness',
        color_discrete_sequence=['#00cc96']
    )
    
    fig.update_layout(
        xaxis_title='Importance (Absolute Coefficient Value)',
        yaxis_title='Financial Factor',
        template='plotly_dark'
    )
    
    return fig

def plot_confusion_matrix(conf_matrix):
    """Create a heatmap visualization of a confusion matrix"""
    # Labels for the confusion matrix
    labels = ['Unfit', 'Fit']
    
    # Create the heatmap
    fig = px.imshow(
        conf_matrix,
        x=labels,
        y=labels,
        color_continuous_scale=['#ff4b4b', '#00cc96'],
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    
    # Add text annotations
    annotations = []
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            annotations.append({
                'x': labels[j],
                'y': labels[i],
                'text': str(conf_matrix[i, j]),
                'showarrow': False,
                'font': {'color': 'white', 'size': 16}
            })
    
    fig.update_layout(
        title='Confusion Matrix',
        annotations=annotations,
        template='plotly_dark'
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
                
                # Actual model training
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Store model in session state
                st.session_state.model = model
                
                # Display model information
                status_text.text("Model training complete!")
                st.subheader("Model Information")
                
                # Show coefficients
                st.write("**Feature Importance**")
                
                # Create feature importance plot
                fig = plot_feature_importance(model, st.session_state.X_train.columns)
                st.plotly_chart(fig, use_container_width=True)
                
                display_success("Model training complete! Your financial fitness model is ready for evaluation.")
                st.button("Continue to Model Evaluation ➡️", on_click=go_to_page, args=('model_eval',), use_container_width=True)
                
            except Exception as e:
                display_error(f"Error in model training: {str(e)}")
    
    st.button("◀️ Back to Data Preparation", on_click=go_to_page, args=('data_prep',))

def model_eval_page():
    st.title("Step 5: Fitness Performance Check 📊")
    
    if st.session_state.model is None:
        display_error("No trained model found! Please go back and train the model first.")
        st.button("◀️ Go to Model Training", on_click=go_to_page, args=('model_training',), use_container_width=True)
        return
    
    st.write("Let's evaluate how well our financial fitness model performs.")
    
    with st.expander("About Model Evaluation", expanded=True):
        st.markdown("""
        ### Understanding Model Performance
        
        Just like tracking progress in the gym, we need metrics to assess our model:
        
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: Correctly identified fit investors (avoiding false positives)
        - **Recall**: Ability to find all fit investors (avoiding false negatives)
        - **F1 Score**: Balance between precision and recall
        
        These metrics help us understand if our fitness assessment is reliable.
        """)
    
    # Evaluate model when button is clicked
    if st.button("Evaluate Financial Fitness Model", use_container_width=True):
        with st.spinner("Evaluating model performance..."):
            try:
                # Make predictions
                y_pred = st.session_state.model.predict(st.session_state.X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                precision = precision_score(st.session_state.y_test, y_pred, zero_division=0)
                recall = recall_score(st.session_state.y_test, y_pred, zero_division=0)
                f1 = f1_score(st.session_state.y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(st.session_state.y_test, y_pred)
                
                # Show progress
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.metric("Precision", f"{precision:.2%}")
                with col2:
                    st.metric("Recall", f"{recall:.2%}")
                    st.metric("F1 Score", f"{f1:.2%}")
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                fig = plot_confusion_matrix(conf_matrix)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.subheader("What Does This Mean?")
                interpretation = f"""
                - The model is **{accuracy:.2%} accurate** in predicting financial fitness.
                - Out of all investors predicted as fit, **{precision:.2%}** actually are fit (precision).
                - Of all truly fit investors, the model identifies **{recall:.2%}** of them (recall).
                
                These results show that our financial fitness assessment is {accuracy >= 0.75 and "reliable" or "still improving"}.
                """
                st.markdown(interpretation)
                
                display_success("Model evaluation complete! Now let's assess your fitness for the selected stock.")
                st.button("Continue to Fitness Assessment ➡️", on_click=go_to_page, args=('fitness_assessment',), use_container_width=True)
                
            except Exception as e:
                display_error(f"Error in model evaluation: {str(e)}")
    
    st.button("◀️ Back to Model Training", on_click=go_to_page, args=('model_training',))

def fitness_assessment_page():
    st.title("Step 6: Your Personal Fitness Assessment 💯")
    
    if st.session_state.model is None or st.session_state.stock_symbol is None:
        display_error("Missing model or stock selection! Please complete previous steps first.")
        st.button("◀️ Go to Model Evaluation", on_click=go_to_page, args=('model_eval',), use_container_width=True)
        return
    
    st.write(f"Let's assess your investment fitness for {st.session_state.stock_symbol}.")
    
    # Get a sample profile from the dataset (first row)
    profile = st.session_state.X_train.iloc[0].copy()
    
    st.subheader("Customize Your Financial Profile")
    st.write("Adjust the sliders below to see how different factors affect your investment fitness:")
    
    # Create input widgets based on available features
    col1, col2 = st.columns(2)
    
    user_profile = {}
    
    # Dynamically create sliders for numeric features
    i = 0
    for feature in profile.index:
        # Determine column for balanced layout
        col = col1 if i % 2 == 0 else col2
        
        with col:
            # Create appropriate input widget based on feature name/type
            if "budget" in feature.lower():
                user_profile[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=1000, 
                    max_value=15000, 
                    value=int(profile[feature] * 5000) if profile[feature] < 3 else 5000,
                    step=100
                )
            elif "risk" in feature.lower():
                user_profile[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=1, 
                    max_value=10, 
                    value=5, 
                    step=1
                )
            elif "horizon" in feature.lower():
                user_profile[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()} (years)", 
                    min_value=1, 
                    max_value=20, 
                    value=5, 
                    step=1
                )
            elif "ratio" in feature.lower():
                user_profile[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.3, 
                    step=0.05
                )
            else:
                # Default for other numeric features
                user_profile[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=-3, 
                    max_value=3, 
                    value=0, 
                    step=1
                )
        i += 1
    
    # Create a DataFrame for the user profile
    user_df = pd.DataFrame([user_profile])
    
    # Make prediction when button is clicked
    if st.button("Assess My Financial Fitness", use_container_width=True):
        with st.spinner("Analyzing your financial profile..."):
            try:
                # Prediction
                prediction = st.session_state.model.predict(user_df)[0]
                probability = st.session_state.model.predict_proba(user_df)[0][1] * 100
                
                # Store results
                st.session_state.prediction_results = {
                    'prediction': prediction,
                    'probability': probability,
                    'profile': user_df
                }
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate analysis progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text(f"Analyzing risk factors... {i+1}%")
                    elif i < 60:
                        status_text.text(f"Evaluating market conditions... {i+1}%")
                    elif i < 90:
                        status_text.text(f"Calculating fitness score... {i+1}%")
                    else:
                        status_text.text(f"Finalizing assessment... {i+1}%")
                    time.sleep(0.02)
                
                # Display result with animation
                st.balloons() if prediction == 1 else None
                
                # Clear status text
                status_text.empty()
                
                # Show detailed results
                st.subheader("Your Fitness Assessment Results")
                
                # Create fitness meter
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    fitness_label = "FIT 💪" if prediction == 1 else "UNFIT 😓"
                    fitness_color = "#00cc96" if prediction == 1 else "#ff4b4b"
                    
                    st.markdown(f"""
                    <div style="background-color: {fitness_color}; padding: 20px; border-radius: 10px; text-align: center;">
                        <h1 style="color: white; margin: 0;">{fitness_label}</h1>
                        <h2 style="color: white; margin: 10px 0;">{probability:.1f}% Confidence</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations based on fitness
                st.subheader("Financial Fitness Recommendations")
                
                if prediction == 1:
                    st.markdown(f"""
                    ### Good news! You're financially fit for investing in {st.session_state.stock_symbol}! 🎉
                    
                    **Key Strengths**:
                    - Your risk tolerance aligns well with this investment
                    - Your investment horizon is appropriate
                    - Your budget allocation appears sufficient
                    
                    **Next Steps**:
                    - Consider diversifying your portfolio beyond just {st.session_state.stock_symbol}
                    - Set up regular monitoring of your investment performance
                    - Establish clear entry and exit strategies
                    """)
                else:
                    st.markdown(f"""
                    ### You may not be financially fit for {st.session_state.stock_symbol} at this time. 🧘‍♂️
                    
                    **Areas to Improve**:
                    - Review your risk tolerance - this investment may be too volatile
                    - Consider extending your investment horizon
                    - Evaluate your budget allocation and potentially adjust
                    
                    **Alternative Approaches**:
                    - Start with lower-risk investments to build your portfolio
                    - Improve your emergency fund before taking on higher risk
                    - Consider working with a financial advisor for personalized guidance
                    """)
                
                # Feature importance for this prediction
                st.subheader("What Factors Influenced Your Fitness Assessment?")
                
                # Show the most important features that influenced the prediction
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Importance': np.abs(st.session_state.model.coef_[0]),
                    'Value': user_df.values[0]
                })
                
                # Sort by importance
                feature_importance = feature_importance.sort_values('Importance', ascending=False).head(5)
                
                # Display table
                st.table(feature_importance[['Feature', 'Value', 'Importance']])
                
                st.button("Continue to Summary ➡️", on_click=go_to_page, args=('summary',), use_container_width=True)
                
            except Exception as e:
                display_error(f"Error in fitness assessment: {str(e)}")
    
    st.button("◀️ Back to Model Evaluation", on_click=go_to_page, args=('model_eval',))

def summary_page():
    st.title("Financial Fitness Journey Complete! 🏅")
    
    if st.session_state.prediction_results is None:
        display_error("Missing assessment results! Please complete the fitness assessment first.")
        st.button("◀️ Go to Fitness Assessment", on_click=go_to_page, args=('fitness_assessment',), use_container_width=True)
        return
    
    # Extract prediction results
    prediction = st.session_state.prediction_results['prediction']
    probability = st.session_state.prediction_results['probability']
    
    # Display summary header
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if prediction == 1:
            st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3c6.svg", width=150)
        else:
            st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3c3.svg", width=150)
    
    with col2:
        st.title("Your Financial Fitness Summary")
        st.write(f"Stock analyzed: **{st.session_state.stock_symbol}**")
        
        fitness_label = "Financially Fit" if prediction == 1 else "Not Financially Fit"
        fitness_color = "#00cc96" if prediction == 1 else "#ff4b4b"
        
        st.markdown(f"""
        <div style="background-color: {fitness_color}; padding: 10px; border-radius: 5px; width: fit-content;">
            <h2 style="color: white; margin: 0; padding: 0 10px;">{fitness_label} ({probability:.1f}%)</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary of journey
    st.subheader("Your Financial Fitness Journey")
    
    journey_steps = [
        "**Health Check-In**: Loaded your financial profile data",
        f"**Target Selection**: Analyzed {st.session_state.stock_symbol} stock",
        "**Data Detox**: Prepared your data for analysis",
        "**Fitness Training**: Trained a machine learning model",
        "**Performance Check**: Evaluated model accuracy",
        "**Fitness Assessment**: Personalized investment fitness analysis"
    ]
    
    for i, step in enumerate(journey_steps, 1):
        st.markdown(f"{i}. {step}")
    
    # Key findings and recommendations
    st.subheader("Key Findings")
    
    if prediction == 1:
        st.markdown("""
        ### Congratulations on your financial fitness! 🎉
        
        Our analysis indicates that your financial profile is well-suited for your selected investment. Your risk tolerance, investment horizon, and budget allocation align with the characteristics of the chosen stock.
        
        **Strengths to Maintain**:
        - Continue your disciplined approach to investing
        - Your diversified risk approach is working well
        - Your investment horizon matches well with market volatility
        
        **Next Steps for Financial Excellence**:
        - Consider a quarterly review of your investment strategy
        - Stay informed about market changes affecting your investments
        - Evaluate opportunities to optimize your portfolio further
        """)
    else:
        st.markdown("""
        ### Thank you for taking your financial fitness seriously 🧘‍♂️
        
        Our analysis suggests your financial profile may not be optimally aligned with your selected investment at this time. However, financial fitness is a journey, not a destination.
        
        **Areas for Improvement**:
        - Consider adjusting your risk tolerance approach
        - Your investment horizon may need to be extended for this type of investment
        - Re-evaluate your budget allocation strategy
        
        **Steps Toward Financial Fitness**:
        - Start with lower-risk investments to build your portfolio
        - Establish a stronger emergency fund before higher-risk investments
        - Consider financial education resources to build investing confidence
        """)
    
    # Final call to action
    st.subheader("Continue Your Financial Fitness Journey")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Start Over", use_container_width=True):
            # Reset session state
            for key in st.session_state.keys():
                if key != 'page':
                    st.session_state[key] = None
            go_to_page('intro')
    
    with col2:
        if st.button("📊 Try Another Stock", use_container_width=True):
            # Keep the loaded data but reset other states
            st.session_state.stock_data = None
            st.session_state.stock_symbol = None
            st.session_state.model = None
            st.session_state.X_train = None
            st.session_state.y_train = None
            st.session_state.X_test = None
            st.session_state.y_test = None
            st.session_state.prediction_results = None
            go_to_page('stock_selection')
    
    with col3:
        if st.button("✏️ Adjust Your Profile", use_container_width=True):
            # Keep data and model but reset prediction
            st.session_state.prediction_results = None
            go_to_page('fitness_assessment')

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
        if st.button("💯 Fitness Assessment", use_container_width=True):
            go_to_page('fitness_assessment')
        if st.button("🏅 Summary", use_container_width=True):
            go_to_page('summary')
        
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
    elif st.session_state.page == 'fitness_assessment':
        fitness_assessment_page()
    elif st.session_state.page == 'summary':
        summary_page()

if __name__ == "__main__":
    main()