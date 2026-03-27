import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sleep Disorder Diagnosis System",
    page_icon="🛌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Beautiful UI
# -----------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Card styling */
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #a855f7, #6366f1, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 10px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #a855f7;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 10px;
    }
    
    /* Input styling */
    .stSelectbox, .stSlider, .stNumberInput {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 40px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    }
    
    /* Success/Error boxes */
    .success-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.2));
        border: 2px solid #22c55e;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2));
        border: 2px solid #ef4444;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #a855f7);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    /* Feature importance section */
    .feature-card {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #6366f1;
    }
    
    /* Health tips */
    .tip-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.1));
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #22c55e;
    }
    
    /* Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load and Prepare Data
# -----------------------------
@st.cache_data
def load_data():
    """Load and preprocess the sleep health dataset"""
    try:
        df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    except FileNotFoundError:
        # Create sample data if file not found
        np.random.seed(42)
        n_samples = 400
        
        df = pd.DataFrame({
            "Person ID": range(1, n_samples + 1),
            "Gender": np.random.choice(["Male", "Female"], n_samples),
            "Age": np.random.randint(25, 60, n_samples),
            "Occupation": np.random.choice([
                "Software Engineer", "Doctor", "Teacher", "Nurse", "Engineer",
                "Lawyer", "Accountant", "Salesperson", "Scientist", "Manager"
            ], n_samples),
            "Sleep Duration": np.round(np.random.uniform(4, 9, n_samples), 1),
            "Quality of Sleep": np.random.randint(4, 10, n_samples),
            "Physical Activity Level": np.random.randint(30, 90, n_samples),
            "Stress Level": np.random.randint(3, 9, n_samples),
            "BMI Category": np.random.choice(["Normal", "Overweight", "Obese"], n_samples),
            "Blood Pressure": [f"{np.random.randint(110, 140)}/{np.random.randint(70, 90)}" for _ in range(n_samples)],
            "Heart Rate": np.random.randint(65, 85, n_samples),
            "Daily Steps": np.random.randint(3000, 10000, n_samples),
            "Sleep Disorder": np.random.choice(["None", "Insomnia", "Sleep Apnea"], n_samples, p=[0.6, 0.25, 0.15])
        })
    
    if "Person ID" in df.columns:
        df.drop("Person ID", axis=1, inplace=True)
    
    # Handle Blood Pressure column if it exists as string
    if df["Blood Pressure"].dtype == object:
        df["Systolic_BP"] = df["Blood Pressure"].apply(lambda x: int(str(x).split("/")[0]) if "/" in str(x) else int(x))
        df["Diastolic_BP"] = df["Blood Pressure"].apply(lambda x: int(str(x).split("/")[1]) if "/" in str(x) else 80)
        df.drop("Blood Pressure", axis=1, inplace=True)
    
    return df

@st.cache_resource
def prepare_model_data():
    """Prepare data for model training"""
    df = load_data().copy()
    
    label_encoders = {}
    categorical_columns = df.select_dtypes(include='object').columns
    
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    X = df.drop("Sleep Disorder", axis=1)
    y = df["Sleep Disorder"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, y, X_scaled, scaler, label_encoders, X.columns.tolist()

# Load data
df_original = load_data()
X, y, X_scaled, scaler, label_encoders, feature_columns = prepare_model_data()

# -----------------------------
# Train Multiple Models
# -----------------------------
@st.cache_resource
def train_all_models():
    """Train multiple ML models and return them with their scores"""
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    }
    
    trained_models = {}
    model_scores = {}
    cv_scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_score = cross_val_score(model, X_scaled, y, cv=5).mean()
        
        trained_models[name] = model
        model_scores[name] = accuracy
        cv_scores[name] = cv_score
    
    # Create Ensemble Model
    ensemble = VotingClassifier(
        estimators=[
            ('rf', trained_models["Random Forest"]),
            ('gb', trained_models["Gradient Boosting"]),
            ('svm', trained_models["SVM"])
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    
    trained_models["Ensemble (Best)"] = ensemble
    model_scores["Ensemble (Best)"] = accuracy_score(y_test, y_pred_ensemble)
    cv_scores["Ensemble (Best)"] = cross_val_score(ensemble, X_scaled, y, cv=5).mean()
    
    return trained_models, model_scores, cv_scores, X_test, y_test

trained_models, model_scores, cv_scores, X_test, y_test = train_all_models()

# Find best model
best_model_name = max(model_scores, key=model_scores.get)
best_model = trained_models[best_model_name]

# -----------------------------
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 2.5rem;">🛌</h1>
        <h2 style="background: linear-gradient(120deg, #a855f7, #6366f1);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   font-weight: 700;">Sleep Health</h2>
        <p style="color: #94a3b8;">AI Diagnosis System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home & Diagnosis", "📊 Model Analytics", "📈 Data Insights", "💡 Health Tips", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Selection
    st.markdown("### 🤖 Select Model")
    selected_model_name = st.selectbox(
        "Choose Algorithm",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index("Ensemble (Best)"),
        label_visibility="collapsed"
    )
    
    selected_model = trained_models[selected_model_name]
    
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 20px;">
        <div class="metric-value">{model_scores[selected_model_name]*100:.1f}%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 Dataset Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Samples", len(df_original))
    with col2:
        st.metric("Features", len(feature_columns))

# -----------------------------
# Main Content
# -----------------------------

if page == "🏠 Home & Diagnosis":
    # Header
    st.markdown('<h1 class="main-header">🛌 Sleep Disorder Diagnosis System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered analysis for detecting sleep disorders using machine learning</p>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{model_scores[selected_model_name]*100:.1f}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(trained_models)}</div>
            <div class="metric-label">ML Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(feature_columns)}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cv_scores[selected_model_name]*100:.1f}%</div>
            <div class="metric-label">CV Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input Form
    st.markdown("### 📝 Enter Your Health Information")
    
    with st.form("diagnosis_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Personal Info")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Select your biological gender")
            age = st.slider("Age", 18, 80, 35, help="Your current age in years")
            occupation = st.selectbox(
                "Occupation",
                label_encoders["Occupation"].classes_,
                help="Your current profession"
            )
            bmi_category = st.selectbox(
                "BMI Category",
                label_encoders["BMI Category"].classes_,
                help="Your Body Mass Index category"
            )
        
        with col2:
            st.markdown("#### 😴 Sleep Metrics")
            sleep_duration = st.slider(
                "Sleep Duration (hours)",
                4.0, 10.0, 7.0, 0.5,
                help="Average hours of sleep per night"
            )
            quality_of_sleep = st.slider(
                "Quality of Sleep",
                1, 10, 7,
                help="Rate your sleep quality (1=Poor, 10=Excellent)"
            )
            stress_level = st.slider(
                "Stress Level",
                1, 10, 5,
                help="Rate your daily stress level (1=Low, 10=High)"
            )
        
        with col3:
            st.markdown("#### ❤️ Health Metrics")
            physical_activity = st.slider(
                "Physical Activity Level",
                10, 100, 50,
                help="Daily physical activity minutes"
            )
            systolic_bp = st.slider(
                "Systolic Blood Pressure",
                90, 180, 120,
                help="Upper blood pressure reading"
            )
            diastolic_bp = st.slider(
                "Diastolic Blood Pressure",
                60, 120, 80,
                help="Lower blood pressure reading"
            )
            heart_rate = st.slider(
                "Heart Rate (BPM)",
                50, 100, 72,
                help="Resting heart rate in beats per minute"
            )
            daily_steps = st.number_input(
                "Daily Steps",
                1000, 20000, 7000, 500,
                help="Average daily step count"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔍 Analyze & Diagnose", use_container_width=True)
    
    # Process Prediction
    if submitted:
        with st.spinner("Analyzing your health data..."):
            # Prepare input data
            input_data = {
                "Gender": gender,
                "Age": age,
                "Occupation": occupation,
                "Sleep Duration": sleep_duration,
                "Quality of Sleep": quality_of_sleep,
                "Physical Activity Level": physical_activity,
                "Stress Level": stress_level,
                "BMI Category": bmi_category,
                "Systolic_BP": systolic_bp,
                "Diastolic_BP": diastolic_bp,
                "Heart Rate": heart_rate,
                "Daily Steps": daily_steps
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical features
            for column in ["Gender", "Occupation", "BMI Category"]:
                if column in input_df.columns:
                    try:
                        input_df[column] = label_encoders[column].transform(input_df[column])
                    except ValueError:
                        # Handle unknown categories
                        input_df[column] = 0
            
            # Ensure correct column order
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Get predictions from all models
            all_predictions = {}
            all_probabilities = {}
            
            for name, model in trained_models.items():
                pred = model.predict(input_scaled)[0]
                all_predictions[name] = pred
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_scaled)[0]
                    all_probabilities[name] = proba
            
            # Main prediction
            prediction = selected_model.predict(input_scaled)[0]
            
            # Decode prediction
            try:
                result_label = label_encoders["Sleep Disorder"].inverse_transform([prediction])[0]
            except:
                result_label = "None"
            
            # Get probability if available
            confidence = 0.0
            if hasattr(selected_model, 'predict_proba'):
                probabilities = selected_model.predict_proba(input_scaled)[0]
                confidence = max(probabilities) * 100
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Results Display
        st.markdown("### 🎯 Diagnosis Results")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if result_label == "None" or result_label.lower() == "none":
                st.markdown(f"""
                <div class="success-box">
                    <h2 style="color: #22c55e; margin-bottom: 10px;">✅ No Sleep Disorder Detected</h2>
                    <p style="font-size: 1.2rem; color: #86efac;">
                        Based on your health metrics, you appear to have healthy sleep patterns.
                    </p>
                    <p style="color: #94a3b8; margin-top: 15px;">
                        Confidence: <strong>{confidence:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    <h2 style="color: #ef4444; margin-bottom: 10px;">⚠️ Potential {result_label} Detected</h2>
                    <p style="font-size: 1.2rem; color: #fca5a5;">
                        Our analysis suggests you may have indicators of {result_label}.
                        Please consult a healthcare professional for proper diagnosis.
                    </p>
                    <p style="color: #94a3b8; margin-top: 15px;">
                        Confidence: <strong>{confidence:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Confidence Gauge
            if hasattr(selected_model, 'predict_proba'):
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    title={'text': "Confidence", 'font': {'color': 'white'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': 'white'},
                        'bar': {'color': "#a855f7"},
                        'bgcolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.3)"},
                            {'range': [50, 75], 'color': "rgba(251, 191, 36, 0.3)"},
                            {'range': [75, 100], 'color': "rgba(34, 197, 94, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': confidence
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    height=250,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Model Consensus
        st.markdown("### 🤖 Model Consensus")
        
        # Count predictions
        pred_counts = {}
        for name, pred in all_predictions.items():
            try:
                label = label_encoders["Sleep Disorder"].inverse_transform([pred])[0]
            except:
                label = "None"
            pred_counts[label] = pred_counts.get(label, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Consensus pie chart
            fig_pie = px.pie(
                values=list(pred_counts.values()),
                names=list(pred_counts.keys()),
                title="Model Predictions Distribution",
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Model predictions table
            pred_df = pd.DataFrame([
                {"Model": name, "Prediction": label_encoders["Sleep Disorder"].inverse_transform([pred])[0]}
                for name, pred in all_predictions.items()
            ])
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # Risk Factors Analysis
        st.markdown("### 📊 Your Health Profile Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart of health metrics
            categories = ['Sleep Duration', 'Sleep Quality', 'Physical Activity', 
                         'Stress (Inv)', 'Heart Health', 'Daily Steps']
            
            # Normalize values for radar chart (0-100 scale)
            values = [
                min(100, (sleep_duration / 10) * 100),
                quality_of_sleep * 10,
                physical_activity,
                (10 - stress_level) * 10,  # Inverted - lower stress is better
                min(100, max(0, 100 - abs(heart_rate - 70) * 2)),  # Optimal around 70
                min(100, (daily_steps / 10000) * 100)
            ]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(168, 85, 247, 0.3)',
                line=dict(color='#a855f7', width=2),
                name='Your Profile'
            ))
            
            # Add ideal profile
            ideal_values = [70, 80, 70, 80, 90, 70]
            fig_radar.add_trace(go.Scatterpolar(
                r=ideal_values + [ideal_values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(34, 197, 94, 0.2)',
                line=dict(color='#22c55e', width=2, dash='dash'),
                name='Ideal Profile'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.2)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                showlegend=True,
                title="Health Metrics Comparison"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            # Risk factors
            st.markdown("#### 🔍 Key Observations")
            
            observations = []
            
            if sleep_duration < 6:
                observations.append(("⚠️", "Low sleep duration - aim for 7-9 hours", "warning"))
            elif sleep_duration > 9:
                observations.append(("⚠️", "High sleep duration - may indicate underlying issues", "warning"))
            else:
                observations.append(("✅", "Sleep duration is within healthy range", "success"))
            
            if quality_of_sleep < 5:
                observations.append(("⚠️", "Poor sleep quality reported", "warning"))
            else:
                observations.append(("✅", "Good sleep quality", "success"))
            
            if stress_level > 7:
                observations.append(("⚠️", "High stress level - consider stress management", "warning"))
            else:
                observations.append(("✅", "Stress level is manageable", "success"))
            
            if physical_activity < 30:
                observations.append(("⚠️", "Low physical activity - aim for 30+ minutes daily", "warning"))
            else:
                observations.append(("✅", "Good physical activity level", "success"))
            
            if daily_steps < 5000:
                observations.append(("⚠️", "Low daily steps - aim for 7,000-10,000", "warning"))
            else:
                observations.append(("✅", "Good daily step count", "success"))
            
            for icon, text, status in observations:
                color = "#22c55e" if status == "success" else "#fbbf24"
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; border-left: 3px solid {color}; 
                            background: rgba(255,255,255,0.05); border-radius: 5px;">
                    {icon} {text}
                </div>
                """, unsafe_allow_html=True)
        
        # Download Results
        st.markdown("### 📥 Download Report")
        
        report_data = input_data.copy()
        report_data["Diagnosis"] = result_label
        report_data["Confidence"] = f"{confidence:.1f}%"
        report_data["Model Used"] = selected_model_name
        
        report_df = pd.DataFrame([report_data])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name="sleep_diagnosis_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON download
            json_data = report_df.to_json(orient='records')
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name="sleep_diagnosis_report.json",
                mime="application/json",
                use_container_width=True
            )

elif page == "📊 Model Analytics":
    st.markdown('<h1 class="main-header">📊 Model Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare performance across different machine learning algorithms</p>', unsafe_allow_html=True)
    
    # Model Comparison
    st.markdown("### 🏆 Model Performance Comparison")
    
    # Accuracy comparison bar chart
    models_df = pd.DataFrame({
        'Model': list(model_scores.keys()),
        'Test Accuracy': [v * 100 for v in model_scores.values()],
        'CV Score': [v * 100 for v in cv_scores.values()]
    }).sort_values('Test Accuracy', ascending=True)
    
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        y=models_df['Model'],
        x=models_df['Test Accuracy'],
        name='Test Accuracy',
        orientation='h',
        marker=dict(
            color=models_df['Test Accuracy'],
            colorscale='Purples',
            line=dict(color='rgba(255,255,255,0.5)', width=1)
        ),
        text=[f'{v:.1f}%' for v in models_df['Test Accuracy']],
        textposition='outside'
    ))
    
    fig_comparison.add_trace(go.Bar(
        y=models_df['Model'],
        x=models_df['CV Score'],
        name='Cross-Validation Score',
        orientation='h',
        marker=dict(
            color='rgba(99, 102, 241, 0.5)',
            line=dict(color='#6366f1', width=2)
        ),
        text=[f'{v:.1f}%' for v in models_df['CV Score']],
        textposition='outside'
    ))
    
    fig_comparison.update_layout(
        title='Model Accuracy Comparison',
        xaxis_title='Accuracy (%)',
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 110]),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed Metrics Table
    st.markdown("### 📋 Detailed Performance Metrics")
    
    metrics_data = []
    for name in model_scores.keys():
        metrics_data.append({
            'Model': name,
            'Test Accuracy': f"{model_scores[name]*100:.2f}%",
            'CV Score (5-fold)': f"{cv_scores[name]*100:.2f}%",
            'Variance': f"{abs(model_scores[name] - cv_scores[name])*100:.2f}%"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Highlight best model
    st.dataframe(
        metrics_df.style.highlight_max(subset=['Test Accuracy', 'CV Score (5-fold)'], color='rgba(168, 85, 247, 0.3)'),
        use_container_width=True,
        hide_index=True
    )
    
    # Feature Importance (for tree-based models)
    st.markdown("### 🎯 Feature Importance Analysis")
    
    rf_model = trained_models["Random Forest"]
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance (Random Forest)',
        color='Importance',
        color_continuous_scale='Purples'
    )
    
    fig_importance.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=500
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("### 🔢 Confusion Matrix")
    
    y_pred = selected_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Get class labels
    class_labels = label_encoders["Sleep Disorder"].classes_
    
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_labels,
        y=class_labels,
        title=f"Confusion Matrix - {selected_model_name}",
        color_continuous_scale='Purples',
        text_auto=True
    )
    
    fig_cm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=500
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)

elif page == "📈 Data Insights":
    st.markdown('<h1 class="main-header">📈 Data Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore patterns and distributions in the sleep health dataset</p>', unsafe_allow_html=True)
    
    df = load_data()
    
    # Dataset Overview
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        disorder_count = len(df[df['Sleep Disorder'] != 'None']) if 'Sleep Disorder' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{disorder_count}</div>
            <div class="metric-label">Disorder Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        healthy_pct = ((len(df) - disorder_count) / len(df) * 100) if len(df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{healthy_pct:.1f}%</div>
            <div class="metric-label">Healthy Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Distribution Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Sleep Disorder Distribution
        if 'Sleep Disorder' in df.columns:
            fig_disorder = px.pie(
                df,
                names='Sleep Disorder',
                title='Sleep Disorder Distribution',
                color_discrete_sequence=px.colors.sequential.Purples_r,
                hole=0.4
            )
            fig_disorder.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig_disorder, use_container_width=True)
    
    with col2:
        # Age Distribution
        fig_age = px.histogram(
            df,
            x='Age',
            color='Sleep Disorder' if 'Sleep Disorder' in df.columns else None,
            title='Age Distribution by Sleep Disorder',
            color_discrete_sequence=px.colors.sequential.Purples_r,
            nbins=20
        )
        fig_age.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Correlation Analysis
    st.markdown("### 🔗 Feature Correlations")
    
    # Prepare numeric data for correlation
    df_numeric = df.select_dtypes(include=[np.number])
    
    fig_corr = px.imshow(
        df_numeric.corr(),
        title='Feature Correlation Heatmap',
        color_continuous_scale='Purples',
        aspect='auto'
    )
    fig_corr.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=600
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Sleep Duration vs Quality Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter = px.scatter(
            df,
            x='Sleep Duration',
            y='Quality of Sleep',
            color='Sleep Disorder' if 'Sleep Disorder' in df.columns else None,
            size='Stress Level',
            title='Sleep Duration vs Quality',
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Occupation analysis
        if 'Occupation' in df.columns and 'Sleep Disorder' in df.columns:
            occupation_disorder = df.groupby(['Occupation', 'Sleep Disorder']).size().reset_index(name='Count')
            fig_occupation = px.bar(
                occupation_disorder,
                x='Occupation',
                y='Count',
                color='Sleep Disorder',
                title='Sleep Disorders by Occupation',
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            fig_occupation.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickangle=45),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_occupation, use_container_width=True)
    
    # Raw Data Explorer
    st.markdown("### 🔍 Data Explorer")
    
    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True, height=400)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Dataset",
            data=csv,
            file_name="sleep_health_dataset.csv",
            mime="text/csv"
        )

elif page == "💡 Health Tips":
    st.markdown('<h1 class="main-header">💡 Sleep Health Tips</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evidence-based recommendations for better sleep</p>', unsafe_allow_html=True)
    
    # Tips Categories
    tab1, tab2, tab3, tab4 = st.tabs(["🌙 Sleep Hygiene", "🍎 Lifestyle", "🧘 Relaxation", "⚠️ Warning Signs"])
    
    with tab1:
        st.markdown("""
        <div class="tip-card">
            <h3>🛏️ Optimize Your Sleep Environment</h3>
            <ul>
                <li><strong>Temperature:</strong> Keep your bedroom between 60-67°F (15-19°C)</li>
                <li><strong>Darkness:</strong> Use blackout curtains or a sleep mask</li>
                <li><strong>Noise:</strong> Consider white noise or earplugs</li>
                <li><strong>Comfort:</strong> Invest in a quality mattress and pillows</li>
            </ul>
        </div>
        
        <div class="tip-card">
            <h3>⏰ Maintain a Consistent Schedule</h3>
            <ul>
                <li>Go to bed and wake up at the same time daily</li>
                <li>Avoid sleeping in on weekends (max 1 hour difference)</li>
                <li>Create a 30-60 minute wind-down routine</li>
                <li>Limit naps to 20-30 minutes before 3 PM</li>
            </ul>
        </div>
        
        <div class="tip-card">
            <h3>📱 Manage Screen Time</h3>
            <ul>
                <li>Avoid screens 1-2 hours before bed</li>
                <li>Use blue light filters on devices</li>
                <li>Keep phones out of the bedroom</li>
                <li>Replace scrolling with reading or meditation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="tip-card">
            <h3>🏃 Exercise Regularly</h3>
            <ul>
                <li>Aim for 30+ minutes of moderate exercise daily</li>
                <li>Morning or afternoon exercise is ideal</li>
                <li>Avoid vigorous exercise within 3 hours of bedtime</li>
                <li>Even light walking improves sleep quality</li>
            </ul>
        </div>
        
        <div class="tip-card">
            <h3>🍽️ Watch Your Diet</h3>
            <ul>
                <li>Avoid heavy meals within 3 hours of bedtime</li>
                <li>Limit caffeine after 2 PM</li>
                <li>Reduce alcohol consumption (disrupts REM sleep)</li>
                <li>Stay hydrated but limit fluids before bed</li>
            </ul>
        </div>
        
        <div class="tip-card">
            <h3>☀️ Get Natural Light</h3>
            <ul>
                <li>Expose yourself to bright light in the morning</li>
                <li>Spend time outdoors during the day</li>
                <li>Natural light helps regulate your circadian rhythm</li>
                <li>Consider a light therapy box in winter</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="tip-card">
            <h3>🧘 Relaxation Techniques</h3>
            <ul>
                <li><strong>Deep Breathing:</strong> 4-7-8 technique (inhale 4s, hold 7s, exhale 8s)</li>
                <li><strong>Progressive Muscle Relaxation:</strong> Tense and release muscle groups</li>
                <li><strong>Meditation:</strong> Guided sleep meditations or body scans</li>
                <li><strong>Journaling:</strong> Write down worries before bed</li>
            </ul>
        </div>
        
        <div class="tip-card">
            <h3>🛁 Pre-Sleep Rituals</h3>
            <ul>
                <li>Take a warm bath or shower 1-2 hours before bed</li>
                <li>Practice gentle stretching or yoga</li>
                <li>Read a physical book (not on screens)</li>
                <li>Listen to calming music or nature sounds</li>
            </ul>
        </div>
        
        <div class="tip-card">
            <h3>🧠 Cognitive Strategies</h3>
            <ul>
                <li>If you can't sleep after 20 minutes, get up briefly</li>
                <li>Practice visualization of peaceful scenes</li>
                <li>Use the bed only for sleep (not work or TV)</li>
                <li>Challenge anxious thoughts about sleep</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2)); 
                    border: 2px solid #ef4444; border-radius: 15px; padding: 20px; margin: 10px 0;">
            <h3 style="color: #ef4444;">🚨 When to Seek Professional Help</h3>
            <p>Consult a healthcare provider if you experience:</p>
            <ul>
                <li>Difficulty sleeping 3+ nights per week for 3+ months</li>
                <li>Loud snoring with gasping or choking sounds</li>
                <li>Excessive daytime sleepiness affecting daily activities</li>
                <li>Restless legs or uncontrollable urge to move</li>
                <li>Sleep walking, talking, or other unusual behaviors</li>
                <li>Falling asleep during inappropriate times (driving, working)</li>
            </ul>
        </div>
        
        <div class="tip-card">
            <h3>📋 Common Sleep Disorders</h3>
            <ul>
                <li><strong>Insomnia:</strong> Difficulty falling or staying asleep</li>
                <li><strong>Sleep Apnea:</strong> Breathing interruptions during sleep</li>
                <li><strong>Restless Leg Syndrome:</strong> Uncomfortable sensations in legs</li>
                <li><strong>Narcolepsy:</strong> Sudden sleep attacks during the day</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sleep Calculator
    st.markdown("### 🧮 Sleep Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        wake_time = st.time_input("What time do you need to wake up?", value=pd.to_datetime("07:00").time())
        
        # Calculate optimal bedtimes (based on 90-minute sleep cycles)
        wake_datetime = pd.to_datetime(f"2024-01-01 {wake_time}")
        
        bedtimes = []
        for cycles in [6, 5, 4]:  # 9h, 7.5h, 6h of sleep
            sleep_duration = cycles * 90 + 15  # Add 15 min to fall asleep
            bedtime = wake_datetime - pd.Timedelta(minutes=sleep_duration)
            bedtimes.append((cycles, bedtime.strftime("%I:%M %p")))
        
        st.markdown("#### Recommended Bedtimes:")
        for cycles, bedtime in bedtimes:
            hours = cycles * 1.5
            st.markdown(f"- **{bedtime}** ({hours:.1f} hours / {cycles} cycles)")
    
    with col2:
        st.markdown("""
        #### 💡 About Sleep Cycles
        
        Sleep occurs in 90-minute cycles. Waking up at the end of a cycle 
        helps you feel more refreshed than waking mid-cycle.
        
        **Optimal sleep:** 5-6 cycles (7.5-9 hours)
        **Minimum healthy:** 4 cycles (6 hours)
        
        *Add 15 minutes to your bedtime to account for falling asleep.*
        """)

elif page == "ℹ️ About":
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This **Sleep Disorder Diagnosis System** is an advanced machine learning application 
        designed to predict potential sleep disorders based on lifestyle and health metrics.
        
        ### 🔬 Methodology
        
        The system employs multiple machine learning algorithms and compares their performance:
        
        - **Random Forest Classifier** - Ensemble of decision trees
        - **Gradient Boosting** - Sequential ensemble learning
        - **Support Vector Machine** - Kernel-based classification
        - **K-Nearest Neighbors** - Instance-based learning
        - **Logistic Regression** - Statistical classification
        - **Neural Network (MLP)** - Deep learning approach
        - **AdaBoost** - Adaptive boosting
        - **Ensemble Voting** - Combined model predictions
        
        ### 📊 Features Analyzed
        
        The model considers various health and lifestyle factors:
        
        | Category | Features |
        |----------|----------|
        | Demographics | Age, Gender, Occupation |
        | Sleep Metrics | Duration, Quality |
        | Health Indicators | BMI, Blood Pressure, Heart Rate |
        | Lifestyle | Physical Activity, Stress Level, Daily Steps |
        
        ### ⚠️ Disclaimer
        
        This tool is for **educational and informational purposes only**. It is not a substitute 
        for professional medical advice, diagnosis, or treatment. Always consult qualified 
        healthcare providers for medical concerns.
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Model Performance
        """)
        
        # Performance summary
        for name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
            st.progress(score, text=f"{name}: {score*100:.1f}%")
        
        st.markdown("""
        ---
        ### 🛠️ Technologies Used
        
        - Python 3.x
        - Streamlit
        - Scikit-learn
        - Pandas
        - NumPy
        - Plotly
        
        ---
        ### 👨‍💻 Developer
        
        Final Year Project
        
        Sleep Disorder Diagnosis Using Machine Learning
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 20px;">
    <p>🛌 Sleep Disorder Diagnosis System | Powered by Machine Learning</p>
    <p style="font-size: 0.8rem;">⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
