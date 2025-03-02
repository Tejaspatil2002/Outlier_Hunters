
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy import stats

# 🎨 Streamlit App Configuration
st.set_page_config(page_title="📊 Flight Profitability Predictor", layout="wide")
st.title("📈 Flight Profitability Prediction App")

# Sidebar: Upload CSV or Manual Entry
st.sidebar.header("Upload Dataset or Enter Data Manually")
choice = st.sidebar.radio("Choose Data Input Method", ("Upload CSV", "Manual Entry"))

df = None  # Initialize dataframe

if choice == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
elif choice == "Manual Entry":
    column_names = ["Revenue (USD)", "Operating Cost (USD)", "Fuel Efficiency (ASK)", 
                    "Aircraft Utilization (Hours/Day)", "Debt-to-Equity Ratio",
                    "Cost per ASK", "Maintenance Downtime (Hours)", "Revenue per ASK",
                    "Fleet Availability (%)", "Net Profit Margin (%)"]
    data = {}
    for col in column_names:
         data[col] = st.sidebar.number_input(f"Enter {col}", value=0.0)
    df = pd.DataFrame([data])

if df is not None:
    st.write("### Dataset Preview")
    st.write(df.head())

    # Data preprocessing tab
    tab1, tab2, tab3 = st.tabs(["Data Preprocessing", "Model Training", "Prediction"])
    
    with tab1:
        st.subheader("Data Preprocessing")
        
        # Handle Missing Values
        st.write("#### Handling Missing Values")
        missing_values = df.isnull().sum()
        st.write("Missing values before imputation:", missing_values)
        df.fillna(df.median(numeric_only=True), inplace=True)
        st.write("Missing values after imputation:", df.isnull().sum())
        
        # Identify Features & Target
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]   # Target (Net Profit Margin)
        
        # Encode Categorical Variables
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        encoders = {}
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            X[col] = encoders[col].fit_transform(X[col])
        
        # Display correlation matrix to identify multicollinearity
        st.write("#### Correlation Matrix (Before handling multicollinearity)")
        correlation_matrix = X.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        
        # Identify highly correlated features
        st.write("#### Identifying Multicollinearity")
        threshold = 0.7
        high_corr_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    colname = correlation_matrix.columns[i]
                    high_corr_features.add(colname)
                    st.write(f"High correlation between {correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.2f}")
        
        # Option to remove features or use PCA
        multicollinearity_method = st.radio(
            "Choose method to handle multicollinearity",
            ("Remove highly correlated features", "Apply PCA", "VIF Analysis")
        )
        
        if multicollinearity_method == "Remove highly correlated features":
            if high_corr_features:
                st.write(f"Removing these highly correlated features: {high_corr_features}")
                X = X.drop(columns=list(high_corr_features))
            else:
                st.write("No highly correlated features found above the threshold.")
        
        elif multicollinearity_method == "Apply PCA":
            n_components = st.slider("Select number of principal components", 
                                     min_value=1, max_value=X.shape[1], value=min(5, X.shape[1]))
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Display explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            st.write("#### PCA Explained Variance")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual explained variance')
            ax.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
            ax.set_xlabel('Principal components')
            ax.set_ylabel('Explained variance ratio')
            ax.legend()
            st.pyplot(fig)
            
            st.write(f"Total variance explained by {n_components} components: {cumulative_variance[-1]:.2%}")
            
            # Create a new DataFrame with PCA components
            X = pd.DataFrame(
                X_pca, 
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
        
        elif multicollinearity_method == "VIF Analysis":
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            # Calculate VIF for each feature
            st.write("#### Variance Inflation Factor (VIF) Analysis")
            X_scaled = StandardScaler().fit_transform(X)
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
            vif_data = vif_data.sort_values("VIF", ascending=False)
            st.write(vif_data)
            
            vif_threshold = st.slider("VIF Threshold for removal", min_value=5.0, max_value=20.0, value=10.0, step=0.5)
            
            features_to_drop = vif_data[vif_data["VIF"] > vif_threshold]["Feature"].tolist()
            if features_to_drop:
                st.write(f"Removing features with VIF > {vif_threshold}: {features_to_drop}")
                X = X.drop(columns=features_to_drop)
            else:
                st.write(f"No features found with VIF > {vif_threshold}")
        
        # Detect and remove outliers
        st.write("#### Outlier Detection and Removal")
        outlier_method = st.radio("Choose outlier detection method", 
                                 ("Z-Score", "IQR (Interquartile Range)", "None"))
        
        df_no_outliers = df.copy()
        outlier_indices = []
        
        if outlier_method == "Z-Score":
            z_threshold = st.slider("Z-Score threshold", min_value=2.0, max_value=4.0, value=3.0, step=0.1)
            
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    z_scores = np.abs(stats.zscore(X[col]))
                    outliers = np.where(z_scores > z_threshold)[0]
                    outlier_indices.extend(outliers)
                    
            outlier_indices = list(set(outlier_indices))  # Remove duplicates
            
            if outlier_indices:
                st.write(f"Found {len(outlier_indices)} outliers using Z-Score method")
                df_no_outliers = df.drop(index=outlier_indices)
                # Recreate X and y without outliers
                X = df_no_outliers.iloc[:, :-1]
                y = df_no_outliers.iloc[:, -1]
                
                # Re-encode categorical columns if any
                for col in categorical_cols:
                    if col in X.columns:
                        X[col] = encoders[col].transform(X[col])
            else:
                st.write("No outliers found using Z-Score method")
        
        elif outlier_method == "IQR":
            iqr_multiplier = st.slider("IQR multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
            
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    
                    col_outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)].index
                    outlier_indices.extend(col_outliers)
            
            outlier_indices = list(set(outlier_indices))  # Remove duplicates
            
            if outlier_indices:
                st.write(f"Found {len(outlier_indices)} outliers using IQR method")
                df_no_outliers = df.drop(index=outlier_indices)
                # Recreate X and y without outliers
                X = df_no_outliers.iloc[:, :-1]
                y = df_no_outliers.iloc[:, -1]
                
                # Re-encode categorical columns if any
                for col in categorical_cols:
                    if col in X.columns:
                        X[col] = encoders[col].transform(X[col])
            else:
                st.write("No outliers found using IQR method")
    
    with tab2:
        st.subheader("Model Training")
        
        # Train-Test Split
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42, step=1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Train Model
        st.write("#### Model Training Options")
        model_choice = st.selectbox(
            "Choose regression model",
            ("Linear Regression", "Ridge Regression", "Lasso Regression")
        )
        
        if model_choice == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_choice == "Ridge Regression":
            from sklearn.linear_model import Ridge
            alpha = st.slider("Ridge alpha", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
            model = Ridge(alpha=alpha)
        elif model_choice == "Lasso Regression":
            from sklearn.linear_model import Lasso
            alpha = st.slider("Lasso alpha", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
            model = Lasso(alpha=alpha)
        
        model.fit(X_train, y_train)
        
        # Predictions & Performance Metrics
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
        
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        
        st.write("#### Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Training Metrics:")
            metrics_df_train = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'R²'],
                'Value': [f"{train_mae:.4f}", f"{train_rmse:.4f}", f"{train_r2:.4f}"]
            })
            st.table(metrics_df_train)
        
        with col2:
            st.write("Testing Metrics:")
            metrics_df_test = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'R²'],
                'Value': [f"{test_mae:.4f}", f"{test_rmse:.4f}", f"{test_r2:.4f}"]
            })
            st.table(metrics_df_test)
        
        # Save Model
        if st.button("Save Model"):
            joblib.dump(model, 'optimized_model.pkl')
            joblib.dump(encoders, 'encoders.pkl')
            st.success("Model saved successfully!")
        
        # Visualize Results
        st.write("#### Model Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Actual vs Predicted (Test Set)")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred_test, alpha=0.6)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted Values')
            st.pyplot(fig)
        
        with col2:
            st.write("Residuals Plot")
            residuals = y_test - y_pred_test
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred_test, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs Predicted Values')
            st.pyplot(fig)
        
        # Feature Importance (if applicable)
        if hasattr(model, 'coef_'):
            st.write("#### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': np.abs(model.coef_)
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)
    
    with tab3:
        st.subheader("🔮 Make a Prediction")
        st.write("Enter values for prediction:")
        
        user_inputs = {}
        for feature in X.columns:
            if feature in categorical_cols:
                options = df[feature].unique().tolist()
                user_inputs[feature] = st.selectbox(f"Select {feature}", options)
            else:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                default_val = float(X[feature].median())
                user_inputs[feature] = st.slider(f"{feature}", min_val, max_val, default_val)
        
        user_data = pd.DataFrame([user_inputs])
        
        if st.button("Predict"):
            # Ensure the input data has the same format as training data
            for col in categorical_cols:
                if col in user_data.columns:
                    user_data[col] = encoders[col].transform(user_data[col])
            
            prediction = model.predict(user_data)
            
            st.success(f"Predicted Net Profit Margin: {prediction[0]:.2f}")
            
            # Create a gauge chart to visualize the prediction
            if prediction[0] >= 0:
                fig, ax = plt.subplots(figsize=(15, 4))
                gauge_colors = ['red', 'darkred', 'brown', 'orangered', 'orange', 'gold', 'yellow','lightyellow', 'green', 'darkgreen', 'lime', 'cyan', 'blue', 'navy','purple', 'violet', 'pink', 'magenta', 'grey', 'black', 'teal', 'turquoise','maroon', 'olive', 'salmon', 'indigo', 'lavender', 'chocolate']  # 28 colors
                thresholds = [0, 5, 10, 15, 20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,500,1000,2000,5000,15000,50000,100000,1000000]

                if len(gauge_colors) < len(thresholds) - 1:
                    raise ValueError(f"gauge_colors must have at least {len(thresholds) - 1} elements, but has {len(gauge_colors)}")

                for i in range(len(thresholds) - 1):
                    ax.axvspan(thresholds[i], thresholds[i+1], alpha=0.2, color=gauge_colors[i])  # Ensure valid index

                
                ax.plot([prediction[0], prediction[0]], [0, 1], 'k-', linewidth=3)
                ax.text(prediction[0], 0.5, f"{prediction[0]:.2f}", ha='center', va='center', fontsize=12, fontweight='bold')
                
                ax.set_xlim(0, max(20, prediction[0] * 1.2))
                ax.set_ylim(0, 1)
                ax.set_title('Profit Margin Gauge')
                ax.set_xticks(thresholds)
                ax.set_yticks([])
                
                labels = ['Poor', 'Fair', 'Good', 'Excellent']

                for i, label in enumerate(labels):  # Ensure i is controlled automatically
                    if i < len(thresholds) - 1:  # Prevent out-of-range access
                        ax.text((thresholds[i] + thresholds[i+1]) / 2, 0.7, label, ha='center', va='center')

                st.pyplot(fig)
            else:
                st.error(f"The predicted profit margin is negative: {prediction[0]:.2f}")
                
            # Show factors that could improve profitability
            if hasattr(model, 'coef_'):
                positive_factors = pd.DataFrame({
                    'Feature': X.columns,
                    'Coefficient': model.coef_
                }).sort_values(by='Coefficient', ascending=False)
                
                st.write("#### Factors that could improve profitability:")
                st.write(positive_factors.head(3))


