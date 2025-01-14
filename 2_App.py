# Import necessary libraries for web app, data processing, visualization, and machine learning.
from flask import Flask, render_template, request, jsonify
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import joblib 
import shap
import sqlite3
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Set non-interactive backend for Matplotlib to use with Flask
matplotlib.use('Agg')  

# Initialize Flask application
app = Flask(__name__)

# Global variables for dataset and file paths
DATA_PATH = "simulated_data.csv"     #
STATIC_FOLDER = os.path.join(os.getcwd(), 'static')


# Load pre-trained model and scaler for predictions
# We are  utilizing a model and scaler that were previously trained and saved in the code file named '1_train_model.py'
# The pre-trained model contains the learned parameters from the training phase, while the scaler is used to standardize or normalize input data to match the format expected by the model
# Loading these pre-trained components allows us to directly make predictions on new data without retraining the model, ensuring consistency and saving time

model_path = "best_customer_satisfaction_model.pkl"
scaler_path = "scaler.pkl"
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError("Model or Scaler files not found! Please ensure the training process has saved them.")


# Ensure Static Folder Exists
def ensure_static_folder():
    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER)


# Function to save Matplotlib plots to static folder
def save_plot(fig, filename):
    ensure_static_folder()  # Ensures the static folder exists
    path = os.path.join(STATIC_FOLDER, filename)
    fig.savefig(path, bbox_inches='tight')  # Save plot with tight bounding box
    plt.close(fig)  # Close the figure to free up memory
    return filename  # Return the filename 


# Function to establish database connection
def get_db_connection():
    conn = sqlite3.connect('customers.db')  # Updated database name
    conn.row_factory = sqlite3.Row
    return conn

# Define routes for the application
@app.route('/')
def home():
    # Render home page template
    return render_template('interface.html')

@app.route('/visualization')
def visualization():
    # Render visualization menu template
    return render_template('visualization_menu.html')

@app.route('/summary')
def summary():
    # Generate dataset summary: stats, missing values, and value counts
    data = pd.read_csv(DATA_PATH)
    stats = data.describe().to_html()
    missing_values = data.isnull().sum().to_frame(name="Missing Values").to_html()
    value_counts = data['customer_satisfaction'].value_counts().to_frame(name="Value Counts").to_html()
    
    return render_template('summary.html', stats=stats, missing_values=missing_values, value_counts=value_counts)

@app.route('/data_visualization')
def data_visualization():
    # Create various visualizations and save plots in static folder
    data = pd.read_csv(DATA_PATH)
    plot_files = {}  # Dictionary to store file paths for plots
    
    # 1. Histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    data.hist(ax=ax)
    plot_files['histogram'] = save_plot(fig, 'histogram.png')

    # 2. Pie Chart (Customer Satisfaction Distribution)
    categories = {
        "1-3 (Low)": 0,
        "4-6 (Moderate)": 0,
        "7-10 (High)": 0,
    }
    for score in data["customer_satisfaction"]:
        if 1 <= score <= 3:
            categories["1-3 (Low)"] += 1
        elif 4 <= score <= 6:
            categories["4-6 (Moderate)"] += 1
        elif 7 <= score <= 10:
            categories["7-10 (High)"] += 1
    labels = categories.keys()
    sizes = categories.values()
    colors = ['red', 'orange', 'green']
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    plt.title("Customer Satisfaction Distribution")
    plot_files['piechart'] = save_plot(fig, 'piechart.png')

    # 3. Density Plots
    numerical_columns = ['age', 'income', 'purchase_history']
    for col in numerical_columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data[col], kde=True, bins=30, ax=ax)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plot_files[f'density_{col}'] = save_plot(fig, f'density_{col}.png')

    # 4. Boxplots for Outliers
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data, ax=ax)
    plt.title("Boxplots for Outlier Detection")
    plot_files['boxplot'] = save_plot(fig, 'boxplot.png')

    # 5. Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Correlation Matrix")
    plot_files['correlation'] = save_plot(fig, 'correlation_matrix.png')

    # 6. Pairplot (Relationships)
    pairplot_path = os.path.join("static", "pairplot.png")
    sns.pairplot(data, vars=numerical_columns, hue="customer_satisfaction", diag_kind="kde", corner=True)
    plt.suptitle("Pairplot of Numerical Features", y=1.02)
    plt.savefig(pairplot_path, bbox_inches='tight')
    plt.close()
    plot_files['pairplot'] = 'pairplot.png'

    # Render template with all plot paths
    return render_template('plots.html', title="Data Visualizations", plot_files=plot_files)

@app.route('/countplots')
def countplots():
    #Generates a count plot for categorical features like customer satisfaction.
    data = pd.read_csv(DATA_PATH)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='customer_satisfaction', data=data, ax=ax)
    countplot_file = save_plot(fig, 'countplot.png')
    return render_template('plot_count.html', title="Countplot for Categorical Features", plot_url=f'static/{countplot_file}')

@app.route('/outliers')
def outliers():
    #Displays a boxplot to analyze outliers in the dataset
    data = pd.read_csv(DATA_PATH)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, ax=ax)
    boxplot_file = save_plot(fig, 'boxplot.png')
    return render_template('plot_count.html', title="Outlier Analysis", plot_url=f'static/{boxplot_file}')

@app.route('/grouped_aggregation')
def grouped_aggregation():
    #Computes grouped statistics (mean and standard deviation) for age, income, and purchase history
    data = pd.read_csv(DATA_PATH)
    grouped_stats = data.groupby('customer_satisfaction').agg({
        'age': ['mean', 'std'],
        'income': ['mean', 'std'],
        'purchase_history': ['mean', 'std']
    }).reset_index()
    return render_template('grouped_aggregation.html', grouped_stats=grouped_stats.to_html())



# Route: SQL Queries Interface
@app.route('/sql', methods=['GET', 'POST'])
def sql_queries():
    results = None
    query_title = None
    
    if request.method == 'POST':
        # Filter Customers by Age Range
        if 'min_age' in request.form and 'max_age' in request.form:
            min_age = request.form['min_age']
            max_age = request.form['max_age']
            query_title = f"Customers between Ages {min_age} and {max_age}"
            conn = get_db_connection()
            results = conn.execute("SELECT customer_id, age FROM customers WHERE age BETWEEN ? AND ?", 
                                   (min_age, max_age)).fetchall()
            conn.close()
        
        # Customers Above a Certain Age
        elif 'min_age_above' in request.form:
            min_age = request.form['min_age_above']
            query_title = f"Customers Above Age {min_age}"
            conn = get_db_connection()
            results = conn.execute("SELECT customer_id, age FROM customers WHERE age > ?", (min_age,)).fetchall()
            conn.close()
        
        # Average Age by Satisfaction Level
        elif 'satisfaction_level' in request.form:
            satisfaction_level = request.form['satisfaction_level']
            query_title = f"Average Age for Satisfaction Level {satisfaction_level}"
            conn = get_db_connection()
            avg_age = conn.execute("SELECT AVG(age) AS avg_age FROM customers WHERE customer_satisfaction = ?", 
                                   (satisfaction_level,)).fetchone()
            conn.close()
            results = [{'Average Age': round(avg_age['avg_age'], 2) if avg_age['avg_age'] else "No data"}]
        
        # Top N Customers by Income
        elif 'top_n' in request.form:
            top_n = request.form['top_n']
            query_title = f"Top {top_n} Customers by Income"
            conn = get_db_connection()
            results = conn.execute("SELECT customer_id, income FROM customers ORDER BY income DESC LIMIT ?", 
                                   (top_n,)).fetchall()
            conn.close()

    return render_template('sql_interface.html', results=results, query_title=query_title)



"""# Load Model and Scaler
model_path = "best_customer_satisfaction_model.pkl"
scaler_path = "scaler.pkl"
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError("Model or Scaler files not found! Please ensure the training process has saved them.")
"""



# Route: Prediction Options Page
@app.route('/prediction')
def prediction_home():
    # This route renders the main prediction options page where users can navigate
    # to different prediction functionalities such as customer predictions, future predictions,
    # customer segmentation, and churn analysis.

    return render_template('prediction_home.html')



# Handles customer prediction based on user-provided input via a form.
@app.route('/predict_customer', methods=['GET', 'POST'])
def predict_customer():
    if request.method == 'POST':
        try:
            # Get user input
            age = float(request.form['age'])
            income = float(request.form['income'])
            purchase_history = float(request.form['purchase_history'])
            age_income_interaction = age * income  # Feature Engineering

            # Prepare input for prediction
            input_data = pd.DataFrame([{
                'age': age,
                'income': income,
                'purchase_history': purchase_history,
                'age_income_interaction': age_income_interaction
            }])
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)
            result = round(prediction[0], 2)

            return render_template('prediction_result.html', prediction=result, age=age, income=income,
                                   purchase_history=purchase_history)

        except ValueError:
            return "Invalid input! Please enter numeric values for all fields."

    # Render input form for GET request
    return render_template('predict_customer.html')



# Route: View Prediction Results
# Displays the saved prediction results and evaluation metrics.
@app.route('/see_prediction_result')
def see_prediction_result():
    try:
        # Load Predictions
        predictions_df = pd.read_csv("predictions.csv")
        
        # Load Evaluation Metrics
        with open("evaluation_metrics.txt", "r") as f:
            metrics = f.readlines()

        # Convert Predictions to HTML
        predictions_html = predictions_df.head(10).to_html(classes="table table-striped", index=False)

        return render_template(
            'see_prediction_result.html',
            metrics=metrics,
            predictions_html=predictions_html
        )
    except FileNotFoundError:
        return "Prediction results or evaluation metrics not found! Please run train_model.py to generate them."

# Route: Predict Future
# Similar to 'predict_customer', this route provides future predictions based on input data.
@app.route('/predict_future', methods=['GET', 'POST'])
def predict_future():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            income = float(request.form['income'])
            purchase_history = float(request.form['purchase_history'])
            age_income_interaction = age * income

            input_data = pd.DataFrame([{
                'age': age,
                'income': income,
                'purchase_history': purchase_history,
                'age_income_interaction': age_income_interaction
            }])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            result = round(prediction[0], 2)

            return render_template('predict_future.html', prediction=result, age=age, income=income,
                                   purchase_history=purchase_history)
        except ValueError:
            return "Invalid input! Please enter numeric values."
    return render_template('predict_future.html')

# Route: Customer Segmentation
# Implements customer segmentation using KMeans clustering.
@app.route('/customer_segmentation')
def customer_segmentation():
    # Load Data
    data = pd.read_csv(DATA_PATH)
    
    # Features for Clustering
    features = data[['age', 'income', 'purchase_history']]
    
    # Scale the Data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(features_scaled)
    
    # Save the Clustering Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(features_scaled[:, 0], features_scaled[:, 1], c=data['cluster'], cmap='viridis')
    plt.colorbar(scatter, label="Cluster")
    ax.set_title('Customer Segmentation')
    ax.set_xlabel('Age')
    ax.set_ylabel('Income')
    plot_file = save_plot(fig, 'customer_segmentation.png')
    
    # Pass the clustered data and plot to the template
    return render_template('customer_segmentation.html', plot_url=f'static/{plot_file}', cluster_data=data.to_html(classes="table table-striped", index=False))


# Route: Purchase Behavior Analysis
@app.route('/purchase_behavior')
def purchase_behavior():
    try:
        # Load Data
        data = pd.read_csv(DATA_PATH)
        
        # Creating a binary churn target
        data['churn'] = (data['customer_satisfaction'] < 6).astype(int)
        
        # Features and Target
        X = data[['age', 'income', 'purchase_history']]
        y = data['churn']
        
        # Train-test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest Classifier
        churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
        churn_model.fit(X_train, y_train)
        
        # Predictions and Evaluation
        y_pred = churn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Convert Report to DataFrame for Display
        report_df = pd.DataFrame(report).transpose()
        
        # Return Results to Template
        return render_template(
            'purchase_behavior.html',
            accuracy=round(accuracy * 100, 2),
            report_html=report_df.to_html(classes="table table-striped", float_format="%.2f", border=0)
        )
    except Exception as e:
        return f"An error occurred: {e}"


# Route: Customer Lifetime Value (CLV) Prediction
@app.route('/clv_prediction')
def clv_prediction():
    try:
        # Load Data
        data = pd.read_csv(DATA_PATH)

        # Features and Target for CLV Prediction
        features = ['age', 'income', 'customer_satisfaction']
        target = 'purchase_history'
        X = data[features]
        y = data[target]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Linear Regression Model
        clv_model = LinearRegression()
        clv_model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred_clv = clv_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred_clv)

        # Combine Actual vs Predicted for Display
        results_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred_clv
        }).head(10)

        # Return Results to Template
        return render_template(
            'clv_prediction.html',
            mae=round(mae, 2),
            results_html=results_df.to_html(classes="table table-striped", index=False, border=0)
        )
    except Exception as e:
        return f"An error occurred: {e}"



# Route: Churn Prediction
@app.route('/churn_prediction')
def churn_prediction():
    try:
        # Load Data
        data = pd.read_csv(DATA_PATH)

        # Creating a binary churn target
        data['churn'] = (data['customer_satisfaction'] < 6).astype(int)

        # Features and Target
        X = data[['age', 'income', 'purchase_history']]
        y = data['churn']

        # Train-test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest Classifier
        churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
        churn_model.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred = churn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Convert Classification Report to a DataFrame for Display
        report_df = pd.DataFrame(report).transpose()

        # Return Results to Template
        return render_template(
            'churn_prediction.html',
            accuracy=round(accuracy * 100, 2),
            report_html=report_df.to_html(classes="table table-striped", float_format="%.2f", border=0)
        )
    except Exception as e:
        return f"An error occurred: {e}"




# Entry point for the application
if __name__ == '__main__':
    app.run(debug=True)