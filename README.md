# Airbnb Price Prediction & Geospatial Analysis

Machine Learning and Spatial Data Science project focused on predicting Airbnb listing prices in Barcelona and New York City using regression models, feature engineering, and H3 geospatial indexing.

---

## Project Overview

This project explores the factors influencing Airbnb prices across two major cities and develops machine learning models capable of estimating listing prices based on property, host, review, and location-related features.

The workflow combines:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Geospatial clustering using H3 indexing
- Regression modeling
- Model evaluation using MAE and RMSE metrics

- ## Dataset

The analysis is based on publicly available Airbnb listing datasets for Barcelona and New York City. The datasets contain detailed information about listings, hosts, pricing, reviews, availability, and geographic location.

### Data Sources
- https://www.kaggle.com/datasets/joebeachcapital/airbnb/data

### Main Features Used
- Listing price
- Room type
- Neighborhood and location coordinates
- Number of reviews
- Availability throughout the year
- Minimum nights
- Host-related statistics
- Review scores and ratings

### Cities Included
- Amsterdam
- Barcelona
- Berlin
- London
- Madrid
- New York City
- Paris
- Roma

### Data Processing
Before modeling, the datasets were cleaned and transformed through:
- Missing value handling
- Outlier filtering
- Feature encoding
- Price normalization
- Geospatial indexing with H3
- Removal of irrelevant or highly sparse variables

The processed datasets were then used for exploratory analysis, spatial visualization, and machine learning model development.

## Methodology

The project workflow follows a complete data science pipeline, starting from data preprocessing and exploratory analysis to geospatial feature engineering and machine learning model evaluation.

### 1. Data Cleaning & Preprocessing
The raw Airbnb datasets required several preprocessing steps before analysis and modeling:
- Handling missing values
- Removing extreme outliers
- Converting categorical variables
- Standardizing numerical features
- Filtering incomplete or inconsistent records

### 2. Exploratory Data Analysis (EDA)
Exploratory analysis was performed to better understand pricing behavior and listing characteristics across both cities.

The analysis included:
- Price distribution analysis
- Neighborhood-level comparisons
- Correlation analysis
- Room type segmentation
- Review and availability trends

### 3. Geospatial Analysis with H3
To improve spatial analysis, the project utilized the H3 geospatial indexing system for location aggregation and visualization.

This enabled:
- Spatial clustering of listings
- Neighborhood-level price aggregation
- Hexagon-based heatmaps
- Improved geographic feature representation

### 4. Feature Engineering
Additional predictive variables were created from the original datasets to improve model performance.

Engineered features included:
- Spatially aggregated pricing indicators
- Host activity metrics
- Review-based indicators
- Availability ratios
- Encoded categorical variables

### 5. Machine Learning Models
Several regression-based machine learning models were trained and evaluated for Airbnb price prediction.

The workflow included:
- Train-test dataset splitting
- Feature selection
- Model training and tuning
- Performance comparison
- Error analysis

The final models were evaluated using MAE and RMSE metrics to measure predictive accuracy.

## Evaluation Metrics

The machine learning models were evaluated using standard regression performance metrics to measure prediction accuracy and error magnitude.

### Mean Absolute Error (MAE)

<img width="151" height="66" alt="image" src="https://github.com/user-attachments/assets/6680a5ee-e646-4e8a-8eb5-b9b00f75bcb1" />

MAE measures the average absolute difference between predicted and actual prices. Lower MAE values indicate more accurate predictions.

### Root Mean Squared Error (RMSE)

<img width="195" height="81" alt="image" src="https://github.com/user-attachments/assets/1a9d6969-87a2-4c68-86d9-329a0dcc127f" />

RMSE penalizes larger prediction errors more strongly, making it useful for identifying models that produce significant outliers or unstable predictions.

### Model Comparison
The metrics were used to:
- Compare regression model performance
- Evaluate generalization capability
- Identify overfitting tendencies
- Select the best-performing predictive model

Lower MAE and RMSE values indicate better predictive performance.

## Results & Findings

The analysis revealed several important factors influencing Airbnb pricing across Barcelona and New York City.

### Key Findings
- Location had one of the strongest impacts on listing prices
- Entire homes and apartments were generally priced significantly higher than private or shared rooms
- Listings with higher review scores and stronger host activity tended to achieve higher prices
- Availability and neighborhood popularity also contributed to pricing differences

### Geospatial Insights
Using H3-based spatial aggregation allowed clearer identification of high-price and low-price regions within each city.

The spatial analysis showed:
- Strong pricing concentration in central tourist areas
- Distinct neighborhood-level pricing clusters
- Significant variation between districts and boroughs
- Improved visualization of regional pricing behavior

### Model Performance
The machine learning models achieved reasonable predictive accuracy for Airbnb price estimation.

The evaluation process highlighted:
- The importance of feature engineering
- The usefulness of spatial features in regression models
- Performance differences between traditional and ensemble-based approaches

### Cross-City Comparison
Comparing Barcelona and New York City demonstrated that:
- Pricing structures differ substantially between cities
- Spatial effects vary depending on urban density and tourism patterns
- Certain predictive features remain consistently important across both datasets

## Visualizations

The project includes multiple visualizations to support exploratory analysis, spatial interpretation, and machine learning evaluation.

### Exploratory Data Analysis
Visualizations created during EDA include:
- Price distribution histograms
- Correlation heatmaps
- Room type comparisons
- Neighborhood-level pricing analysis
- Availability and review trends

### Geospatial Visualizations
Spatial analysis was performed using H3 indexing and geographic aggregation techniques.

Generated visual outputs include:
- H3 hexagon heatmaps
- Spatial price distribution maps
- Neighborhood clustering visualizations
- City-level pricing comparisons

### Model Evaluation Visualizations
To compare machine learning models and interpret results, the project also includes:
- Feature importance charts
- Predicted vs actual price plots
- Error distribution graphs
- Model performance comparison charts

## Technologies Used

The project was developed using Python and several data science and geospatial analysis libraries.

### Core Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- GeoPandas
- H3
- Matplotlib
- Seaborn
- Jupyter Notebook

### Tools & Techniques
- Machine Learning Regression
- Geospatial Aggregation
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Spatial Visualization

## Team
Project team:

- Botond Biró
- Saahas Bondalapati
- Zsombor Tóth

MSc Business Informatics, Corvinus University of Budapest

Course:
Data Science Project in Business, by Associate Professor Tibor Kovács Spring 2026
