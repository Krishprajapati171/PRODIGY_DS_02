# Step-by-step pipeline: Traffic Accident Analysis and Injury Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import folium
from folium.plugins import HeatMap

# Step 1: Load data
df = pd.read_csv("traffic_accidents.csv")

# Step 2: Data Cleaning and Feature Extraction
df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')
df['crash_hour'] = df['crash_hour'].fillna(df['crash_date'].dt.hour)
df['crash_day_of_week'] = df['crash_date'].dt.dayofweek

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['time_of_day'] = df['crash_hour'].apply(get_time_of_day)
df['crash_weekday_name'] = df['crash_date'].dt.day_name()
df['any_injury'] = df['injuries_total'].apply(lambda x: 1 if x > 0 else 0)

# Step 3: Exploratory Data Analysis (visualizations)
sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(x='time_of_day', data=df, order=['Morning', 'Afternoon', 'Evening', 'Night'], palette='Set2')
plt.title('Number of Accidents by Time of Day')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(x='crash_weekday_name', data=df, order=order_days, palette='Set3')
plt.title('Number of Accidents by Day of Week')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
top_weather = df['weather_condition'].value_counts().head(10).index
sns.countplot(y='weather_condition', data=df[df['weather_condition'].isin(top_weather)], order=top_weather, palette='coolwarm')
plt.title('Top Weather Conditions in Accidents')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(y='roadway_surface_cond', data=df, order=df['roadway_surface_cond'].value_counts().index, palette='viridis')
plt.title('Accidents by Road Surface Condition')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(y='lighting_condition', data=df, order=df['lighting_condition'].value_counts().index, palette='magma')
plt.title('Accidents by Lighting Condition')
plt.tight_layout()
plt.show()

# Step 4: Feature Engineering for Modeling
features = [
    'weather_condition', 'lighting_condition', 'roadway_surface_cond',
    'time_of_day', 'crash_day_of_week', 'crash_hour']
target = 'any_injury'

# Step 5: Encode Categorical Features
df_encoded = df[features + [target]].copy()
for col in ['weather_condition', 'lighting_condition', 'roadway_surface_cond', 'time_of_day']:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

df_encoded.dropna(inplace=True)

# Step 6: Train/Test Split and Modeling
X = df_encoded.drop(columns=[target])
y = df_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 7: Evaluation
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Injury', 'Injury'], yticklabels=['No Injury', 'Injury'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Step 8: Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feat_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance, palette='crest')
plt.title('Feature Importance in Predicting Accident Injuries')
plt.tight_layout()
plt.show()

# Step 9: Hotspot Visualization (if location data exists)
if 'latitude' in df.columns and 'longitude' in df.columns:
    df_location = df[['latitude', 'longitude']].dropna()
    m = folium.Map(location=[df_location['latitude'].mean(), df_location['longitude'].mean()], zoom_start=11)
    heat_data = [[row['latitude'], row['longitude']] for index, row in df_location.iterrows()]
    HeatMap(heat_data, radius=10).add_to(m)
    m.save("accident_hotspots.html")
    print("Accident hotspot map saved as 'accident_hotspots.html'")
else:
    print("Location data not available. Skipping hotspot visualization.")

# Step 10: Project Summary Output
print("""
PROJECT SUMMARY:
- Data cleaned and features extracted (crash time, day, lighting, weather)
- Injury predicted using Random Forest model
- Model evaluated with classification metrics and confusion matrix
- Most important features identified
- Hotspot map created if coordinates available
""")
