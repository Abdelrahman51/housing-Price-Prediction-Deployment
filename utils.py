# Major Libraries
import pandas as pd
import os
# sklearn -- for pipeline and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector


FILE_PATH = os.path.join(os.getcwd(), 'housing.csv')
df = pd.read_csv(FILE_PATH)


df['ocean_proximity'] = df['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN')


df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedroms_per_rooms'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']


X = df.drop(columns=['median_house_value'], axis=1)  # Features
y = df['median_house_value']  # target


X_train, X_test, Y_train, y_test = train_test_split(
    X, y, shuffle=True, test_size=0.15, random_state=42)


num_cols = [col for col in X_train.columns if X_train[col].dtype in [
    'int32', 'int64', 'float32', 'float64']]
categ_cols = [col for col in X_train.columns if X_train[col].dtype not in [
    'int32', 'int64', 'float32', 'float64']]


num_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(num_cols)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categ_pipline = Pipeline(steps=[
    ('selector', DataFrameSelector(categ_cols)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(sparse=False))
])

total_pipline = FeatureUnion(transformer_list=[
    ('num', num_pipeline),
    ('categ', categ_pipline)
])


X_train_final = total_pipline.fit_transform(X_train)

def preprocess_new(X_new):
    return total_pipline.transform(X_new)
