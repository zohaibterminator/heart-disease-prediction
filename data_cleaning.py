import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import kagglehub
import pickle


# Encoder Class
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns, target_column):
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.ohe = OneHotEncoder(sparse_output=False)
        self.le = LabelEncoder()
        self.encoded_feature_names = []  # Store encoded feature names

    def fit(self, X, y=None):
        self.ohe.fit(X[self.categorical_columns])
        self.le.fit(X[self.target_column])
        self.encoded_feature_names = self.ohe.get_feature_names_out(self.categorical_columns).tolist()  # Store encoded feature names
        return self

    def transform(self, X):
        encoded = self.ohe.transform(X[self.categorical_columns])

        encoded_df = pd.DataFrame(
            encoded, 
            columns=self.encoded_feature_names, 
            index=X.index
        )
        
        result = pd.concat([
            X.drop(self.categorical_columns + [self.target_column], axis=1),
            encoded_df
        ], axis=1)
        result[self.target_column] = self.le.transform(X[self.target_column])
        return result

    
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features, encoded_features, target_column, num_k=5, cat_k=5):
        """
        :param numeric_features: List of numeric feature names
        :param encoded_features: List of encoded feature names
        :param target_column: Target column name
        :param num_k: Number of top numeric features to select
        :param cat_k: Number of top encoded features to select
        """
        self.numeric_features = numeric_features
        self.encoded_features = encoded_features  # Use encoded features
        self.target_column = target_column
        self.num_k = num_k
        self.cat_k = cat_k
        self.chi2_selector = None
        self.numeric_selector = None

    def fit(self, X, y=None):
        # Pearson correlation for numeric features
        self.numeric_selector = X[self.numeric_features].corrwith(X[self.target_column]).abs().nlargest(self.num_k).index.tolist()

        # Chi-Square for encoded categorical features
        X_encoded = X[self.encoded_features]
        y = X[self.target_column]
        
        # Apply chi-squared test and select top k features
        self.chi2_selector = SelectKBest(chi2, k=self.cat_k).fit(X_encoded, y)
        return self

    def transform(self, X):
        # Select top numeric features based on Pearson correlation
        X_selected_num = X[self.numeric_selector]
        y = X[self.target_column]

        # Select top encoded categorical features based on Chi-Square
        X_encoded = X[self.encoded_features]
        X_selected_cat = pd.DataFrame(self.chi2_selector.transform(X_encoded), columns=self.chi2_selector.get_feature_names_out(), index=X.index)

        # Concatenate selected numeric and categorical features
        return pd.concat([X_selected_num, X_selected_cat, y], axis=1)

# Splitter Class
class Splitter(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, test_size=0.3, random_state=42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        y = X[self.target_column]
        X = X.drop(self.target_column, axis=1)
        return tuple(train_test_split(X, y, test_size=self.test_size, random_state=self.random_state))


# Scaler Class
class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type='standard'):
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, tuple) and len(X) == 4:
            X_train, X_test, y_train, y_test = X
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            return self.scaler.fit_transform(X)


# Full pipeline with feature selection
class FullPipeline:
    def __init__(self, categorical_columns, target_column, numeric_features, num_k=5, cat_k=5):
        self.encoder = Encoder(categorical_columns, target_column)
        self.feature_selector = None  # Initialize after encoding to access encoded names
        self.splitter = Splitter(target_column)
        self.scaler = Scaler()
        self.numeric_features = numeric_features
        self.num_k = num_k
        self.cat_k = cat_k

    def fit_transform(self, X):
        # Apply encoding and retrieve encoded feature names
        X = self.encoder.fit_transform(X)
        self.feature_selector = FeatureSelector(
            numeric_features=self.numeric_features, 
            encoded_features=self.encoder.encoded_feature_names,
            target_column=self.encoder.target_column,
            num_k=self.num_k, cat_k=self.cat_k
        )
        X = self.feature_selector.fit_transform(X)
        X_train, X_test, y_train, y_test = self.splitter.transform(X)
        return self.scaler.transform((X_train, X_test, y_train, y_test))

class FullPipeline:
    def __init__(self, categorical_columns, target_column, numeric_features, num_k=5, cat_k=5):
        self.encoder = Encoder(categorical_columns, target_column)
        self.feature_selector = None  # Initialize after encoding to access encoded names
        self.splitter = Splitter(target_column)
        self.scaler = Scaler()
        self.numeric_features = numeric_features
        self.num_k = num_k
        self.cat_k = cat_k

    def fit_transform(self, X):
        X = self.encoder.fit_transform(X)

        pickle.dump(self.encoder, open("encoder.pkl", "wb"))

        self.feature_selector = FeatureSelector(
            numeric_features=self.numeric_features, 
            encoded_features=self.encoder.encoded_feature_names,
            target_column=self.encoder.target_column,
            num_k=self.num_k, cat_k=self.cat_k
        )
        X = self.feature_selector.fit_transform(X)

        pickle.dump(self.feature_selector, open("feature_selector.pkl", "wb"))

        X_train, X_test, y_train, y_test = self.splitter.transform(X)

        pickle.dump(self.splitter, open("splitter.pkl", "wb"))

        X_train_scaled, X_test_scaled, y_train, y_test = self.scaler.transform((X_train, X_test, y_train, y_test))

        pickle.dump(self.scaler, open("scaler.pkl", "wb"))

        return (X_train_scaled, X_test_scaled, y_train, y_test)


def main():
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    df = pd.read_csv(path + r"\heart.csv")

    df.drop_duplicates(inplace=True) # dropping the duplicates

    # defining the pipeline
    pipeline = FullPipeline(
        categorical_columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
        target_column='HeartDisease',
        numeric_features=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
        num_k=3,  # Select top 3 numeric features
        cat_k=3   # Select top 3 categorical features
    )

    # transforming the data
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df) 

    with open("transformed_data.pkl", "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)


if __name__ == "__main__":
    main()