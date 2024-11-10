import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import dill
import kagglehub


def encode_data(X, categorical_columns, target_column):
    ohe = OneHotEncoder(sparse_output=False)
    le = LabelEncoder()
    
    # Fit encoders and transform data
    encoded_categorical = ohe.fit_transform(X[categorical_columns])
    encoded_df = pd.DataFrame(encoded_categorical, columns=ohe.get_feature_names_out(categorical_columns), index=X.index)

    # Concatenate encoded columns with remaining data
    result = pd.concat([X.drop(categorical_columns + [target_column], axis=1), encoded_df], axis=1)
    result[target_column] = le.fit_transform(X[target_column])

    # Save encoders
    dill.dump(ohe, open("encoder.pkl", "wb"))

    return result, ohe.get_feature_names_out(categorical_columns).tolist()


def select_features(X, numeric_features, encoded_features, target_column, num_k=5, cat_k=5):
    # Select numeric features based on Pearson correlation
    selected_numeric_features = X[numeric_features].corrwith(X[target_column]).abs().nlargest(num_k).index.tolist()

    # Select categorical features based on Chi-Square
    X_encoded = X[encoded_features]
    y = X[target_column]
    chi2_selector = SelectKBest(chi2, k=cat_k).fit(X_encoded, y)
    selected_categorical_features = chi2_selector.get_feature_names_out()

    # Concatenate selected features
    selected_data = pd.concat([X[selected_numeric_features], X[selected_categorical_features], y], axis=1)

    return selected_data


def split_data(X, target_column, test_size=0.3, random_state=42):
    y = X[target_column]
    X = X.drop(target_column, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test, scaler_type='standard'):
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    dill.dump(scaler, open("scaler.pkl", "wb"))

    return X_train_scaled, X_test_scaled


def preprocess_pipeline(df, categorical_columns, target_column, numeric_features, num_k=5, cat_k=5):
    # Encoding
    df_encoded, encoded_feature_names = encode_data(df, categorical_columns, target_column)

    # Feature selection
    df_selected = select_features(df_encoded, numeric_features, encoded_feature_names, target_column, num_k, cat_k)

    # Data splitting
    X_train, X_test, y_train, y_test = split_data(df_selected, target_column)

    # Scaling
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# Main function for running the pipeline and saving transformed data
def main():
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    df = pd.read_csv(path + r"\heart.csv")
    df.drop_duplicates(inplace=True)  # dropping duplicates

    # Apply preprocessing
    X_train, X_test, y_train, y_test = preprocess_pipeline(
        df,
        categorical_columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
        target_column='HeartDisease',
        numeric_features=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
        num_k=3,  # Select top 3 numeric features
        cat_k=3   # Select top 3 categorical features
    )

    # Save the preprocessed data
    with open("transformed_data.pkl", "wb") as f:
        dill.dump((X_train, X_test, y_train, y_test), f)


if __name__ == "__main__":
    main()