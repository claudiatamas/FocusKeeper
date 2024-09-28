import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder


def predict_recommendation_and_tiredness(input_data):
    try:
        df = pd.read_excel('daily_stats.xlsx')
    except Exception as e:
        raise ValueError("An error occurred while reading the Excel file:", e)

    df.drop(['date'], axis=1, inplace=True)

    categorical_columns = ['diet_quality', 'mood', 'anxiety_level', 'stress_level', 'energy_level', 'recommendation']
    numerical_columns = ['sleep_hours', 'exercise_done', 'hydration', 'caffeine_intake', 'alcohol_intake', 'medications',
                         'social_interaction', 'relaxation_time', 'focus_time', 'screen_time']

    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    for column in numerical_columns:
        df[column] = df[column].astype(int)

    X = df.drop(['recommendation', 'tiredness'], axis=1)
    y_recommendation = df['recommendation']
    y_tiredness = df['tiredness']

    model_r = KNeighborsRegressor(n_neighbors=5)
    model_r.fit(X, y_recommendation)

    model_t = KNeighborsRegressor(n_neighbors=5)
    model_t.fit(X, y_tiredness)


    input_data_encoded = {}
    for column, value in input_data.items():
        if column in categorical_columns[:-1]:
            input_data_encoded[column] = label_encoders[column].transform([value])[0]
        else:
            input_data_encoded[column] = value

    recommendation = model_r.predict([list(input_data_encoded.values())])
    tiredness = model_t.predict([list(input_data_encoded.values())])


    decoded_recommendation = label_encoders['recommendation'].inverse_transform([int(recommendation[0])])[0]

    tiredness_message = "You seem tired. Take some rest and recharge!"
    if tiredness[0] == 0:
        tiredness_message = "You seem fresh and energetic!"

    return decoded_recommendation, tiredness_message
