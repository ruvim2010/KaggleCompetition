import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

training_data = pandas.read_csv('train.csv').drop(['PassengerId', 'Ticket', 'Cabin'], axis='columns')
test_data = pandas.read_csv('test.csv')
passenger_id = test_data['PassengerId']
test_data = test_data.drop(['PassengerId', 'Ticket', 'Cabin'], axis='columns')
prediction_target = training_data.pop('Survived').values


def encode_data(dataset):
    dataset['Title'] = dataset['Name'].str.extract(r' ([A-Za-z]+)\.')
    dataset['Name'] = dataset['Name'].str.split(',').str[0]
    dataset['Age'] = dataset.groupby('Title')['Age'].transform(
        lambda x: x.fillna(x.median() if not x.isna().all() else dataset['Age'].median()))
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
    dataset['Embarked'] = dataset['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset['Name'])
    dataset['Name'] = label_encoder.transform(dataset['Name'])
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset['Title'])
    dataset['Title'] = label_encoder.transform(dataset['Title'])
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())


encode_data(training_data)
encode_data(test_data)

pipeline = Pipeline([
    ('model', LogisticRegression()),
])

parameter_grid = [
    {
        'model': [LogisticRegression()],
        'model__penalty': ['l1', 'l2'],
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'lbfgs', 'saga'],
        'model__max_iter': [10000],
        'model__class_weight': [None, 'balanced'],
    },
    {
        'model': [RandomForestClassifier()],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': [1, 3, 5, 7, 10],
        'model__bootstrap': [True, False],
        'model__class_weight': [None, 'balanced']
    }
]

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(training_data, prediction_target)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)
print("Best estimator:", grid_search.best_estimator_)

survived = grid_search.best_estimator_.predict(test_data)

submission = pandas.DataFrame({
    'PassengerId': passenger_id,
    'Survived': survived,
})
submission.to_csv('submission.csv', index=False)
