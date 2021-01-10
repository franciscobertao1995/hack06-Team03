import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
########################################
# Begin database stuff
DB = SqliteDatabase('predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    label = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)
# End database stuff
########################################
########################################
# Unpickle the previously-trained model
with open('columns.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# End model un-pickling
########################################
########################################
# Input validation functions

def check_valid_columns(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
            
    valid_columns = {
        "observation_id",
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
    }
        
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""

def check_numerical_values(observation):
    """
        Validates that observation contains valid age value 
        
        Returns:
        - assertion value: True if age is valid, False otherwise
        - error message: empty if age is valid, False otherwise
    """
    
    valid_range_map = {
        "age": [0, 110],
        "fnlwgt": [0,9999999],
        "education-num": [0,16], # we might choose >16 in case there are extra classes
        "capital-gain": [0,99999],
        "capital-loss": [0,99999],
        "hours-per-week": [0,120],
    }

    for key, valid_range in valid_range_map.items():
        value = observation[key]
        
        if value < valid_range[0] or value > valid_range[1]:
            error = "Invalid value provided for {}: {}. Allowed values are between: {} and {}".format(
                key, value, ",".join(["'{}'".format(v) for v in valid_range]))
            return False, error
    return True, ""

def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = {
        "sex": ["Male", "Female"],
        "race": ['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other']
    }
    
    for key, valid_categories in valid_category_map.items():
        value = observation[key]
        if value not in valid_categories:
            error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
            return False, error

    return True, ""
# End input validation functions
########################################
########################################
# Begin webserver stuff
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()

    # verification routines
    valid_columns_ok, error = check_valid_columns(obs_dict)
    if not valid_columns_ok:
        response = {'error': error}
        return jsonify(response)
    
    valid_numerical_ok, error = check_numerical_values(obs_dict)
    if not valid_numerical_ok:
        response = {'error': error}
        return jsonify(response)
    
    valid_categorical_ok, error = check_categorical_values(obs_dict)
    if not valid_categorical_ok:
        response = {'error': error}
        return jsonify(response)

    # read data
    _id = obs_dict.pop('observation_id')
    obs = pd.DataFrame([obs_dict], columns=columns).astype(dtypes)
    
    # compute prediction
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response = {'observation_id': _id, 'prediction': bool(prediction)}
    
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)
    
@app.route('/update', methods=['POST'])
def update():
    obs_dict = request.get_json()
    
    try:
        p = Prediction.get(Prediction.observation_id == obs_dict['id'])
        p.label = obs_dict['label']
        p.save()
        
        response = obs_dict
        
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs_dict['id'])
        return jsonify({'error': error_msg})
    
if __name__ == "__main__":
    app.run()