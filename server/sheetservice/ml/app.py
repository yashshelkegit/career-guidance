import pickle
import pandas as pd
import joblib

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sample input dictionary
input_dict = {
    'Timestamp': '24/04/2025 16:49:49',
    'Mathematics': '5',
    'Biology': '5',
    'Chemistry': '5',
    'Language': '5',
    'Economics/Finance': '5',
    'Social Studies': '5',
    'Research/Ideation': '5',
    'Music/Dance': '5',
    'Drawing/Art': '5',
    'Crafting': '5',
    'Acting/Drama': '5',
    'Communication': '5',
    'Creativity': '4',
    'Design Thinking': '4',
    'Debate': '3',
    'Public Speaking': '2',
    'Physical fitness': '1',
    'Writing Skill': '2',
    'Coding': '3',
    'Exercise/gym': 'Sometimes',
    'Discipline': 'Not disciplined',
    'Player': 'Individual player',
    'Mental strength': 'mentally non-resilient',
    'Personality': 'Extrovert',
    'Led any team': 'No',
    'Emotional': 'Not emotional',
    'Preferred Nation': 'Advance',
    'Debate/Drama': 'Rare',
    'Social Work': 'Rare',
    'Hackathons': 'Rare',
    'Drawing/Arts': 'Rare',
    'Sports or Outdoor Activities': 'Rare',
    'Chess/Strategic Games': 'Rare',
    'Use design software': 'Yes',
    'Social media screen time': '5+',
    'Time spent on interest': '10+',
    'Content pieces per month': '10+',
    'Gender': 'male',
    'DOB': '22/01/2003',
    'Logical Thinking': '3',
    'Verbal Ability': '3',
    'Reasoning Skills': '3',
    'Quantiative Aptitude': '1',
    'Email': 'Bruetmaxx@hotmail.com',
    '': '',
    'Column 1': ''
}

def round_prediction(pred):
    try:
        return round(float(pred), 2)
    except:
        return pred

def law_predict(input_dict):
    with open(os.path.join(BASE_DIR, "pickles/law.pkl"), "rb") as f:
        pipeline = pickle.load(f)
    relevant_features = [
        'Social Studies',
        'Verbal Ability',
        'Reasoning Skills',
        'Debate',
        'Public Speaking',
        'Research/Ideation',
        'Communication',
        'Led any team',
        'Mental strength'
    ]
    data = {key: input_dict[key] for key in relevant_features}
    df = pd.DataFrame([data])
    return round_prediction(pipeline.predict(df)[0])

def media_predict(input_dict):
    with open(os.path.join(BASE_DIR, "pickles/media_communication_pipeline.pkl"), "rb") as f:
        pipeline = pickle.load(f)
    try:
        relevant_features = pipeline.feature_names_in_
    except AttributeError:
        relevant_features = [
            'Language',
            'Communication',
            'Public Speaking',
            'Writing Skill',
            'Acting/Drama',
            'Creativity',
            'Personality'
        ]
    data = {key: input_dict.get(key, 0) for key in relevant_features}
    df = pd.DataFrame([data])
    return round_prediction(pipeline.predict(df)[0])

def sports_predict(input_dict):
    with open(os.path.join(BASE_DIR, "pickles/sports.pkl"), "rb") as file:
        pipeline = pickle.load(file)
    relevant_features = [
        'Physical fitness',
        'Exercise/gym',
        'Sports or Outdoor Activities',
        'Discipline',
        'Mental strength',
        'Player',
        'Emotional'
    ]
    data = {key: input_dict[key] for key in relevant_features}
    df = pd.DataFrame([data])
    return round_prediction(pipeline.predict(df)[0])

def tech_predict(input_dict):
    with open(os.path.join(BASE_DIR, "pickles/tech.pkl"), "rb") as file:
        pipeline = pickle.load(file)
    relevant_features = [
        'Mathematics',
        'Coding',
        'Logical Thinking',
        'Quantiative Aptitude',
        'Research/Ideation',
        'Design Thinking',
        'Use design software',
        'Hackathons',
        'Drawing/Art',
        'Crafting'
    ]
    data = {key: input_dict[key] for key in relevant_features}
    df = pd.DataFrame([data])
    return round_prediction(pipeline.predict(df)[0])

def predict(input_dict = dict):
    with open(os.path.join(BASE_DIR, "pickles/creativity.pkl"), "rb") as file:
        creativity_pipeline = pickle.load(file)
    creativity_features = [
        'Drawing/Art',
        'Creativity',
        'Design Thinking',
        'Music/Dance',
        'Crafting',
        'Writing Skill',
        'Use design software',
        'Mathematics',
        'Logical Thinking'
    ]
    creativity_data = {key: input_dict[key] for key in creativity_features}
    creativity_df = pd.DataFrame([creativity_data])
    creativity_prediction = round_prediction(creativity_pipeline.predict(creativity_df)[0])

    with open(os.path.join(BASE_DIR, "pickles/business.pkl"), "rb") as file:
        business_pipeline = pickle.load(file)
    business_features = [
        'Economics/Finance',
        'Communication',
        'Public Speaking',
        'Reasoning Skills',
        'Led any team',
        'Player',
        'Personality',
        'Mental strength',
        'Preferred Nation',
        'Emotional'
    ]
    business_data = {key: input_dict[key] for key in business_features}
    business_df = pd.DataFrame([business_data])
    business_prediction = round_prediction(business_pipeline.predict(business_df)[0])

    healthcare_pipeline = joblib.load(os.path.join(BASE_DIR, "pickles/healthcare_knn_model.pkl"))
    healthcare_features = [
        'Biology',
        'Chemistry',
        'Mathematics',
        'Research/Ideation',
        'Logical Thinking',
        'Reasoning Skills',
        'Discipline'
    ]
    healthcare_data = {key: input_dict[key] for key in healthcare_features}
    healthcare_df = pd.DataFrame([healthcare_data])
    healthcare_prediction = round_prediction(healthcare_pipeline.predict(healthcare_df)[0])

    law_prediction = law_predict(input_dict)
    media_prediction = media_predict(input_dict)
    tech_prediction = tech_predict(input_dict)
    sports_prediction = sports_predict(input_dict)

    return {
        'Creativity': creativity_prediction,
        'Business': business_prediction,
        'Healthcare': healthcare_prediction,
        'Law': law_prediction,
        'Media and Communication': media_prediction,
        'Technology': tech_prediction,
        'Sports': sports_prediction
    }

# Example usage
if __name__ == "__main__":
    predictions = predict(input_dict)
    print(predictions)
