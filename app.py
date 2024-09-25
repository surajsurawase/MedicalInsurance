from flask import Flask, request, render_template, jsonify
from utils import MedicalInsurance, load_dataset
import config
from flask_cors import CORS

df = load_dataset()
med_ins = MedicalInsurance()

app = Flask(__name__)
CORS(app)  

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/smoker_options")
def smoker_options():
    med_ins.load_data()
    return jsonify(list(med_ins.column_data['smoker'].keys()))

@app.route("/gender_options")
def gender_options():
    df['gender'] = df['gender'].apply(lambda x : x.lower())
    
    return jsonify(list(df['gender'].unique()))

@app.route("/region_options")
def region_option():
    return jsonify(list(df['region'].unique()))

@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.form
    print(data)

    age = data['age']
    gender = data['gender']
    bmi = data['bmi']
    children = data['children']
    smoker = data['smoker']
    region = data['region']

    
    pred_charges = med_ins.get_predicted_charges(age, gender, bmi, children, smoker, region)
    
    print(f"Predicted charges: {pred_charges}")  # Debugging log
    return jsonify({'predicted Medical Insurance Charges': pred_charges})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=config.FLASK_PORT_NUMBER, debug=True)