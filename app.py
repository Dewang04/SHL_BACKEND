# app.py using Flask

from flask import Flask, request, jsonify
from flask_cors import CORS 
from SHL_RECOMMENDER_Copy1 import SHLRecommender, format_recommendations

app = Flask(__name__)

# Initialize your recommender
CORS(app)
recommender = SHLRecommender("SHL_DATASET_TOKENIZED.csv")

@app.route('/recommend', methods=['POST'])
def recommend_assessments():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        results = recommender.recommend(query, k=3)
        recommendations = format_recommendations(results)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)

