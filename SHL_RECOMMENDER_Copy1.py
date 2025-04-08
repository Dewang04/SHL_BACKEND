import pandas as pd
import numpy as np
import re
from ast import literal_eval
import textwrap
import os
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini
genai.configure(api_key="AIzaSyA2i0t2xY0GHcPUXjnXucmbjMeaH_PmEPA")  # Replace with your actual API key

# Set up Gemini model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

# Updated safety settings to match the correct Gemini API format
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings,
)


class SHLRecommender:
    def __init__(self, csv_path="SHL_DATASET_TOKENIZED.csv"):
        """Initialize with CSV file containing preprocessed assessment data."""
        self.assessments = self._load_and_preprocess_csv(csv_path)
        # Use TF-IDF vectorizer for embeddings
        self.embedding_model = TfidfVectorizer()
        self._precompute_embeddings()

    def _load_and_preprocess_csv(self, csv_path):
        """Load and validate CSV data, converting string representations if needed."""
        try:
            df = pd.read_csv(csv_path)

            # Convert string representations to lists for Job-Level and Languages
            if 'Job-Level' in df.columns:
                df['Job-Level'] = df['Job-Level'].apply(
                    lambda x: x.split(',') if isinstance(x, str) else ['Not Specified']
                )
            if 'Languages' in df.columns:
                df['Languages'] = df['Languages'].apply(
                    lambda x: x.split(',') if isinstance(x, str) else ['English']
                )

            # Convert boolean columns for Remote Testing and Adaptive/IRT
            bool_mapping = {'yes': True, 'no': False, 'Yes': True, 'No': False, True: True, False: False}
            if 'Remote Testing' in df.columns:
                df['Remote Testing'] = df['Remote Testing'].map(
                    lambda x: bool_mapping.get(x, False) if pd.notna(x) else False
                )
            if 'Adaptive/IRT' in df.columns:
                df['Adaptive/IRT'] = df['Adaptive/IRT'].map(
                    lambda x: bool_mapping.get(x, False) if pd.notna(x) else False
                )

            # Process Duration field: extract numeric values
            if 'Duration' in df.columns:
                df['Duration'] = df['Duration'].apply(
                    lambda x: self._extract_duration(x) if pd.notna(x) else np.nan
                )

            # Convert DataFrame to list of dictionaries
            assessments = df.to_dict("records")
            return assessments

        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return []

    def _extract_duration(self, duration_text):
        """Extract numeric duration value from various text formats."""
        if not isinstance(duration_text, str):
            return duration_text
        numbers = re.findall(r'\d+', duration_text)
        if numbers:
            return int(numbers[0])
        return np.nan

    def _precompute_embeddings(self):
        """Generate embeddings for all assessments using CSV column names."""
        texts = []
        for assess in self.assessments:
            text = (
                f"{assess.get('Individual Test Solutions', '')} "
                f"{assess.get('Description', '')} "
                f"Job Levels: {' '.join(assess.get('Job-Level', []))} "
                f"Languages: {' '.join(assess.get('Languages', []))} "
                f"Test Type: {assess.get('Test Type', '')}"
            )
            texts.append(text)
        if texts:
            self.embedding_model.fit(texts)
            matrix = self.embedding_model.transform(texts)
            self.embeddings = {}
            for i, assess in enumerate(self.assessments):
                test_name = assess.get('Individual Test Solutions', f"Test_{i}")
                self.embeddings[test_name] = matrix[i].toarray()[0]
        else:
            self.embeddings = {}

    def parse_constraints(self, query):
        """Extract constraints from natural language query."""
        constraints = {}
        query_lower = query.lower()

        # Duration extraction
        duration_match = re.search(r'(\d+)\s*minute', query_lower)
        if duration_match:
            constraints['max_duration'] = int(duration_match.group(1))

        # Test type extraction
        test_types = []
        if 'cognitive' in query_lower or 'ability' in query_lower:
            test_types.append('cognitive')
        if 'personality' in query_lower:
            test_types.append('personality')
        if 'skill' in query_lower or 'technical' in query_lower:
            test_types.append('skill')
        if test_types:
            constraints['test_types'] = test_types

        # Remote testing extraction
        if 'remote' in query_lower:
            constraints['remote'] = True
        elif 'in-person' in query_lower or 'onsite' in query_lower:
            constraints['remote'] = False

        # Adaptive testing extraction
        if 'adaptive' in query_lower or 'irt' in query_lower:
            constraints['adaptive'] = True

        return constraints

    def filter_candidates(self, constraints):
        """Apply hard constraints to reduce candidate pool using CSV column names."""
        candidates = []
        for assess in self.assessments:
            match = True

            # Duration constraint
            if 'max_duration' in constraints:
                duration = assess.get('Duration')
                if pd.isna(duration) or duration > constraints['max_duration']:
                    match = False

            # Test type constraint
            if 'test_types' in constraints:
                test_type = assess.get('Test Type', '').lower()
                if not any(tt in test_type for tt in constraints['test_types']):
                    match = False

            # Remote testing constraint
            if 'remote' in constraints:
                if assess.get('Remote Testing') != constraints['remote']:
                    match = False

            # Adaptive constraint
            if 'adaptive' in constraints:
                if constraints['adaptive'] and not assess.get('Adaptive/IRT'):
                    match = False

            if match:
                candidates.append(assess.get("Individual Test Solutions"))
        return candidates

    def calculate_similarity(self, query, test_name):
        """Calculate semantic similarity between query and assessment."""
        query_vector = self.embedding_model.transform([query]).toarray()[0]
        test_vector = self.embeddings.get(test_name)
        if test_vector is None:
            return 0.0
        return cosine_similarity([query_vector], [test_vector])[0][0]

    def calculate_score(self, query, assessment, constraints):
        """Calculate composite recommendation score using CSV columns."""
        test_name = assessment.get("Individual Test Solutions")
        # 1. Semantic similarity (50%)
        semantic_sim = self.calculate_similarity(query, test_name)

        # 2. Constraint matching (30%)
        constraint_score = 0
        if "max_duration" in constraints:
            duration_val = assessment.get("Duration", float("inf"))
            if pd.isna(duration_val):
                duration_val = float("inf")
            duration_ratio = min(1, duration_val / constraints["max_duration"])
            constraint_score += 0.15 * (2 - duration_ratio)
        if "test_types" in constraints:
            test_type = assessment.get("Test Type", "").lower()
            matched_types = sum(1 for tt in constraints["test_types"] if tt in test_type)
            if matched_types > 0:
                constraint_score += 0.15 * (matched_types / len(constraints["test_types"]))

        # 3. Importance weighting (20%) - based on description length
        importance = 0.0
        desc = assessment.get("Description", "")
        if desc and isinstance(desc, str):
            importance = min(1.0, len(desc) / 500)

        return (0.5 * semantic_sim + 0.3 * constraint_score + 0.2 * importance)

    def recommend(self, query, k=3):
        """Main recommendation endpoint using CSV data columns."""
        constraints = self.parse_constraints(query)
        candidates = self.filter_candidates(constraints)

        # Relax constraints if no candidates
        if not candidates and constraints.get("max_duration"):
            relaxed_constraints = constraints.copy()
            relaxed_constraints["max_duration"] += 15
            candidates = self.filter_candidates(relaxed_constraints)
        if not candidates:
            candidates = [assess.get("Individual Test Solutions") for assess in self.assessments]

        scored = []
        for candidate_name in candidates:
            assessment = next((a for a in self.assessments if a.get("Individual Test Solutions") == candidate_name), None)
            if assessment:
                score = self.calculate_score(query, assessment, constraints)
                scored.append((assessment, score))
        ranked = sorted(scored, key=lambda x: -x[1])

        results = []
        for assessment, score in ranked[:k]:
            result = {
                "name": assessment.get("Individual Test Solutions", "Unnamed Test"),
                "url": assessment.get("URL", "#"),
                "duration": assessment.get("Duration", "Not specified"),
                "adaptive": assessment.get("Adaptive/IRT", False),
                "remote": assessment.get("Remote Testing", False),
                "test_type": assessment.get("Test Type", "Not specified"),
                "job_levels": assessment.get("Job-Level", ["Not specified"]),
                "languages": assessment.get("Languages", ["English"]),
                "description": assessment.get("Description", ""),
                "score": score
            }
            results.append({"assessment": result, "score": score})
        return results


def format_recommendations(results):
    """Format recommendations for Gemini conversation as a list."""
    if not results:
        return ["No matching assessments found."]
    formatted = []
    for i, result in enumerate(results, 1):
        assess = result["assessment"]
        formatted.append(
            f"{i}. {assess['name']}\n"
            f"   - Duration: {assess['duration']} mins\n"
            f"   - Type: {assess['test_type']}\n"
            f"   - Job Levels: {', '.join(assess['job_levels'])}\n"
            f"   - Languages: {', '.join(assess['languages'])}\n"
            f"   - Remote: {'Yes' if assess['remote'] else 'No'}\n"
            f"   - Adaptive: {'Yes' if assess['adaptive'] else 'No'}\n"
            f"   - URL: {assess['url']}"
        )
    return formatted



def generate_chat_response(prompt, recommendations):
    """Generate conversational response using Gemini."""
    context = f"""
    You are SHL's AI Assessment Consultant. Your role is to:
    1. Provide personalized assessment recommendations based on user queries.
    2. Explain why each assessment is suitable for their needs.
    3. Answer follow-up questions about assessments.
    4. Maintain a professional, helpful tone.

    Current recommendations based on user query:
    {recommendations}
    """
    try:
        convo = model.start_chat(history=[])
        response = convo.send_message(
            f"{context}\n\nUser message: {prompt}\n"
            "Provide a helpful, conversational response explaining these recommendations:"
        )
        return response.text
    except Exception as e:
        return f"I encountered an error generating a response. Here are the technical recommendations:\n\n{recommendations}"


def main():
    # Initialize recommender using the CSV file with your assessment data.
    recommender = SHLRecommender("SHL_DATASET_TOKENIZED.csv")
    print("SHL Assessment Chatbot Initialized")
    print("Type 'exit' to end the conversation\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Thank you for using SHL's assessment recommender. Goodbye!")
            break
        if not user_input:
            print("Chatbot: Please enter your query about SHL assessments.")
            continue
        try:
            results = recommender.recommend(user_input, k=3)
            rec_list = format_recommendations(results)
            bot_response = generate_chat_response(user_input, "\n".join(rec_list))
            print("\nChatbot:")
            # Print each recommendation on its own line
            for rec in rec_list:
                print(rec)
                print()  # extra newline for readability
            print("\nAdditional Comments:")
            for paragraph in textwrap.wrap(bot_response, width=80):
                print(paragraph)
            print()
        except Exception as e:
            print(f"Chatbot: I encountered an error processing your request. {str(e)}")



if __name__ == "__main__":
    main()
