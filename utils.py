import joblib
import random
import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL_PATH = "model/semantic_model.pkl"

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings, outputs = joblib.load(MODEL_PATH)


def generate_reply(user_input):
    if not user_input.strip():
        return "Say something 😅", 0

    # Encode query
    query_vec = model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarity
    similarities = util.cos_sim(query_vec, embeddings)[0]

    sim_scores = similarities.cpu().numpy()

    # Get top matches
    top_indices = sim_scores.argsort()[-3:][::-1]

    # Choose randomly from best matches
    chosen_index = random.choice(top_indices)

    reply = outputs[chosen_index]
    confidence = float(sim_scores[chosen_index])

    # Confidence threshold
    if confidence < 0.40:
        return "I’m not sure what to say 😅", round(confidence * 100, 2)

    return reply, round(confidence * 100, 2)