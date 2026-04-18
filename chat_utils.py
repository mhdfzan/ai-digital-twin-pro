import joblib, random
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings, outputs = joblib.load("model/semantic_model.pkl")

def chat_reply(text):
    q = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q, embeddings)[0]

    top = sims.topk(3)
    idx = random.choice(top.indices.tolist())

    reply = outputs[idx]
    conf = float(top.values.max())

    if conf < 0.4:
        return "not sure what to say 😅", conf

    return reply, conf