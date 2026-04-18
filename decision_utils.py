import joblib

vec, model = joblib.load("model/decision_model.pkl")

def predict(context, a, b):
    text = f"{context} [A] {a} [B] {b}"
    X = vec.transform([text])

    probs = model.predict_proba(X)[0]
    idx = probs.argmax()

    decision = a if idx == 0 else b
    confidence = probs.max()

    # Explanation logic
    if "money" in context.lower():
        reason = "you usually prefer saving money"
    elif "study" in context.lower():
        reason = "you prioritize productivity"
    elif "home" in context.lower():
        reason = "you prefer staying comfortable"
    else:
        reason = "based on your past choices"

    return decision, round(confidence*100,2), reason