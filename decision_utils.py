import joblib
import os

MODEL_PATH = "model/decision_model.pkl"
_vec   = None
_model = None


def _load():
    global _vec, _model
    if _vec is None and os.path.exists(MODEL_PATH):
        _vec, _model = joblib.load(MODEL_PATH)


def predict(context, option_a, option_b):
    _load()

    if _vec is None or _model is None:
        # If decision model not trained, just pick first option
        return option_a, 60.0, "Based on your usual preferences."

    text  = f"{context} [A] {option_a} [B] {option_b}"
    X     = _vec.transform([text])
    probs = _model.predict_proba(X)[0]
    idx   = int(probs.argmax())

    decision   = option_a if idx == 0 else option_b
    confidence = round(float(probs.max()) * 100, 1)

    ctx = context.lower()
    if any(w in ctx for w in ["money", "buy", "spend", "save", "cost"]):
        reason = "You tend to be mindful about spending."
    elif any(w in ctx for w in ["study", "work", "assignment", "productive"]):
        reason = "You usually prioritize productivity."
    elif any(w in ctx for w in ["home", "stay", "rest", "relax", "sleep"]):
        reason = "You generally prefer staying comfortable."
    elif any(w in ctx for w in ["gym", "exercise", "workout", "health"]):
        reason = "You value keeping healthy habits."
    elif any(w in ctx for w in ["eat", "food", "hungry", "meal"]):
        reason = "Based on your usual food preferences."
    else:
        reason = "Based on your past decision patterns."

    return decision, confidence, reason