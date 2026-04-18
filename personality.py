import random

def apply_style(text, personality="casual"):
    # tone
    if personality == "casual":
        if random.random() > 0.6:
            text += " bro"
    elif personality == "formal":
        text = "Sure. " + text
    elif personality == "funny":
        text += " 😂"

    # emoji
    if random.random() > 0.5:
        text += " 😄"

    return text