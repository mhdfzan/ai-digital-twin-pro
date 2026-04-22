import random


def apply_style(text, personality="casual"):
    if personality == "casual":
        if random.random() > 0.6:
            text += " bro"
        if random.random() > 0.55:
            text += " 😄"
    elif personality == "formal":
        text = "Sure. " + text
    elif personality == "funny":
        text += random.choice([" lol 😂", " haha", " bruh 😅", " no cap"])
    return text