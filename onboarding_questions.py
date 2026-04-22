"""
onboarding_questions.py

Each question has:
  - id:       unique key
  - category: grouping label
  - question: what the UI asks the user
  - hint:     placeholder / example shown in input
  - pairs:    list of (input_template, output_template) where {answer} is replaced
              with the user's actual response.
              This is how answers become training data.
"""

QUESTIONS = [

    # ── GREETINGS ──────────────────────────────────────────────────
    {
        "id": "greet_hi",
        "category": "Greetings",
        "question": "When someone says 'hi' to you, what do you usually reply?",
        "hint": "e.g. hey!, yo, what's up, heyy",
        "pairs": [
            ("hi",    "{answer}"),
            ("hello", "{answer}"),
            ("hey",   "{answer}"),
            ("yo",    "{answer}"),
        ]
    },
    {
        "id": "greet_morning",
        "category": "Greetings",
        "question": "How do you reply to 'good morning'?",
        "hint": "e.g. morning!, gm, rise and shine",
        "pairs": [
            ("good morning", "{answer}"),
            ("gm",           "{answer}"),
            ("morning",      "{answer}"),
        ]
    },
    {
        "id": "greet_howru",
        "category": "Greetings",
        "question": "When someone asks 'how are you?', what's your usual reply?",
        "hint": "e.g. I'm good bro, doing great, not bad",
        "pairs": [
            ("how are you",     "{answer}"),
            ("how r u",         "{answer}"),
            ("you good?",       "{answer}"),
            ("how are you doing", "{answer}"),
        ]
    },
    {
        "id": "greet_wyd",
        "category": "Greetings",
        "question": "What do you say when someone asks 'what are you doing?'",
        "hint": "e.g. just chilling, nothing much, working",
        "pairs": [
            ("what are you doing", "{answer}"),
            ("wyd",                "{answer}"),
            ("what you up to",     "{answer}"),
            ("whatcha doing",      "{answer}"),
        ]
    },

    # ── AGREEMENTS ─────────────────────────────────────────────────
    {
        "id": "agree_yes",
        "category": "Agreements",
        "question": "How do you say 'yes' in your own way?",
        "hint": "e.g. yeah bro, yep, absolutely, for sure",
        "pairs": [
            ("yes?",     "{answer}"),
            ("you sure?","{answer}"),
            ("really?",  "{answer}"),
            ("ok?",      "{answer}"),
        ]
    },
    {
        "id": "agree_coming",
        "category": "Agreements",
        "question": "When someone asks 'are you coming?', how do you reply?",
        "hint": "e.g. yeah I'll be there, on my way, yep coming",
        "pairs": [
            ("are you coming",      "{answer}"),
            ("will you come",       "{answer}"),
            ("are you coming today","{answer}"),
            ("coming?",             "{answer}"),
        ]
    },
    {
        "id": "agree_help",
        "category": "Agreements",
        "question": "When someone asks for your help, what do you say?",
        "hint": "e.g. sure, yeah what's up, tell me",
        "pairs": [
            ("can you help me", "{answer}"),
            ("help me",         "{answer}"),
            ("I need help",     "{answer}"),
            ("can you assist",  "{answer}"),
        ]
    },

    # ── DISAGREEMENTS ──────────────────────────────────────────────
    {
        "id": "disagree_no",
        "category": "Disagreements",
        "question": "How do you say 'no' or decline something in your style?",
        "hint": "e.g. nah bro, nope, not really, can't do it",
        "pairs": [
            ("no?",        "{answer}"),
            ("nah",        "{answer}"),
            ("not coming", "{answer}"),
            ("can't?",     "{answer}"),
        ]
    },
    {
        "id": "disagree_busy",
        "category": "Disagreements",
        "question": "When someone asks if you're free but you're busy, what do you say?",
        "hint": "e.g. kinda busy rn, bit tied up, not now",
        "pairs": [
            ("are you free",  "{answer}"),
            ("free now?",     "{answer}"),
            ("are you busy",  "{answer}"),
            ("you available?","{answer}"),
        ]
    },

    # ── REACTIONS ──────────────────────────────────────────────────
    {
        "id": "react_funny",
        "category": "Reactions",
        "question": "When something is funny, how do you react in text?",
        "hint": "e.g. lmao, haha dead, bruh 💀, lol fr",
        "pairs": [
            ("that's funny",     "{answer}"),
            ("lol",              "{answer}"),
            ("haha",             "{answer}"),
            ("that cracked me up","{answer}"),
        ]
    },
    {
        "id": "react_surprised",
        "category": "Reactions",
        "question": "When something surprises you, what do you say?",
        "hint": "e.g. no way!, wait what, bro seriously?, ain't no way",
        "pairs": [
            ("did you hear that",  "{answer}"),
            ("can you believe it", "{answer}"),
            ("that's crazy",       "{answer}"),
            ("are you serious",    "{answer}"),
        ]
    },
    {
        "id": "react_happy",
        "category": "Reactions",
        "question": "When you're happy or excited about something, how do you express it?",
        "hint": "e.g. let's gooo, yesss, fire 🔥, that's so sick",
        "pairs": [
            ("that's great news", "{answer}"),
            ("we won",            "{answer}"),
            ("it worked",         "{answer}"),
            ("that's amazing",    "{answer}"),
        ]
    },
    {
        "id": "react_sad",
        "category": "Reactions",
        "question": "When something bad happens, how do you react?",
        "hint": "e.g. that's rough, rip, that sucks, damn",
        "pairs": [
            ("that's bad news",  "{answer}"),
            ("I failed",         "{answer}"),
            ("it didn't work",   "{answer}"),
            ("that's terrible",  "{answer}"),
        ]
    },

    # ── LOCATION / PLANS ───────────────────────────────────────────
    {
        "id": "loc_where",
        "category": "Location & Plans",
        "question": "When someone asks where you are, how do you usually reply?",
        "hint": "e.g. at home bro, on the way, nearby, chilling at home",
        "pairs": [
            ("where are you",    "{answer}"),
            ("where r u",        "{answer}"),
            ("where you at",     "{answer}"),
            ("your location",    "{answer}"),
        ]
    },
    {
        "id": "plans_food",
        "category": "Location & Plans",
        "question": "When someone suggests food / eating, what do you say?",
        "hint": "e.g. I'm in, let's eat, I'm starving, what are we getting",
        "pairs": [
            ("let's eat",     "{answer}"),
            ("food?",         "{answer}"),
            ("hungry?",       "{answer}"),
            ("let's order",   "{answer}"),
        ]
    },
    {
        "id": "plans_hangout",
        "category": "Location & Plans",
        "question": "When someone suggests hanging out, what do you reply?",
        "hint": "e.g. yeah let's go, sure, I'm down, when and where",
        "pairs": [
            ("let's hang out",    "{answer}"),
            ("wanna meet up",     "{answer}"),
            ("come over",         "{answer}"),
            ("shall we go out",   "{answer}"),
        ]
    },

    # ── THANKS & APOLOGIES ─────────────────────────────────────────
    {
        "id": "thanks",
        "category": "Thanks & Apologies",
        "question": "When someone thanks you, how do you reply?",
        "hint": "e.g. no worries, anytime, don't mention it",
        "pairs": [
            ("thanks",       "{answer}"),
            ("thank you",    "{answer}"),
            ("thx",          "{answer}"),
            ("thanks a lot", "{answer}"),
        ]
    },
    {
        "id": "sorry",
        "category": "Thanks & Apologies",
        "question": "When someone apologizes to you, what do you say?",
        "hint": "e.g. it's fine, no worries, all good, don't stress",
        "pairs": [
            ("sorry",     "{answer}"),
            ("my bad",    "{answer}"),
            ("oops",      "{answer}"),
            ("I'm sorry", "{answer}"),
        ]
    },

    # ── FAREWELLS ──────────────────────────────────────────────────
    {
        "id": "bye",
        "category": "Farewells",
        "question": "How do you say goodbye in your style?",
        "hint": "e.g. later bro, peace, see ya, take care",
        "pairs": [
            ("bye",             "{answer}"),
            ("goodbye",         "{answer}"),
            ("see you later",   "{answer}"),
            ("catch you later", "{answer}"),
            ("good night",      "{answer}"),
            ("gn",              "{answer}"),
        ]
    },

    # ── FILLER / PERSONALITY ───────────────────────────────────────
    {
        "id": "filler_okay",
        "category": "Filler Phrases",
        "question": "When you just want to say 'okay' or acknowledge something, what do you type?",
        "hint": "e.g. ok cool, got it, noted, aight",
        "pairs": [
            ("okay",    "{answer}"),
            ("ok",      "{answer}"),
            ("alright", "{answer}"),
            ("fine",    "{answer}"),
            ("sure",    "{answer}"),
        ]
    },
    {
        "id": "filler_wait",
        "category": "Filler Phrases",
        "question": "When you need someone to wait, what do you say?",
        "hint": "e.g. one sec, hold on, gimme a min, brb",
        "pairs": [
            ("wait",        "{answer}"),
            ("hold on",     "{answer}"),
            ("just a sec",  "{answer}"),
            ("one moment",  "{answer}"),
        ]
    },
    {
        "id": "filler_bored",
        "category": "Filler Phrases",
        "question": "When you're bored and texting someone, what do you say?",
        "hint": "e.g. I'm so bored, nothing to do, man same, dying of boredom",
        "pairs": [
            ("I'm bored",    "{answer}"),
            ("so bored rn",  "{answer}"),
            ("nothing to do","{answer}"),
            ("bored af",     "{answer}"),
        ]
    },
    {
        "id": "filler_tired",
        "category": "Filler Phrases",
        "question": "When you're tired, how do you express it?",
        "hint": "e.g. dead tired, exhausted bro, I can't, I need sleep",
        "pairs": [
            ("I'm tired",    "{answer}"),
            ("so tired",     "{answer}"),
            ("exhausted",    "{answer}"),
            ("need sleep",   "{answer}"),
        ]
    },

    # ── IDENTITY ───────────────────────────────────────────────────
    {
        "id": "identity_name",
        "category": "Identity",
        "question": "When someone asks who you are or what your name is, what do you say?",
        "hint": "e.g. I'm [your name], it's me, you know who I am",
        "pairs": [
            ("who are you",       "{answer}"),
            ("what's your name",  "{answer}"),
            ("introduce yourself","{answer}"),
        ]
    },
    {
        "id": "identity_slang",
        "category": "Identity",
        "question": "What's a word or phrase you say ALL the time? (your signature slang)",
        "hint": "e.g. fr, lowkey, bro, no cap, deadass, on god",
        "pairs": [
            ("say something",      "{answer}"),
            ("talk to me",         "{answer}"),
            ("say your thing",     "{answer}"),
            ("what do you always say", "{answer}"),
        ]
    },
]


def get_training_pairs_from_answers(answers: dict) -> list:
    """
    answers: { question_id: user_answer_string }
    Returns a list of (input, output) tuples ready to write to user_data.txt
    """
    pairs = []
    for q in QUESTIONS:
        qid    = q["id"]
        answer = answers.get(qid, "").strip()
        if not answer:
            continue
        for inp_tpl, out_tpl in q["pairs"]:
            output = out_tpl.replace("{answer}", answer)
            pairs.append((inp_tpl, output))
    return pairs