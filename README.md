# 🧠 AI Digital Twin Pro

A personalized AI system that mimics user communication style and predicts decision-making behavior.

---

## 🚀 Features

- 💬 **Chat-based AI Twin**  
  Mimics user tone, slang, and conversation style  

- 🧠 **Decision Prediction Engine**  
  Predicts choices based on past behavior  

- 👤 **Multi-User Profiles**  
  Each user has:
  - Name
  - Bio
  - Avatar
  - Personality type  

- 🎭 **Personality Modeling**
  - Casual
  - Formal
  - Funny  
  (affects responses dynamically)

- 🌐 **Premium Web UI**
  - ChatGPT-style interface  
  - Sidebar profile view  
  - Decision panel  

- 💾 **Persistent Storage**
  - Chat history saved in SQLite  
  - User data stored securely  

---

## 🛠️ Tech Stack

- Python  
- Flask  
- Sentence Transformers (NLP)  
- Scikit-learn  
- SQLite  
- HTML / CSS / JavaScript  

---

## 📂 Project Structure
ai-digital-twin-pro/
│
├── app.py                 # Main Flask app
├── database.py            # Database setup (SQLite)
├── chat_utils.py          # Chat model logic
├── decision_utils.py      # Decision prediction logic
├── personality.py         # Personality styling
│
├── train_chat.py          # Train chat model
├── train_decision.py      # Train decision model
│
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── .gitignore             # Ignore unnecessary files
│
├── templates/             # Frontend HTML files
│   ├── index.html         # Main UI
│   ├── login.html         # Login page
│   └── signup.html        # Signup page
│
├── data/                  # Training datasets
│   ├── user_data.txt
│   └── decisions.txt
│
├── model/                 # Trained models (ignored in Git)
│   ├── semantic_model.pkl
│   └── decision_model.pkl
│
├── assets/                # Screenshots/images for README
│   └── ui.png
│
└── venv/                  # Virtual environment (ignored)