# 📚 Smart Question Analyzer – AI-Powered Question Deduplication & Answer System

This application helps teachers and students by analyzing educational PDFs, extracting questions, identifying duplicates, and generating answers using AI. It's built with the purpose of enhancing learning accessibility, especially for underprivileged communities.

---

## ✨ Features

- 📄 Upload and process multiple PDF question papers.
- 🧠 Automatically extract questions using NLP.
- 🔍 Detect repeated/duplicate questions using TF-IDF and cosine similarity.
- 🤖 Train an ML model to predict repeated questions.
- 💬 Generate high-quality answers using Cohere AI (`command-r` model).
- 💾 Save models for future use.

---

## 💡 Purpose: Helping the Community

This tool is built to assist:

- 📚 **Students from rural or low-income backgrounds** who lack guided educational resources.
- 👩‍🏫 **Teachers** to avoid repeating questions and easily identify important ones.
- 🧪 **NGOs and educators** to prepare mock exams efficiently.
- 🧠 **Learners** who benefit from AI-powered, detailed explanations.

It supports education equity by leveraging AI in places where teaching staff and study material may be limited.

---

## 🧰 Tech Stack

### ⚙️ Backend
| Technology | Purpose |
|------------|---------|
| Python | Core development language |
| pdfplumber | Extracts text from PDFs |
| scikit-learn | TF-IDF, cosine similarity, ML classifier |
| pickle | Save/load ML models |
| requests | API calls to Cohere |

### 💡 AI/NLP
| Tool | Purpose |
|--------------|---------|
| Cohere API | Answer generation (command-r model) |
| TfidfVectorizer | Text vectorization |
| Cosine Similarity | Detect duplicate questions |

### 💽 Frontend / Interface
| Tool | Purpose |
|------|---------|
| Google Colab / Streamlit | Interface for file upload and output display |
| (Optional) React or Flask | Extendable frontend and deployment options |

---

## 📦 Folder Structure (Suggested)

```
📁 project-root/
│
├── app.py # Main program (Colab or Streamlit)
├── model_training.py # ML model training script
├── utils/ # Helpers for cleaning, extraction, etc.
├── data/ # Sample PDFs and test files
├── models/ # Trained model + vectorizer
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 🔹 Step 1: Clone the Repo

```bash
git clone https://github.com/yourusername/smart-question-analyzer.git
cd smart-question-analyzer
```

### 🔹 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 3: Set Your API Key

In your script or `.env` file:
```env
COHERE_API_KEY=your_cohere_key_here
```

Or directly edit the `COHERE_API_KEY` in the script.

### 🔹 Step 4: Launch the App

For Streamlit:
```bash
streamlit run app.py
```

Or run directly in Google Colab.

---

## 🧪 Sample Use Case

Upload 2–3 previous year science/English papers:

- ✅ Extracts 100+ unique questions.
- ↻ Detects 20+ repeated/variant questions.
- 🤖 Trains a model to predict likely repeats.
- 💬 Lets you pick a question and get an AI-generated answer.

---

## 📚 Mock Test Paper Generator (Extension)

You can also add support to generate **7 full mock papers** each for:

- **Science** – 30 questions per paper
- **English** – 30 questions per paper

These can be randomly selected from extracted unique questions or prioritized based on frequency (repeated across years).

✅ Coming Soon in v2.

---

## ✅ requirements.txt

```txt
pdfplumber
scikit-learn
requests
numpy
pandas
streamlit
cohere
```

---

## 🤝 Contributing

Pull requests are welcome! Please fork the repo and create a branch for your feature.

---



## 🌍 Built with ❤️ to Empower Education Everywhere

> Education is a right, not a privilege.  
This tool uses AI to support quality education access for those who need it most.
