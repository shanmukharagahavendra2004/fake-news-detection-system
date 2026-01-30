# 📰 Fake News Detection System (ML + MLOps)

An end-to-end **Fake News Detection System** that classifies news articles as **Real** or **Fake** using **Machine Learning** and **Natural Language Processing (NLP)**.
The project follows **MLOps best practices** with automated training using **GitHub Actions CI pipeline**.

---

🔗 **Live Demo (Frontend):** https://fake-news-detectionsystem.vercel.app/ 
🔗 **Backend API:** https://shanmukharaghavendrar-fake-news-detection-system.hf.space  

🔗 **Repository:** [https://github.com/shanmukharagahavendra2004/fake-news-detection-system](https://github.com/shanmukharagahavendra2004/fake-news-detection-system)
🔗 **CI Pipeline:** GitHub Actions (Automated Training & Artifacts)

---

## 🚀 Features

* 🧠 NLP-based text preprocessing (cleaning, tokenization, stopword removal)
* 📊 Machine Learning model for fake news classification
* ⚙️ Automated CI pipeline using **GitHub Actions**
* 📦 Trained model artifact upload
* 🔁 Reproducible ML workflow
* 🐳 Docker-ready setup
* 🧪 Model evaluation support

---

## 🏗️ System Architecture

```
┌──────────────┐
│   Dataset    │
│ fake / true  │
└──────┬───────┘
       ▼
┌──────────────┐
│ Preprocessing│
│  (NLP + NLTK)│
└──────┬───────┘
       ▼
┌──────────────┐
│ ML Training  │
│ (Scikit-learn)
└──────┬───────┘
       ▼
┌──────────────┐
│ Model Output │
│   Artifacts  │
└──────┬───────┘
       ▼
┌──────────────┐
│ GitHub CI/CD │
│ Actions MLOps│
└──────────────┘
```

---

## 🧰 Tech Stack

* **Language**: Python 3.9
* **ML / NLP**: Scikit-learn, NLTK
* **Data Processing**: Pandas, NumPy
* **MLOps**: GitHub Actions
* **Containerization**: Docker
* **Experiment Tracking**: MLflow (optional)

---

## 🔁 CI Pipeline (GitHub Actions)

The CI pipeline automatically runs on every push to the `main` branch.

### 🔄 Pipeline Steps

1. Checkout repository
2. Setup Python environment
3. Install dependencies
4. Download required NLTK resources
5. Train ML model
6. Upload trained model as artifact

```yaml
Trigger → Install → Preprocess → Train → Upload Model
```

---

## 📋 Dataset

Stored inside:

```
fake-news-mlops/dataset/
```

* `fake.csv` → Fake news articles
* `true.csv` → Real news articles

---

## 📁 Project Structure

```
fake-news-detection-system/
│
├── fake-news-frontend/          # Frontend (optional / future)
│
├── fake-news-mlops/
│   ├── dataset/
│   │   ├── fake.csv
│   │   └── true.csv
│   │
│   ├── models/                 # Trained models
│   │
│   ├── src/
│   │   ├── train.py            # Model training
│   │   ├── preprocess.py       # NLP preprocessing
│   │   └── evaluate.py         # Evaluation
│   │
│   ├── app.py                  # Inference / API
│   ├── requirements.txt
│   ├── Dockerfile
│
├── .github/
│   └── workflows/
│       └── main.yml            # CI pipeline
│
└── README.md
```

---

## ▶️ How the ML Pipeline Works

1. **Load Dataset** (`fake.csv`, `true.csv`)
2. **Text Cleaning**

   * Lowercasing
   * Removing punctuation
   * Stopword removal (NLTK)
3. **Feature Extraction**

   * TF-IDF Vectorization
4. **Model Training**

   * Machine Learning classifier
5. **Evaluation**

   * Accuracy and performance metrics
6. **Model Saving**

   * Stored in `models/`
7. **Artifact Upload**

   * Uploaded via GitHub Actions

---

## ⚙️ Local Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shanmukharagahavendra2004/fake-news-detection-system.git
cd fake-news-detection-system
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r fake-news-mlops/requirements.txt
```

### 4️⃣ Download NLTK Resources

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 5️⃣ Train the Model

```bash
python fake-news-mlops/src/train.py
```

---

## 📦 Model Output

* Trained models saved in:

```
fake-news-mlops/models/
```

* CI pipeline uploads the trained model as a **GitHub Actions artifact**

---

## 🐳 Docker Support

Build and run using Docker:

```bash
docker build -t fake-news-detector fake-news-mlops/
docker run fake-news-detector
```

---

## 🧪 Testing

```bash
python fake-news-mlops/src/evaluate.py
```

---

## 👤 Author

**Shanmukha Raghavendra**

* GitHub: [@shanmukharagahavendra2004](https://github.com/shanmukharagahavendra2004)
* LinkedIn: [https://www.linkedin.com/in/shanmukha-raghavendra-ravutu-2b5153289/](https://www.linkedin.com/in/shanmukha-raghavendra-ravutu-2b5153289/)
* Email: [shanmukharaghavendra.r@gmail.com](mailto:shanmukharaghavendra.r@gmail.com)

---

## 🙏 Acknowledgments

* NLTK for NLP utilities
* Scikit-learn for ML models
* GitHub Actions for CI/CD
* Open-source ML community

---

<div align="center">
  Made with ❤️ by Shanmukha Raghavendra  
  ⭐ Star this repository if you find it useful!
</div>

---
