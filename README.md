# ML Project 📊

An end-to-end machine learning pipeline: data exploration, model training/comparison, and a Flask web app for predictions, deployed on AWS Elastic Beanstalk.

---

## What it does

- **Notebook** (`notebook/`) for exploratory data analysis
- **Training pipeline** (`src/`) that trains and compares multiple models (CatBoost, XGBoost, scikit-learn estimators) and picks the best performer
- **Flask app** (`app.py`) that serves predictions through a web form (`templates/`)
- **Deployment-ready** via `.ebextensions` for AWS Elastic Beanstalk

---

## Tech Stack

| | |
|---|---|
| Data | pandas, numpy |
| Modeling | CatBoost, XGBoost, scikit-learn |
| Visualization | seaborn, matplotlib |
| Serving | Flask |
| Deployment | AWS Elastic Beanstalk |

---

## Setup & Run

```bash
pip install -r requirements.txt
python app.py
```

---

## 👨‍💻 About the Developer

Built by **Harsh M Singh** — B.Tech CSE (Data Science), Lokmanya Tilak College of Engineering, Mumbai.

- 🔗 GitHub: [github.com/singh-105](https://github.com/singh-105)
- 💼 AI Intern @ Deep Cytes

Feel free to connect, star the repo, or open an issue!
