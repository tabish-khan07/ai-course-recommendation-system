# Project 3 - AI Recommendation Logic

This command-line project is a content-based course recommendation system. It turns a user's interests into TF-IDF vectors, scores every course using cosine similarity, and displays the Top 3 matches.

## Run it

```powershell
cd "C:\Users\tabis\OneDrive\Documents\decode_project\project_3_recommendation"
python recommendation_system.py
```

Enter at least three interests. Example inputs:

```text
Python
data
machine learning
```

## How it meets the brief

- Takes a minimum of three user interests.
- Uses content-based filtering rather than collaborative filtering.
- Builds TF-IDF vectors from a shared course-keyword vocabulary.
- Uses cosine similarity to score and rank every course.
- Sorts the scores and displays only the Top 3 recommendations.
- Uses only Python's standard library; no package installation is required.
