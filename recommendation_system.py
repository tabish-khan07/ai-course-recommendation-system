"""Project 3: a content-based course recommendation system."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Course:
    """Represent one course and the keywords that describe its content."""

    title: str
    description: str
    keywords: str


COURSES = (
    Course(
        "Python Automation Fundamentals",
        "Build practical scripts that automate repetitive work.",
        "python automation scripting productivity",
    ),
    Course(
        "Machine Learning Foundations",
        "Learn supervised learning, data preparation, and model evaluation.",
        "python machine learning data algorithms classification",
    ),
    Course(
        "Web Design Essentials",
        "Create responsive websites with modern layouts and user interfaces.",
        "web design html css frontend user interface",
    ),
    Course(
        "Cloud Computing Basics",
        "Understand cloud services, deployment, and scalable infrastructure.",
        "cloud deployment infrastructure devops services",
    ),
    Course(
        "Data Analysis with Python",
        "Explore, clean, and visualise data to discover useful insights.",
        "python data analysis visualization statistics pandas",
    ),
    Course(
        "Java Programming and Algorithms",
        "Develop Java programs and solve algorithmic problems efficiently.",
        "java programming algorithms data structures coding",
    ),
    Course(
        "Cybersecurity Fundamentals",
        "Learn security principles, network protection, and ethical practices.",
        "cybersecurity security network ethical hacking privacy",
    ),
    Course(
        "Mobile App Development",
        "Design and develop mobile applications with modern tools.",
        "mobile app development android user interface programming",
    ),
    Course(
        "Deep Learning Introduction",
        "Explore neural networks, tensors, and practical AI concepts.",
        "python deep learning neural networks ai tensors",
    ),
    Course(
        "Database Management Systems",
        "Work with relational databases, SQL queries, and data modelling.",
        "database sql data management backend queries",
    ),
)


def tokenize(text: str) -> list[str]:
    """Convert text to lowercase keyword tokens."""
    return re.findall(r"[a-z]+", text.lower())


def build_idf(documents: Iterable[Sequence[str]]) -> dict[str, float]:
    """Calculate smooth inverse-document-frequency values for a corpus."""
    document_list = list(documents)
    document_frequency: Counter[str] = Counter()
    for document in document_list:
        document_frequency.update(set(document))

    total_documents = len(document_list)
    return {
        term: log((1 + total_documents) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }


def tf_idf_vector(tokens: Sequence[str], idf: dict[str, float]) -> dict[str, float]:
    """Build a TF-IDF vector using the vocabulary shared by all courses."""
    if not tokens:
        return {}

    frequencies = Counter(token for token in tokens if token in idf)
    token_count = sum(frequencies.values())
    if token_count == 0:
        return {}

    return {
        term: (frequency / token_count) * idf[term]
        for term, frequency in frequencies.items()
    }


def cosine_similarity(first: dict[str, float], second: dict[str, float]) -> float:
    """Return cosine similarity between two sparse TF-IDF vectors."""
    dot_product = sum(value * second.get(term, 0.0) for term, value in first.items())
    first_length = sqrt(sum(value * value for value in first.values()))
    second_length = sqrt(sum(value * value for value in second.values()))
    if first_length == 0 or second_length == 0:
        return 0.0
    return dot_product / (first_length * second_length)


def recommend(interests: Sequence[str], limit: int = 3) -> list[tuple[Course, float, list[str]]]:
    """Rank courses by TF-IDF cosine similarity to the user's interests."""
    if len(interests) < 3:
        raise ValueError("Please provide at least three interests.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    course_tokens = [tokenize(course.keywords) for course in COURSES]
    idf = build_idf(course_tokens)
    user_tokens = tokenize(" ".join(interests))
    user_vector = tf_idf_vector(user_tokens, idf)

    if not user_vector:
        raise ValueError("None of those interests match the available course keywords.")

    ranked_courses = []
    user_terms = set(user_tokens)
    for course, tokens in zip(COURSES, course_tokens):
        score = cosine_similarity(user_vector, tf_idf_vector(tokens, idf))
        matching_terms = sorted(user_terms.intersection(tokens))
        ranked_courses.append((course, score, matching_terms))

    return sorted(ranked_courses, key=lambda result: result[1], reverse=True)[:limit]


def get_interests() -> list[str]:
    """Collect the minimum three interests required by the project brief."""
    print("Enter at least three interests, such as Python, cloud, or web design.")
    interests = []
    while len(interests) < 3:
        interest = input(f"Interest {len(interests) + 1}: ").strip()
        if interest:
            interests.append(interest)
        else:
            print("Please enter an interest.")
    return interests


def main() -> None:
    """Run the interactive recommendation system."""
    print("\n=== AI Course Recommendation System ===\n")
    interests = get_interests()

    try:
        results = recommend(interests)
    except ValueError as error:
        print(f"\n{error}")
        print("Try terms like Python, data, web, cloud, Java, security, or mobile.")
        return

    print("\nTop 3 recommendations:\n")
    for position, (course, score, matching_terms) in enumerate(results, start=1):
        matches = ", ".join(matching_terms) if matching_terms else "related course content"
        print(f"{position}. {course.title} - {score:.1%} match")
        print(f"   {course.description}")
        print(f"   Matching interests: {matches}\n")


if __name__ == "__main__":
    main()
