# Job Fitment Scoring Project

This project implements a Job Fitment Scoring System that automatically evaluates a candidate's compatibility with job descriptions. It leverages machine learning and NLP to generate structured data from JDs, filters candidates based on mandatory skills, and computes similarity-based fitment scores using Milvus and Sentence Transformers.

## 🚀 Features

- Extracts structured information (skills, experience, etc.) from unstructured job descriptions.
- Filters candidates from PostgreSQL based on required criteria.
- Computes fitment scores using Sentence-BERT embeddings and Milvus similarity search.
- Scalable and modular codebase.

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: DSFactory, dspy, sentence-transformers
- **Database**: PostgreSQL
- **Vector Store**: Milvus
- **Environment**: `.env` for DB and Milvus URIs


1. **Clone the repository**

git clone https://github.com/your-username/job-fitment-scoring-project.git
cd job-fitment-scoring-project

run the command:
pip install -r requirements.txt

create .env file with the following:
GEMINI_API_KEY= "your api key"

then finally run command:

