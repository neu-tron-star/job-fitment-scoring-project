import os
import dspy
import fitz  # PyMuPDF
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Load Gemini API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure DSPy with Gemini model
# Using a specific model like 'gemini-1.5-flash-latest' is recommended
lm = dspy.LM('gemini-1.5-flash-latest', api_key=api_key, temperature=0)
dspy.settings.configure(lm=lm)

# FastAPI app setup
app = FastAPI(
    title="Job Fitment Scoring API",
    description="Paste JD text and upload CV PDFs to get candidate fitment scores.",
    version="1.2.0"
)

# Allow all CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# THIS IS THE CRUCIAL PART FOR THE ROOT URL
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI, click 127.0.0.1:8000/docs to test API"}

# DSPy Signature for scoring
class ResumeScorer(dspy.Signature):
    """Scores a resume against a job description based on multiple criteria."""
    resume_text: str = dspy.InputField()
    job_description_text: str = dspy.InputField()

    skill_score: float = dspy.OutputField(desc="Relevant Technical Skills (0-20).")
    experience_score: float = dspy.OutputField(desc="Experience in Similar Roles (0-20).")
    achievement_score: float = dspy.OutputField(desc="Achievements and Impact (0-20).")
    communication_score: float = dspy.OutputField(desc="Communication & Presentation (0-20).")
    education_score: float = dspy.OutputField(desc="Educational Background (0-10).")
    overallfit_score: float = dspy.OutputField(desc="Overall Role Fit (0-10).")

# Chain of Thought using DSPy
scorer = dspy.ChainOfThought(ResumeScorer)

# Helper function to read PDF text
def read_pdf_text(file: UploadFile) -> str:
    """Reads the text content from an uploaded PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        with fitz.open(tmp_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
    finally:
        os.remove(tmp_path)
    return text

# POST endpoint for job fitment score
@app.post("/job_fitment_score")
async def score_cvs(
    jd_text: str = Form(..., description="Paste the Job Description text here."),
    cv_files: List[UploadFile] = File(..., description="One or more CV PDFs"),
):
    """
    Receives a job description and a list of CV files,
    then returns a fitment score for each CV.
    """
    results = []
    for cv in cv_files:
        try:
            cv_text = read_pdf_text(cv)
            result = scorer(resume_text=cv_text, job_description_text=jd_text)

            total = (
                float(result.skill_score) +
                float(result.experience_score) +
                float(result.achievement_score) +
                float(result.communication_score) +
                float(result.education_score) +
                float(result.overallfit_score)
            )

            results.append({
                "filename": cv.filename,
                "technical_skills": result.skill_score,
                "experience": result.experience_score,
                "achievements": result.achievement_score,
                "communication": result.communication_score,
                "education": result.education_score,
                "overall_fit": result.overallfit_score,
                "total_score": total
            })
        except Exception as e:
            results.append({
                "filename": cv.filename,
                "error": str(e)
            })
    return {"results": results}