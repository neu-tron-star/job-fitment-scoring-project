import os
import dspy
import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure DSPy
lm = dspy.LM('gemini/gemini-2.0-flash', api_key=api_key, temperature=0)
dspy.settings.configure(lm=lm)

# FastAPI setup
app = FastAPI(
    title="Job Fitment Scoring API",
    description="Paste JD text and upload CV PDFs to get candidate fitment scores.",
    version="1.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DSPy scoring signature
class ResumeScorer(dspy.Signature):
    resume_text: str = dspy.InputField()
    job_description_text: str = dspy.InputField()

    skill_score: float = dspy.OutputField(desc="'Relevant Technical Skills' (0-20).")
    experience_score: float = dspy.OutputField(desc="'Experience in Similar Roles' (0-20).")
    achievment_score: float = dspy.OutputField(desc="Achievements and Impact (0-20).")
    communication_score: float = dspy.OutputField(desc="Communication & Presentation (0-20).")
    education_score: float = dspy.OutputField(desc="Educational Background (0-10).")
    overallfit_score: float = dspy.OutputField(desc="Overall Role Fit (0-10).")

scorer = dspy.ChainOfThought(ResumeScorer, parse_output=True)

# Utility: Extract text from uploaded PDF
def read_pdf_text(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        with fitz.open(tmp_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
    finally:
        os.remove(tmp_path)

    return text

# API Endpoint
@app.post("/job_fitment_score")
async def score_cvs(
    jd_text: str = Form(..., description="Paste the Job Description text here."),
    cv_files: List[UploadFile] = File(..., description="One or more CV PDFs"),
):
    results = []

    for cv in cv_files:
        try:
            cv_text = read_pdf_text(cv)
            result = scorer(resume_text=cv_text, job_description_text=jd_text)

            total = (
                float(result.skill_score) +
                float(result.experience_score) +
                float(result.achievment_score) +
                float(result.communication_score) +
                float(result.education_score) +
                float(result.overallfit_score)
            )

            results.append({
                "filename": cv.filename,
                "technical_skills": result.skill_score,
                "experience": result.experience_score,
                "achievements": result.achievment_score,
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
