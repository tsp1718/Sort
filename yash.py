from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import time
import re
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
#
# Import LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn
# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Experience(BaseModel):
    company: str
    title: str
    description: str
    duration: str

class StudentDetails(BaseModel):
    socialProfiles: Dict[str, str]
    skills: List[str]
    collegeName: str
    yearOfStudy: str
    certificates: List[str]
    achievements: str
    experience: List[Experience]

class ApplicantDetails(BaseModel):
    fullName: str
    email: str
    studentDetails: StudentDetails

class JobData(BaseModel):
    jobTitle: str
    jobId: str
    applicants: List[ApplicantDetails]

class SortRequest(BaseModel):
    candidates: List[JobData]
    job_description: str

class SortResponse(BaseModel):
    name: str
    ats_score: int

def extract_candidate_durations(data: List[JobData]) -> List[Dict]:
    candidates_list = []
    for job in data:
        for applicant in job.applicants:
            experiences = applicant.studentDetails.experience
            duration_str = ", ".join(exp.duration for exp in experiences) if experiences else "No experience"
            
            candidate = {
                "name": applicant.fullName,
                "experience": duration_str,
                "skills": applicant.studentDetails.skills,
                "collegeName": applicant.studentDetails.collegeName,
                "yearOfStudy": applicant.studentDetails.yearOfStudy,
                "certificates": applicant.studentDetails.certificates,
                "achievements": applicant.studentDetails.achievements
            }
            candidates_list.append(candidate)
    
    return candidates_list

def analyze_candidates_batch(candidates_batch: List[Dict], job_description: str):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=1000,
        google_api_key=google_api_key
    )
    candidates_str = "\n".join([f"Candidate Name: {c['name']}, Experience: {c['experience']}, Skills: {', '.join(c['skills'])}" for c in candidates_batch])
    
    template = f"""
    You are an AI assistant specialized in candidate analysis for recruitment.
    Analyze the following candidates against the job description and provide an ATS Score (0 to 100):
    
    Candidates:
    {candidates_str}
    
    Job Description:
    {job_description}
    
    Output format:
    Candidate Name: <Name>
    ATS Score: <score>
    """
    
    prompt = PromptTemplate(input_variables=[], template=template)
    chain = prompt | llm
    response = chain.invoke({})
    
    return response.content

def extract_ats_score(candidate_analysis: str) -> int:
    match = re.search(r"ATS Score:\s*(\d+)", candidate_analysis)
    return int(match.group(1)) if match else 0

@app.post("/sort", response_model=List[SortResponse])
async def sort_candidates(request: SortRequest):
    print(request.candidates)
    candidates = extract_candidate_durations(request.candidates)
    job_description = request.job_description
    
    results = []
    rank = []
    batch_size = 5
    max_retries = 5
    
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        
        for attempt in range(max_retries):
            try:
                batch_response = analyze_candidates_batch(batch, job_description)
                break
            except Exception as e:
                print(f"Error: {e}")
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        else:
            raise HTTPException(status_code=500, detail="Failed to analyze candidates after retries")
        
        candidate_analyses = re.split(r"\nCandidate Name:\s*", batch_response)
        candidate_analyses = [s.strip() for s in candidate_analyses if s.strip()]
        
        for analysis in candidate_analyses:
            name_match = re.match(r"([^:\n]+)", analysis)
            candidate_name = name_match.group(1).strip() if name_match else "Unknown"
            ats_score = extract_ats_score(analysis)
            results.append({"name": candidate_name, "ats_score": ats_score})
            rank.append({"name": candidate_name, "ats_score": ats_score})
    
    rank.sort(key=lambda x: x["ats_score"], reverse=True)
    
    return rank

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)