from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import time
import re
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
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

def extract_candidates_and_job_description(data):
    candidates_list = []
    # Loop through each candidate group in the data
    for candidate_group in data.get("candidates", []):
        # Loop through each applicant in the candidate group
        for applicant in candidate_group.get("applicants", []):
            student = applicant.get("studentDetails", {})
            extracted_candidate = {
                "name": applicant.get("fullName"),
                "experience": student.get("experience", []),
                "skills": student.get("skills", []),
                "collegeName": student.get("collegeName"),
                "yearOfStudy": student.get("yearOfStudy"),
                "certificates": student.get("certificates", []),
                "achievements": student.get("achievements")
            }
            candidates_list.append(extracted_candidate)
    
    # Extract job description from the outer JSON object
    job_description = data.get("job_description")
    return candidates_list, job_description


def extract_json_from_markdown(text):
    """Extract JSON from markdown code blocks or plain text."""
    # Try to extract JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    
    if match:
        return match.group(1).strip()
    
    # If no markdown code block is found, return the original text
    return text.strip()


def analyze_candidates_batch(candidates_batch: List[Dict], job_description: str):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=1000,
        google_api_key=google_api_key
    )
    # Prepare candidate information.
    candidates_str = "\n\n".join([
        f"Candidate Name: {c['name']}\n"
        f"Experience: {json.dumps(c['experience'])}\n"
        f"Skills: {', '.join(c['skills'])}\n"
        f"College Name: {c['collegeName']}\n"
        f"Year of Study: {c['yearOfStudy']}\n"
        f"Certificates: {', '.join(c['certificates'])}\n"
        f"Achievements: {c['achievements']}"
        for c in candidates_batch
    ])
    
    # Updated template to be more explicit about not using code blocks
    template = """
    You are an AI assistant specialized in candidate analysis for recruitment.
    Analyze the following candidates against the job description and provide an ATS Score (0 to 100) for each candidate.
    
    Candidates:
    {candidates}
    
    Job Description:
    {job_description}
    
    Please output a valid JSON array where each element is an object with the following keys:
      - "Candidate Name": the candidate's full name
      - "ATS Score": an integer score from 0 to 100
    
    IMPORTANT: Return ONLY the raw JSON array with no markdown formatting, code blocks, or additional text.
    
    Example of the expected format:
    [
      {{"Candidate Name": "John Doe", "ATS Score": 85}},
      {{"Candidate Name": "Jane Smith", "ATS Score": 90}}
    ]
    """
    
    # Define the prompt with the correct input variables
    prompt = PromptTemplate(
        input_variables=["candidates", "job_description"],
        template=template
    )
    
    # Invoke the chain with the correct input variables
    chain = prompt | llm
    response = chain.invoke({
        "candidates": candidates_str,
        "job_description": job_description
    })
    
    # Extract JSON from the response content (which might be wrapped in code blocks)
    json_str = extract_json_from_markdown(response.content)
    
    # Parse the JSON response from the LLM output.
    try:
        candidate_list = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {e}. Extracted JSON string: {json_str}")
    
    return candidate_list


def extract_ats_score(candidate_analysis: str) -> int:
    match = re.search(r"ATS Score:\s*(\d+)", candidate_analysis)
    return int(match.group(1)) if match else 0

@app.post("/sort", response_model=List[SortResponse])
async def sort_candidates(request: SortRequest):
    # Use model_dump() instead of dict() for Pydantic v2 compatibility
    candidates, job_description = extract_candidates_and_job_description(request.model_dump())
    
    results = []
    batch_size = 5
    max_retries = 5
    
    # We accumulate candidate results from each batch.
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                candidate_list = analyze_candidates_batch(batch, job_description)
                break
            except Exception as e:
                print(f"Error: {e}")
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        else:
            raise HTTPException(status_code=500, detail="Failed to analyze candidates after retries")
        
        results.extend(candidate_list)
    
    # Sort the results based on ATS score in descending order.
    results.sort(key=lambda x: x["ATS Score"], reverse=True)
    
    # Map keys to the expected response model.
    sorted_results = [{"name": c["Candidate Name"], "ats_score": c["ATS Score"]} for c in results]
    return sorted_results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)