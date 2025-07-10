from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import json
import asyncio
from datetime import datetime
import uuid
import google.generativeai as genai
import os
from contextlib import asynccontextmanager

# Configure Gemini API
genai.configure(api_key="AIzaSyDDo-ya0HfN5inz0XwZkW3EuIocPjAf-mc")


print("🔍 Available Gemini models:")
for model in genai.list_models():
    print("🧠", model.name)




# In-memory storage (use Redis/database in production)
emails_db = {}
analysis_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting AI Customer Support API...")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="AI Customer Support API",
    description="Dynamic industry detection and response generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class EmailSubmission(BaseModel):
    from_email: EmailStr
    subject: str
    body: str

class EmailResponse(BaseModel):
    id: str
    from_email: str
    subject: str
    body: str
    timestamp: str
    industry: str
    sentiment: str
    status: str = "new"

class AnalysisRequest(BaseModel):
    email_id: str

class AnalysisResponse(BaseModel):
    industry: str
    confidence: float
    categorization: Dict[str, str]
    analysis: Dict[str, Any]
    suggested_actions: List[Dict[str, Any]]
    initial_reply_draft: str
    next_steps: List[str]
    estimated_resolution: str

class ActionExecution(BaseModel):
    email_id: str
    action_id: str
    action_label: str

class ActionResponse(BaseModel):
    tool_response: Dict[str, Any]
    updated_reply_draft: str

# Email templates by industry
EMAIL_TEMPLATES = {
    "banking": [
        {
            "subject": "Account Access Issue - Unable to Login",
            "body": "Hello, I'm unable to access my online banking account. I've tried resetting my password multiple times but still can't log in. My account number is 123456789. Please help me regain access as I need to make an urgent transfer.",
            "category": "Account Access"
        },
        {
            "subject": "Credit Card Blocked - Need Immediate Help",
            "body": "Hi, My credit card ending in 1234 has been blocked since this morning. I tried making a purchase but the transaction was declined. I have important payments due today. Can you please unblock my card immediately?",
            "category": "Card Services"
        },
        {
            "subject": "Loan Application Status Inquiry",
            "body": "Dear Team, I submitted my home loan application (Reference: HL-2024-567) three weeks ago but haven't received any updates. Could you please check the status and let me know if any additional documents are required?",
            "category": "Loan Services"
        }
    ],
    "insurance": [
        {
            "subject": "Auto Insurance Claim - Accident Report",
            "body": "Hello, I was involved in a car accident yesterday and need to file a claim. The incident happened at 3 PM on Main Street. The other driver was at fault. My policy number is POL-789456. Please guide me through the claim process.",
            "category": "Auto Claims"
        },
        {
            "subject": "Health Insurance Coverage Question",
            "body": "Hi, I'm scheduled for surgery next month and want to confirm my coverage. The procedure code is CPT-12345. My policy number is HIB-654321. What would be my out-of-pocket expenses?",
            "category": "Health Coverage"
        },
        {
            "subject": "Premium Payment Issue - Auto Debit Failed",
            "body": "Dear Sir/Madam, My auto-debit for insurance premium failed last month. I've updated my bank details but want to ensure there's no lapse in coverage. Policy number: LIFE-987654. Please confirm payment status.",
            "category": "Premium Services"
        }
    ],
    "it_support": [
        {
            "subject": "VPN Connection Problem - Cannot Access Network",
            "body": "Hi IT Team, I'm working from home and cannot connect to the company VPN. Getting error code 691. This is blocking my access to internal systems. Please help urgently as I have a client presentation at 2 PM.",
            "category": "Network Access"
        },
        {
            "subject": "Email Account Locked - Need Immediate Resolution",
            "body": "Hello, My email account has been locked due to multiple failed login attempts. Username: john.doe. I need access restored immediately as I'm expecting important client communications. Please reset my account.",
            "category": "Account Management"
        },
        {
            "subject": "Software License Expired - Need Renewal",
            "body": "Hi, The license for Adobe Creative Suite on my workstation has expired. I need this software for ongoing projects. Employee ID: EMP123. Please renew the license or provide alternative access.",
            "category": "Software Licensing"
        }
    ]
}

async def call_gemini_api(prompt: str) -> str:
    """Call Gemini API with error handling"""
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        print("🔹 Gemini raw response:\n", response.text)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback response
        return json.dumps({
            "industry": "unknown",
            "confidence": 0.5,
            "categorization": {
                "level1": "General",
                "level2": "Support",
                "level3": "Inquiry"
            },
            "analysis": {
                "sentiment": "neutral",
                "urgency": "medium",
                "summary": "Customer inquiry received",
                "customer_type": "individual",
                "issue_complexity": "moderate"
            },
            "suggested_actions": [
                {
                    "id": "acknowledge",
                    "label": "Acknowledge Receipt",
                    "description": "Send acknowledgment to customer",
                    "api_endpoint": "/api/acknowledge",
                    "priority": "primary"
                }
            ],
            "initial_reply_draft": "Thank you for contacting us. We have received your inquiry and will respond shortly.",
            "next_steps": ["Review inquiry", "Contact customer"],
            "estimated_resolution": "24 hours"
        })

def detect_industry(subject: str, body: str) -> str:
    """Quick industry detection based on keywords"""
    text = f"{subject} {body}".lower()
    
    banking_keywords = ['account', 'bank', 'credit', 'debit', 'loan', 'mortgage', 'transaction', 'payment', 'balance']
    insurance_keywords = ['claim', 'policy', 'premium', 'coverage', 'accident', 'medical', 'health', 'auto insurance']
    it_keywords = ['password', 'login', 'vpn', 'system', 'software', 'network', 'email', 'access', 'computer']
    
    banking_score = sum(1 for word in banking_keywords if word in text)
    insurance_score = sum(1 for word in insurance_keywords if word in text)
    it_score = sum(1 for word in it_keywords if word in text)
    
    if banking_score >= insurance_score and banking_score >= it_score:
        return "banking"
    elif insurance_score >= it_score:
        return "insurance"
    elif it_score > 0:
        return "it_support"
    else:
        return "other"

def generate_sample_emails(industry: str) -> List[Dict]:
    """Generate sample emails for detected industry"""
    base_emails = {
        "banking": [
            {
                "id": "SR000001",
                "status": "In Progress",
                "category": "TransactionIssue",
                "from_email": "customer1@email.com",
                "subject": "Transaction Failed - Need Refund",
                "body": "Hello, I made a payment of $500 yesterday but the transaction failed. The amount was debited from my account but the recipient did not receive it. Please help me get a refund.",
                "timestamp": "11:30:22",
                "sentiment": "negative",
                "priority": "high",
                "attachments": ["transaction_receipt.pdf"],
                "industry": "banking",
                "unread": False
            },
            {
                "id": "SR000002",
                "status": "Resolved",
                "category": "AccountActivation",
                "from_email": "newcustomer@email.com",
                "subject": "Account Successfully Activated",
                "body": "Thank you for activating my savings account. I can now access online banking and mobile app. Great service!",
                "timestamp": "10:15:08",
                "sentiment": "positive",
                "priority": "low",
                "attachments": [],
                "industry": "banking",
                "unread": False
            }
        ],
        "insurance": [
            {
                "id": "SR000001",
                "status": "New",
                "category": "ClaimStatus",
                "from_email": "policyholder@email.com",
                "subject": "Auto Claim Status Update Required",
                "body": "I submitted claim CLM-2024-789 two weeks ago for my car accident. Haven't received any updates. Please provide status and next steps.",
                "timestamp": "09:45:12",
                "sentiment": "neutral",
                "priority": "medium",
                "attachments": ["police_report.pdf", "photos.zip"],
                "industry": "insurance",
                "unread": True
            },
            {
                "id": "SR000002",
                "status": "In Progress",
                "category": "PolicyInquiry",
                "from_email": "member@email.com",
                "subject": "Coverage Verification for Surgery",
                "body": "I need to verify if my upcoming knee surgery is covered under my current health plan. Surgery date is March 25th. Policy: HP-456789.",
                "timestamp": "14:20:33",
                "sentiment": "neutral",
                "priority": "medium",
                "attachments": ["doctor_note.pdf"],
                "industry": "insurance",
                "unread": False
            }
        ],
        "it_support": [
            {
                "id": "SR000001",
                "status": "New",
                "category": "SystemAccess",
                "from_email": "employee@company.com",
                "subject": "Unable to Access Shared Drive",
                "body": "I cannot access the shared network drive since this morning. Getting access denied error. This is blocking my work on the quarterly report.",
                "timestamp": "08:30:45",
                "sentiment": "negative",
                "priority": "high",
                "attachments": [],
                "industry": "it_support",
                "unread": True
            },
            {
                "id": "SR000002",
                "status": "Resolved",
                "category": "SoftwareRequest",
                "from_email": "designer@company.com",
                "subject": "Software Installation Completed",
                "body": "Thank you for installing the new design software on my workstation. Everything is working perfectly now. Great support!",
                "timestamp": "16:10:22",
                "sentiment": "positive",
                "priority": "low",
                "attachments": [],
                "industry": "it_support",
                "unread": False
            }
        ]
    }
    
    return base_emails.get(industry, base_emails["banking"])

# API Routes
@app.get("/")
async def root():
    return {"message": "AI Customer Support API is running"}

@app.get("/api/templates")
async def get_templates():
    """Get email templates for all industries"""
    return EMAIL_TEMPLATES

@app.post("/api/submit-email", response_model=EmailResponse)
async def submit_email(email: EmailSubmission):
    """Submit a new email for analysis"""
    try:
        # Generate unique ID
        email_id = str(uuid.uuid4())
        
        # Quick industry detection
        industry = detect_industry(email.subject, email.body)
        
        # Create email record
        email_record = {
            "id": email_id,
            "from_email": email.from_email,
            "subject": email.subject,
            "body": email.body,
            "timestamp": datetime.now().isoformat(),
            "industry": industry,
            "sentiment": "neutral",  # Will be updated by analysis
            "status": "new"
        }
        
        # Store in database
        emails_db[email_id] = email_record
        
        return EmailResponse(**email_record)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/emails/{industry}")
async def get_emails_by_industry(industry: str):
    """Get sample emails for a specific industry"""
    try:
        sample_emails = generate_sample_emails(industry)
        return {"emails": sample_emails}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_email(request: AnalysisRequest):
    """Analyze an email using AI"""
    try:
        email_id = request.email_id
        
        # Check cache first
        if email_id in analysis_cache:
            return AnalysisResponse(**analysis_cache[email_id])
        
        # Get email from database
        if email_id not in emails_db:
            raise HTTPException(status_code=404, detail="Email not found")
        
        email = emails_db[email_id]
        
        # Create analysis prompt
        prompt = f"""
        Analyze this customer support email and provide appropriate categorization and actions based on detected industry.

        Email ID: {email['id']}
        Subject: {email['subject']}
        Body: {email['body']}
        From: {email['from_email']}
        Current Status: {email['status']}

        Respond with ONLY a valid JSON object:
        {{
            "industry": "banking/insurance/it_support/healthcare/retail/other",
            "confidence": 0.95,
            "categorization": {{
                "level1": "main department",
                "level2": "category", 
                "level3": "specific issue"
            }},
            "analysis": {{
                "sentiment": "positive/neutral/negative",
                "urgency": "low/medium/high",
                "summary": "brief summary",
                "customer_type": "individual/business/premium",
                "issue_complexity": "simple/moderate/complex"
            }},
            "suggested_actions": [
                {{
                    "id": "action_id",
                    "label": "Button Label",
                    "description": "What this does",
                    "api_endpoint": "/api/endpoint",
                    "priority": "primary/secondary"
                }}
            ],
            "initial_reply_draft": "Professional reply draft",
            "next_steps": ["step1", "step2"],
            "estimated_resolution": "time estimate"
        }}

        DO NOT include backticks, code blocks, or any text outside the JSON object.
        """
        
        # Call Gemini API
        response = await call_gemini_api(prompt)
        
        # Clean and parse response
        clean_response = response.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response.replace('```json\n', '').replace('\n```', '')
        if clean_response.startswith('```'):
            clean_response = clean_response.replace('```\n', '').replace('\n```', '')
        
        try:
            analysis_result = json.loads(clean_response)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response: {clean_response}")
            print("⚠️ JSON decode error:", e)
            print("🟡 Gemini gave this bad response:\n", clean_response)

            # Use fallback analysis
            analysis_result = json.loads(await call_gemini_api("Return a simple JSON with basic analysis"))
        
        # Cache the result
        analysis_cache[email_id] = analysis_result
        
        return AnalysisResponse(**analysis_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/execute-action", response_model=ActionResponse)
async def execute_action(request: ActionExecution):
    """Execute an action and update reply draft"""
    try:
        email_id = request.email_id
        
        # Get email and analysis
        if email_id not in emails_db:
            raise HTTPException(status_code=404, detail="Email not found")
        
        if email_id not in analysis_cache:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        email = emails_db[email_id]
        analysis = analysis_cache[email_id]
        
        # Create action execution prompt
        prompt = f"""
        A customer support agent clicked the "{request.action_label}" button for this {analysis['industry']} customer issue:
        
        Customer Issue: "{analysis['analysis']['summary']}"
        Original Email: "{email['body']}"
        Email ID: {email['id']}
        Current Industry: {analysis['industry']}
        
        Generate realistic mock API response data and an updated reply draft.
        
        Respond with ONLY valid JSON:
        {{
            "tool_response": {{
                "status": "success/error",
                "data": "relevant data object based on the tool type",
                "message": "brief status message",
                "timestamp": "current timestamp"
            }},
            "updated_reply_draft": "Updated professional reply incorporating the fetched data"
        }}
        
        DO NOT include backticks or code blocks.
        """
        
        # Call Gemini API
        response = await call_gemini_api(prompt)
        
        # Clean and parse response
        clean_response = response.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response.replace('```json\n', '').replace('\n```', '')
        if clean_response.startswith('```'):
            clean_response = clean_response.replace('```\n', '').replace('\n```', '')
        
        try:
            action_result = json.loads(clean_response)
        except json.JSONDecodeError:
            # Fallback response
            action_result = {
                "tool_response": {
                    "status": "success",
                    "data": "Action executed successfully",
                    "message": f"Executed {request.action_label}",
                    "timestamp": datetime.now().isoformat()
                },
                "updated_reply_draft": "Thank you for your inquiry. We have processed your request and will respond shortly."
            }
        
        return ActionResponse(**action_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
