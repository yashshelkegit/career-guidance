from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from google.oauth2.service_account import Credentials
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from fastapi.responses import HTMLResponse
from motor.motor_asyncio import AsyncIOMotorClient
# from pymongo import MongoClient
from datetime import datetime
# import datetime
from bson import ObjectId
from fastapi.templating import Jinja2Templates
from database import collection
import gspread
import os
import json
import re
import joblib

from ml.app import predict

input_dict = {
    'Timestamp': '24/04/2025 16:49:49',
    'Mathematics': '5',
    'Biology': '5',
    'Chemistry': '5',
    'Language': '5',
    'Economics/Finance': '5',
    'Social Studies': '5',
    'Research/Ideation': '5',
    'Music/Dance': '5',
    'Drawing/Art': '5',
    'Crafting': '5',
    'Acting/Drama': '5',
    'Communication': '5',
    'Creativity': '4',
    'Design Thinking': '4',
    'Debate': '3',
    'Public Speaking': '2',
    'Physical fitness': '1',
    'Writing Skill': '2',
    'Coding': '3',
    'Exercise/gym': 'Sometimes',
    'Discipline': 'Not disciplined',
    'Player': 'Individual player',
    'Mental strength': 'mentally non-resilient',
    'Personality': 'Extrovert',
    'Led any team': 'No',
    'Emotional': 'Not emotional',
    'Preferred Nation': 'Advance',
    'Debate/Drama': 'Rare',
    'Social Work': 'Rare',
    'Hackathons': 'Rare',
    'Drawing/Arts': 'Rare',
    'Sports or Outdoor Activities': 'Rare',
    'Chess/Strategic Games': 'Rare',
    'Use design software': 'Yes',
    'Social media screen time': '5+',
    'Time spent on interest': '10+',
    'Content pieces per month': '10+',
    'Gender': 'male',
    'DOB': '22/01/2003',
    'Logical Thinking': '3',
    'Verbal Ability': '3',
    'Reasoning Skills': '3',
    'Quantiative Aptitude': '1',
    'Email': 'Bruetmaxx@hotmail.com',
    '': '',
    'Column 1': ''
}

# print(predict(input_dict))

load_dotenv()

groq = os.getenv('groq_api_key')
app_password = os.getenv('app_password')
email = os.getenv('email')

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# client = MongoClient("mongodb://localhost:27017")
# db = client["career"]
# collection = db["guidance"]

conf = ConnectionConfig(
    MAIL_USERNAME=email,
    MAIL_PASSWORD=app_password,
    MAIL_FROM="careerguidance33@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

url =  "https://baseline-barrier-layout-quiet.trycloudflare.com"

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq,
    model_name='llama-3.3-70b-versatile'
)
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file"
]

creds = Credentials.from_service_account_file("creds.json", scopes=scope)
client = gspread.authorize(creds)

sheet_id = "1gzBeR53s2oc2k68VR0QJqH2k0UTvatZmTd8hTtq8cbY"
sheet = client.open_by_key(sheet_id).sheet1
print(sheet)

@app.get("/read-sheet")
def read_sheet():
    data = sheet.get_all_records()
    return {"data": data}

@app.get("/trigger")
async def trigger(background_tasks: BackgroundTasks):
    all_data = sheet.get_all_values()
    headers = all_data[0]
    last_row = all_data[-1]
    row_data = dict(zip(headers, last_row))
    # print(row_data)
    row_data["Preferred Nation"] = "Advance"
    suitability_scores = predict(row_data)
    print(suitability_scores)
    client_email = row_data["Email"]
    response = llm.invoke(f"""
                          Based on this realtime news data: {row_data} and suitability score: {suitability_scores} and also justify as per the suitability score
                            Return a JSON object with the following keys:
                            - "job_roles": A string describing relevant job roles and skills to learn.
                            - "business_opportunities": A string describing business opportunities and where to start.
                            - "project_ideas": A string describing some project ideas.
                            - "future_trends": A string describing upcoming trends.

                            Respond only with the JSON object, no preamble.
                          """)
    print(row_data)
    json = extract_json(response.content)

    # Store in DB asynchronously
    url = await store_in_db(client_email, response.content, score_data={})
    print(url)
    if url:
        email_body = f"""
                    <p>Dear {client_email},</p>
                    <p>Thank you for submitting the Google Form and contributing valuable data to our project. We're excited to share with you a personalized career guidance report based on your input.</p>
                    <p>You can view your curated path by clicking the link below:</p>
                    <p><a href="{url}">{url}</a></p>
                    <p><b>This is temporary link and will expire soon</b></p>
                    <p>We appreciate your involvement</p>
                    <hr>
                    <p><b>This is system generated email!</b></p>
                    <p>Warm regards</p>                    
                    """
        await send_email(client_email, email_body, background_tasks)
    return {"last_row": row_data}


@app.get("/test-db")
async def test_db_connection():
    try:
        count = await collection.count_documents({})
        return {"status": "connected", "document_count": count}
    except Exception as e:
        return {"status": "error", "details": str(e)}
    
@app.get("/test-email")
async def test_email(background_tasks: BackgroundTasks):
    await send_email("youremail@gmail.com", "This is a test email", background_tasks)
    return {"status": "email sent"}


@app.get("/guidance/{item_id}", response_class=HTMLResponse)
async def get_guidance_page(request: Request, item_id: str):
    try:
        obj_id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID format")

    document = await collection.find_one({"_id": obj_id})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    response_raw = document.get('response_content', '')
    response_content = {}

    if isinstance(response_raw, dict):
        response_content = response_raw
    elif isinstance(response_raw, str):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_raw.strip())
        try:
            response_content = json.loads(cleaned)
        except Exception as e:
            print(f"Error parsing response_content: {e}")
            response_content = {}

    timestamp = document.get('timestamp')
    if timestamp and hasattr(timestamp, 'strftime'):
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    return templates.TemplateResponse("guidance.html", {
        "request": request,
        "data": document,
        "timestamp": timestamp,
        "response_content": response_content
    })



async def send_email(email, body, background_tasks: BackgroundTasks):
    try:
        message = MessageSchema(
            subject="Response for your Traits and Data",
            recipients=[email],
            body=body,
            subtype="html"
        )
        fm = FastMail(conf)
        await fm.send_message(message)
        # background_tasks.add_task(fm.send_message, message)
        print(f"Email scheduled to be sent to {email}")
    except Exception as e:
        print(f"Error while scheduling email to {email}: {str(e)}")


def extract_json(response_text: str) -> dict:
    try:
        start_index = response_text.find("{")
        end_index = response_text.rfind("}") + 1  
        if start_index == -1 or end_index == -1:
            return {"status": "error", "message": "No valid JSON object found.", "data": {"answer": ""}}
        json_str = response_text[start_index:end_index]
        print(json_str)
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"JSON decode error while extraction: {str(e)}", "data": {"answer": ""}}
    

async def store_in_db(email, response_content, score_data):
    try:
        document = {
            "email": email,
            "response_content": response_content,
            "score_data": score_data,
            "timestamp": datetime.now()
        }
        result = await collection.insert_one(document)
        print(result.inserted_id)
        return f"{url}/guidance/{str(result.inserted_id)}"  # Return the URL with the inserted ID
    except Exception as e:
        print(f"Error storing in DB: {e}")
        return None

