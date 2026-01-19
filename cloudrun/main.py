#!/usr/bin/env python3
"""
Claude Code Study - Cloud Run API Server
FastAPI server with Firestore backend for Cloud Run deployment
"""

import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import firestore_client as db
from evaluator import trigger_evaluation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    if os.environ.get("ENVIRONMENT", "development").lower() == "production":
        raise RuntimeError("JWT_SECRET environment variable must be set in production")
    JWT_SECRET = secrets.token_hex(32)
    logger.warning("Using auto-generated JWT_SECRET. Set JWT_SECRET env var for production.")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Validation patterns
PARTICIPANT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,30}$')
GITHUB_URL_PATTERN = re.compile(r'^https://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+/?$')
NAME_PATTERN = re.compile(r'^[a-zA-Z가-힣\s]{1,50}$')


# ============== Pydantic Models ==============

class SubmissionRequest(BaseModel):
    week: int
    github_url: str

    @field_validator('week')
    @classmethod
    def validate_week(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Week must be between 1 and 5')
        return v

    @field_validator('github_url')
    @classmethod
    def validate_github_url(cls, v):
        if not GITHUB_URL_PATTERN.match(v):
            raise ValueError('Invalid GitHub URL format')
        return v


class ParticipantRegister(BaseModel):
    participant_id: str
    name: str
    github_username: Optional[str] = None

    @field_validator('participant_id')
    @classmethod
    def validate_participant_id(cls, v):
        if not PARTICIPANT_ID_PATTERN.match(v):
            raise ValueError('Participant ID must be 3-30 characters (letters, numbers, _, -)')
        return v


class UserRegister(BaseModel):
    user_id: str
    first_name: str
    last_name: str
    password: str

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not PARTICIPANT_ID_PATTERN.match(v):
            raise ValueError('User ID must be 3-30 characters (letters, numbers, _, -)')
        return v

    @field_validator('first_name', 'last_name')
    @classmethod
    def validate_name(cls, v):
        if not NAME_PATTERN.match(v):
            raise ValueError('Name must contain only letters and spaces (1-50 characters)')
        return v.strip()

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 4:
            raise ValueError('Password must be at least 4 characters')
        return v


class UserLogin(BaseModel):
    user_id: str
    password: str


# ============== FastAPI App ==============

app = FastAPI(
    title="Claude Code Study API",
    description="API for submission and leaderboard (Cloud Run)",
    version="2.0.0"
)

# CORS Configuration
ALLOWED_ORIGINS = [
    "http://localhost:8003",
    "http://127.0.0.1:8003",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

# Add Cloudflare Pages URL if configured
CLOUDFLARE_PAGES_URL = os.environ.get("CLOUDFLARE_PAGES_URL")
if CLOUDFLARE_PAGES_URL:
    ALLOWED_ORIGINS.append(CLOUDFLARE_PAGES_URL.rstrip("/"))

# Add additional CORS origins
if os.environ.get("CORS_ORIGINS"):
    ALLOWED_ORIGINS.extend([
        origin.strip()
        for origin in os.environ.get("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ])

# Allow all origins in development
ALLOW_CREDENTIALS = True
if os.environ.get("ALLOW_ALL_ORIGINS", "").lower() == "true":
    ALLOWED_ORIGINS = ["*"]
    ALLOW_CREDENTIALS = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Authentication ==============

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def create_jwt_token(user_data: dict) -> str:
    """Create a JWT token for a user."""
    payload = {
        "user_id": user_data.get("user_id"),
        "full_name": user_data["full_name"],
        "first_name": user_data["first_name"],
        "last_name": user_data["last_name"],
        "role": user_data.get("role", "participant"),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[dict]:
    """Verify a JWT token and return the payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Dependency to get current user from JWT token."""
    if not authorization:
        return None
    if not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    return verify_jwt_token(token)


async def require_admin(current_user: Optional[dict] = Depends(get_current_user)) -> dict:
    """Dependency to require admin role."""
    if not current_user:
        raise HTTPException(401, "Not authenticated")
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return current_user


async def get_profile_image_url(first_name: str, last_name: str = "") -> Optional[str]:
    """Check if profile image exists on sundong.kim."""
    full_combined = f"{first_name}{last_name}".lower().replace(" ", "")
    first_only = first_name.lower().replace(" ", "")

    name_variants = [full_combined, first_only, first_name.lower()]
    name_variants = list(dict.fromkeys(name_variants))

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0), follow_redirects=True) as client:
            for name in name_variants:
                for ext in ['png', 'jpg', 'jpeg']:
                    url = f"https://sundong.kim/assets/img/members/{name}.{ext}"
                    try:
                        response = await client.head(url)
                        if response.status_code == 200:
                            return url
                    except httpx.HTTPError:
                        continue
    except Exception as e:
        logger.warning(f"Profile image lookup failed: {e}")
    return None


# ============== Auth Routes ==============

@app.post("/api/auth/register")
async def register_user(data: UserRegister):
    """Register a new user."""
    full_name = f"{data.first_name} {data.last_name}"

    # Check duplicate user_id
    if db.get_user(data.user_id):
        raise HTTPException(400, "User ID already exists")

    # Check duplicate full_name
    if db.get_user_by_full_name(full_name):
        raise HTTPException(400, "User already exists")

    # Get profile image
    profile_image = await get_profile_image_url(data.first_name, data.last_name)

    # Determine role
    role = "admin" if data.user_id.lower() == "iamseungpil" else "participant"

    user_data = {
        "user_id": data.user_id,
        "full_name": full_name,
        "first_name": data.first_name,
        "last_name": data.last_name,
        "password_hash": hash_password(data.password),
        "profile_image": profile_image,
        "role": role,
        "registered_at": datetime.now(timezone.utc).isoformat()
    }

    db.create_user(user_data)

    token = create_jwt_token(user_data)

    return {
        "status": "registered",
        "user_id": data.user_id,
        "full_name": full_name,
        "profile_image": profile_image,
        "token": token
    }


@app.post("/api/auth/login")
async def login_user(data: UserLogin):
    """Login a user by user_id."""
    user = db.get_user(data.user_id)

    if not user:
        raise HTTPException(401, "Invalid credentials")

    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")

    token = create_jwt_token(user)

    return {
        "status": "logged_in",
        "user_id": user.get("user_id"),
        "full_name": user["full_name"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "profile_image": user.get("profile_image"),
        "role": user.get("role", "participant"),
        "token": token
    }


@app.get("/api/auth/me")
async def get_current_profile(current_user: Optional[dict] = Depends(get_current_user)):
    """Get current user profile."""
    if not current_user:
        raise HTTPException(401, "Not authenticated")

    user = db.get_user(current_user.get("user_id"))
    if not user:
        raise HTTPException(401, "Not authenticated")

    return {
        "user_id": user.get("user_id"),
        "full_name": user["full_name"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "profile_image": user.get("profile_image"),
        "role": user.get("role", "participant"),
        "registered_at": user.get("registered_at")
    }


# ============== Participants ==============

@app.post("/api/participants/register")
async def register_participant(data: ParticipantRegister):
    """Register a new participant."""
    if db.get_participant(data.participant_id):
        raise HTTPException(400, "Participant ID already exists")

    participant_data = {
        "id": data.participant_id,
        "name": data.name,
        "github": data.github_username,
        "scores": {}
    }

    db.create_participant(participant_data)

    return {"status": "registered", "participant_id": data.participant_id}


@app.get("/api/participants")
async def list_participants():
    """List all participants."""
    return db.list_participants()


# ============== Admin Routes ==============

@app.post("/api/admin/challenge/{week}/start")
async def admin_start_challenge(week: int, admin: dict = Depends(require_admin)):
    """Admin starts a challenge."""
    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    challenge = db.get_challenge(week)
    if challenge["status"] != "not_started":
        raise HTTPException(400, f"Challenge already {challenge['status']}")

    result = db.start_challenge(week, admin.get("user_id"))

    return {
        "status": "started",
        "week": week,
        "start_time": result["start_time"],
        "started_by": admin.get("user_id")
    }


@app.post("/api/admin/challenge/{week}/end")
async def admin_end_challenge(week: int, admin: dict = Depends(require_admin)):
    """Admin ends a challenge."""
    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    challenge = db.get_challenge(week)
    if challenge["status"] != "started":
        raise HTTPException(400, "Challenge is not active")

    result = db.end_challenge(week)

    return {
        "status": "ended",
        "week": week,
        "end_time": result["end_time"]
    }


@app.get("/api/admin/users")
async def admin_list_users(admin: dict = Depends(require_admin)):
    """Admin gets list of all registered users."""
    users = db.list_users()
    return [
        {
            "user_id": u.get("user_id"),
            "full_name": u.get("full_name"),
            "first_name": u.get("first_name"),
            "last_name": u.get("last_name"),
            "role": u.get("role", "participant"),
            "profile_image": u.get("profile_image"),
            "registered_at": u.get("registered_at")
        }
        for u in users
    ]


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: str, admin: dict = Depends(require_admin)):
    """Admin deletes a user."""
    if user_id.lower() == admin.get("user_id", "").lower():
        raise HTTPException(400, "Cannot delete yourself")

    if not db.delete_user(user_id):
        raise HTTPException(404, "User not found")

    return {"status": "deleted", "user_id": user_id}


# ============== Challenge Status ==============

@app.get("/api/challenges/status")
async def get_all_challenges_status():
    """Get status of all challenges."""
    return db.get_all_challenges()


@app.get("/api/challenge/{week}/status")
async def get_challenge_status(week: int):
    """Get status of a specific week's challenge."""
    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    challenge = db.get_challenge(week)
    return {"week": week, **challenge}


# ============== Personal Timer ==============

@app.post("/api/challenge/{week}/start-personal")
async def start_personal_timer(
    week: int,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Start personal timer for a user."""
    if not current_user:
        raise HTTPException(401, "Authentication required")

    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(401, "Invalid user token")

    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    challenge = db.get_challenge(week)

    if challenge["status"] == "not_started":
        raise HTTPException(400, "Challenge has not started yet. Wait for admin to start.")
    if challenge["status"] == "ended":
        raise HTTPException(400, "Challenge has ended. Cannot start personal timer.")

    result = db.start_personal_timer(week, user_id)

    return {
        "status": result["status"],
        "week": week,
        "user_id": user_id,
        "started_at": result["started_at"],
        "message": "Personal timer was already started" if result["status"] == "already_started" else None
    }


@app.get("/api/challenge/{week}/my-status")
async def get_my_challenge_status(
    week: int,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Get current user's personal challenge status."""
    if not current_user:
        raise HTTPException(401, "Authentication required")

    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(401, "Invalid user token")

    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    challenge = db.get_challenge(week)

    response = {
        "week": week,
        "challenge_status": challenge["status"],
        "personal_status": "not_started",
        "personal_start_time": None,
        "elapsed_seconds": None
    }

    personal_starts = challenge.get("personal_starts", {})

    if user_id in personal_starts:
        user_personal = personal_starts[user_id]
        response["personal_status"] = user_personal.get("status", "in_progress")
        response["personal_start_time"] = user_personal.get("started_at")

        if response["personal_start_time"] and response["personal_status"] == "in_progress":
            try:
                start_time = datetime.fromisoformat(response["personal_start_time"])
                now = datetime.now(timezone.utc)
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                elapsed = now - start_time
                response["elapsed_seconds"] = round(elapsed.total_seconds())
            except (ValueError, TypeError):
                response["elapsed_seconds"] = None

    return response


# ============== Submissions ==============

@app.post("/api/submissions/submit")
async def submit_solution(
    data: SubmissionRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Submit a solution via GitHub URL."""
    if not current_user:
        raise HTTPException(401, "Authentication required to submit solutions")

    challenge = db.get_challenge(data.week)

    if challenge["status"] == "not_started":
        raise HTTPException(400, "Challenge has not started yet.")
    if challenge["status"] == "ended":
        raise HTTPException(400, "Challenge has ended. No more submissions allowed.")

    participant_id = current_user.get("user_id")
    personal_starts = challenge.get("personal_starts", {})
    user_personal = personal_starts.get(participant_id)

    if not user_personal:
        raise HTTPException(400, "You must start your timer first before submitting.")
    if user_personal.get("status") == "submitted":
        raise HTTPException(400, "You have already submitted for this week.")

    # Calculate elapsed time
    submission_time = datetime.now(timezone.utc)
    personal_start_time = datetime.fromisoformat(user_personal["started_at"])
    if personal_start_time.tzinfo is None:
        personal_start_time = personal_start_time.replace(tzinfo=timezone.utc)
    elapsed_seconds = (submission_time - personal_start_time).total_seconds()
    elapsed_minutes = elapsed_seconds / 60

    # Save metadata
    metadata = {
        "participant_id": participant_id,
        "week": data.week,
        "github_url": data.github_url,
        "submitted_at": submission_time.isoformat(),
        "personal_start_time": user_personal["started_at"],
        "global_start_time": challenge.get("start_time"),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_minutes": round(elapsed_minutes, 1),
        "status": "submitted"
    }

    db.save_submission_metadata(data.week, participant_id, metadata)
    db.update_personal_status(data.week, participant_id, "submitted")

    # Trigger evaluation in background
    background_tasks.add_task(trigger_evaluation, data.week, participant_id, data.github_url)

    return {
        "status": "submitted",
        "message": "Evaluation will start shortly",
        "participant_id": participant_id,
        "week": data.week,
        "elapsed_minutes": round(elapsed_minutes, 1)
    }


@app.get("/api/submissions/{week}")
async def list_submissions(week: int):
    """List all submissions for a week."""
    return db.list_submissions(week)


# ============== Evaluations & Leaderboard ==============

@app.get("/api/evaluations/{week}/{participant_id}")
async def get_evaluation(week: int, participant_id: str):
    """Get evaluation result for a participant."""
    result = db.get_evaluation(week, participant_id)
    if not result:
        raise HTTPException(404, "Evaluation not found")
    return result


@app.get("/api/leaderboard/season")
async def get_season_leaderboard():
    """Get overall season leaderboard."""
    return db.get_season_leaderboard()


@app.get("/api/leaderboard/{week}")
async def get_week_leaderboard(week: int):
    """Get leaderboard for a specific week."""
    return db.get_week_leaderboard(week)


# ============== Health Check ==============

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "environment": os.environ.get("ENVIRONMENT", "development")
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
