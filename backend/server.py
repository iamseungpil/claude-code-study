#!/usr/bin/env python3
"""
Claude Code Study - API Server
FastAPI server for submission handling and leaderboard
"""

import json
import logging
import os
import re
import subprocess
import shutil
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()

import bcrypt
import jwt
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

# Import SQLite database module
from database import (
    init_db,
    create_user as db_create_user,
    get_user as db_get_user,
    get_user_by_email as db_get_user_by_email,
    create_submission as db_create_submission,
    get_submission as db_get_submission,
    get_submission_history as db_get_submission_history,
    get_submission_count as db_get_submission_count,
    get_week_leaderboard as db_get_week_leaderboard,
    get_season_leaderboard as db_get_season_leaderboard,
    get_time_rank_by_elapsed
)

# JWT Configuration
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    if os.environ.get("ENVIRONMENT", "development").lower() == "production":
        raise RuntimeError("JWT_SECRET environment variable must be set in production")
    JWT_SECRET = secrets.token_hex(32)
    logging.warning("Using auto-generated JWT_SECRET. Set JWT_SECRET env var for production.")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Configuration
BASE_DIR = Path(__file__).parent.parent
SUBMISSIONS_DIR = BASE_DIR / "submissions"
EVALUATIONS_DIR = BASE_DIR / "evaluations"
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

# 정적 파일 서빙 설정 (Cloudflare 배포 시 비활성화)
SERVE_STATIC = os.environ.get("SERVE_STATIC", "true").lower() == "true"

# Validation patterns
PARTICIPANT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,30}$')
GITHUB_URL_PATTERN = re.compile(r'^https://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+/?$')
NAME_PATTERN = re.compile(r'^[a-zA-Z가-힣\s]{1,50}$')

# Pydantic Models
def validate_week_range(v: int) -> int:
    """Shared week validation (1-5)."""
    if not 1 <= v <= 5:
        raise ValueError('Week must be between 1 and 5')
    return v


class SubmissionRequest(BaseModel):
    week: int
    github_url: str

    @field_validator('week')
    @classmethod
    def validate_week(cls, v):
        return validate_week_range(v)

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


class StartChallenge(BaseModel):
    participant_id: str
    week: int

    @field_validator('week')
    @classmethod
    def validate_week(cls, v):
        return validate_week_range(v)


class EndChallenge(BaseModel):
    participant_id: str
    week: int

    @field_validator('week')
    @classmethod
    def validate_week(cls, v):
        return validate_week_range(v)

# Authentication Models
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure directories exist
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize SQLite database
    init_db()
    logging.info("SQLite database initialized")

    # Initialize participants.json if not exists
    participants_file = DATA_DIR / "participants.json"
    if not participants_file.exists():
        with open(participants_file, 'w') as f:
            json.dump({"participants": []}, f)

    # Initialize users.json if not exists
    users_file = DATA_DIR / "users.json"
    if not users_file.exists():
        with open(users_file, 'w') as f:
            json.dump({"users": []}, f)

    # Initialize challenges.json if not exists
    challenges_file = DATA_DIR / "challenges.json"
    if not challenges_file.exists():
        with open(challenges_file, 'w') as f:
            json.dump({
                f"week{i}": {"status": "not_started", "start_time": None, "end_time": None}
                for i in range(1, 6)
            }, f, indent=2)

    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="Claude Code Study API",
    description="API for submission and leaderboard",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (환경변수로 유연하게 관리)
ALLOWED_ORIGINS = [
    "http://localhost:8003",
    "http://127.0.0.1:8003",
    "http://localhost:8002",  # 하위 호환성
    "http://127.0.0.1:8002",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://localhost:5500",  # VS Code Live Server
    "http://127.0.0.1:5500",
    "https://claude-code-study.pages.dev",  # Cloudflare Pages production
]

# Cloudflare Pages 도메인 추가
CLOUDFLARE_PAGES_URL = os.environ.get("CLOUDFLARE_PAGES_URL")
if CLOUDFLARE_PAGES_URL:
    ALLOWED_ORIGINS.append(CLOUDFLARE_PAGES_URL.rstrip("/"))

# 추가 CORS origins (쉼표 구분)
if os.environ.get("CORS_ORIGINS"):
    ALLOWED_ORIGINS.extend([
        origin.strip()
        for origin in os.environ.get("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ])

# 개발용: 모든 origin 허용 (주의: 프로덕션에서 사용 금지)
# 와일드카드 CORS와 credentials는 함께 사용할 수 없음
ALLOW_CREDENTIALS = True
if os.environ.get("ALLOW_ALL_ORIGINS", "").lower() == "true":
    ALLOWED_ORIGINS = ["*"]
    ALLOW_CREDENTIALS = False  # 와일드카드 시 credentials 비활성화
    logging.warning("SECURITY: ALLOW_ALL_ORIGINS enabled. Credentials disabled for CORS.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Challenges Helper Functions ==============

def load_challenges() -> dict:
    """Load challenges data from JSON file."""
    challenges_file = DATA_DIR / "challenges.json"
    try:
        with open(challenges_file) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            f"week{i}": {"status": "not_started", "start_time": None, "end_time": None}
            for i in range(1, 6)
        }


def save_challenges(challenges: dict):
    """Save challenges data to JSON file."""
    challenges_file = DATA_DIR / "challenges.json"
    with open(challenges_file, 'w') as f:
        json.dump(challenges, f, indent=2, ensure_ascii=False)


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
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
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
    """Check if profile image exists on sundong.kim.

    The actual image naming convention is: {firstname}{lastname}.{ext}
    e.g., "seungpillee.png", "sundongkim.png"

    Returns:
        Profile image URL if found, None otherwise.
        Never raises exceptions - registration should not fail due to profile image lookup.
    """
    # Build name variants to try
    # Primary: firstname + lastname combined (e.g., "seungpillee")
    full_combined = f"{first_name}{last_name}".lower().replace(" ", "")
    first_only = first_name.lower().replace(" ", "")

    name_variants = [
        full_combined,                    # seungpillee
        first_only,                       # seungpil
        first_name,                       # Seungpil
        first_name.lower(),               # seungpil
        first_name.capitalize(),          # Seungpil
    ]
    # Remove duplicates while preserving order
    name_variants = list(dict.fromkeys(name_variants))

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0),
            follow_redirects=True
        ) as client:
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
        logging.warning(f"Profile image lookup failed for '{first_name} {last_name}': {type(e).__name__}: {e}")

    return None


@app.post("/api/auth/register")
async def register_user(data: UserRegister):
    """Register a new user."""
    users_file = DATA_DIR / "users.json"

    try:
        with open(users_file) as f:
            db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        db = {"users": []}

    full_name = f"{data.first_name} {data.last_name}"

    # Check duplicate user_id
    for u in db["users"]:
        if u.get("user_id", "").lower() == data.user_id.lower():
            raise HTTPException(400, "User ID already exists")

    # Check duplicate full_name
    for u in db["users"]:
        if u["full_name"].lower() == full_name.lower():
            raise HTTPException(400, "User already exists")

    # Get profile image (uses firstname+lastname combined, e.g., "seungpillee.png")
    profile_image = await get_profile_image_url(data.first_name, data.last_name)

    # Determine role: user_id "iamseungpil" is admin, others are participants
    role = "admin" if data.user_id.lower() == "iamseungpil" else "participant"

    db["users"].append({
        "user_id": data.user_id,
        "full_name": full_name,
        "first_name": data.first_name,
        "last_name": data.last_name,
        "password_hash": hash_password(data.password),
        "profile_image": profile_image,
        "role": role,
        "registered_at": datetime.now().isoformat()
    })

    with open(users_file, 'w') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    # Create token
    token = create_jwt_token({
        "user_id": data.user_id,
        "full_name": full_name,
        "first_name": data.first_name,
        "last_name": data.last_name,
        "role": role
    })

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
    users_file = DATA_DIR / "users.json"

    try:
        with open(users_file) as f:
            db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(401, "Invalid credentials")

    # Find user by user_id
    user = None
    for u in db["users"]:
        if u.get("user_id", "").lower() == data.user_id.lower():
            user = u
            break

    if not user:
        raise HTTPException(401, "Invalid credentials")

    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")

    # Create token
    token = create_jwt_token({
        "user_id": user.get("user_id"),
        "full_name": user["full_name"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "role": user.get("role", "participant")
    })

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

    users_file = DATA_DIR / "users.json"

    try:
        with open(users_file) as f:
            db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(401, "Not authenticated")

    # Find user by user_id (primary identifier)
    for u in db["users"]:
        if u.get("user_id", "").lower() == current_user.get("user_id", "").lower():
            return {
                "user_id": u.get("user_id"),
                "full_name": u["full_name"],
                "first_name": u["first_name"],
                "last_name": u["last_name"],
                "profile_image": u.get("profile_image"),
                "role": u.get("role", "participant"),
                "registered_at": u.get("registered_at")
            }

    raise HTTPException(401, "Not authenticated")


# ============== Participants ==============

@app.post("/api/participants/register")
async def register_participant(data: ParticipantRegister):
    """Register a new participant."""
    participants_file = DATA_DIR / "participants.json"
    
    with open(participants_file) as f:
        db = json.load(f)
    
    # Check duplicate
    for p in db["participants"]:
        if p["id"] == data.participant_id:
            raise HTTPException(400, "Participant ID already exists")
    
    db["participants"].append({
        "id": data.participant_id,
        "name": data.name,
        "github": data.github_username,
        "registered_at": datetime.now().isoformat(),
        "scores": {}
    })
    
    with open(participants_file, 'w') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    
    return {"status": "registered", "participant_id": data.participant_id}


@app.get("/api/participants")
async def list_participants():
    """List all participants."""
    participants_file = DATA_DIR / "participants.json"
    
    with open(participants_file) as f:
        db = json.load(f)
    
    return db["participants"]


# ============== Challenge Management (Admin) ==============

@app.post("/api/admin/challenge/{week}/start")
async def admin_start_challenge(week: int, admin: dict = Depends(require_admin)):
    """Admin starts a challenge for all participants."""
    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    challenges = load_challenges()
    week_key = f"week{week}"

    if challenges[week_key]["status"] != "not_started":
        raise HTTPException(400, f"Challenge already {challenges[week_key]['status']}")

    start_time = datetime.now().isoformat()
    challenges[week_key] = {
        "status": "started",
        "start_time": start_time,
        "end_time": None,
        "started_by": admin.get("user_id")
    }
    save_challenges(challenges)

    return {
        "status": "started",
        "week": week,
        "start_time": start_time,
        "started_by": admin.get("user_id")
    }


@app.post("/api/admin/challenge/{week}/end")
async def admin_end_challenge(week: int, admin: dict = Depends(require_admin)):
    """Admin ends a challenge."""
    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    challenges = load_challenges()
    week_key = f"week{week}"

    if challenges[week_key]["status"] != "started":
        raise HTTPException(400, "Challenge is not active")

    end_time = datetime.now().isoformat()
    challenges[week_key]["status"] = "ended"
    challenges[week_key]["end_time"] = end_time
    save_challenges(challenges)

    return {
        "status": "ended",
        "week": week,
        "end_time": end_time
    }


@app.post("/api/admin/challenge/{week}/restart")
async def admin_restart_challenge(week: int, admin: dict = Depends(require_admin)):
    """
    Admin restarts a challenge by clearing all data.
    - Deletes all submissions for the week
    - Deletes all evaluations for the week
    - Resets challenge to not_started state
    """
    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    deleted_submissions = 0
    deleted_evaluations = 0

    # Delete submissions directory for this week
    submissions_dir = Path("submissions") / f"week{week}"
    if submissions_dir.exists():
        for item in submissions_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                deleted_submissions += 1

    # Delete evaluations for this week
    evaluations_dir = Path("evaluations") / f"week{week}"
    if evaluations_dir.exists():
        for item in evaluations_dir.glob("*.json"):
            item.unlink()
            deleted_evaluations += 1

    # Reset challenge to initial state
    challenges = load_challenges()
    week_key = f"week{week}"
    challenges[week_key] = {
        "status": "not_started",
        "start_time": None,
        "end_time": None,
        "started_by": None,
        "personal_starts": {}
    }
    save_challenges(challenges)

    return {
        "status": "restarted",
        "week": week,
        "deleted_submissions": deleted_submissions,
        "deleted_evaluations": deleted_evaluations,
        "message": f"Week {week} has been reset. All submissions and evaluations cleared."
    }


# ============== User Management (Admin) ==============

@app.get("/api/admin/users")
async def admin_list_users(admin: dict = Depends(require_admin)):
    """Admin gets list of all registered users."""
    users_file = DATA_DIR / "users.json"

    try:
        with open(users_file) as f:
            db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    # Return user info without password_hash
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
        for u in db.get("users", [])
    ]


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: str, admin: dict = Depends(require_admin)):
    """Admin deletes a user."""
    users_file = DATA_DIR / "users.json"

    try:
        with open(users_file) as f:
            db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(404, "User not found")

    # Prevent self-deletion
    if user_id.lower() == admin.get("user_id", "").lower():
        raise HTTPException(400, "Cannot delete yourself")

    # Find and remove user
    original_len = len(db.get("users", []))
    db["users"] = [u for u in db.get("users", []) if u.get("user_id", "").lower() != user_id.lower()]

    if len(db["users"]) == original_len:
        raise HTTPException(404, "User not found")

    with open(users_file, 'w') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    return {"status": "deleted", "user_id": user_id}


# ============== Challenge Status (Public) ==============

@app.get("/api/challenges/status")
async def get_all_challenges_status():
    """Get status of all challenges."""
    return load_challenges()


@app.get("/api/challenge/{week}/status")
async def get_challenge_status(week: int):
    """Get status of a specific week's challenge."""
    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    challenges = load_challenges()
    week_key = f"week{week}"

    return {
        "week": week,
        **challenges[week_key]
    }


# ============== Personal Timer APIs ==============

@app.post("/api/challenge/{week}/start-personal")
async def start_personal_timer(
    week: int,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Start personal timer for a user in a specific week's challenge.

    This allows each user to have their own start time within the global challenge window.
    The personal timer starts when the user clicks 'Start' on their interface.

    Returns:
        - started_at: ISO timestamp of when the user started
        - If already started, returns existing start time (idempotent)
    """
    # 1. JWT 인증 확인
    if not current_user:
        raise HTTPException(401, "Authentication required")

    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(401, "Invalid user token")

    # 2. week 유효성 검사
    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    # 3. 챌린지 데이터 로드
    challenges = load_challenges()
    week_key = f"week{week}"
    challenge = challenges[week_key]

    # 4. 챌린지 상태 확인 (started인지)
    if challenge["status"] == "not_started":
        raise HTTPException(400, "Challenge has not started yet. Wait for admin to start.")

    if challenge["status"] == "ended":
        raise HTTPException(400, "Challenge has ended. Cannot start personal timer.")

    # 5. personal_starts 딕셔너리 초기화 (하위 호환성)
    if "personal_starts" not in challenge:
        challenge["personal_starts"] = {}

    # 6. 이미 시작했으면 기존 시간 반환 (중복 방지 - 멱등성 보장)
    if user_id in challenge["personal_starts"]:
        existing = challenge["personal_starts"][user_id]
        return {
            "status": "already_started",
            "week": week,
            "user_id": user_id,
            "started_at": existing["started_at"],
            "message": "Personal timer was already started"
        }

    # 7. 새로운 개인 시작 시간 기록 (UTC)
    started_at = datetime.now(timezone.utc).isoformat()
    challenge["personal_starts"][user_id] = {
        "started_at": started_at,
        "status": "in_progress"
    }

    # 8. challenges.json 저장
    challenges[week_key] = challenge
    save_challenges(challenges)

    return {
        "status": "started",
        "week": week,
        "user_id": user_id,
        "started_at": started_at
    }


@app.get("/api/challenge/{week}/my-status")
async def get_my_challenge_status(
    week: int,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Get current user's personal challenge status for a specific week.

    Returns:
        - challenge_status: Global challenge status (not_started/started/ended)
        - personal_status: User's personal status (not_started/in_progress/submitted)
        - personal_start_time: When user started (if applicable)
        - elapsed_seconds: Seconds since user started (if in_progress)
    """
    # 1. JWT 인증 확인
    if not current_user:
        raise HTTPException(401, "Authentication required")

    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(401, "Invalid user token")

    # 2. week 유효성 검사
    if not 1 <= week <= 5:
        raise HTTPException(400, "Week must be between 1 and 5")

    # 3. 챌린지 데이터 로드
    challenges = load_challenges()
    week_key = f"week{week}"
    challenge = challenges[week_key]

    # 4. 기본 응답 구조
    response = {
        "week": week,
        "challenge_status": challenge["status"],
        "personal_status": "not_started",
        "personal_start_time": None,
        "elapsed_seconds": None
    }

    # 5. personal_starts가 없으면 빈 딕셔너리 처리 (하위 호환성)
    personal_starts = challenge.get("personal_starts", {})

    # 6. 해당 사용자의 개인 시작 상태 조회
    if user_id in personal_starts:
        user_personal = personal_starts[user_id]
        response["personal_status"] = user_personal.get("status", "in_progress")
        response["personal_start_time"] = user_personal.get("started_at")

        # 7. elapsed_seconds 계산 (시작했으면) - UTC 기준
        if response["personal_start_time"] and response["personal_status"] == "in_progress":
            try:
                start_time = datetime.fromisoformat(response["personal_start_time"])
                # Ensure timezone-aware comparison
                now = datetime.now(timezone.utc)
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                elapsed = now - start_time
                response["elapsed_seconds"] = round(elapsed.total_seconds())
            except (ValueError, TypeError):
                # ISO format 파싱 실패 시 None 유지
                response["elapsed_seconds"] = None

    return response


# ============== Submissions ==============

def clone_github_repo(github_url: str, target_dir: Path) -> bool:
    """Clone a GitHub repository."""
    try:
        # Remove existing if any
        if target_dir.exists():
            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Clone] Attempting to clone: {github_url}")
        print(f"[Clone] Target directory: {target_dir}")

        # Clone
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"[Clone] FAILED - returncode: {result.returncode}")
            print(f"[Clone] stderr: {result.stderr}")
            print(f"[Clone] stdout: {result.stdout}")
            return False

        print(f"[Clone] SUCCESS")
        return True
    except subprocess.TimeoutExpired:
        print(f"[Clone] TIMEOUT - exceeded 60 seconds")
        return False
    except Exception as e:
        print(f"[Clone] EXCEPTION: {type(e).__name__}: {e}")
        return False


def trigger_evaluation(week: int, participant_id: str):
    """Trigger Claude Code evaluation in background."""
    try:
        subprocess.run(
            ["python3", str(BASE_DIR / "backend" / "evaluator.py"), 
             "evaluate", str(week), participant_id],
            cwd=str(BASE_DIR),
            timeout=300
        )
    except Exception as e:
        print(f"Evaluation error: {e}")


@app.post("/api/submissions/submit")
async def submit_solution(
    data: SubmissionRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Submit a solution via GitHub URL. Requires authentication."""
    # Authentication check
    if not current_user:
        raise HTTPException(401, "Authentication required to submit solutions")

    # Check challenge status
    challenges = load_challenges()
    week_key = f"week{data.week}"
    challenge = challenges[week_key]

    if challenge["status"] == "not_started":
        raise HTTPException(400, "Challenge has not started yet. Wait for admin to start.")

    if challenge["status"] == "ended":
        raise HTTPException(400, "Challenge has ended. No more submissions allowed.")

    # Use authenticated user's user_id as participant_id
    participant_id = current_user.get("user_id")

    # Check if user has started their personal timer
    personal_starts = challenge.get("personal_starts", {})
    user_personal = personal_starts.get(participant_id)

    if not user_personal:
        raise HTTPException(400, "You must start your timer first before submitting.")

    # Allow resubmission - no longer block on "submitted" status

    # Calculate elapsed time from PERSONAL start (not global) - UTC 기준
    submission_time = datetime.now(timezone.utc)
    personal_start_time = datetime.fromisoformat(user_personal["started_at"])
    # Ensure timezone-aware comparison
    if personal_start_time.tzinfo is None:
        personal_start_time = personal_start_time.replace(tzinfo=timezone.utc)
    elapsed_seconds = (submission_time - personal_start_time).total_seconds()
    elapsed_minutes = elapsed_seconds / 60

    submission_dir = SUBMISSIONS_DIR / f"week{data.week}" / participant_id
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata to track submission history
    metadata_file = submission_dir / "metadata.json"
    existing_history = []
    submission_number = 1

    if metadata_file.exists():
        with open(metadata_file) as f:
            existing_metadata = json.load(f)
            existing_history = existing_metadata.get("submission_history", [])
            # If no history but has submitted_at, create first entry from existing data
            if not existing_history and existing_metadata.get("submitted_at"):
                existing_history.append({
                    "submission_number": 1,
                    "github_url": existing_metadata.get("github_url"),
                    "submitted_at": existing_metadata.get("submitted_at"),
                    "elapsed_seconds": existing_metadata.get("elapsed_seconds"),
                    "elapsed_minutes": existing_metadata.get("elapsed_minutes")
                })
            submission_number = len(existing_history) + 1

    # Create new submission entry
    new_submission = {
        "submission_number": submission_number,
        "github_url": data.github_url,
        "submitted_at": submission_time.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_minutes": round(elapsed_minutes, 1)
    }
    existing_history.append(new_submission)

    # Create metadata with personal time reference and history
    metadata = {
        "participant_id": participant_id,
        "week": data.week,
        "github_url": data.github_url,
        "submitted_at": submission_time.isoformat(),
        "personal_start_time": user_personal["started_at"],
        "global_start_time": challenge.get("start_time"),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_minutes": round(elapsed_minutes, 1),
        "status": "submitted",
        "submission_number": submission_number,
        "submission_history": existing_history
    }

    # Clone repository FIRST (before updating status)
    code_dir = submission_dir / "code"
    success = clone_github_repo(data.github_url, code_dir)

    if not success:
        raise HTTPException(400, "Failed to clone repository")

    # Only update status AFTER successful clone
    challenges[week_key]["personal_starts"][participant_id]["status"] = "submitted"
    save_challenges(challenges)

    metadata["status"] = "cloned"

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save to SQLite database
    try:
        db_create_submission(
            user_id=participant_id,
            week=data.week,
            submission_number=submission_number,
            github_url=data.github_url,
            submitted_at=submission_time.isoformat(),
            elapsed_seconds=round(elapsed_seconds, 1),
            elapsed_minutes=round(elapsed_minutes, 1),
            personal_start_time=user_personal["started_at"]
        )
        logging.info(f"Saved submission to SQLite: {participant_id} week{data.week} try{submission_number}")
    except Exception as e:
        logging.warning(f"SQLite save failed: {e}")

    # Trigger evaluation in background
    background_tasks.add_task(trigger_evaluation, data.week, participant_id)

    return {
        "status": "submitted",
        "message": "Evaluation will start shortly",
        "participant_id": participant_id,
        "week": data.week,
        "elapsed_minutes": round(elapsed_minutes, 1),
        "submission_number": submission_number,
        "is_resubmission": submission_number > 1,
        "submission_history": existing_history
    }


@app.get("/api/submissions/{week}")
async def list_submissions(week: int):
    """List all submissions for a week."""
    submissions_dir = SUBMISSIONS_DIR / f"week{week}"
    
    if not submissions_dir.exists():
        return []
    
    submissions = []
    for participant_dir in submissions_dir.iterdir():
        if participant_dir.is_dir():
            metadata_file = participant_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    submissions.append(json.load(f))
    
    return submissions


@app.get("/api/submissions/{week}/{participant_id}/history")
async def get_submission_history(week: int, participant_id: str):
    """Get submission history for a specific participant.

    Each submission entry may have embedded evaluation data (new format).
    For backward compatibility, also checks eval file for latest submission.
    """
    submission_dir = SUBMISSIONS_DIR / f"week{week}" / participant_id
    metadata_file = submission_dir / "metadata.json"

    if not metadata_file.exists():
        return {"submission_history": [], "total_submissions": 0}

    with open(metadata_file) as f:
        metadata = json.load(f)

    history = metadata.get("submission_history", [])

    # If no history array but has submission data, create initial entry
    if not history and metadata.get("submitted_at"):
        history = [{
            "submission_number": 1,
            "github_url": metadata.get("github_url"),
            "submitted_at": metadata.get("submitted_at"),
            "elapsed_seconds": metadata.get("elapsed_seconds"),
            "elapsed_minutes": metadata.get("elapsed_minutes")
        }]

    # For backward compatibility: if latest submission doesn't have embedded evaluation,
    # try to read from eval file
    if history and not history[-1].get("evaluation"):
        eval_file = EVALUATIONS_DIR / f"week{week}" / f"{participant_id}.json"
        if eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
                # Add latest evaluation to the most recent submission
                history[-1]["evaluation"] = {
                    "total": eval_data.get("scores", {}).get("total"),
                    "rubric": eval_data.get("scores", {}).get("rubric"),
                    "time_rank": eval_data.get("scores", {}).get("time_rank"),
                    "time_rank_bonus": eval_data.get("scores", {}).get("time_rank_bonus"),
                    "status": eval_data.get("status"),
                    "evaluated_at": eval_data.get("evaluated_at")
                }

    return {
        "submission_history": history,
        "total_submissions": len(history),
        "personal_start_time": metadata.get("personal_start_time")
    }


# ============== Evaluations & Leaderboard ==============

@app.get("/api/evaluations/{week}/{participant_id}")
async def get_evaluation(week: int, participant_id: str):
    """Get evaluation result for a participant."""
    eval_file = EVALUATIONS_DIR / f"week{week}" / f"{participant_id}.json"
    
    if not eval_file.exists():
        raise HTTPException(404, "Evaluation not found")
    
    with open(eval_file) as f:
        return json.load(f)


def _get_week_leaderboard_data(week: int) -> list:
    """Helper function to get leaderboard data for a specific week.

    Scoring System:
    - Rubric Score: Up to 80 points (evaluated by Claude)
    - Time Rank Bonus: Up to 20 points (based on submission order)
      - 1st: +20, 2nd: +17, 3rd: +14, 4th: +11, 5th: +8, 6th+: +5
    """
    evaluations_dir = EVALUATIONS_DIR / f"week{week}"

    if not evaluations_dir.exists():
        return []

    results = []
    for eval_file in evaluations_dir.glob("*.json"):
        with open(eval_file) as f:
            data = json.load(f)
            if data.get("status") == "completed":
                scores = data.get("scores", {})
                # Skip entries without valid scores
                if not scores or "total" not in scores:
                    continue
                results.append({
                    "participant_id": data.get("participant") or data.get("participant_id"),
                    "total": scores.get("total", 0),
                    "rubric": scores.get("rubric", 0),
                    "time_rank": scores.get("time_rank", 0),
                    "time_rank_bonus": scores.get("time_rank_bonus", scores.get("time_bonus", 0)),
                    "evaluated_at": data.get("evaluated_at")
                })

    # Sort by total score descending
    results.sort(key=lambda x: x["total"], reverse=True)

    # Add rank (overall rank by score, not submission time)
    for i, r in enumerate(results):
        r["rank"] = i + 1
        if i == 0:
            r["medal"] = "🥇"
        elif i == 1:
            r["medal"] = "🥈"
        elif i == 2:
            r["medal"] = "🥉"
        else:
            r["medal"] = ""

    return results


# NOTE: /api/leaderboard/season must be defined BEFORE /api/leaderboard/{week}
# to prevent FastAPI from matching "season" as a week parameter
@app.get("/api/leaderboard/season")
async def get_season_leaderboard():
    """Get overall season leaderboard (all weeks combined)."""
    participants_file = DATA_DIR / "participants.json"
    
    with open(participants_file) as f:
        db = json.load(f)
    
    # Calculate season points
    season_scores = {}
    
    for week in range(1, 6):
        week_leaderboard = _get_week_leaderboard_data(week)
        
        for entry in week_leaderboard:
            pid = entry["participant_id"]
            if pid not in season_scores:
                season_scores[pid] = {
                    "participant_id": pid,
                    "total_points": 0,
                    "weeks_completed": 0,
                    "weekly_scores": {}
                }
            
            # Weekly ranking points
            rank = entry["rank"]
            if rank == 1:
                points = 10
            elif rank == 2:
                points = 7
            elif rank == 3:
                points = 5
            else:
                points = 3
            
            season_scores[pid]["total_points"] += points
            season_scores[pid]["weeks_completed"] += 1
            season_scores[pid]["weekly_scores"][f"week{week}"] = {
                "rank": rank,
                "points": points,
                "score": entry["total"]
            }
    
    # Sort by total points
    results = list(season_scores.values())
    results.sort(key=lambda x: x["total_points"], reverse=True)
    
    # Add season rank
    for i, r in enumerate(results):
        r["season_rank"] = i + 1
        if i == 0:
            r["title"] = "🎸 Master of Vibe Coding"
        elif i == 1:
            r["title"] = "🥈 Runner-up"
        elif i == 2:
            r["title"] = "🥉 3rd Place"
        else:
            r["title"] = ""
    
    return results


@app.get("/api/leaderboard/{week}")
async def get_week_leaderboard(week: int):
    """Get leaderboard for a specific week."""
    return _get_week_leaderboard_data(week)


# ============== Static Files (Local Development Only) ==============

# Allowed static HTML files (prevents arbitrary file access)
ALLOWED_HTML_FILES = {
    "index.html", "leaderboard.html", "admin.html",
    "week1.html", "week2.html", "week3.html", "week4.html", "week5.html",
    "week1-learn.html", "week2-learn.html", "week3-learn.html",
    "week4-learn.html", "week5-learn.html"
}

if SERVE_STATIC:
    @app.get("/")
    async def serve_index():
        """Serve main page."""
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/config.js")
    async def serve_config_js():
        """Serve config.js."""
        return FileResponse(FRONTEND_DIR / "config.js", media_type="application/javascript")

    @app.get("/{filename:path}")
    async def serve_static_file(filename: str):
        """Serve allowed static HTML files."""
        if filename in ALLOWED_HTML_FILES:
            return FileResponse(FRONTEND_DIR / filename)
        raise HTTPException(404, "File not found")

    @app.get("/challenges/week1/uigen.zip")
    async def serve_week1_uigen():
        """Serve Week 1 UIGen project zip."""
        zip_path = BASE_DIR / "challenges" / "week1" / "uigen.zip"
        if not zip_path.exists():
            raise HTTPException(404, "uigen.zip not found")
        return FileResponse(zip_path, filename="uigen.zip", media_type="application/zip")

    @app.get("/challenges/week2/{filename}")
    async def serve_week2_challenge(filename: str):
        """Serve Week 2 challenge files."""
        file_path = BASE_DIR / "challenges" / "week2" / filename
        if not file_path.exists():
            raise HTTPException(404, f"{filename} not found")
        return FileResponse(file_path, filename=filename)


# ============== Health Check ==============

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
