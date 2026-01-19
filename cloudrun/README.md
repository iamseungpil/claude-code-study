# Claude Code Study - Cloud Run Backend

Google Cloud Run에서 실행되는 백엔드 API 서버입니다.

## 구조

```
cloudrun/
├── Dockerfile           # 컨테이너 이미지 정의
├── requirements.txt     # Python 의존성
├── main.py             # FastAPI 앱 (Firestore 연동)
├── firestore_client.py  # Firestore CRUD 연산
├── evaluator.py        # Claude Code CLI 평가 시스템
├── rubrics/            # 평가 루브릭 (week1-5)
├── cloudbuild.yaml     # Cloud Build CI/CD 설정
└── README.md
```

## 사전 요구사항

1. Google Cloud 프로젝트
2. Firestore 데이터베이스 (Native mode)
3. Secret Manager에 등록된 시크릿:
   - `jwt-secret`: JWT 서명용 시크릿
   - `anthropic-api-key`: Claude API 키

## 배포 방법

### 1. 수동 배포

```bash
# GCP 프로젝트 설정
gcloud config set project YOUR_PROJECT_ID

# Firestore 생성 (첫 배포 시)
gcloud firestore databases create --location=asia-northeast3

# Secret Manager에 시크릿 등록
echo -n "your-jwt-secret" | gcloud secrets create jwt-secret --data-file=-
echo -n "your-anthropic-api-key" | gcloud secrets create anthropic-api-key --data-file=-

# Cloud Run 배포
gcloud run deploy claude-code-study-api \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 900 \
  --set-env-vars "ENVIRONMENT=production,ALLOW_ALL_ORIGINS=true" \
  --set-secrets "JWT_SECRET=jwt-secret:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest"
```

### 2. Cloud Build 사용 (CI/CD)

```bash
# Cloud Build로 배포
gcloud builds submit --config cloudbuild.yaml
```

## 환경 변수

| 변수 | 설명 | 필수 |
|------|------|------|
| `JWT_SECRET` | JWT 서명용 시크릿 | O |
| `ANTHROPIC_API_KEY` | Claude API 키 | O |
| `ENVIRONMENT` | 환경 (production/development) | X |
| `ALLOW_ALL_ORIGINS` | 모든 CORS 허용 (true/false) | X |
| `CLOUDFLARE_PAGES_URL` | Frontend URL (CORS용) | X |

## 로컬 테스트

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
export JWT_SECRET="test-secret"
export ANTHROPIC_API_KEY="your-api-key"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# 서버 실행
python main.py
```

## API 엔드포인트

- `POST /api/auth/register` - 사용자 등록
- `POST /api/auth/login` - 로그인
- `GET /api/auth/me` - 현재 사용자 정보
- `POST /api/admin/challenge/{week}/start` - 챌린지 시작 (관리자)
- `POST /api/admin/challenge/{week}/end` - 챌린지 종료 (관리자)
- `GET /api/challenges/status` - 모든 챌린지 상태
- `POST /api/challenge/{week}/start-personal` - 개인 타이머 시작
- `GET /api/challenge/{week}/my-status` - 개인 상태 조회
- `POST /api/submissions/submit` - 제출
- `GET /api/evaluations/{week}/{participant_id}` - 평가 결과
- `GET /api/leaderboard/{week}` - 주간 리더보드
- `GET /api/leaderboard/season` - 시즌 리더보드
- `GET /api/health` - 헬스 체크

## 비용 (무료 티어)

- Cloud Run: 2M 요청/월, 360,000 GB-초
- Firestore: 1GB 저장, 50K 읽기/일
- Secret Manager: 6개 시크릿 버전, 10K 액세스/월

일반적인 스터디 그룹 사용량은 무료 티어 내에서 처리됩니다.
