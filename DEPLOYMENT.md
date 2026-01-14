# Cloudflare 하이브리드 배포 가이드

이 문서는 Claude Code Study 프로젝트를 Cloudflare Pages(프론트엔드) + 로컬 서버(백엔드)로 배포하는 방법을 설명합니다.

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                        인터넷                                   │
├──────────────────────────┬──────────────────────────────────────┤
│                          │                                      │
│   ┌──────────────────────▼────────────────────┐                │
│   │       Cloudflare Pages (CDN)              │                │
│   │   https://your-project.pages.dev          │                │
│   │                                           │                │
│   │   - index.html                            │                │
│   │   - leaderboard.html                      │                │
│   │   - week1.html, week2-5.html              │                │
│   │   - config.js (API_BASE 설정)              │                │
│   └───────────────────────────────────────────┘                │
│                          │                                      │
│                          │ API 요청                             │
│                          ▼                                      │
│   ┌───────────────────────────────────────────┐                │
│   │     Cloudflare Tunnel                      │                │
│   │   https://your-tunnel.trycloudflare.com    │                │
│   └──────────────────────┬────────────────────┘                │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    로컬 서버 (your-computer)                      │
│                                                                  │
│   ┌─────────────────────────────────────────┐                   │
│   │         FastAPI Server (:8003)           │                   │
│   │                                          │                   │
│   │   - /api/auth/*                          │                   │
│   │   - /api/participants/*                  │                   │
│   │   - /api/challenge/*                     │                   │
│   │   - /api/submissions/*                   │                   │
│   │   - /api/leaderboard/*                   │                   │
│   └─────────────────────────────────────────┘                   │
│                          │                                       │
│                          ▼                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │         Claude Code CLI                  │                   │
│   │    (평가 실행 - API 비용 없음)              │                   │
│   └─────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

## 사전 요구 사항

- Cloudflare 계정 (무료)
- Claude Code Pro/Max 구독 (로컬 CLI 사용)
- Python 3.8+
- cloudflared CLI 설치

## 1. 백엔드 서버 설정 (로컬 컴퓨터)

### 1.1 의존성 설치
```bash
cd backend
pip install -r requirements.txt
```

### 1.2 프로덕션 환경변수 설정
```bash
# 필수 환경변수
export ENVIRONMENT=production
export JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# CORS 설정 (Cloudflare Pages URL)
export CORS_ORIGINS="https://your-project.pages.dev"

# 정적 파일 서빙 비활성화 (Cloudflare Pages에서 처리)
export SERVE_STATIC=false
```

### 1.3 서버 실행
```bash
python server.py
```

## 2. Cloudflare Tunnel 설정

### 2.1 cloudflared 설치

**macOS:**
```bash
brew install cloudflare/cloudflare/cloudflared
```

**Linux:**
```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
```

**Windows:**
```powershell
winget install cloudflare.cloudflared
```

### 2.2 빠른 터널 시작 (Quick Tunnel)
```bash
cloudflared tunnel --url http://localhost:8003
```

출력 예시:
```
Your quick Tunnel has been created! Visit it at:
https://random-words-here.trycloudflare.com
```

이 URL을 `frontend/config.js`에 설정합니다.

### 2.3 영구 터널 설정 (선택)

영구 터널을 원하면:
```bash
# 로그인
cloudflared login

# 터널 생성
cloudflared tunnel create claude-code-study

# 설정 파일 생성
cat > ~/.cloudflared/config.yml << EOF
tunnel: <TUNNEL_ID>
credentials-file: ~/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8003
  - service: http_status:404
EOF

# 터널 실행
cloudflared tunnel run claude-code-study
```

## 3. Cloudflare Pages 배포 (프론트엔드)

### 3.1 frontend/config.js 수정

```javascript
// Cloudflare Tunnel URL을 여기에 설정
const CONFIGURED_API_BASE = 'https://your-tunnel.trycloudflare.com';
```

### 3.2 Cloudflare Pages 배포

**방법 1: Cloudflare Dashboard**
1. [Cloudflare Pages](https://pages.cloudflare.com) 접속
2. "Create a project" 클릭
3. GitHub 연동 또는 Direct Upload
4. Build 설정:
   - Framework preset: None
   - Build command: (비워두기)
   - Build output directory: `frontend`

**방법 2: Wrangler CLI**
```bash
npm install -g wrangler
wrangler login
cd frontend
wrangler pages deploy . --project-name=claude-code-study
```

## 4. 환경별 설정 요약

| 환경변수 | 로컬 개발 | 프로덕션 |
|---------|----------|---------|
| `ENVIRONMENT` | development | production |
| `JWT_SECRET` | (자동 생성) | **필수 설정** |
| `CORS_ORIGINS` | (미설정) | `https://your.pages.dev` |
| `SERVE_STATIC` | true | false |
| `ALLOW_ALL_ORIGINS` | (미설정) | (미설정) |
| `CLOUDFLARE_PAGES_URL` | (미설정) | `https://your.pages.dev` |

## 5. 테스트

### 5.1 로컬 테스트
```bash
# 터미널 1: 백엔드 서버
cd backend
python server.py

# 터미널 2: Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8003

# 브라우저에서 접속
# - 로컬: http://localhost:8003
# - 터널: https://random.trycloudflare.com
```

### 5.2 API 테스트
```bash
# 터널 URL로 API 호출
curl https://your-tunnel.trycloudflare.com/api/leaderboard/season
```

## 6. 보안 체크리스트

- [ ] `JWT_SECRET` 환경변수 설정됨
- [ ] `ENVIRONMENT=production` 설정됨
- [ ] `CORS_ORIGINS`에 정확한 Pages URL 설정됨
- [ ] HTTPS 사용 (Cloudflare에서 자동 처리)
- [ ] 민감한 정보가 git에 커밋되지 않음

## 7. 문제 해결

### CORS 오류
```
Access to fetch has been blocked by CORS policy
```
**해결:**
- `CORS_ORIGINS` 환경변수에 Pages URL 추가
- 프로토콜(https://) 포함 확인
- 후행 슬래시 없이 설정

### 터널 연결 불안정
**해결:**
- Quick Tunnel 대신 영구 터널 사용
- `cloudflared tunnel run` 명령으로 실행

### JWT 토큰 오류
**해결:**
- `JWT_SECRET` 환경변수 확인
- 서버 재시작 시 같은 시크릿 사용

## 8. 비용

| 서비스 | 비용 |
|-------|------|
| Cloudflare Pages | 무료 (월 500회 빌드) |
| Cloudflare Tunnel | 무료 |
| Claude Code CLI | Pro/Max 구독에 포함 |
| 로컬 서버 전기료 | 약간 |

## 9. 다른 컴퓨터로 백엔드 이전

1. 새 컴퓨터에 Python 환경 설정
2. 프로젝트 복사 또는 git clone
3. `pip install -r requirements.txt`
4. 환경변수 설정 (JWT_SECRET 동일하게 유지!)
5. cloudflared 설치 및 터널 재설정
6. `frontend/config.js`의 터널 URL 업데이트
7. Cloudflare Pages 재배포
