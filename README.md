# Claude Code Study Group

5주 Claude Code 스터디 평가 시스템

## 🚀 Quick Start

### 1. 의존성 설치
```bash
cd backend
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
# backend 폴더에서
python server.py
```

### 3. 브라우저에서 접속
```
http://localhost:8003
```

## 📁 구조

```
claude-code-study/
├── frontend/           # 웹 페이지
│   ├── index.html      # 메인 페이지
│   └── leaderboard.html # 리더보드
├── backend/            # API 서버
│   ├── server.py       # FastAPI 서버
│   ├── evaluator.py    # 평가 로직
│   └── watcher.py      # 제출 감시
├── submissions/        # 참가자 제출물
├── evaluations/        # 평가 결과
├── rubrics/            # 평가 기준
├── challenges/         # 챌린지 자료
└── .claude/            # Claude Code 설정
```

## 🔌 API Endpoints

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/participants/register` | 참가자 등록 |
| POST | `/api/challenge/start` | 챌린지 시작 |
| POST | `/api/challenge/end` | 챌린지 종료 |
| POST | `/api/submissions/submit` | 솔루션 제출 |
| GET | `/api/leaderboard/{week}` | 주간 리더보드 |
| GET | `/api/leaderboard/season` | 시즌 리더보드 |

## 🎮 사용법

### 참가자 등록
```bash
curl -X POST http://localhost:8003/api/participants/register \
  -H "Content-Type: application/json" \
  -d '{"participant_id": "user001", "name": "홍길동"}'
```

### 챌린지 시작
```bash
curl -X POST http://localhost:8003/api/challenge/start \
  -H "Content-Type: application/json" \
  -d '{"participant_id": "user001", "week": 1}'
```

### 솔루션 제출
```bash
curl -X POST http://localhost:8003/api/submissions/submit \
  -H "Content-Type: application/json" \
  -d '{"participant_id": "user001", "week": 1, "github_url": "https://github.com/user/repo"}'
```

## 🤖 자동 평가

### Watcher 실행 (백그라운드)
```bash
python backend/watcher.py
```

### 수동 평가
```bash
python backend/evaluator.py evaluate 1 user001
```

## 📊 리더보드

`http://localhost:8003/leaderboard.html` 에서 확인

- Season: 전체 시즌 순위
- Week 1-5: 주간 순위

## 🏆 점수 체계

### 주간 순위 포인트
- 1등: 10점
- 2등: 7점
- 3등: 5점
- 완료: 3점

### 시간 보너스
- 제한시간 70% 이내: +10점
- 제한시간 85% 이내: +5점
- 초과: -5점/5분
