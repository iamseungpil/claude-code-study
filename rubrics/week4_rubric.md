# Week 4 Rubric: Bot/App MVP

## Challenge Summary
Discord 봇 또는 웹 대시보드 MVP를 팀으로 개발하고, CLAUDE.md로 팀 컨텍스트를 공유하는 챌린지

## Time Limit: 60 minutes

## Scoring: 90 points (rubric) + 10 points (time bonus)

---

## Evaluation Process

### Step 1: MVP 실행 검증 (필수)
```bash
cd submissions/week4/{participant_id}

# Option A: Discord Bot
npm install
node bot.js  # 또는 npm start
# 기대: 봇이 온라인 상태로 전환

# Option B: Web Dashboard
npm install
npm run dev  # 또는 npm start
# 기대: http://localhost:3000 접속 가능

# CLAUDE.md 확인
cat CLAUDE.md
# 기대: 프로젝트 컨텍스트 문서화
```

**MVP 실행 실패 시**: MVP Functionality 점수 50% 감점

### Step 2: 코드 리뷰 (Claude)
실행 검증 후, 아래 Rubric에 따라 검토

---

## Rubric Breakdown (90 points)

### MVP Validation (필수 조건)
| Test | Expected | Impact |
|------|----------|--------|
| npm install | 성공 | 실패 시 평가 중단 |
| npm start/dev | 성공 | 실패 시 50% 감점 |
| CLAUDE.md | 존재 | CLAUDE.md Quality 영향 |

### MVP Functionality (30 points)
| Item | Points | Criteria |
|------|--------|----------|
| Core Feature 1 | 10 | 첫 번째 핵심 기능 동작 |
| Core Feature 2 | 10 | 두 번째 핵심 기능 동작 |
| Core Feature 3 | 10 | 세 번째 핵심 기능 동작 |

**Discord Bot 체크리스트:**
- [ ] 봇이 온라인 상태로 연결
- [ ] 최소 3개 명령어 동작 (예: /search, /summary, /notify)
- [ ] 응답 속도 적절함

**Web Dashboard 체크리스트:**
- [ ] 메인 페이지 렌더링
- [ ] 데이터 표시 기능
- [ ] 인터랙션 기능 (버튼, 폼 등)

### CLAUDE.md Quality (20 points)
| Item | Points | Criteria |
|------|--------|----------|
| Project Overview | 5 | 프로젝트 목적 및 구조 설명 |
| Coding Conventions | 5 | 코딩 스타일 가이드 |
| Team Context | 5 | 팀원 역할, 작업 분담 |
| Known Issues | 5 | 알려진 이슈 및 해결 방법 |

**체크리스트:**
- [ ] 프로젝트 목적 명시
- [ ] 폴더 구조 설명
- [ ] 사용 기술 스택 명시
- [ ] 팀원별 담당 영역 기록

**기대 CLAUDE.md 구조:**
```markdown
# Project: [Name]

## Team Members
- Alice: Backend API
- Bob: Frontend UI

## Tech Stack
- Discord.js / Next.js
- TypeScript

## Conventions
- Commit: Conventional Commits
- Branch: feature/*, fix/*

## Known Issues
- API rate limit 주의
```

### Automation - Headless Mode (20 points)
| Item | Points | Criteria |
|------|--------|----------|
| Headless Usage | 10 | -p 플래그로 자동화 실행 |
| Script Integration | 5 | package.json scripts 활용 |
| Workflow | 5 | 반복 작업 자동화 |

**체크리스트:**
- [ ] claude -p "task" 사용 흔적
- [ ] 자동화된 개발 스크립트
- [ ] README에 자동화 방법 설명

**기대 패턴:**
```json
// package.json
{
  "scripts": {
    "dev": "next dev",
    "claude:feature": "claude -p 'Implement feature X'",
    "claude:test": "claude -p 'Write tests for Y'"
  }
}
```

### Teamwork - Git (20 points)
| Item | Points | Criteria |
|------|--------|----------|
| Branch Strategy | 8 | feature/fix 브랜치 사용 |
| Commit Quality | 6 | 의미 있는 커밋 메시지 |
| PR/Merge | 6 | PR 또는 merge 기록 |

**체크리스트:**
- [ ] main/develop 외 feature 브랜치 존재
- [ ] Conventional Commits 형식 (feat:, fix:, docs:)
- [ ] 여러 커밋 기록 (팀 협업 증거)

---

## Time Bonus (10 points)
| Completion | Bonus |
|------------|-------|
| ≤70% time (42min) | +10 |
| ≤85% time (51min) | +5 |
| On time | 0 |
| Late | -5/5min |

---

## Passing Criteria
- **Minimum Pass**: MVP 실행 + 2개 이상 기능 (45+ points)
- **Excellence**: 모든 기능 + CLAUDE.md 충실 + Git 활용 (75+ points)

---

## Evaluation Notes
1. **팀 프로젝트** - 개인 작업도 허용하지만 Git 기록 필요
2. **CLAUDE.md 중요** - 팀 컨텍스트 공유 여부 확인
3. **Headless Mode** - -p 플래그 사용 여부 확인
4. **데모 영상** - 제출 시 데모 영상 포함 권장

---

## Output JSON Format
```json
{
  "rubric_score": 72,
  "project_type": "discord_bot",
  "validation": {
    "npm_install": "pass",
    "npm_start": "pass",
    "claude_md_exists": true
  },
  "breakdown": {
    "mvp_functionality": 25,
    "claude_md_quality": 17,
    "automation_headless": 15,
    "teamwork_git": 15
  },
  "feedback": "Discord 봇 3개 기능 구현. CLAUDE.md 충실. Headless 모드 활용.",
  "strengths": [
    "모든 핵심 기능 동작",
    "팀 컨텍스트 잘 문서화",
    "Conventional Commits 사용"
  ],
  "improvements": [
    "PR 워크플로우 추가 권장",
    "에러 핸들링 보강"
  ]
}
```

## Failure Example
```json
{
  "rubric_score": 28,
  "project_type": "web_dashboard",
  "validation": {
    "npm_install": "pass",
    "npm_start": "fail",
    "error": "Error: Cannot find module 'react'",
    "claude_md_exists": false
  },
  "breakdown": {
    "mvp_functionality": 12,
    "claude_md_quality": 0,
    "automation_headless": 6,
    "teamwork_git": 10
  },
  "feedback": "빌드 실패로 기능 테스트 불가. CLAUDE.md 누락.",
  "strengths": ["Git 커밋 기록 존재"],
  "improvements": ["의존성 설치 문제 해결", "CLAUDE.md 작성 필수"]
}
```
