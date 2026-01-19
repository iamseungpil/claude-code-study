# Week 4 Rubric: MVP Development with MCP

## Challenge Summary
2개 이상의 MCP 서버를 활용하여 Discord 봇 또는 웹 대시보드 MVP를 개발하고, Skills를 사용하여 개발 워크플로우를 자동화하는 챌린지

## Time Limit: 60 minutes

## Scoring: 90 points (rubric) + 10 points (time bonus)

---

## Evaluation Process

### Step 1: MVP 실행 검증 (필수)
```bash
cd submissions/week4/{participant_id}

# MCP 설정 확인
cat .claude/settings.local.json
# 기대: 2+ MCP 서버 설정

# Option A: Discord Bot
npm install --cache /tmp/npm-cache
node bot.js  # 또는 npm start
# 기대: 봇이 온라인 상태로 전환

# Option B: Web Dashboard
npm install --cache /tmp/npm-cache
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

### Deliverables Check (필수 조건)
| Item | Expected | Impact |
|------|----------|--------|
| MCP Settings | 2+ 서버 | MCP Integration 점수 영향 |
| npm install | 성공 | 실패 시 평가 중단 |
| npm start/dev | 성공 | 실패 시 50% 감점 |
| CLAUDE.md | 존재 | Code Quality 점수 영향 |

### MCP Integration (25 points)
| Item | Points | Criteria |
|------|--------|----------|
| Server Count | 10 | 2개 이상 MCP 서버 설정 |
| Integration | 10 | MCP 도구 실제 사용 |
| Configuration | 5 | 올바른 설정 구조 |

**체크리스트:**
- [ ] .claude/settings.local.json에 2+ MCP 서버
- [ ] MCP 도구 활용한 기능 구현
- [ ] API 키 환경 변수 처리

**기대 설정 패턴:**
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "notion": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-notion"],
      "env": { "NOTION_API_KEY": "${NOTION_API_KEY}" }
    }
  }
}
```

### MVP Functionality (30 points)
| Item | Points | Criteria |
|------|--------|----------|
| Core Feature 1 | 10 | 첫 번째 핵심 기능 동작 |
| Core Feature 2 | 10 | 두 번째 핵심 기능 동작 |
| Core Feature 3 | 10 | 세 번째 핵심 기능 동작 |

**Discord Bot 체크리스트:**
- [ ] 봇이 온라인 상태로 연결
- [ ] 최소 3개 명령어 동작 (예: /search, /summary, /notify)
- [ ] MCP 연동 기능 포함

**Web Dashboard 체크리스트:**
- [ ] 메인 페이지 렌더링
- [ ] MCP로 데이터 가져오기
- [ ] 인터랙션 기능 (버튼, 폼 등)

### Skills Usage (15 points)
| Item | Points | Criteria |
|------|--------|----------|
| Built-in Skills | 8 | /commit, /review-pr 등 사용 |
| Custom Skills | 4 | SKILL.md 작성 (선택) |
| Workflow | 3 | Skills 기반 개발 워크플로우 |

**체크리스트:**
- [ ] /commit 또는 /review-pr 사용 흔적
- [ ] Git 커밋 메시지 품질 (Conventional Commits)
- [ ] 선택: 커스텀 SKILL.md 존재

### Code Quality (20 points)
| Item | Points | Criteria |
|------|--------|----------|
| Error Handling | 8 | 적절한 에러 처리 |
| Documentation | 7 | CLAUDE.md, README |
| Code Structure | 5 | 깔끔한 코드 구조 |

**체크리스트:**
- [ ] try-catch 에러 핸들링
- [ ] CLAUDE.md 프로젝트 컨텍스트
- [ ] README.md 설치/실행 가이드
- [ ] 코드 모듈화

**기대 CLAUDE.md 구조:**
```markdown
# Project: [Name]

## Purpose
Brief description of the MVP

## Tech Stack
- Discord.js / Next.js
- MCP: Notion, GitHub

## How to Run
1. Set environment variables
2. npm install
3. npm start

## MCP Configuration
- Notion: For project tracking
- GitHub: For issue management
```

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
- **Minimum Pass**: 2 MCP + MVP 실행 + 2개 이상 기능 (45+ points)
- **Excellence**: 2+ MCP 활용 + 3개 기능 + Skills 사용 (75+ points)

---

## Evaluation Notes
1. **MCP 중요** - 2개 이상 MCP 서버 활용 필수
2. **Skills 사용** - /commit, /review-pr 등 활용 확인
3. **MVP 집중** - 복잡한 기능보다 핵심 기능 동작 중시
4. **문서화** - CLAUDE.md, README 존재 여부 확인

---

## Output JSON Format
```json
{
  "rubric_score": 75,
  "project_type": "discord_bot",
  "validation": {
    "npm_install": "pass",
    "npm_start": "pass",
    "mcp_servers": 2,
    "claude_md_exists": true
  },
  "breakdown": {
    "mcp_integration": 22,
    "mvp_functionality": 27,
    "skills_usage": 12,
    "code_quality": 14
  },
  "feedback": "2개 MCP 서버 활용. 3개 핵심 기능 동작. Skills 활용 확인.",
  "strengths": [
    "GitHub + Notion MCP 연동",
    "모든 핵심 기능 동작",
    "/commit 사용으로 일관된 커밋"
  ],
  "improvements": [
    "에러 핸들링 강화 필요",
    "CLAUDE.md 내용 보강"
  ]
}
```

## Failure Example
```json
{
  "rubric_score": 32,
  "project_type": "web_dashboard",
  "validation": {
    "npm_install": "pass",
    "npm_start": "fail",
    "error": "Error: Cannot find module 'react'",
    "mcp_servers": 1,
    "claude_md_exists": false
  },
  "breakdown": {
    "mcp_integration": 8,
    "mvp_functionality": 12,
    "skills_usage": 5,
    "code_quality": 7
  },
  "feedback": "MCP 1개만 사용, 빌드 실패. CLAUDE.md 누락.",
  "strengths": ["MCP 설정 시도"],
  "improvements": ["2개 이상 MCP 필수", "빌드 오류 수정", "CLAUDE.md 작성"]
}
```
