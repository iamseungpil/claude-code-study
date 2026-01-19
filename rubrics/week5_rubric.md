# Week 5 Rubric: Final Project + Live Demo

## Challenge Summary
5주간 배운 모든 기능을 통합한 최종 프로젝트를 완성하고, 라이브 데모를 진행하는 챌린지

## Time Limit: 75 minutes

## Scoring: 90 points (rubric) + 10 points (time bonus)

---

## Evaluation Process

### Step 1: 프로젝트 실행 검증 (필수)
```bash
cd submissions/week5/{participant_id}

# 프로젝트 빌드 및 실행
npm install
npm run build
npm start  # 또는 npm run dev

# CI/CD 설정 확인
cat .github/workflows/*.yml

# 배포 상태 확인
# (Vercel, Railway, Render 등 배포 URL 확인)
```

**빌드 실패 시**: Project Completeness 점수 50% 감점

### Step 2: 데모 및 코드 리뷰 (Claude)
실행 검증 후, 아래 Rubric에 따라 검토

---

## Rubric Breakdown (90 points)

### Build Validation (필수 조건)
| Test | Expected | Impact |
|------|----------|--------|
| npm install | 성공 | 실패 시 평가 중단 |
| npm run build | 성공 | 실패 시 50% 감점 |
| Deployment | 확인 | Feature Integration 영향 |

### Project Completeness (30 points)
| Item | Points | Criteria |
|------|--------|----------|
| Core Features | 15 | 핵심 기능 완성도 |
| Deployment | 10 | 실제 배포 상태 |
| Documentation | 5 | README, 사용 가이드 |

**체크리스트:**
- [ ] 모든 핵심 기능 동작
- [ ] 배포 URL 접속 가능 (또는 로컬 실행)
- [ ] README.md에 설치/실행 방법

**배포 플랫폼 예시:**
- Vercel, Netlify (Web)
- Railway, Render, Fly.io (Server)
- npm/PyPI (CLI tool)
- 24/7 hosting (Discord bot)

### Feature Integration (25 points)
| Item | Points | Criteria |
|------|--------|----------|
| Week 1-2 Features | 8 | CLAUDE.md, Plan Mode, @refs, Hooks |
| Week 3-4 Features | 9 | Skills, SKILL.md, MCP servers (Notion, Playwright, GitHub) |
| Week 5 Features | 8 | CI/CD, GitHub Actions, Headless, Deploy |

**통합 기능 체크리스트:**
- [ ] CLAUDE.md 존재 및 활용
- [ ] Hooks 설정 (.claude/settings.local.json)
- [ ] Skills 활용 (/commit, /review-pr 등)
- [ ] MCP 서버 연동 (Semantic Scholar, Notion, Playwright, GitHub 등)
- [ ] CI/CD 파이프라인 (.github/workflows/)
- [ ] Headless 모드 스크립트

**기대 기능 배치:**
| Week | Features |
|------|----------|
| 1 | CLAUDE.md, Plan Mode, @refs, Context management |
| 2 | Hooks (PreToolUse, PostToolUse, exit codes) |
| 3 | Skills (SKILL.md, /commit), MCP Basics (Semantic Scholar) |
| 4 | MCP Advanced (Notion, Playwright, GitHub), Multi-MCP |
| 5 | CI/CD, GitHub Actions, Headless, Deploy |

### CI/CD & Automation (15 points)
| Item | Points | Criteria |
|------|--------|----------|
| Pipeline Setup | 8 | GitHub Actions 워크플로우 |
| Test Automation | 4 | 자동 테스트 실행 |
| Deploy Automation | 3 | 자동 배포 설정 |

**체크리스트:**
- [ ] .github/workflows/*.yml 존재
- [ ] push/PR 트리거 설정
- [ ] npm test 또는 빌드 검증 단계
- [ ] 배포 자동화 (선택)

**기대 워크플로우:**
```yaml
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npm run build
      - run: npm test
```

### Demo Quality (20 points)
| Item | Points | Criteria |
|------|--------|----------|
| Presentation | 8 | 명확한 설명, 구조화된 진행 |
| Live Demo | 8 | 실제 동작 시연 |
| Q&A Handling | 4 | 질문 대응 능력 |

**데모 체크리스트:**
- [ ] 프로젝트 개요 설명 (2분)
- [ ] 아키텍처/기능 설명 (3분)
- [ ] 라이브 기능 시연 (5분)
- [ ] 회고 및 배운 점 (2분)

**데모 팁:**
- /clear로 세션 초기화
- 백업 녹화 준비
- Claude Code 기능 활용 시연

---

## Time Bonus (10 points)
| Completion | Bonus |
|------------|-------|
| ≤70% time (52min) | +10 |
| ≤85% time (64min) | +5 |
| On time | 0 |
| Late | -5/5min |

---

## Season Champion Criteria
- **총 5주 합산 점수** 기준 순위 결정
- **Champion**: 1위
- **Runner-up**: 2-3위
- 동점 시 최종 주차 점수로 결정

---

## Passing Criteria
- **Minimum Pass**: 배포 완료 + 데모 진행 (50+ points)
- **Excellence**: 모든 기능 통합 + CI/CD + 우수한 데모 (80+ points)

---

## Evaluation Notes
1. **5주 통합** - 이전 주차 기능들이 실제로 사용되었는지 확인
2. **배포 상태** - 실제 접속 가능한 URL 필요
3. **데모 품질** - 라이브 시연 능력 평가
4. **회고** - 배운 점과 개선점 정리 여부

---

## Output JSON Format
```json
{
  "rubric_score": 78,
  "project_info": {
    "name": "Research Notification Bot",
    "type": "discord_bot",
    "deploy_url": "https://discord.gg/xxx"
  },
  "validation": {
    "npm_install": "pass",
    "npm_build": "pass",
    "deployment": "verified"
  },
  "breakdown": {
    "project_completeness": 25,
    "feature_integration": 22,
    "cicd_automation": 13,
    "demo_quality": 18
  },
  "features_used": [
    "CLAUDE.md",
    "Hooks (PreToolUse)",
    "MCP (Semantic Scholar)",
    "GitHub Actions CI",
    "Headless Mode"
  ],
  "feedback": "5주 기능 잘 통합. CI/CD 구성 완료. 데모 명확하고 인상적.",
  "strengths": [
    "MCP + Subagent 조합 활용",
    "GitHub Actions 자동 빌드",
    "라이브 데모 매끄러움"
  ],
  "improvements": [
    "테스트 커버리지 추가",
    "에러 핸들링 문서화"
  ]
}
```

## Failure Example
```json
{
  "rubric_score": 35,
  "project_info": {
    "name": "Web Dashboard",
    "type": "web_app",
    "deploy_url": null
  },
  "validation": {
    "npm_install": "pass",
    "npm_build": "fail",
    "error": "TypeScript compilation error",
    "deployment": "not_found"
  },
  "breakdown": {
    "project_completeness": 10,
    "feature_integration": 10,
    "cicd_automation": 5,
    "demo_quality": 10
  },
  "features_used": [
    "CLAUDE.md"
  ],
  "feedback": "빌드 실패, 배포 미완료. 기능 통합 부족.",
  "strengths": ["프로젝트 시도"],
  "improvements": ["빌드 에러 수정 필수", "배포 완료 필요"]
}
```

---

## Champion Award Calculation

### Season Total Score
```
Total = Week1 + Week2 + Week3 + Week4 + Week5
```

### Awards
| Rank | Title | Prize |
|------|-------|-------|
| 1st | Champion | Certificate + Special Prize |
| 2nd | Runner-up | Certificate |
| 3rd | Runner-up | Certificate |

### Tiebreaker
1. Week 5 점수
2. Week 4 점수
3. Week 3 점수
4. 제출 시간 (빠른 순)
