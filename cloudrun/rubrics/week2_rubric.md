# Week 2 Rubric: Hooks Mastery

## Challenge Summary
Claude Code Hook 시스템을 이해하고, 보안/자동화 Hook을 구현하는 챌린지

## Time Limit: 60 minutes

## Scoring: 80 points (rubric) + 20 points (time rank bonus)

---

## Evaluation Process

### Step 1: Hook 실행 검증 (필수)
```bash
cd submissions/week2/{participant_id}

# Stage 1: Security Hook 테스트
echo '{"tool_input":{"file_path":"/path/to/.env"}}' | node hooks/read_hook.js
# 기대: exit code 2 + 에러 메시지

echo '{"tool_input":{"file_path":"/path/to/safe.txt"}}' | node hooks/read_hook.js
# 기대: exit code 0

# Settings 유효성 검사
node -e "JSON.parse(require('fs').readFileSync('.claude/settings.json'))"
```

**Hook 실행 실패 시**: 해당 Stage 점수 0점

### Step 2: 코드 리뷰 (Claude)
실행 검증 후, 아래 Rubric에 따라 코드 검토

---

## Rubric Breakdown (80 points)

### Hook Validation (필수 조건)
| Test | Expected | Impact |
|------|----------|--------|
| `read_hook.js` with .env | exit 2 | Stage 1 점수 영향 |
| `read_hook.js` with safe file | exit 0 | Stage 1 점수 영향 |
| `settings.json` valid | JSON parse OK | Stage 2 점수 영향 |

### Stage 1: Security Hook (25 points)
| Item | Points | Criteria |
|------|--------|----------|
| Implementation | 15 | .env 파일 읽기 차단 로직 구현 |
| Exit Code | 5 | exit(2)로 차단, exit(0)로 허용 |
| Message | 5 | 차단 시 에러 메시지 출력 |

**체크리스트:**
- [ ] read_hook.js의 TODO 완성
- [ ] file_path에서 ".env" 체크
- [ ] process.exit(2)로 차단
- [ ] console.error로 메시지 출력

**기대 코드 패턴:**
```javascript
if (filePath.includes(".env")) {
  console.error("Blocked: Cannot read .env file")
  process.exit(2)
}
process.exit(0)
```

### Stage 2: Query Hook 활성화 (30 points)
| Item | Points | Criteria |
|------|--------|----------|
| Activation | 10 | process.exit(0) 제거 |
| Understanding | 10 | Claude Agent SDK 사용법 이해 |
| Settings | 10 | settings.json에 Hook 등록 확인 |

**체크리스트:**
- [ ] query_hook.js의 early exit (line 9) 제거
- [ ] SDK의 query() 함수 이해
- [ ] Hook이 Write/Edit 시 동작
- [ ] abortController 추가 (선택)

### Stage 3: Custom Hook (15 points)
| Item | Points | Criteria |
|------|--------|----------|
| Creation | 8 | 새 Hook 파일 생성 |
| Functionality | 7 | PostToolUse에서 동작 |

**체크리스트:**
- [ ] 새 Hook 파일 존재 (예: log_hook.js, format_hook.js)
- [ ] settings.json의 PostToolUse에 등록
- [ ] 의미 있는 기능 구현 (로깅, 포맷팅, 통계 등)

### CLAUDE.md Quality (10 points)
| Item | Points | Criteria |
|------|--------|----------|
| Hook 원리 | 4 | PreToolUse vs PostToolUse 설명 |
| Exit Code | 3 | 0=allow, 2=block 설명 |
| Debug Tips | 3 | jq, echo 등 디버깅 팁 |

**기대 내용:**
- Hook이 stdin으로 JSON을 받는 구조 설명
- exit code 의미 (0 = allow, 2 = block)
- 디버깅 방법: `jq .`, `echo`, log 파일 활용

---

## Time Rank Bonus (20 points)
| Rank | Bonus |
|------|-------|
| 1st | +20 |
| 2nd | +17 |
| 3rd | +14 |
| 4th | +11 |
| 5th | +8 |
| 6th+ | +5 |

---

## Passing Criteria
- **Minimum Pass**: Stage 1 완료 (25+ points)
- **Excellence**: Stage 1 + Stage 2 + CLAUDE.md (65+ points)

---

## Evaluation Notes
1. **Hook 실행 필수** - read_hook.js를 실제로 실행하여 exit code 검증
2. **설정 파일** - settings.json이 valid JSON인지 확인
3. **참고 파일** - tsc.js를 참고했는지 확인
4. **queries_COMPLETED.zip** - 정답 참고용 (비교 평가에 활용)

---

## Output JSON Format
```json
{
  "rubric_score": 55,
  "hook_tests": {
    "read_hook_block_env": "pass",
    "read_hook_allow_safe": "pass",
    "settings_json_valid": "pass"
  },
  "breakdown": {
    "stage_1_security_hook": 25,
    "stage_2_query_hook": 20,
    "stage_3_custom_hook": 5,
    "claude_md_quality": 5
  },
  "feedback": "Hook 테스트 통과. Security Hook 완벽 구현. Query Hook 활성화됨.",
  "strengths": [
    ".env 차단 테스트 통과 (exit 2)",
    "일반 파일 허용 테스트 통과 (exit 0)",
    "settings.json 유효"
  ],
  "improvements": [
    "Custom Hook 기능 추가 필요",
    "CLAUDE.md에 디버깅 팁 보강"
  ]
}
```

## Hook Test Failure Example
```json
{
  "rubric_score": 10,
  "hook_tests": {
    "read_hook_block_env": "fail",
    "read_hook_allow_safe": "pass",
    "settings_json_valid": "pass",
    "error": "Expected exit code 2, got 0"
  },
  "breakdown": {
    "stage_1_security_hook": 0,
    "stage_2_query_hook": 5,
    "stage_3_custom_hook": 0,
    "claude_md_quality": 5
  },
  "feedback": "Security Hook 테스트 실패. .env 파일 차단 미구현.",
  "strengths": ["settings.json 구조 이해"],
  "improvements": [".env 차단 로직 구현 필수"]
}
```

---

## Reference: queries_COMPLETED.zip 비교 포인트

### read_hook.js 완성본 (lines 12-16):
```javascript
if (readPath.includes(".env")) {
  console.error("You cannot read the .env file");
  process.exit(2);
}
```

### query_hook.js 수정 포인트:
- Line 9: `process.exit(0)` 제거
- abortController 추가 (선택)

### settings.json Hook 구조:
```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Read", "hooks": [...] },
      { "matcher": "Write|Edit", "hooks": [...] }
    ],
    "PostToolUse": [...]
  }
}
```
