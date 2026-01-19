# Week 1 Rubric: UIGen Feature Sprint

## Challenge Summary
UIGen (AI Component Generator)에 3가지 기능을 구현하는 Memory-Driven Development 챌린지

## Time Limit: 70 minutes

## Scoring: 80 points (rubric) + 20 points (time rank bonus)

---

## Evaluation Process

### Step 1: 코드 실행 검증 (필수)
```bash
cd submissions/week1/{participant_id}
npm install          # 의존성 설치
npm run build        # TypeScript 컴파일
```

**빌드 실패 시**: 최대 40점까지만 획득 가능 (Stage 점수 50% 감점)

### Step 2: 코드 리뷰 (Claude)
빌드 통과 후, 아래 Rubric에 따라 코드 검토

---

## Rubric Breakdown (80 points)

### Build Validation (필수 조건)
| Item | Result | Impact |
|------|--------|--------|
| `npm install` | Pass/Fail | Fail 시 평가 중단 |
| `npm run build` | Pass/Fail | Fail 시 Stage 점수 50% 감점 |

### Stage 1: Clear All Files (20 points)
| Item | Points | Criteria |
|------|--------|----------|
| Button | 8 | Trash2 아이콘, HeaderActions.tsx에 위치 |
| Dialog | 8 | 확인 다이얼로그, 경고 메시지, Cancel/Delete 버튼 |
| Functionality | 4 | reset() 호출로 파일 초기화 |

**체크리스트:**
- [ ] Trash2 아이콘 버튼 존재
- [ ] Dialog 컴포넌트 사용
- [ ] "Delete All" 클릭 시 reset() 호출
- [ ] "Cancel" 클릭 시 다이얼로그만 닫힘

### Stage 2: Download as ZIP (25 points)
| Item | Points | Criteria |
|------|--------|----------|
| Package | 5 | JSZip 설치 및 import |
| Implementation | 15 | getAllFiles() 사용, ZIP 생성, 다운로드 트리거 |
| Error Handling | 5 | 파일 없을 때 처리, 경로 정규화 |

**체크리스트:**
- [ ] JSZip 패키지 설치됨
- [ ] Download 아이콘 버튼 존재
- [ ] getAllFiles()로 파일 Map 획득
- [ ] ZIP 파일 자동 다운로드
- [ ] 파일명에 타임스탬프 또는 고정명

### Stage 3: Keyboard Shortcuts (20 points) - BONUS
| Item | Points | Criteria |
|------|--------|----------|
| Shortcut | 8 | Cmd/Ctrl+K로 열기, ESC로 닫기 |
| UI | 7 | Command 컴포넌트 사용, 검색 가능 |
| Commands | 5 | Clear All, Download ZIP 명령어 동작 |

**체크리스트:**
- [ ] Cmd/Ctrl+K 단축키 동작
- [ ] Command Palette UI 표시
- [ ] 최소 2개 명령어 (Clear All, Download)
- [ ] ESC로 닫기

### CLAUDE.md Quality (15 points)
| Item | Points | Criteria |
|------|--------|----------|
| Learnings | 10 | 에러 해결 기록, 패턴 발견 문서화 |
| Structure | 5 | 체계적 구성, 재사용 가능한 팁 |

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
- **Minimum Pass**: Stage 1 + Stage 2 완료 (45+ points)
- **Excellence**: 모든 Stage + CLAUDE.md 충실 (70+ points)

---

## Evaluation Notes
1. **코드 실행 필수** - `npm install` → `npm run build` 순서로 실행
2. **빌드 실패 = 감점** - 컴파일 에러 시 Stage 점수 50% 감점
3. **패턴 준수** - 기존 코드 스타일 따르는지 확인
4. **Memory-Driven** - CLAUDE.md에 학습 내용 기록 여부 중요

---

## Output JSON Format
\`\`\`json
{
  "rubric_score": 65,
  "build_status": {
    "npm_install": "pass",
    "npm_build": "pass"
  },
  "breakdown": {
    "stage_1_clear_all": 20,
    "stage_2_download_zip": 25,
    "stage_3_keyboard": 10,
    "claude_md_quality": 10
  },
  "feedback": "빌드 성공. Stage 1, 2 완성. Stage 3 부분 구현. CLAUDE.md 양호.",
  "strengths": [
    "TypeScript 컴파일 에러 없음",
    "Dialog 컴포넌트 올바르게 사용",
    "JSZip으로 ZIP 다운로드 구현 완료"
  ],
  "improvements": [
    "Stage 3 Command Palette 완성 필요",
    "에러 핸들링 추가 권장"
  ]
}
\`\`\`

## Build Failure Example
\`\`\`json
{
  "rubric_score": 32,
  "build_status": {
    "npm_install": "pass",
    "npm_build": "fail",
    "build_error": "Type error: Property 'reset' does not exist on type..."
  },
  "breakdown": {
    "stage_1_clear_all": 10,
    "stage_2_download_zip": 12,
    "stage_3_keyboard": 5,
    "claude_md_quality": 5
  },
  "feedback": "빌드 실패로 50% 감점. 타입 에러 수정 필요.",
  "strengths": ["기능 구현 시도함"],
  "improvements": ["TypeScript 타입 에러 수정 필수"]
}
\`\`\`
