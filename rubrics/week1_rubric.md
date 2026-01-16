# Week 1 Rubric: UIGen Feature Sprint

## Challenge Summary
UIGen (AI Component Generator)에 3가지 기능을 구현하는 Memory-Driven Development 챌린지

## Time Limit: 70 minutes

## Scoring: 80 points (rubric) + 20 points (time rank bonus)

---

## Rubric Breakdown (80 points)

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
1. **코드 실행 없음** - 파일 구조와 코드 로직만 검토
2. **패턴 준수** - 기존 코드 스타일 따르는지 확인
3. **Memory-Driven** - CLAUDE.md에 학습 내용 기록 여부 중요

---

## Output JSON Format
\`\`\`json
{
  "rubric_score": 65,
  "breakdown": {
    "stage_1_clear_all": 20,
    "stage_2_download_zip": 25,
    "stage_3_keyboard": 10,
    "claude_md_quality": 10
  },
  "feedback": "Stage 1, 2를 완성했고 Stage 3는 부분 구현. CLAUDE.md 문서화 양호.",
  "strengths": [
    "Dialog 컴포넌트 올바르게 사용",
    "JSZip으로 ZIP 다운로드 구현 완료"
  ],
  "improvements": [
    "Stage 3 Command Palette 완성 필요",
    "에러 핸들링 추가 권장"
  ]
}
\`\`\`
