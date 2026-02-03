# Week 3 Rubric: Paper Survey with Skills + Subagents + MCP

## Challenge Summary
**Skills**, **Subagents**, **MCP**를 활용하여 10개 이상의 논문을 조사하고 서베이 문서를 작성하는 챌린지.
참가자는 **커스텀 Skill**을 필수로, **MCP 설정**을 선택적으로 repo에 포함하여 제출한다.

## Time Limit: 60 minutes

## Scoring: 90 points (rubric) + 10 points (time bonus)

---

## Evaluation Process

### Step 1: 결과물 확인 (필수)
```bash
cd submissions/week3/{participant_id}

# 필수 파일 존재 확인
ls survey.md
ls references.bib

# 논문 개수 확인
grep -c "@article\|@inproceedings\|@misc" references.bib
# 기대: 10 이상

# Skill 파일 확인 (필수)
ls .claude/skills/*/SKILL.md

# MCP 설정 확인 (선택)
cat .claude/settings.local.json 2>/dev/null || echo "MCP config not found (optional)"
```

**결과물 미존재 시**: 해당 항목 0점

### Step 2: 품질 리뷰 (Claude)
결과물 확인 후, 아래 Rubric에 따라 검토

---

## 제출 필수 파일

| 파일 | 필수 여부 | 설명 |
|------|----------|------|
| `survey.md` | ✅ 필수 | 서베이 문서 본문 |
| `references.bib` | ✅ 필수 | BibTeX 형식 참고문헌 (10개 이상, 실제 논문만) |
| `.claude/skills/*/SKILL.md` | ✅ 필수 | 커스텀 Skill 정의 (서베이 작성용) |
| `.claude/settings.local.json` | ⭕ 선택 | MCP 서버 설정 (보너스 점수) |

**중요**: `references.bib`의 논문은 반드시 **실제 존재하는 논문**이어야 함 (arXiv ID 또는 DOI 포함).
가짜/hallucinated 논문은 해당 논문에 대해 0점 처리.

---

## Rubric Breakdown (90 points)

### A. Tool Setup — Skill + MCP + Subagent (20 points)

Skills, MCP, Subagent를 올바르게 설정하고 활용했는지 평가한다.

#### Skill 작성 (10 points) - 필수
| Item | Points | Criteria |
|------|--------|----------|
| SKILL.md 존재 | 5 | `.claude/skills/*/SKILL.md` 존재, YAML frontmatter 올바름 |
| Quality | 5 | Skill 내용이 서베이 작성 워크플로우에 실질적으로 유용 |

**체크리스트:**
- [ ] `.claude/skills/*/SKILL.md` 파일 존재
- [ ] YAML frontmatter 포함 (`name`, `description` 등)
- [ ] Skill 내용이 구체적이고 재사용 가능한 지시사항 포함

**기대 Skill 패턴:**
```markdown
---
name: survey-writer
description: Write academic survey documents from paper summaries
allowed-tools: [Read, Write, Grep, Glob, Task]
user-invocable: true
---

# Survey Writer Skill

## 역할
수집한 논문들을 기반으로 체계적인 서베이 문서를 작성한다.

## 작성 원칙
1. 각 논문의 핵심 기여(contribution)를 한 문장으로 정리
2. 논문들을 방법론/접근법 기준으로 분류
3. 분류별로 흐름을 만들어 서술
4. 연구 갭과 향후 방향을 도출
...
```

#### MCP 설정 (5 points) - 선택
| Item | Points | Criteria |
|------|--------|----------|
| Configuration | 3 | `.claude/settings.local.json`에 MCP 설정 존재 |
| Functionality | 2 | 논문 검색에 실제 활용한 흔적 |

**MCP 미사용 시**: 이 항목 0점 (다른 방법으로 논문 수집 시 감점 없음)

**기대 설정 패턴 (arXiv):**
```json
{
  "mcpServers": {
    "arxiv": {
      "command": "uv",
      "args": ["tool", "run", "arxiv-mcp-server", "--storage-path", "./papers"]
    }
  }
}
```

**기대 설정 패턴 (Semantic Scholar):**
```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "npx",
      "args": ["-y", "@smithery-ai/semantic-scholar"]
    }
  }
}
```

#### Subagent 활용 (5 points) - 권장
| Item | Points | Criteria |
|------|--------|----------|
| Usage Evidence | 3 | Task tool을 사용하여 병렬 분석한 흔적 |
| Effectiveness | 2 | 효율적인 병렬 처리 전략 |

**평가 기준:**
- [ ] 여러 논문을 동시에 분석하기 위해 Subagent 활용
- [ ] CLAUDE.md나 대화 기록에서 Task tool 사용 흔적
- [ ] 병렬 처리로 시간 효율성 향상

---

### B. 구조적 완성도 — Writing Structure (35 points)

서베이 문서가 **학술 글쓰기 원칙**에 따라 구조적으로 잘 짜여져 있는지 평가한다.

| Item | Points | Criteria |
|------|--------|----------|
| 서론 (Introduction) | 10 | 연구 주제 정의, 서베이 목적, 범위(scope) 명시 |
| 논리적 구성 (Organization) | 10 | 섹션 간 논리적 흐름, 분류 체계의 일관성 |
| 참고문헌 (References) | 10 | BibTeX 10개 이상, 본문 인용과 매칭, **실제 논문만** |
| 결론 (Conclusion) | 5 | 종합 요약, 핵심 메시지 명확 |

**체크리스트:**
- [ ] 서론에서 "무엇을, 왜 조사했는지" 명확히 서술
- [ ] 본론이 의미 있는 카테고리로 분류되어 있음 (단순 나열 ✕)
- [ ] 각 섹션이 자연스럽게 연결됨 (전환 문장, 비교/대조)
- [ ] `references.bib`에 10개 이상 엔트리, 본문에서 인용
- [ ] **모든 논문이 실제 존재함** (arXiv ID 또는 DOI 확인 가능)
- [ ] 결론에서 전체 서베이를 관통하는 핵심 메시지 제시

**감점 기준:**
- 논문을 번호순으로 단순 나열만 한 경우: Organization 최대 3점
- 인용 없이 논문 제목만 언급: References 최대 5점
- 서론/결론 없이 바로 논문 요약으로 시작: Introduction/Conclusion 0점
- **가짜 논문 인용**: 해당 논문 관련 점수 0점

---

### C. 연구 인사이트 — Research Insight (35 points)

서베이가 **충분히 새롭고 insight 위주로 올바르게 정리**되어 있는지 평가한다.
단순 요약이 아닌, 비판적 분석과 종합적 시각이 핵심.

| Item | Points | Criteria |
|------|--------|----------|
| 논문 요약 품질 (Summaries) | 12 | 각 논문의 핵심 기여·방법론·결과를 정확하게 요약 |
| 비판적 분석 (Critical Analysis) | 12 | 논문 간 비교, 장단점·한계점 분석, 연구 갭 식별 |
| 종합 인사이트 (Synthesis) | 11 | 분야 트렌드 도출, 향후 연구 방향 제시, 독자적 관점 |

**체크리스트:**
- [ ] 각 논문에 대해: 목표, 방법론, 핵심 결과가 구분되어 서술
- [ ] 표면적 요약을 넘어 "이 논문이 왜 중요한지" 설명
- [ ] 논문 간 공통점/차이점을 명시적으로 비교
- [ ] 현재 연구의 한계점이나 미해결 문제(research gap) 지적
- [ ] 향후 연구 방향을 구체적으로 제시 (막연한 "향후 연구 필요" ✕)
- [ ] 단순 복사·붙여넣기가 아닌 자신의 언어로 재서술

**감점 기준:**
- 논문 abstract를 그대로 복사: 해당 논문 요약 0점
- 모든 논문을 동일 패턴으로 나열 (기계적 요약): Summaries 최대 5점
- 비교·분석 없이 개별 요약만 존재: Critical Analysis 0점
- 트렌드/방향 제시 없음: Synthesis 최대 3점

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
- **Minimum Pass**: 10+ 실제 논문 + survey.md + 기본 구조 존재 (45+ points)
- **Excellence**: Skill + Subagent 활용 + 구조적 글쓰기 + 비판적 인사이트 (75+ points)

---

## Evaluation Notes
1. **Skill 평가** — `.claude/skills/*/SKILL.md`가 커밋되어 있어야 함. 파일이 없으면 해당 항목 0점.
2. **MCP 평가** — 선택사항. 설정 파일 없어도 다른 방법으로 논문 수집 시 감점 없음.
3. **Subagent 평가** — Task tool 활용 흔적 확인. CLAUDE.md나 conversation에서 확인.
4. **논문 검증** — 가짜 논문은 0점 처리. arXiv ID 또는 DOI로 확인 가능해야 함.
5. **구조 vs 인사이트** — 구조가 잘 잡혀 있어도 인사이트가 없으면 B만 만점, C는 저점.

---

## Output JSON Format
```json
{
  "rubric_score": 72,
  "deliverables": {
    "survey_md": true,
    "references_bib": true,
    "paper_count": 12,
    "real_paper_count": 12,
    "skill_md": true,
    "mcp_settings": true,
    "subagent_used": true
  },
  "breakdown": {
    "tool_setup_skill": 9,
    "tool_setup_mcp": 4,
    "tool_setup_subagent": 4,
    "structure_introduction": 8,
    "structure_organization": 9,
    "structure_references": 10,
    "structure_conclusion": 4,
    "insight_summaries": 10,
    "insight_critical_analysis": 8,
    "insight_synthesis": 6
  },
  "feedback": "Skill과 Subagent를 잘 활용함. 구조적으로 탄탄하나 비판적 분석이 일부 부족.",
  "strengths": [
    "survey-writer Skill이 잘 구성됨",
    "Task tool로 병렬 논문 분석 수행",
    "논문 분류 체계가 명확하고 일관됨",
    "12편 실제 논문 수집, BibTeX 형식 정확"
  ],
  "improvements": [
    "논문 간 방법론 비교가 더 구체적이면 좋겠음",
    "research gap 지적이 막연함 — 구체적 미해결 문제 제시 필요"
  ]
}
```

## Failure Example
```json
{
  "rubric_score": 25,
  "deliverables": {
    "survey_md": true,
    "references_bib": true,
    "paper_count": 8,
    "real_paper_count": 6,
    "skill_md": false,
    "mcp_settings": false,
    "subagent_used": false
  },
  "breakdown": {
    "tool_setup_skill": 0,
    "tool_setup_mcp": 0,
    "tool_setup_subagent": 0,
    "structure_introduction": 3,
    "structure_organization": 3,
    "structure_references": 5,
    "structure_conclusion": 0,
    "insight_summaries": 5,
    "insight_critical_analysis": 3,
    "insight_synthesis": 6
  },
  "feedback": "Skill 미작성, 가짜 논문 2편 포함. 논문을 번호순 나열, 비교 분석 없음.",
  "strengths": ["survey.md 작성 시도", "일부 논문 요약 존재"],
  "improvements": [
    "10개 이상 실제 논문 수집 필수",
    ".claude/skills/에 커스텀 Skill 작성 필요",
    "Task tool로 병렬 분석 시도 권장",
    "논문 간 비교·분석 추가 필요",
    "가짜 논문 제거 — arXiv ID/DOI 확인 필수"
  ]
}
```
