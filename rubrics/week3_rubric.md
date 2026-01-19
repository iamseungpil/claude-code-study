# Week 3 Rubric: Paper Survey with Skills + MCP

## Challenge Summary
Skills와 MCP (Model Context Protocol)를 활용하여 Semantic Scholar API로 10개 이상의 논문을 조사하고 서베이 문서를 작성하는 챌린지

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

# Skill 파일 확인
ls .claude/skills/*/SKILL.md

# MCP 설정 확인
cat .claude/settings.local.json
```

**결과물 미존재 시**: 해당 항목 0점

### Step 2: 품질 리뷰 (Claude)
결과물 확인 후, 아래 Rubric에 따라 검토

---

## Rubric Breakdown (90 points)

### Deliverables Check (필수 조건)
| Item | Expected | Impact |
|------|----------|--------|
| survey.md | 존재 | Survey Document 점수 영향 |
| references.bib | 존재 | Paper Collection 점수 영향 |
| 논문 수 | 10+ | Paper Collection 점수 영향 |
| SKILL.md | 존재 | Skills Usage 점수 영향 |
| MCP settings | 존재 | MCP Setup 점수 영향 |

### MCP Setup (20 points)
| Item | Points | Criteria |
|------|--------|----------|
| Configuration | 10 | Semantic Scholar MCP 설정 |
| Integration | 5 | API 연동 및 검색 수행 |
| Automation | 5 | 자동화된 데이터 수집 |

**체크리스트:**
- [ ] .claude/settings.local.json에 MCP 설정
- [ ] Semantic Scholar 서버 연동 확인
- [ ] 논문 검색 및 상세정보 조회 활용

**기대 설정 패턴:**
```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-semantic-scholar"],
      "env": { "SEMANTIC_SCHOLAR_API_KEY": "${SEMANTIC_SCHOLAR_API_KEY}" }
    }
  }
}
```

### Skills Usage (20 points)
| Item | Points | Criteria |
|------|--------|----------|
| Custom Skill | 10 | SKILL.md 작성 |
| Skill Structure | 5 | 올바른 frontmatter 형식 |
| Integration | 5 | MCP와 연동하여 활용 |

**체크리스트:**
- [ ] .claude/skills/*/SKILL.md 존재
- [ ] user-invocable: true 설정 (또는 description 기반 discovery)
- [ ] 문서 작성 자동화에 활용

**기대 Skill 패턴:**
```markdown
---
name: survey-writer
description: Write academic survey documents from paper summaries
allowed-tools: [Read, Write, WebFetch]
user-invocable: true
---

# Survey Writer Skill
Instructions for Claude on how to write survey documents...
```

### Paper Collection (25 points)
| Item | Points | Criteria |
|------|--------|----------|
| Quantity | 10 | 10개 이상 논문 수집 |
| Relevance | 10 | 주제와의 관련성 |
| Diversity | 5 | 다양한 연구 방향 포함 |

**체크리스트:**
- [ ] references.bib에 10개 이상 엔트리
- [ ] 모든 논문이 주제와 관련됨
- [ ] 다양한 저자/연도/접근법 포함

### Survey Document (25 points)
| Item | Points | Criteria |
|------|--------|----------|
| Introduction | 8 | 연구 주제 소개 및 배경 |
| Summaries | 9 | 논문별 요약 품질 |
| Classification | 4 | 논문 분류 체계 |
| Conclusion | 4 | 인사이트 및 결론 |

**체크리스트:**
- [ ] 서베이 주제 명확히 정의
- [ ] 각 논문별 요약 (목표, 방법론, 결과)
- [ ] 논문들을 카테고리로 분류
- [ ] 종합적인 결론 및 향후 방향

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
- **Minimum Pass**: 10+ 논문 + survey.md 존재 (45+ points)
- **Excellence**: Skills + MCP 활용 + 품질 높은 서베이 (75+ points)

---

## Evaluation Notes
1. **Skills 평가** - SKILL.md 존재 및 구조 확인
2. **MCP 평가** - settings 파일에 MCP 설정 여부 확인
3. **논문 품질** - 관련성 낮은 논문은 카운트하지 않음
4. **표절 주의** - 요약이 원문 복사가 아닌지 확인
5. **BibTeX 형식** - references.bib 파싱 가능 여부

---

## Output JSON Format
```json
{
  "rubric_score": 75,
  "deliverables": {
    "survey_md": true,
    "references_bib": true,
    "paper_count": 12,
    "skill_md": true,
    "mcp_settings": true
  },
  "breakdown": {
    "mcp_setup": 18,
    "skills_usage": 17,
    "paper_collection": 22,
    "survey_document": 18
  },
  "feedback": "Skills와 MCP 잘 활용함. 12개 논문 수집, 요약 품질 양호.",
  "strengths": [
    "Custom survey-writer skill 작성",
    "Semantic Scholar MCP 활용",
    "논문 분류 체계 명확",
    "BibTeX 형식 정확"
  ],
  "improvements": [
    "일부 요약에서 핵심 기여 불명확",
    "결론 섹션 보강 필요"
  ]
}
```

## Failure Example
```json
{
  "rubric_score": 30,
  "deliverables": {
    "survey_md": true,
    "references_bib": true,
    "paper_count": 5,
    "skill_md": false,
    "mcp_settings": false
  },
  "breakdown": {
    "mcp_setup": 0,
    "skills_usage": 0,
    "paper_collection": 15,
    "survey_document": 15
  },
  "feedback": "논문 수 부족, Skills와 MCP 미사용.",
  "strengths": ["survey.md 작성 시도", "기본 구조 존재"],
  "improvements": ["10개 이상 논문 수집 필수", "Skills 작성 필요", "MCP 설정 필요"]
}
```
