# Week 3 Rubric: Paper Survey with MCP

## Challenge Summary
MCP (Model Context Protocol)와 Semantic Scholar API를 활용하여 10개 이상의 논문을 조사하고 서베이 문서를 작성하는 챌린지

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
```

**결과물 미존재 시**: 해당 항목 0점

### Step 2: 품질 리뷰 (Claude)
결과물 확인 후, 아래 Rubric에 따라 검토

---

## Rubric Breakdown (90 points)

### Deliverables Check (필수 조건)
| Item | Expected | Impact |
|------|----------|--------|
| survey.md | 존재 | Paper Coverage 점수 영향 |
| references.bib | 존재 | Paper Coverage 점수 영향 |
| 논문 수 | 10+ | Paper Coverage 점수 영향 |

### Paper Coverage (30 points)
| Item | Points | Criteria |
|------|--------|----------|
| Quantity | 15 | 10개 이상 논문 수집 |
| Relevance | 10 | 주제와의 관련성 |
| Diversity | 5 | 다양한 연구 방향 포함 |

**체크리스트:**
- [ ] references.bib에 10개 이상 엔트리
- [ ] 모든 논문이 주제와 관련됨
- [ ] 다양한 저자/연도/접근법 포함

### Summary Quality (25 points)
| Item | Points | Criteria |
|------|--------|----------|
| Accuracy | 10 | 논문 내용의 정확한 요약 |
| Key Points | 10 | 핵심 기여/방법론 추출 |
| Consistency | 5 | 일관된 요약 형식 |

**체크리스트:**
- [ ] 각 논문별 요약 존재
- [ ] 연구 목표, 방법론, 결과 포함
- [ ] 객관적이고 정확한 서술

### MCP Usage (20 points)
| Item | Points | Criteria |
|------|--------|----------|
| Configuration | 8 | Semantic Scholar MCP 설정 |
| Automation | 7 | 자동화된 검색 워크플로우 |
| Efficiency | 5 | 효율적인 API 활용 |

**체크리스트:**
- [ ] .claude/settings.json 또는 settings.local.json에 MCP 설정
- [ ] Semantic Scholar API 연동 확인
- [ ] 수동이 아닌 자동화된 검색 흔적

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

### Survey Structure (15 points)
| Item | Points | Criteria |
|------|--------|----------|
| Introduction | 5 | 연구 주제 소개 및 배경 |
| Classification | 5 | 논문 분류 체계 |
| Conclusion | 5 | 인사이트 및 결론 |

**체크리스트:**
- [ ] 서베이 주제 명확히 정의
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
- **Excellence**: 모든 항목 충실 + MCP 활용 (75+ points)

---

## Evaluation Notes
1. **논문 품질** - 관련성 낮은 논문은 카운트하지 않음
2. **MCP 사용** - settings 파일에 MCP 설정 여부 확인
3. **표절 주의** - 요약이 원문 복사가 아닌지 확인
4. **BibTeX 형식** - references.bib 파싱 가능 여부

---

## Output JSON Format
```json
{
  "rubric_score": 70,
  "deliverables": {
    "survey_md": true,
    "references_bib": true,
    "paper_count": 12
  },
  "breakdown": {
    "paper_coverage": 25,
    "summary_quality": 20,
    "mcp_usage": 15,
    "survey_structure": 10
  },
  "feedback": "12개 논문 수집, 요약 품질 양호, MCP 설정 확인됨.",
  "strengths": [
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
  "rubric_score": 25,
  "deliverables": {
    "survey_md": true,
    "references_bib": false,
    "paper_count": 5
  },
  "breakdown": {
    "paper_coverage": 10,
    "summary_quality": 10,
    "mcp_usage": 0,
    "survey_structure": 5
  },
  "feedback": "논문 수 부족, BibTeX 파일 누락, MCP 미사용.",
  "strengths": ["survey.md 작성 시도"],
  "improvements": ["10개 이상 논문 수집 필수", "MCP 설정 필요"]
}
```
