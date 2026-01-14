# Week 1 Rubric: Project Folder Cleanup (v2 - Enhanced)

## Challenge Summary
Given a messy ML project folder (41 files), analyze its purpose, identify important files, remove clutter, and write clear documentation.

## Project: messy-ml-project-v2
- **파일 수**: 41개
- **실제 필요 파일**: ~15개
- **삭제 대상 파일**: ~20개
- **검토 필요 파일**: ~6개

## Time Limit
**60분** (기존 45분에서 연장 - 복잡도 증가)

## Required Outputs
1. `analysis.md` - 프로젝트 분석 보고서
2. `README.md` - 업데이트된 프로젝트 문서
3. `cleanup_plan.md` - 정리 계획 (선택사항, 가산점)

---

## Scoring Criteria (90 points + 10 time bonus)

### 1. Project Analysis (40 points)

| Points | Criteria |
|--------|----------|
| 10 | **Purpose Identification**: 프로젝트 목적 정확히 파악 (MNIST CNN 학습) |
| 10 | **Core Files**: 핵심 파일 식별 (train_final_REAL.py, model.py, utils.py) |
| 10 | **Dependency Analysis**: 파일 간 의존성 정확히 추적 |
| 10 | **Code Smell Detection**: 중복 코드, 사용되지 않는 코드 식별 |

**평가 질문:**
- `train*.py` 중 어떤 파일이 실제 사용되는지 파악했는가?
- `helper.py`, `common.py`, `utils.py`의 중복 코드를 발견했는가?
- 설정 파일들의 관계를 이해했는가?
- Magic numbers의 의미를 파악했는가?

### 2. Cleanup Quality (30 points)

| Points | Criteria |
|--------|----------|
| 10 | **Accuracy**: 중요 파일을 삭제 대상으로 표시하지 않음 |
| 10 | **Completeness**: 불필요한 파일 대부분 발견 |
| 10 | **Reasoning**: 각 삭제에 명확한 근거 제시 |

**삭제 대상 파일 (정답):**
```
반드시 삭제:
- src/train.py (불완전, 미사용)
- src/train_v2.py (중간 버전, 미사용)
- src/train_v2_backup.py (백업, 불필요)
- src/train_final.py (가짜 최종)
- src/train_final_v2.py (가짜 최종 v2)
- src/train_old_dont_use.py (deprecated)
- src/model_backup.py (model.py와 동일)
- src/model_old.py (deprecated)
- src/utils_old.py (deprecated)
- data/loader_backup.py (loader.py와 동일)
- config/config_old.json (구버전)
- requirements_old.txt (구버전)
- tests/test_model_old.py (구버전)
- logs/*.log (오래된 로그)

병합 필요:
- src/helper.py + src/common.py → utils.py에 통합

검토 필요:
- src/model_v2.py (BatchNorm 버전 - 유지 또는 병합)
- src/utils_v2.py (개선 버전 - 유지 또는 병합)
- data/preprocess_new.py (WIP - 유지 또는 삭제)
```

**Red Flags (감점):**
- 실제 사용 파일 삭제 표시 (-5점/개)
  - train_final_REAL.py
  - model.py
  - utils.py
  - config/config.json
  - requirements.txt
- 명백한 중복 파일 누락 (-3점/개)
- 삭제 근거 없음 (-5점)

### 3. README Quality (20 points)

| Points | Criteria |
|--------|----------|
| 5 | **Project Description**: 명확한 1-2문장 요약 |
| 5 | **Usage Instructions**: 프로젝트 실행 방법 |
| 5 | **File Structure**: 정리 후 파일 구조 설명 |
| 5 | **Requirements**: 의존성 목록 |

**좋은 README 포함 요소:**
- 프로젝트 목적 설명
- 설치 및 실행 방법
- 주요 파일 설명
- 설정 방법 안내

---

## Code Smells 체크리스트 (참고용)

본 프로젝트에 포함된 코드 스멜:

### Bloaters
- [ ] Long Method: `train_final_REAL.py`의 `main()` 함수
- [ ] Long Parameter List: `load_data()`, `train_one_epoch()` 함수

### Dispensables
- [ ] Dead Code: `unused_method_1()`, `unused_method_2()`, 여러 deprecated 함수
- [ ] Duplicate Code: `helper.py`와 `common.py`, `utils.py`의 중복 함수
- [ ] Comments: TODO, FIXME, HACK 주석이 코드 대신 사용됨

### Other Issues
- [ ] Magic Numbers: `0.1307`, `0.3081`, `42` 등 하드코딩된 값
- [ ] Poor Naming: `do_something()`, `helper_function_1()` 등
- [ ] Global Variables: `global_model`, `global_optimizer` 등

---

## Output JSON Format

```json
{
  "rubric_score": 78,
  "breakdown": {
    "analysis": 35,
    "cleanup": 25,
    "readme": 18
  },
  "code_smells_found": [
    "duplicate_code",
    "dead_code",
    "magic_numbers"
  ],
  "files_correctly_identified": {
    "keep": ["train_final_REAL.py", "model.py", "utils.py"],
    "delete": ["train.py", "train_v2.py", "model_backup.py"],
    "merge": ["helper.py", "common.py"]
  },
  "feedback": "코드 스멜을 잘 식별했습니다. helper.py와 common.py의 중복을 발견한 것이 좋습니다.",
  "strengths": [
    "핵심 파일 정확히 식별",
    "의존성 분석 우수",
    "Magic numbers 식별"
  ],
  "improvements": [
    "README에 설정 방법 추가 필요",
    "config 파일 정리 방안 제시 필요"
  ]
}
```

---

## Evaluation Checklist

### Analysis (각 항목 확인)
- [ ] 프로젝트 목적 명확히 기술
- [ ] 메인 진입점(train_final_REAL.py) 식별
- [ ] 파일 의존성 맵핑
- [ ] 중복 파일 식별
- [ ] 임시/캐시 파일 식별
- [ ] 코드 스멜 2개 이상 식별

### Cleanup (각 항목 확인)
- [ ] False positive 없음 (중요 파일 삭제 표시 없음)
- [ ] 중복 파일 플래그됨
- [ ] 구버전 파일 플래그됨
- [ ] 로그 파일 처리 언급
- [ ] 각 삭제에 이유 있음
- [ ] 병합 가능 파일 식별 (helper.py/common.py)

### README (각 항목 확인)
- [ ] 프로젝트 설명 존재
- [ ] 사용/실행 방법 존재
- [ ] 파일 구조 문서화
- [ ] 의존성 목록

---

## 참고: 정리 후 이상적인 구조

```
messy-ml-project-v2/ (정리 후)
├── src/
│   ├── train.py           # train_final_REAL.py에서 리네임
│   ├── model.py           # 유지
│   ├── model_v2.py        # 선택적 유지 (BatchNorm 버전)
│   └── utils.py           # helper.py, common.py 통합
├── data/
│   ├── preprocess.py      # 유지
│   └── loader.py          # 유지
├── config/
│   ├── config.py          # 또는 config.json으로 통일
│   ├── config.json
│   ├── config_prod.json   # 유지
│   └── config_dev.json    # 유지
├── tests/
│   └── test_model.py      # 업데이트 필요
├── scripts/
│   ├── run_training.sh    # 업데이트 필요
│   └── evaluate.py
├── outputs/
├── requirements.txt
├── requirements_dev.txt
├── README.md              # 업데이트됨
└── setup.py
```

---

## 난이도 대비 기대 점수

| 레벨 | 예상 시간 | 점수 범위 |
|------|----------|----------|
| 초급 | 55-60분 | 50-65점 |
| 중급 | 40-50분 | 65-80점 |
| 고급 | 30-40분 | 80-95점 |
