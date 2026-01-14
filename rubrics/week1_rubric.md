# Week 1 Rubric: ML Project Refactoring Challenge

## Challenge Summary
복잡하게 얽힌 MNIST 학습 프로젝트(`messy-ml-project-v2`)를 분석하고, 사용되는 코드와 불필요한 코드를 분리하여 깔끔하게 리팩토링합니다.

## Challenge Details
- **총 파일 수**: 68개 (Python 51개)
- **실제 사용 파일**: ~10개
- **삭제 가능 파일**: ~25개
- **중복 함수**: ~25개

## Time Limit
60 minutes

## Required Outputs
1. 정리된 프로젝트 폴더 구조
2. 학습 코드가 정상 동작해야 함
3. 테스트 통과
4. README.md

---

## Scoring Criteria

### Total: 100 points = Rubric (80) + Time Rank (20)

---

## Rubric Score (80 points)

### 1. Dead Code Removal (25 points)

| Points | Criteria |
|--------|----------|
| 25 | 사용되지 않는 파일/함수 90% 이상 제거 |
| 20 | 70-89% 제거 |
| 15 | 50-69% 제거 |
| 10 | 30-49% 제거 |
| 5 | 일부만 제거 |
| 0 | 제거 시도 없음 |

**Check:**
- [ ] train_*.py 중복 파일 정리 (16개 → 1-2개)
- [ ] model_*.py 중복 파일 정리 (4개 → 1개)
- [ ] utils_*.py, helper.py, common.py 통합 (5개 → 1개)
- [ ] 사용되지 않는 config 파일 제거
- [ ] 오래된 로그 파일 제거

**삭제 대상 파일들:**
```
src/train.py, train_v2.py, train_v2_backup.py, train_final.py,
train_final_v2.py, train_old_dont_use.py
src/model_v2.py, model_backup.py, model_old.py
src/utils_v2.py, utils_old.py, helper.py, common.py
src/trainer_legacy.py, optimizer_old.py, scheduler_old.py
src/metrics_v2.py, metrics_old.py, visualize_results.py
src/layers_experimental.py, checkpoint_utils.py
src/augment_v2.py, sampler_old.py
data/preprocess_new.py, loader_backup.py
config/config_old.json
tests/test_model_old.py
logs/*.log
```

### 2. Code Organization (20 points)

| Points | Criteria |
|--------|----------|
| 20 | 4개 영역 모두 명확히 분리 (Model, Data, Training, Evaluation) |
| 15 | 3개 영역 분리 |
| 10 | 2개 영역 분리 |
| 5 | 기본적인 파일 분리만 |
| 0 | 구조화 없음 |

**이상적인 구조:**
```
project/
├── src/
│   ├── model.py          # 모델 정의
│   ├── dataset.py        # 데이터 로딩
│   ├── trainer.py        # 학습 로직
│   ├── metrics.py        # 평가 지표
│   └── utils.py          # 유틸리티
├── config/
│   └── config.yaml       # 설정
├── tests/
│   └── test_model.py     # 테스트
├── train.py              # 메인 스크립트
└── requirements.txt
```

### 3. Functionality Preserved (20 points)

| Points | Criteria |
|--------|----------|
| 20 | 학습 코드 정상 동작 + 테스트 통과 |
| 15 | 학습 가능하나 일부 기능 손실 |
| 10 | 코드 실행은 되나 학습 불가 |
| 5 | Import 오류 있음 |
| 0 | 실행 불가 |

**Check:**
- [ ] `python train.py` 실행 가능
- [ ] MNIST 데이터 다운로드/로드 정상
- [ ] 모델 학습 및 검증 동작
- [ ] 체크포인트 저장/로드 동작
- [ ] 테스트 실행 가능

### 4. Duplicate Code Consolidation (10 points)

| Points | Criteria |
|--------|----------|
| 10 | 중복 함수들을 하나로 통합, 재사용성 확보 |
| 7 | 대부분 통합, 일부 중복 남음 |
| 4 | 일부만 통합 |
| 0 | 중복 코드 그대로 |

**중복 함수 예시:**
- `get_device()` - utils.py, helper.py, common.py에 모두 존재
- `count_params()` - 여러 파일에 중복
- `load_data()` - train 파일들에 중복
- `setup_logging()` - 여러 곳에 중복

### 5. Documentation (5 points)

| Points | Criteria |
|--------|----------|
| 5 | README 포함, 실행 방법/구조 설명 완비 |
| 3 | README 있으나 내용 부족 |
| 0 | README 없음 |

**Check:**
- [ ] 프로젝트 설명
- [ ] 설치 방법 (requirements.txt)
- [ ] 실행 방법
- [ ] 프로젝트 구조 설명

---

## Time Rank Score (20 points)

제출 순서에 따른 점수:

| Rank | Points |
|------|--------|
| 1st | 20 |
| 2nd | 17 |
| 3rd | 14 |
| 4th | 11 |
| 5th | 8 |
| 6th+ | 5 |
| Timeout/No submission | 0 |

**Note:** 동점자 처리
- 동일 시간 제출 시 동일 등수
- 다음 등수는 건너뜀 (예: 1등 2명이면 다음은 3등)

---

## Output JSON Format

```json
{
  "participant_id": "user001",
  "week": 1,
  "rubric_score": 72,
  "time_rank": 2,
  "time_rank_score": 17,
  "total_score": 89,
  "breakdown": {
    "dead_code_removal": 20,
    "code_organization": 18,
    "functionality_preserved": 20,
    "duplicate_consolidation": 9,
    "documentation": 5
  },
  "files_before": 68,
  "files_after": 15,
  "files_removed": 53,
  "tests": {
    "total": 5,
    "passed": 5,
    "failed": 0
  },
  "functionality": {
    "training_works": true,
    "data_loading_works": true,
    "checkpoint_works": true
  },
  "feedback": "불필요한 파일 제거 우수. 4개 영역 분리 완료. 학습 정상 동작.",
  "strengths": [
    "68개 → 15개로 파일 정리",
    "중복 함수 통합",
    "깔끔한 폴더 구조"
  ],
  "improvements": [
    "config.yaml로 하드코딩된 값 이동 필요",
    "타입 힌트 추가 권장"
  ]
}
```

---

## Evaluation Checklist

### Dead Code Removal (불필요 코드 제거)
- [ ] train_*.py 중복 파일 제거 (16개 중 15개)
- [ ] model_*.py 중복 파일 제거 (4개 중 3개)
- [ ] utils/helper/common 통합 (5개 → 1개)
- [ ] 오래된 config 파일 제거
- [ ] 로그 파일 제거

### Code Organization (코드 구조화)
- [ ] 명확한 폴더 구조
- [ ] Model 영역 분리
- [ ] Data 영역 분리
- [ ] Training 영역 분리
- [ ] Evaluation 영역 분리

### Functionality (기능 보존)
- [ ] train.py 실행 가능
- [ ] 모델 학습 동작
- [ ] 테스트 통과
- [ ] 체크포인트 동작

### Code Quality (코드 품질)
- [ ] 중복 함수 통합
- [ ] 매직 넘버 제거/상수화
- [ ] 일관된 코딩 스타일

### Documentation (문서화)
- [ ] README.md 존재
- [ ] 실행 방법 설명
- [ ] requirements.txt 정리

---

## Reference Solution
`_solution/` 폴더 참고 (채점 시에만 확인)

### Solution Structure
```
_solution/
├── train.py              # 메인 학습 스크립트
├── config/
│   └── config.yaml       # 설정 파일
├── src/
│   ├── __init__.py
│   ├── model.py          # MNISTClassifier
│   ├── dataset.py        # 데이터 로딩
│   ├── trainer.py        # Trainer 클래스
│   ├── metrics.py        # 평가 지표
│   └── utils.py          # 유틸리티
├── tests/
│   ├── test_model.py
│   └── test_utils.py
└── requirements.txt
```

### Key Improvements in Solution
1. 68개 → 12개 파일로 정리
2. 4개 영역 명확히 분리
3. 중복 함수 완전 제거
4. 타입 힌트 추가
5. 설정 파일로 하드코딩 제거
