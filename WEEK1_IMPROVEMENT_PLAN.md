# 1주차 과제 개선 계획서

## 📊 현재 상태 분석

### 기존 프로젝트 (messy-ml-project) 평가

| 항목 | 현재 상태 | 평가 |
|------|----------|------|
| 파일 수 | 8개 | ⚠️ 너무 적음 |
| 코드 스멜 유형 | 버전 중복, 백업 파일 | ⚠️ 단조로움 |
| 실제 프로젝트 유사성 | 낮음 | ❌ 개선 필요 |
| 교육적 가치 | 중간 | ⚠️ 보완 필요 |
| 난이도 적절성 | 너무 쉬움 | ⚠️ 45분 내 너무 여유 |

### 현재 파일 구조
```
messy-ml-project/
├── train.py           (11줄, 불완전)
├── train_v2.py        (19줄, 중간 버전)
├── train_final.py     (22줄, 가짜 최종)
├── train_final_real.py (102줄, 실제 메인)
├── model.py           (26줄, 현재 사용)
├── model_backup.py    (26줄, 동일 백업)
├── utils.py           (49줄, 현재 사용)
└── utils_old.py       (10줄, deprecated)
```

**문제점:**
1. 코드가 너무 깔끔함 - 실제 "지저분한" 코드 경험 부족
2. 찾아야 할 것이 너무 명확함 (파일명에서 힌트 제공)
3. 실제 프로젝트에서 발생하는 복잡한 상황 미반영

---

## 🎯 개선 방향

### 추가해야 할 코드 스멜 (Code Smells)

검색 결과에서 발견한 핵심 코드 스멜 카테고리 ([refactoring.guru](https://refactoring.guru/refactoring/smells) 기반):

#### 1. Bloaters (비대화된 코드)
- **Long Method**: 100줄 이상의 거대한 함수
- **Large Class**: 너무 많은 책임을 가진 클래스
- **Long Parameter List**: 5개 이상의 매개변수

#### 2. Object-Orientation Abusers
- **Switch Statements**: 반복되는 switch/if-else 체인
- **Parallel Inheritance**: 중복된 상속 구조

#### 3. Change Preventers
- **Divergent Change**: 하나의 클래스가 여러 이유로 변경됨
- **Shotgun Surgery**: 하나의 변경이 여러 파일에 영향

#### 4. Dispensables (불필요한 것들)
- **Dead Code**: 사용되지 않는 코드
- **Duplicate Code**: 중복 코드
- **Comments**: 코드 대신 주석으로 설명하는 패턴

#### 5. Couplers (결합도 문제)
- **Feature Envy**: 다른 클래스의 데이터를 과도하게 사용
- **Inappropriate Intimacy**: 클래스 간 과도한 의존

---

## 🔧 구체적 개선안

### 확장된 프로젝트 구조 (제안)

```
messy-ml-project-v2/
├── src/
│   ├── train.py              # 최신이 아닌 버전
│   ├── train_v2.py           # 중간 버전
│   ├── train_v2_backup.py    # v2 백업
│   ├── train_final.py        # "최종"이라지만...
│   ├── train_final_v2.py     # 최종의 두번째 버전
│   ├── train_final_REAL.py   # 실제 메인 (500줄짜리 God Object)
│   ├── train_old_dont_use.py # 사용금지 파일
│   ├── model.py
│   ├── model_v2.py           # 새 모델 버전
│   ├── model_backup.py       # 백업
│   ├── model_old.py          # 옛날 모델
│   ├── utils.py
│   ├── utils_v2.py
│   ├── utils_old.py
│   ├── helper.py             # utils와 기능 중복
│   └── common.py             # helper와 기능 중복
├── data/
│   ├── preprocess.py
│   ├── preprocess_new.py
│   ├── loader.py
│   └── loader_backup.py
├── config/
│   ├── config.py             # 하드코딩된 값들
│   ├── config.json
│   ├── config_prod.json
│   ├── config_dev.json
│   └── config_old.json
├── notebooks/
│   ├── experiment1.ipynb
│   ├── experiment2.ipynb
│   ├── test.ipynb
│   └── Untitled.ipynb
├── logs/
│   ├── train_2024_01.log
│   ├── train_2024_02.log
│   └── debug.log
├── outputs/
│   ├── model_v1.pt
│   ├── model_v2.pt
│   ├── model_final.pt
│   ├── model_final_v2.pt
│   └── checkpoint_epoch_5.pt
├── tests/
│   ├── test_model.py         # 오래된 테스트
│   └── test_model_old.py     # 더 오래된 테스트
├── requirements.txt
├── requirements_dev.txt
├── requirements_old.txt
├── setup.py
├── README.md                  # 오래되고 부정확한 문서
├── TODO.txt
├── notes.txt
└── .env.example
```

### 추가할 코드 스멜 예시

#### 1. God Object (train_final_REAL.py)
```python
# 500줄짜리 파일에서 모든 것을 처리:
# - 데이터 로딩
# - 전처리
# - 모델 정의
# - 학습
# - 평가
# - 저장
# - 로깅
# - 시각화
```

#### 2. Magic Numbers
```python
def train():
    for epoch in range(100):  # 왜 100인가?
        if loss < 0.01:        # 왜 0.01인가?
            lr = lr * 0.1      # 왜 0.1인가?
```

#### 3. Dead Code
```python
def unused_function():
    """이 함수는 어디서도 호출되지 않음"""
    pass

# TODO: 나중에 구현
# FIXME: 버그 있음
# DEPRECATED: 사용하지 마세요
```

#### 4. Duplicate Code (helper.py vs common.py)
두 파일에 거의 동일한 함수들이 존재

#### 5. Poor Naming
```python
def do_stuff(x, y, z, a, b):
    temp = x + y
    temp2 = temp * z
    res = temp2 / a - b
    return res
```

---

## 📚 추가 교육 필요 내용

1주차 과제를 효과적으로 수행하기 위해 참가자들에게 사전에 알려줘야 할 내용:

### 필수 선행 지식
1. **Code Smell 개념**: 코드에서 나쁜 패턴을 인식하는 방법
2. **리팩토링 원칙**: Martin Fowler의 리팩토링 카탈로그 기본
3. **Claude Code 사용법**: 파일 분석, 의존성 추적 명령어

### 추천 사전 학습 자료
- [Refactoring Guru - Code Smells](https://refactoring.guru/refactoring/smells)
- [Martin Fowler - Code Smell](https://martinfowler.com/bliki/CodeSmell.html)
- [freeCodeCamp - Clean Code Course](https://www.freecodecamp.org/news/level-up-your-javascript-detect-smells-and-write-clean-code/)

### 실습 전 안내사항
1. 파일을 삭제하기 전 반드시 의존성 확인
2. 이름만 보고 판단하지 말 것 (train_old.py가 실제로 사용될 수도)
3. Git history 확인하는 습관

---

## 📋 다음 단계

1. [x] 현재 프로젝트 분석 완료
2. [x] 개선 방향 수립 완료
3. [ ] 확장된 프로젝트 생성
4. [ ] 평가 기준(rubric) 업데이트
5. [ ] 참가자 가이드 문서 작성

---

## 🔗 참고 자료

- [CodelyTV/refactoring-code-smells](https://github.com/CodelyTV/refactoring-code-smells) - 실습용 코드 스멜 예제
- [Refactoring.guru](https://refactoring.guru/refactoring/smells) - 코드 스멜 카탈로그
- [Technical Debt Examples](https://brainhub.eu/library/technical-debt-examples) - 기술 부채 실사례
- [Knight Capital Case](https://www.stepsize.com/blog/technical-debt-horror-stories) - 기술 부채 공포 스토리
