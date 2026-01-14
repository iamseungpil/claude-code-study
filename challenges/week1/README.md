# Week 1 Challenge: Legacy Code Separation & Refactoring

## 미션

당신은 퇴사한 개발자가 남긴 `everything.py` 파일을 인수인계 받았습니다.
이 파일에는 **두 개의 완전히 다른 프로젝트**가 하나의 파일에 섞여 있습니다.

### 해야 할 일

1. **분석**: 코드를 분석하여 두 개의 서로 다른 도메인을 식별하세요
2. **분리**: 각 프로젝트를 별도의 폴더/모듈로 분리하세요
3. **리팩토링**: 코드 가독성을 개선하세요
4. **검증**: 모든 테스트가 통과하는지 확인하세요
5. **문서화**: 각 프로젝트의 README.md를 작성하세요

## 제한 시간

**45분**

## 테스트 실행

```bash
# 현재 상태 테스트
python everything.py test

# 또는
python -m unittest everything
```

## 기대하는 결과물 구조 (예시)

```
submission/
├── gilded_rose/          # 프로젝트 1
│   ├── __init__.py
│   ├── gilded_rose.py
│   ├── item.py
│   ├── test_gilded_rose.py
│   └── README.md
├── tennis/               # 프로젝트 2
│   ├── __init__.py
│   ├── tennis_game.py
│   ├── test_tennis.py
│   └── README.md
└── README.md             # 전체 설명
```

## 평가 기준

| 항목 | 배점 |
|------|------|
| 프로젝트 분리 | 20점 |
| 파일 구조화 | 15점 |
| 테스트 통과 | 25점 |
| 리팩토링 품질 | 15점 |
| 문서화 | 5점 |
| **루브릭 합계** | **80점** |
| 시간 등수 점수 | 20점 |
| **총점** | **100점** |

## 힌트

- `everything.py`를 실행해보면 두 프로젝트가 무엇인지 감을 잡을 수 있습니다
- 테스트 코드를 먼저 분석하면 각 프로젝트의 기능을 이해하는 데 도움이 됩니다
- Claude Code를 활용하여 코드 분석 및 리팩토링을 진행하세요!

## 실행 방법

```bash
# 데모 실행 (프로젝트 이해용)
python everything.py

# 테스트 실행
python everything.py test
```
