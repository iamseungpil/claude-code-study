# Week 1 Challenge: Project Folder Cleanup

## 🎯 Mission

당신은 이직한 회사에서 전임자가 남긴 ML 프로젝트를 인수받았습니다.
프로젝트 폴더가 매우 지저분하여 어떤 파일이 실제로 사용되는지 파악하기 어렵습니다.

**당신의 임무:**
1. 프로젝트의 목적과 구조를 분석하세요
2. 실제로 사용되는 핵심 파일을 식별하세요
3. 불필요한 파일을 정리할 계획을 세우세요
4. 깔끔한 문서를 작성하세요

## ⏰ Time Limit

**60분**

## 📁 Challenge Project

```
challenges/week1/messy-ml-project-v2/
```

이 폴더에는 41개의 파일이 있습니다. 그 중 많은 파일이 중복이거나 사용되지 않습니다.

## ✅ Required Outputs

### 1. analysis.md
프로젝트 분석 보고서. 다음 내용을 포함해야 합니다:
- 프로젝트의 목적
- 핵심 파일 목록과 역할
- 파일 간 의존성
- 발견한 문제점 (코드 스멜)

### 2. README.md
정리된 프로젝트 문서. 다음 내용을 포함해야 합니다:
- 프로젝트 설명
- 설치 및 실행 방법
- 파일 구조 설명
- 의존성 목록

### 3. (선택) cleanup_plan.md
정리 계획 문서 - 작성 시 가산점

## 💡 Hints

### Claude Code 활용 팁

```bash
# 파일 구조 확인
ls -la src/

# 특정 함수가 어디서 호출되는지 확인
grep -r "SimpleCNN" .

# 파일 내용 빠르게 확인
cat src/train.py

# import 관계 확인
grep -r "from " src/*.py | grep import
```

### 주의사항
- 파일 이름만 보고 판단하지 마세요 (`train_final.py`가 실제 최종이 아닐 수 있습니다)
- 실제 import와 사용 여부를 확인하세요
- 주석과 TODO를 잘 읽어보세요

## 🔍 What to Look For

### 코드 스멜 (Code Smells)
나쁜 코드의 징후를 찾아보세요:

1. **중복 코드**: 비슷한 기능의 파일/함수가 여러 개
2. **죽은 코드**: 어디서도 호출되지 않는 함수
3. **Magic Numbers**: 설명 없는 숫자값 (`0.1307`, `42`)
4. **긴 함수**: 100줄 이상의 함수
5. **과도한 매개변수**: 5개 이상의 인자를 받는 함수

### 파일 분류
각 파일을 다음 중 하나로 분류하세요:
- **Keep**: 필수적으로 유지해야 하는 파일
- **Delete**: 확실히 삭제해도 되는 파일
- **Merge**: 다른 파일과 병합해야 하는 파일
- **Review**: 추가 검토가 필요한 파일

## 📊 Scoring

| 항목 | 점수 |
|------|------|
| 프로젝트 분석 | 40점 |
| 정리 품질 | 30점 |
| README 품질 | 20점 |
| 시간 보너스 | 10점 |

### 시간 보너스
- 42분 이내 (70%): +10점
- 51분 이내 (85%): +5점
- 60분 이내: 0점
- 초과: -5점/5분

## 📚 Pre-study Materials

과제 전에 다음 자료를 읽어보면 도움이 됩니다:

1. [Refactoring Guru - Code Smells](https://refactoring.guru/refactoring/smells)
2. [Martin Fowler - Code Smell](https://martinfowler.com/bliki/CodeSmell.html)

## 🚀 Getting Started

1. 프로젝트 폴더로 이동
   ```bash
   cd challenges/week1/messy-ml-project-v2
   ```

2. 파일 구조 확인
   ```bash
   find . -type f -name "*.py" | head -20
   ```

3. Claude Code 시작
   ```bash
   claude
   ```

4. 분석 시작!
   ```
   이 프로젝트를 분석해줘. 어떤 파일이 실제로 사용되고 있고, 어떤 파일이 불필요한지 파악해줘.
   ```

Good luck! 🍀
