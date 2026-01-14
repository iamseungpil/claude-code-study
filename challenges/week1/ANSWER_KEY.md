# Week 1 Answer Key (채점자용)

⚠️ **이 문서는 채점자/운영자 전용입니다. 참가자에게 공개하지 마세요.**

## 프로젝트 정답

### 핵심 파일 (Keep)
| 파일 | 역할 | 비고 |
|------|------|------|
| `src/train_final_REAL.py` | **메인 학습 스크립트** | 실제 사용되는 유일한 train 파일 |
| `src/model.py` | CNN 모델 정의 | SimpleCNN 클래스 |
| `src/utils.py` | 유틸리티 함수 | load_data, save_model 등 |
| `data/preprocess.py` | 전처리 함수 | |
| `data/loader.py` | 데이터 로더 | |
| `config/config.py` | 설정 값 | 하드코딩된 상수들 |
| `config/config.json` | 설정 파일 | |
| `config/config_prod.json` | 프로덕션 설정 | |
| `config/config_dev.json` | 개발 설정 | |
| `tests/test_model.py` | 테스트 | 업데이트 필요 |
| `scripts/evaluate.py` | 평가 스크립트 | |
| `scripts/run_training.sh` | 실행 스크립트 | |
| `requirements.txt` | 의존성 | |
| `requirements_dev.txt` | 개발 의존성 | |
| `setup.py` | 패키지 설정 | |

### 삭제 대상 (Delete)
| 파일 | 이유 |
|------|------|
| `src/train.py` | 불완전, 미사용 |
| `src/train_v2.py` | 중간 버전, 미사용 |
| `src/train_v2_backup.py` | 백업 파일, 불필요 |
| `src/train_final.py` | 가짜 최종, 미사용 |
| `src/train_final_v2.py` | 가짜 최종 v2, 미사용 |
| `src/train_old_dont_use.py` | deprecated 명시 |
| `src/model_backup.py` | model.py와 100% 동일 |
| `src/model_old.py` | deprecated, OldModel 미사용 |
| `src/utils_old.py` | deprecated 명시 |
| `data/loader_backup.py` | loader.py와 동일 |
| `config/config_old.json` | 구버전 형식 |
| `requirements_old.txt` | 구버전 의존성 |
| `tests/test_model_old.py` | deprecated |
| `logs/*.log` | 오래된 로그 (선택적) |
| `TODO.txt` | 코드에 통합하거나 이슈로 이동 |
| `notes.txt` | 개인 메모, 불필요 |

### 병합 대상 (Merge)
| 파일들 | 병합 방향 |
|--------|----------|
| `src/helper.py` + `src/common.py` | → `src/utils.py`에 통합 |
| `src/utils_v2.py` | → `src/utils.py`에 통합 |

### 검토 필요 (Review)
| 파일 | 판단 |
|------|------|
| `src/model_v2.py` | BatchNorm 버전 - 성능 비교 후 결정 |
| `data/preprocess_new.py` | WIP 상태 - 완성 또는 삭제 |

## 코드 스멜 정답

### 1. Bloaters
- **Long Method**: `train_final_REAL.py`의 `main()` 함수 (~100줄)
- **Long Parameter List**:
  - `load_data()` - 10개 매개변수
  - `train_one_epoch()` - 8개 매개변수
  - `SimpleCNN.__init__()` - 10개 매개변수

### 2. Dispensables
- **Dead Code**:
  - `unused_method_1()`, `unused_method_2()` in SimpleCNN
  - `deprecated_forward()` in SimpleCNN
  - `OldModel` class
  - `setup_logging_old()`, `load_data_old()`, `main_old()` 등
  - `test_function()` - tests 폴더로 이동 필요

- **Duplicate Code**:
  - `helper.py`의 `get_device()` = `common.py`의 `get_device()` = `utils_old.py`의 `get_device()`
  - `helper.py`의 `count_params()` = `utils.py`의 `count_parameters()`
  - `helper.py`의 `compute_accuracy()` = `common.py`의 `calculate_accuracy()`
  - `helper.py`의 `normalize_data()` = `data/preprocess.py`의 `normalize()`
  - `save_model()` = `save_model_simple()` in train_final_REAL.py

### 3. Other Issues
- **Magic Numbers**:
  - `0.1307` - MNIST mean
  - `0.3081` - MNIST std
  - `42` - random seed
  - `0.01` - threshold
  - `0.1` - LR decay factor

- **Global Variables**:
  - `global_model`, `global_optimizer`, `global_loss_history`
  - `current_epoch`, `is_training`, `debug_mode`

- **Poor Naming**:
  - `do_something(a, b, c, d, e, f, g)`
  - `helper_function_1()`, `helper_function_2()`
  - `MYSTERIOUS_CONSTANT`

## 채점 가이드

### 분석 (40점)

| 점수 | 기준 |
|------|------|
| 35-40 | train_final_REAL.py 식별 + 의존성 완벽 분석 + 코드스멜 3개 이상 |
| 25-34 | 핵심 파일 대부분 식별 + 의존성 부분 분석 + 코드스멜 1-2개 |
| 15-24 | 핵심 파일 일부 식별 + 의존성 분석 부족 |
| 0-14 | 핵심 파일 미식별 또는 잘못된 분석 |

### 정리 (30점)

| 점수 | 기준 |
|------|------|
| 25-30 | 삭제 대상 80%+ 정확 + 병합 대상 식별 + 근거 완벽 |
| 18-24 | 삭제 대상 60%+ 정확 + 근거 제시 |
| 10-17 | 삭제 대상 일부 정확 + 근거 부족 |
| 0-9 | 중요 파일 삭제 표시 또는 대부분 미식별 |

### README (20점)

| 점수 | 기준 |
|------|------|
| 17-20 | 4개 항목 모두 충실 + 실행 가능한 명령어 |
| 12-16 | 3개 항목 충실 |
| 6-11 | 1-2개 항목만 충실 |
| 0-5 | 형식만 갖춤 또는 부정확 |
