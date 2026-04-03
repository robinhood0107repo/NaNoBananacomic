# NaNoBananacomic 작업 규칙

## 1. 목적

이 문서는 저장소 운영, Git, 의존성, 산출물, Codex 작업 방식에 대한 **실행 규칙**을 고정한다.

- 아키텍처 방향은 `doc/계획.md`
- 구현 상세는 `doc/구현계획.md`
- 실제 작업 절차와 저장소 운영 규칙은 `doc/rules.md`

즉, 이 문서는 “무엇을 만들 것인가”보다 “어떻게 작업하고 관리할 것인가”를 정한다.

---

## 2. 문서 우선순위

작업 전에 아래 순서로 문서를 확인한다.

1. `doc/rules.md`
2. `doc/구현계획.md`
3. `doc/계획.md`
4. `GUI_PLAN/` 내부 목업 자료

우선순위 규칙:

- 저장소 운영, Git, 의존성, 배포 규칙은 `doc/rules.md`를 따른다.
- 구현 구조, 하네스, 품질 게이트는 `doc/구현계획.md`를 따른다.
- 전체 방향, 책임 분리, 산출물 계약은 `doc/계획.md`를 따른다.
- `GUI_PLAN/`은 시각 목업이며 고정 명세가 아니다.

Codex를 포함한 모든 자동화/에이전트 작업은 큰 변경 전에 위 문서를 먼저 읽는 것을 기본 규칙으로 한다.

추가 기본 규칙:

- 진행 상태의 단일 기준 문서는 `doc/계획진행현황.md`다.
- 큰 구현 단계가 끝나면 `doc/계획진행현황.md` 체크리스트를 즉시 갱신한다.
- Phase 단위의 의미 있는 변경이 끝나면 반드시 Git 커밋을 만든다.
- 원격 푸시가 가능한 상태라면 같은 턴에서 바로 push까지 진행한다.

---

## 3. 프로젝트 경계

### 3.1 새 시스템 우선

- 새 파이프라인은 `src/` 기준으로 새로 작성한다.
- 기존 코드는 `legacy/`에 격리하며 실행 경로에 포함하지 않는다.
- 새 코드에서 `legacy/`를 import하지 않는다.

### 3.2 역할 분리

로컬 책임:

- 말풍선 검출
- union mask 생성
- 말풍선-only RGBA 생성
- Nano Banana 결과 alpha 복구
- 원본과 최종 레이어 합성
- 검증과 diff 확인

Nano Banana 책임:

- OCR
- 번역
- 식질
- 말풍선 내부 인페이팅

### 3.3 외부 연동 모드

Phase 3 외부 연동은 아래 두 경로를 모두 허용한다.

- 유료 API 자동 경로
- 무료 웹 수동 경로

규칙:

- provider, model, API 입력은 GUI에서 받는다.
- 실제 API 키 평문은 `project.json`이나 `pages/*.page.json`에 저장하지 않는다.
- 무료 웹 경로는 prompt package 생성과 수동 결과 가져오기를 정식 경로로 본다.
- 수동 웹 결과는 GUI 파일 선택 또는 `imports/nano/` 붙여넣기로 수용한다.
- 수동 웹 결과는 원본과 해상도나 캔버스 위치가 달라진 상태로 들어오는 것을 정상 입력으로 본다.
- 로컬은 수동 import 결과를 원본 페이지 좌표계에 맞추는 책임을 가진다.

### 3.4 수동 import 파일명 규칙

`page_id`는 원본 파일명 stem과 같다.

예:

- `1.webp` -> `page_id = 1`
- `K1466567_g2_144301.jpg` -> `page_id = K1466567_g2_144301`

권장 파일명:

- `<page_id>_submitted.png`

허용 파일명:

- `<page_id>.<ext>`
- `<page_id>.png`
- `<page_id>.jpg`
- `<page_id>.webp`
- `<page_id>_submitted.<ext>`
- `<page_id>_translated.<ext>`
- `<page_id>_result.<ext>`
- `<page_id>_output.<ext>`
- `<page_id>_edited.<ext>`
- `<page_id>_web.<ext>`
- `<page_id>_nano.<ext>`
- `<page_id>_final.<ext>`
- `<page_id>_balloons_only.<ext>`
- `<page_id>_raw.<ext>`
- `<page_id>_*.png`
- `<page_id>-*.png`
- `<page_id> *.png`

실제 탐색은 확장자를 엄격히 제한하지 않는다. 파일명이 `page_id` 자체이거나 `page_id_`, `page_id-`, `page_id `로 시작하면 공통 이미지 파일로 간주해 후보로 본다.

여러 파일이 동시에 있으면 탐색 우선순위는 아래다.

1. `<page_id>_submitted.<ext>`
2. `<page_id>_translated`, `<page_id>_result`, `<page_id>_output`, `<page_id>_edited`, `<page_id>_web`, `<page_id>_nano`, `<page_id>_final`, `<page_id>_balloons_only`
3. `<page_id>_raw.<ext>`
4. `<page_id>` 자체 파일
5. 그 외 `page_id` prefix 후보 중 최신 파일

### 3.5 GUI 자료의 위치

- `GUI_PLAN/`은 참고용 디자인 자료다.
- 실제 구현은 `PySide6` 중심으로 진행한다.
- `GUI_PLAN/`은 기본적으로 Git 추적 대상이 아니다.

---

## 4. 저장소와 프로젝트 폴더 규칙

이 저장소 루트와 실제 작업 프로젝트 폴더는 분리한다.

- 저장소 루트: 코드, 문서, 테스트, 런처를 보관
- 사용자가 GUI에서 선택한 폴더: 실제 만화 프로젝트 루트

프로젝트 폴더에는 아래 산출물이 생성될 수 있다.

- `project.json`
- `pages/*.page.json`
- `imports/nano/*`
- `artifacts/`
- `logs/`
- `result/`

원칙:

- 중간 산출물은 프로젝트 폴더에 저장한다.
- 저장소 루트에는 원칙적으로 사용자 결과물을 쌓지 않는다.
- 루트에 생성된 실행 산출물은 기본적으로 Git에 올리지 않는다.

---

## 5. Git 추적 규칙

### 5.1 기본 포함/제외

Git에 올리는 대상:

- `src/`
- `tests/`
- `doc/`
- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `launch.bat`
- 이후 추가되는 설정/CI/도구 파일

기본 제외 대상:

- `legacy/`
- `GUI_PLAN/`
- `.venv/`
- `.vendor/`
- `.bootstrap/`
- `__pycache__/`
- 테스트/린트/타입체크 캐시
- 빌드 결과물
- 로컬 로그
- 실행 중 생성된 프로젝트 산출물

위 제외 정책은 `.gitignore`에 반영하고 유지한다.

### 5.2 대용량 파일

다음 유형은 일반 Git 대신 `Git LFS` 또는 별도 데이터 관리로 다룬다.

- 샘플 원본 이미지
- 회귀 기준 이미지
- 모델 가중치
- 대형 benchmark 산출물

데이터셋/실험 추적이 본격화되면 `DVC`를 사용한다.

---

## 6. Git 브랜치 및 커밋 규칙

### 6.1 브랜치 규칙

- 안정 브랜치는 `main`
- 작업 브랜치는 `codex/...`
- 기능, 수정, 실험은 가급적 `main`에서 직접 하지 않는다.

브랜치 이름 예시:

- `codex/setup-repo-rules`
- `codex/add-alpha-restore`
- `codex/benchmark-bw-detector`

### 6.2 원격 저장소 규칙

기본 원격 저장소는 아래를 사용한다.

- `origin = https://github.com/robinhood0107repo/NaNoBananacomic.git`

원격 저장소가 바뀌면 `doc/rules.md`에도 반영한다.

### 6.3 커밋 규칙

- 한 커밋에는 하나의 논리적 변경만 담는다.
- 문서 변경과 코드 변경이 강하게 연결되지 않으면 분리한다.
- 캐시, 임시 파일, 생성 결과물은 커밋하지 않는다.
- 실험용 대형 파일은 일반 Git에 직접 올리지 않는다.

권장 커밋 흐름:

1. 변경 범위 확인
2. 필요한 파일만 stage
3. 품질 게이트 실행
4. 커밋

### 6.4 푸시 전 최소 점검

가능한 한 아래를 통과한 뒤 푸시한다.

- `ruff format`
- `ruff check --fix`
- `mypy`
- `pytest`

도구가 아직 설정되지 않은 단계라면, 어떤 검증을 생략했는지 기록한다.

### 6.5 마일스톤 형상관리 규칙

- `Phase 1`, `Phase 1.5`, `Phase 2`처럼 문서에 이름이 있는 단계는 중간 저장 없이 오래 끌지 않는다.
- 논리적 마일스톤이 끝나면 아래 순서를 기본으로 따른다.

1. 관련 문서 갱신
2. `doc/계획진행현황.md` 체크리스트 갱신
3. 테스트/검증 실행
4. Git commit
5. Git push

- push가 실패하면 실패 원인을 기록하고, 로컬 commit까지는 반드시 남긴다.
- `main`에서 직접 장시간 작업하지 말고 기본적으로 `codex/...` 브랜치를 사용한다.

---

## 7. 의존성 관리 규칙

### 7.1 기준 파일

의존성의 기준은 아래처럼 관리한다.

- 개발/프로젝트 선언의 기준: `pyproject.toml`
- 배포/재현 설치용 고정 목록: `requirements.txt`

`pyproject.toml`은 범위 기반 선언을 유지할 수 있지만, `requirements.txt`는 반드시 **정확한 버전 고정**을 사용한다.

### 7.2 `requirements.txt` 유지 규칙

다음 경우에는 반드시 `requirements.txt`를 갱신한다.

- 런타임 의존성을 추가했을 때
- 런타임 의존성 버전을 바꿨을 때
- 배포 환경에서 필요한 패키지 구성이 바뀌었을 때

원칙:

- `requirements.txt`는 나중 배포와 재현 설치를 위한 파일이다.
- 최소한 현재 런타임에 필요한 패키지는 모두 포함한다.
- 버전은 `package==x.y.z` 형태로 고정한다.
- `pyproject.toml`과 `requirements.txt` 내용이 어긋나면 즉시 동기화한다.

### 7.3 환경/실행 규칙

- 표준 Python 버전은 `3.12.x`
- WSL/Linux 개발 실행은 `uv run ...` 또는 프로젝트 가상환경 Python 사용을 허용한다.
- Windows 사용자 진입점은 `launch.bat`다.
- `launch.bat`는 인자 없이 실행하면 로컬 `.venv`를 준비한 뒤 `PySide6` GUI를 실행한다.
- `launch.bat <subcommand>` 형태는 기존 CLI를 그대로 실행한다.
- GUI 최근 프로젝트, 창 크기, 도킹 레이아웃은 `QSettings`에 저장한다.
- API 키는 GUI 세션 메모리 또는 `GEMINI_API_KEY` 환경변수로만 사용하고, `project.json`에는 저장하지 않는다.

향후 lockfile을 도입하면 원칙은 아래와 같다.

- 선언은 `pyproject.toml`
- 잠금은 `uv lock`
- 배포용 고정 목록은 필요 시 lock 기반으로 export

### 7.4 CUDA / GPU 사전 점검 규칙

CUDA를 실제로 사용하는 작업 전에는 아래를 먼저 확인한다.

1. `/dev/dxg` 존재 여부
2. `nvidia-smi` 정상 동작 여부
3. `torch.cuda.is_available()` 결과
4. 필요한 경우 실제 detector 추론 후 GPU 메모리 증가 여부

원칙:

- GPU 빌드가 설치되어 있어도 위 4개가 통과하지 않으면 “CUDA 사용 가능”으로 간주하지 않는다.
- Codex 샌드박스/WSL 상태가 바뀌면 이 점검을 다시 수행한다.
- GPU 작업 결과를 문서화할 때는 가능하면 `doc/계획진행현황.md`에 최신 상태를 남긴다.

---

## 8. 코드 및 산출물 규칙

### 8.1 코드 구조

- 라이브러리 코드는 `src/comic_pipeline/` 아래에 둔다.
- 테스트는 `tests/` 아래에 둔다.
- 문서는 `doc/`에 둔다.
- 실행 규칙 변경은 반드시 `doc/rules.md`에 반영한다.

### 8.2 산출물 규칙

프로젝트 산출물은 아래 구분을 유지한다.

- `artifacts/`: 중간 결과
- `logs/`: 실행 로그
- `result/`: 최종 완성본
- `pages/`: 페이지 manifest
- `artifacts/composite/`: 최종 합성 추적본
- `artifacts/debug/step6_<timestamp>_summary.json`: 배치 검증 요약

파일 계약은 항상 원본 페이지 좌표계와 해상도를 기준으로 맞춘다.

### 8.3 검증 우선순위

로컬 품질 우선순위는 아래와 같다.

1. 말풍선 mask 정확도
2. 비말풍선 영역 원본 보존
3. alpha 복구 안정성
4. 최종 합성 안정성
5. 처리 속도

---

## 9. 품질 게이트 규칙

기본 품질 게이트:

- `ruff`
- `mypy`
- `pytest`

도입 예정/권장 도구:

- `pre-commit`
- `coverage.py`
- `pytest-regressions`
- `hypothesis`
- `nox`

원칙:

- 타입, 포맷, 테스트 실패를 방치한 채 누적하지 않는다.
- 이미지 처리 코드는 회귀 테스트 대상으로 확대한다.
- detector 추론 wrapper는 단순 line coverage보다 contract 검증을 우선한다.

---

## 10. Codex 작업 규칙

Codex가 이 저장소에서 작업할 때는 아래를 기본으로 한다.

1. `doc/rules.md`, `doc/구현계획.md`, `doc/계획.md`를 먼저 확인한다.
2. `legacy/`는 참고만 하고 새 코드 경로에 섞지 않는다.
3. `GUI_PLAN/`은 시각 참고 자료로만 사용한다.
4. 저장소에 올릴 파일과 올리지 않을 파일을 구분한다.
5. 의존성이 바뀌면 `requirements.txt` 갱신 여부를 함께 점검한다.
6. 실행 산출물과 캐시를 커밋하지 않는다.
7. 규칙이 바뀌면 코드만 고치지 말고 이 문서도 함께 갱신한다.

---

## 11. 문서 갱신 규칙

아래 변경이 생기면 `doc/rules.md`를 함께 업데이트한다.

- Git 전략 변경
- 원격 저장소 변경
- 의존성 관리 방식 변경
- 배포 방식 변경
- 산출물 저장 정책 변경
- Codex 작업 규칙 변경

아키텍처 자체가 바뀌면 `doc/계획.md`와 `doc/구현계획.md`도 같이 갱신한다.

---

## 12. 현재 기준 체크리스트

현재 저장소 기준으로 즉시 적용되는 규칙은 아래와 같다.

- `legacy/`와 `GUI_PLAN/`은 Git에서 제외한다.
- 캐시, 로컬 환경, 실행 산출물은 Git에서 제외한다.
- `requirements.txt`는 고정 버전 파일로 유지한다.
- 저장소 기본 원격은 `robinhood0107repo/NaNoBananacomic`이다.
- 작업 전에는 `doc/rules.md`를 먼저 본다.
