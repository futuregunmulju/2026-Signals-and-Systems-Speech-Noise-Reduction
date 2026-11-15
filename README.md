# 2026-Signals-and-Systems-Speech-Noise-Reduction
(2026 Fall) Traditional speech noise reduction challenge

# 9조 — 신호 및 시스템 1차 챌린지

전통적 신호 처리 기반 음성 잡음 제거 프로그램입니다.

## 제출 알고리즘

Band-pass → MMSE-STSA → MMSE-LSA 순서로 필터링합니다.

- 샘플링 주파수 — 16 kHz
- Band-pass — 4차 Butterworth, 120–3,800 Hz
- STFT — Hann window, FFT 512, hop 128
- 잡음 추정 — 완전한 프레임 중 에너지가 낮은 20%
- SNR 평활화 계수 — 0.98
- 최소 게인 — 0.05

## 실행 파일 (2 ways)

`[9조]_Challenge1_Speech_Noise_Reduction_Algorithms.ipynb`

## 1. Google Colab 실행 방법

1. GitHub에서 위 노트북을 열고 `Open in Colab`을 클릭합니다.
2. 첫 번째 코드 셀을 실행해 패키지를 설치합니다.
3. 두 번째 코드 셀을 실행해 input·output 폴더를 생성합니다.
4. 왼쪽 파일 메뉴를 새로고침하고 `/content/input` 폴더에 테스트 WAV를 업로드합니다.
5. 세 번째 코드 셀에서 필터 함수를 정의합니다.
6. 네 번째 코드 셀을 실행해 모든 입력 WAV를 처리합니다.
7. `/content/output` 폴더에서 결과 WAV를 확인합니다.

GitHub에 있는 input·output 폴더는 Colab에 자동 복사되지 않습니다.
Colab에서는 두 번째 셀이 실행 환경에 폴더를 생성합니다.

## 2. 로컬 Jupyter 실행 방법

1. 저장소를 다운로드하고 압축을 풉니다.
2. 저장소 폴더에서 터미널을 열고 패키지를 설치합니다.

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Jupyter에서 제출 노트북을 엽니다.
4. 첫 번째와 두 번째 코드 셀을 실행합니다.
5. 두 번째 셀에 출력되는 입력 폴더 경로에 테스트 WAV를 넣습니다.
6. 세 번째와 네 번째 코드 셀을 실행합니다.
7. 출력 폴더에서 결과 WAV를 확인합니다.

## 입출력 규칙

- input 폴더 바로 아래의 모든 `.wav` 파일을 처리합니다.
- 입력 WAV마다 output 폴더에 같은 파일명의 WAV를 생성합니다.
- 입력과 같은 샘플링 주파수, 채널 수, 샘플 수를 유지합니다.
- 여러 채널은 채널별로 처리합니다.
- 결과는 32-bit float WAV로 저장합니다.
- 같은 이름의 출력 파일이 있으면 이번 결과로 덮어씁니다.
- 실행 마지막에 처리 성공 개수와 출력 경로가 표시됩니다.
- 입력은 16 kHz WAV를 사용합니다.

## Validation 결과

| 항목 | 결과 |
|---|---|
| 평가 데이터 | Validation 164쌍 |
| 평가 지표 | 16 kHz PESQ-WB |
| 처리 후 평균 PESQ | 1.612048 |
| 채점 구간 | valid_samples가 있으면 해당 구간, 없으면 전체 파형 |

위 점수는 Validation 평가 시 메모리상의 필터 출력으로 계산했습니다.
제출 프로그램은 테스트 WAV를 처리해 저장하며,
최종 PESQ 평가는 조교님이 보유한 Clean 정답으로 수행합니다.
