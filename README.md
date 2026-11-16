# 2026-Signals-and-Systems-Speech-Noise-Reduction
(2026 Fall) Traditional speech noise reduction challenge

# 음성 잡음 제거 (고전 신호처리, 딥러닝/머신러닝 미사용)

- 팀: **9조** 구조대
- 라운드: 2026 신호 및 시스템 1차 챌린지
- 제출 버전: 2차 제출 (tag `v2`)

## 실행 방법

```bash
pip install -r requirements.txt        # numpy, scipy, soundfile
python denoise.py --input_dir input --output_dir output
```

- `input_dir`의 모든 `.wav`를 같은 파일명으로 `output_dir`에 저장합니다. 길이, 샘플레이트, 채널 수, subtype은 입력과 동일하게 유지합니다.
- `best_params.json`(튜닝된 설정)이 `denoise.py`와 **같은 폴더**에 있어야 합니다. 있으면 자동으로 기본값을 덮어씁니다.
- 학습된 모델, 외부 데이터, 잡음 템플릿은 사용하지 않습니다. 사용 라이브러리: numpy, scipy, soundfile.

## 파일

| 파일 | 내용 |
|---|---|
| `denoise.py` | 잡음 제거 알고리즘 전체와 CLI |
| `best_params.json` | 튜닝된 파라미터(config) |
| `requirements.txt` | 의존 패키지 |

## 알고리즘 개요

STFT 영역에서 한 번의 이득(gain) 계산으로 처리하는 단일 파이프라인입니다. 잡음 유형별 모듈은 이득을 따로 곱하는 필터가 아니라, **잡음/말소리 추정치만 보정**합니다(필터 직렬 연결 없음).

1. **고역통과(HPF, 100 Hz, 영위상)**: 바람/럼블 제거.
2. **톤 억제(`_tonal_suppress`)**: 희소하고 시간적으로 연속인 스펙트럼 피크(정상 톤 잡음)를 노치.
3. **잡음 PSD 추정(`_noise_psd`)**: 클립 전체를 보는 비인과 최소통계 + 말소리 부재 가중 시간 평균 (Martin, 2001 계열).
4. **사이렌 궤적 추적(`_siren_track`)**: 스펙트럼 돌출도 → 하모닉 콤 현저도 → 스윕 속도를 제한한 Viterbi 경로. 클립에서 검출된 경우에만 사이렌 전력을 잡음 PSD에 더함.
5. **사전 SNR 추정**: Decision-directed (Ephraim & Malah, 1984) + 케프스트럼 평활 ML-SNR 혼합 (Breithaupt et al., 2007 계열).
6. **하모닉 콤(`_harmonic_mult`)**: 프레임별 케프스트럼 F0로 말소리 하모닉 사이 대역의 사전 SNR을 낮춤(유성음 신뢰도로 가중).
7. **MMSE-LSA 이득** (Ephraim & Malah, 1985) + 음성 휴지 구간 전용 이득 하한.
8. 저이득 영역 이득 평활(musical noise 억제), 소량의 원신호 leak.

## 파라미터 튜닝

- 파라미터는 Train 데이터 일부에서 **무작위/지역 탐색**으로 골랐고, Validation(164쌍)으로 확인했습니다.
- 모듈(케프스트럼, 휴지 바닥값, 하모닉, 사이렌) 채택 여부는 Validation 평균 개선량이 기준(+0.003~0.005)을 넘을 때만 결정했습니다. Validation을 선택에 사용했으므로 **Validation 점수는 약간 낙관적일 수 있습니다.**
- 학습 가능한 모델은 없고, 튜닝 대상은 수십 개의 스칼라 파라미터뿐입니다.

## Validation 결과 (PESQ-WB, 164쌍)

| 단계 | 평균 PESQ-WB |
|---|---|
| 잡음 입력(노이즈 그대로) | 약 1.395 |
| 기본 직렬 파이프라인 | 1.687 |
| 파라미터 탐색 | 1.7772 |
| + 케프스트럼 평활, 휴지 구간 바닥값 | 1.8255 |
| + 하모닉 콤 | 1.8323 |
| + 사이렌 추적 | 1.8334 |
| **최종 (하모닉 + 사이렌)** | **1.8395** |

최종 버전 SNR별:

| 목표 SNR (dB) | 0 | 5 | 10 | 15 |
|---|---|---|---|---|
| 평균 PESQ-WB | 1.253 | 1.571 | 1.865 | 2.154 |

## 재현

`denoise.py`, `best_params.json`, `requirements.txt`를 같은 폴더에 두고 위 실행 명령을 사용하세요. 알고리즘은 결정적(deterministic)이며 난수를 쓰지 않습니다.
