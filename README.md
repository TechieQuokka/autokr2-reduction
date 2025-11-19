# DeepFilterNet3 Noise Reduction CLI

DeepFilterNet3 기반 음성 노이즈 제거 도구

## 🎯 기능

- **딥러닝 기반 노이즈 억제**: DeepFilterNet3를 사용한 고품질 노이즈 제거
- **노이즈 게이트**: 선택적 amplitude 기반 노이즈 게이트 적용
- **JSON 설정 지원**: 재현 가능한 배치 처리를 위한 설정 파일
- **CLI 오버라이드**: 명령줄에서 설정값 덮어쓰기 가능
- **GPU 가속**: CUDA 지원 (자동 감지 또는 수동 선택)

## 📦 설치

### 1. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

**주요 의존성:**
- `deepfilternet` - DeepFilterNet3 모델
- `soundfile` - 오디오 I/O
- `numpy` - 배열 처리
- `torch` - PyTorch 백엔드

## 🚀 사용법

### 기본 사용

```bash
python denoise.py \
  --input vocals_raw.wav \
  --config settings.json \
  --output vocals_denoised.wav
```

### JSON 설정 파일 형식

```json
{
  "noise_reduction": {
    "strength": 0.5,
    "apply_gate": true,
    "gate_threshold": -40,
    "reason": "No significant static noise"
  }
}
```

**파라미터 설명:**
- `strength` (0.0~1.0): 노이즈 억제 강도
  - 0.0 = 최소 억제
  - 1.0 = 최대 억제 (음성 왜곡 가능)
  - **권장**: 0.5~0.7
- `apply_gate` (boolean): 노이즈 게이트 적용 여부
- `gate_threshold` (dB): 게이트 임계값 (기본: -40dB)

### 고급 사용법

#### 노이즈 억제 강도 조정

```bash
python denoise.py \
  --input input.wav \
  --config config.json \
  --output output.wav \
  --strength 0.8
```

#### 노이즈 게이트 비활성화

```bash
python denoise.py \
  --input input.wav \
  --config config.json \
  --output output.wav \
  --no-gate
```

#### 게이트 임계값 변경

```bash
python denoise.py \
  --input input.wav \
  --config config.json \
  --output output.wav \
  --gate-threshold -35
```

#### CPU 강제 사용

```bash
python denoise.py \
  --input input.wav \
  --config config.json \
  --output output.wav \
  --device cpu
```

#### 상세 로그 출력

```bash
python denoise.py \
  --input input.wav \
  --config config.json \
  --output output.wav \
  --verbose
```

## 📋 전체 옵션

```
필수 옵션:
  --input PATH          입력 오디오 파일 (WAV 권장)
  --config PATH         JSON 설정 파일
  --output PATH         출력 오디오 파일

선택 옵션 (JSON 오버라이드):
  --strength FLOAT      노이즈 억제 강도 (0.0~1.0)
  --gate-threshold DB   노이즈 게이트 임계값 (dB)
  --no-gate            노이즈 게이트 비활성화

실행 옵션:
  --device {auto,cpu,cuda}  연산 디바이스 (기본: auto)
  --verbose                상세 로그 출력
```

## ⚙️ 처리 과정

1. **JSON 설정 로드**: 노이즈 제거 파라미터 읽기
2. **오디오 로드**: 입력 오디오 파일 읽기 및 전처리
   - 샘플레이트 48kHz 변환 (DeepFilterNet 최적화)
   - 스테레오 → 모노 변환 (필요시)
3. **DeepFilterNet3 노이즈 억제**: 딥러닝 기반 노이즈 제거
4. **노이즈 게이트 적용** (선택): 잔여 저레벨 노이즈 억제
5. **오디오 저장**: 처리된 결과 저장

## ⚠️ 주의사항

### 과도한 노이즈 억제 부작용

노이즈 억제 강도(`strength`)가 너무 높으면:
- ❌ 음성 왜곡 ("물속에서 말하는 듯한 소리")
- ❌ 뮤지컬 노이즈 (버블링 효과)
- ❌ 자음 손실 (ㅅ, ㅆ, ㅊ 등)

**권장 사항:**
- 기본값 0.5~0.7 사용
- 점진적으로 증가시키면서 테스트
- 음성 품질과 노이즈 제거의 균형 유지

### 노이즈 게이트 설정

게이트 임계값(`gate_threshold`)이 너무 높으면:
- ❌ 음성의 일부가 잘릴 수 있음
- ❌ 특히 작은 소리나 숨소리가 제거됨

**권장 사항:**
- 기본값 -40dB 권장
- 조용한 음성의 경우 -45dB ~ -50dB
- 큰 소리의 경우 -35dB ~ -30dB

## 🏗️ 프로젝트 구조

```
reduction/
├── denoise.py              # 메인 CLI 스크립트
├── denoiser/
│   ├── __init__.py        # 패키지 초기화
│   ├── config.py          # JSON 설정 파서
│   ├── audio.py           # 오디오 I/O
│   └── core.py            # DeepFilterNet3 래퍼
├── requirements.txt        # 의존성 목록
├── README.md              # 이 문서
└── .gitignore
```

## 🔧 트러블슈팅

### CUDA 오류

CUDA 관련 오류 발생시:
```bash
python denoise.py ... --device cpu
```

### 메모리 부족

긴 오디오 파일 처리시 메모리 부족이 발생하면:
- 짧은 구간으로 나누어 처리
- CPU 사용 (`--device cpu`)
- 배치 크기 조정 (코드 수정 필요)

### DeepFilterNet 설치 오류

```bash
# 특정 버전 설치
pip install deepfilternet==0.5.6

# 또는 소스에서 설치
pip install git+https://github.com/Rikorose/DeepFilterNet.git
```

## 📝 예시

### 예시 1: 기본 노이즈 제거

```bash
python denoise.py \
  --input recording.wav \
  --config config.json \
  --output clean.wav \
  --verbose
```

### 예시 2: 강한 노이즈 제거

```bash
python denoise.py \
  --input noisy_vocals.wav \
  --config config.json \
  --output clean_vocals.wav \
  --strength 0.8 \
  --gate-threshold -35
```

### 예시 3: 부드러운 노이즈 제거

```bash
python denoise.py \
  --input vocals.wav \
  --config config.json \
  --output processed.wav \
  --strength 0.4 \
  --no-gate
```

## 📄 라이선스

This project uses DeepFilterNet3, which is licensed under the MIT License.

## 🙏 감사의 말

이 프로젝트는 [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)을 기반으로 합니다.
