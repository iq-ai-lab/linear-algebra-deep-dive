# 7.5 RoPE: Rotary Position Embedding

> "위치 정보를 덧셈이 아닌 회전으로 인코딩하면, 상대 위치가 자연스럽게 살아난다."

---

## 1. 학습 목표

- **Rotary Position Embedding (RoPE)** 의 수식과 구현을 엄밀히 유도한다.
- 쿼리·키의 내적이 **상대 위치 $n - m$만의 함수**로 환원됨을 증명한다.
- 복소수 표현과 실수 $2 \times 2$ 블록 회전 표현의 동치성을 본다.
- RoPE의 **장거리 감쇠(long-range decay)** 와 주파수 선택 $\theta_i$의 해석을 다룬다.
- 실제 LLaMA, GPT-NeoX, Qwen 등의 구현 세부와 **긴 컨텍스트 확장** (NTK-aware, YaRN) 기법을 조망한다.

---

## 2. 배경: 절대 vs 상대 위치 부호화

### 2.1 Sinusoidal (절대)

Transformer 원 논문: 위치 $m$에 대한 $d$-차원 벡터

$$
\text{PE}(m)_{2i} = \sin(m/\theta_i), \quad \text{PE}(m)_{2i+1} = \cos(m/\theta_i), \quad \theta_i = 10000^{2i/d}
$$

을 임베딩에 **더한다**: $x_m \leftarrow x_m + \text{PE}(m)$. 절대 위치를 인코딩.

### 2.2 Learned 절대 위치

BERT: $\text{PE}(m) \in \mathbb{R}^d$를 학습. 최대 길이 제한.

### 2.3 상대 위치

Shaw et al. (2018): Attention 점수에

$$
s_{ij} = q_i^T k_j + q_i^T r_{i-j}
$$

$r$은 상대 위치 벡터. 단 복잡하고 구현이 어려움.

### 2.4 RoPE의 목표

"**쿼리와 키에 어떤 변환**을 적용하면 그 내적이 **상대 위치만의 함수**가 되는가?"

---

## 3. RoPE 유도

### 3.1 요구 조건

쿼리 $q_m$ (위치 $m$)와 키 $k_n$ (위치 $n$)에 대해 함수 $f_q, f_k$를 찾자:

$$
\langle f_q(q, m), f_k(k, n)\rangle = g(q, k, n - m)
$$

내적이 **$n - m$만의 함수**인 $g$.

### 3.2 2차원에서의 해

$d = 2$를 먼저 해 보자. 복소수 표현 $q = q_1 + i q_2$를 사용. **회전**

$$
f_q(q, m) = e^{im\theta} q, \quad f_k(k, n) = e^{in\theta} k
$$

그러면

$$
\langle f_q(q, m), \overline{f_k(k, n)}\rangle = e^{im\theta} q \cdot e^{-in\theta} \bar k = e^{i(m-n)\theta} q \bar k
$$

실수부:

$$
\text{Re}[f_q(q, m) \overline{f_k(k, n)}] = q_1 k_1 \cos((m-n)\theta) + \ldots
$$

상대 위치 $m - n$만의 함수.

### 3.3 고차원으로 일반화

$d$가 짝수라 하자. $d/2$개 주파수 $\theta_i = 10000^{-2i/d}$ ($i = 0, \ldots, d/2 - 1$).

$q \in \mathbb{R}^d$를 $d/2$개 2차원 쌍으로 나누고 각 쌍에 회전:

$$
R_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
$$

블록 대각 행렬:

$$
R_m = \begin{pmatrix} R_m^{(0)} & & \\ & R_m^{(1)} & \\ & & \ddots \\ & & & R_m^{(d/2-1)} \end{pmatrix}
$$

### 3.4 RoPE 정의

$$
\text{RoPE}(x, m) = R_m x
$$

Attention 점수:

$$
\langle R_m q, R_n k\rangle = q^T R_m^T R_n k = q^T R_{n-m} k
$$

(왜냐면 $R_m^T R_n = R_{-m} R_n = R_{n-m}$, 블록 독립성).

### 3.5 왜 이게 상대 위치?

블록 $i$에 대해:

$$
\text{Re}[(q_{2i} + i q_{2i+1}) \overline{(k_{2i} + i k_{2i+1})} e^{i(n-m)\theta_i}]
$$

주파수 $\theta_i$마다 **$n - m$에 의한 위상차**가 있다. 모든 블록의 합이 점수.

---

## 4. 주파수 선택

### 4.1 Geometric Progression

$$
\theta_i = 10000^{-2i/d}, \quad i = 0, \ldots, d/2 - 1
$$

- $i = 0$: $\theta_0 = 1$ (주기 $2\pi$, 빠른 변화)
- $i = d/2 - 1$: $\theta_{d/2-1} = 10000^{-(d-2)/d} \approx 1/10000$ (주기 $2\pi \cdot 10000$, 느린 변화)

이렇게 **다양한 시간 스케일**을 동시에 커버.

### 4.2 Long-range Decay

거리 $|n - m|$가 커지면 내적의 평균 크기가 감소:

$$
\mathbb{E}[\langle R_m q, R_n k\rangle] \approx 0 \text{ for large } |n - m|
$$

(랜덤 $q, k$에 대해). 이 감쇠는 각 주파수가 위상이 다르게 회전하여 **평균적으로 상쇄**되기 때문.

### 4.3 Causal decay profile

RoPE는 sinusoidal에 비해 **더 부드러운** 장거리 감쇠를 보이는 것이 관찰됨 (Su et al. 2021).

---

## 5. 구현

### 5.1 "반쪽 회전" 트릭

실제 구현에서는 각 블록을 $(x_{2i}, x_{2i+1})$ 쌍으로 보는 대신 **상위-하위 반쪽**:

$$
x = [x_1, x_2, \ldots, x_{d/2}, x_{d/2 + 1}, \ldots, x_d]
$$

회전:

```
x_rot = [x_1 * cos - x_{d/2+1} * sin, ..., x_{d/2} * cos - x_d * sin,
         x_{d/2+1} * cos + x_1 * sin, ..., x_d * cos + x_{d/2} * sin]
```

즉

$$
\text{rotate\_half}(x) = \begin{pmatrix} -x_2 \\ x_1 \end{pmatrix}_{\text{block}} \text{ or } \begin{pmatrix} -x_{d/2+1:} \\ x_{:d/2} \end{pmatrix}_{\text{half}}
$$

(구현에 따라).

### 5.2 효율성

- 곱셈만, 추가 행렬 없음
- 선형 메모리
- Pre-compute $\cos, \sin$ tables ($[S_{\max}, d]$)

---

## 6. RoPE 변종과 긴 컨텍스트

### 6.1 Position Interpolation (PI)

훈련 길이 $L_{\text{train}}$을 넘는 추론 길이 $L_{\text{ext}}$에 대해 모든 위치를 **선형 스케일**:

$$
m' = m \cdot L_{\text{train}} / L_{\text{ext}}
$$

즉 위치를 "압축"하여 기존 분포에 맞춤. 간단하지만 정보 손실.

### 6.2 NTK-Aware Scaling

높은 주파수 블록은 그대로, 낮은 주파수 블록만 스케일:

$$
\theta_i \leftarrow \theta_i \cdot (L_{\text{ext}} / L_{\text{train}})^{2i/(d-2)}
$$

Neural Tangent Kernel 분석에서 유도. 고주파 정보는 보존.

### 6.3 YaRN (Yet another RoPE extensioN)

NTK-aware + temperature scaling + partial rotation. 32K~100K 컨텍스트로 확장에 효과적.

### 6.4 RoPE의 한계

- 짝수 $d$만 가능
- 학습 분포 밖 절대 위치에서 extrapolation 어려움 (PI, NTK 보완)
- 상대 위치가 아닌 **절대** 정보가 필요할 땐 불리

---

## 7. 수학적 성질

### 7.1 RoPE의 선형성과 은폐

$R_m$은 직교행렬. $\|R_m q\| = \|q\|$. 따라서 쿼리·키의 노름은 위치에 무관.

### 7.2 덧셈 정리의 의미

$R_m R_n = R_{m+n}$. 이는 **군 구조** $(SO(2))^{d/2}$의 표현.

### 7.3 고유값과 고유벡터

각 $R_m^{(i)}$의 고유값은 $e^{\pm im\theta_i}$ (복소수). 실수 고유벡터는 없음 (90° 회전 제외).

### 7.4 Fourier 해석

RoPE는 본질적으로 **시퀀스 축의 이산 푸리에 변환**. 각 주파수 블록이 하나의 Fourier component.

---

## 8. 다른 모델의 위치 인코딩

### 8.1 ALiBi (Attention with Linear Biases)

위치에 의한 **bias를 attention 점수에 더함**:

$$
s_{ij} = q_i^T k_j - \alpha|i - j|
$$

$\alpha$는 head별 다른 기울기. RoPE와 달리 query/key를 변형하지 않음. 매우 긴 컨텍스트로 extrapolation 잘 됨.

### 8.2 Learned 상대 위치 (T5 bucket)

$i - j$의 값을 버킷화하여 각 버킷에 학습 가능한 스칼라. Attention 점수에 더함.

### 8.3 비교

| 방식 | 복잡도 | Extrapolation | 특징 |
|---|---|---|---|
| Sinusoidal | O(Sd) | 제한적 | 간단 |
| Learned Abs | O(Sd) | 불가 | 훈련 길이에 고정 |
| Shaw relative | O(S^2 d) | 좋음 | 복잡 |
| RoPE | O(Sd) | NTK로 확장 | 우아, 주류 |
| ALiBi | O(S^2) (스칼라) | 우수 | 극단 길이 대응 |

---

## 9. RoPE와 Attention의 결합

### 9.1 Standard MHA with RoPE

```
Q' = RoPE(Q W^Q, positions)
K' = RoPE(K W^K, positions)
V  = V W^V
scores = Q' K'^T / sqrt(d_k)
...
```

### 9.2 왜 V에는 적용 안 하는가?

내적에서만 상대 위치가 나타나면 충분. $V$는 값 벡터이므로 위치 변환 필요 없음.

### 9.3 2D, 3D RoPE

이미지나 비디오에서 2D / 3D 위치로 확장. 각 축마다 주파수 셋 분리.

---

## 10. Python 실험

### 10.1 RoPE 기본 구현

```python
import numpy as np

def precompute_rope(d, max_len, base=10000):
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2) / d))
    t = np.arange(max_len)
    freqs = np.outer(t, inv_freq)  # (max_len, d/2)
    cos = np.cos(freqs)  # (max_len, d/2)
    sin = np.sin(freqs)
    return cos, sin

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return np.concatenate([-x2, x1], axis=-1)

def apply_rope(x, cos, sin):
    # cos, sin: (seq, d/2) 혹은 broadcastable
    # half-style 구현
    d = x.shape[-1]
    cos_full = np.repeat(cos, 2, axis=-1)[..., :d]
    sin_full = np.repeat(sin, 2, axis=-1)[..., :d]
    return x * cos_full + rotate_half(x) * sin_full

np.random.seed(0)
d = 8
S = 10

cos, sin = precompute_rope(d, S)
q = np.random.randn(S, d)
k = np.random.randn(S, d)

q_rot = apply_rope(q, cos, sin)
k_rot = apply_rope(k, cos, sin)

print(f"q shape: {q.shape}")
print(f"q_rot shape: {q_rot.shape}")

# 노름 보존 확인
for i in range(S):
    print(f"위치 {i}: ||q|| = {np.linalg.norm(q[i]):.4f}, ||q_rot|| = {np.linalg.norm(q_rot[i]):.4f}")
```

### 10.2 상대 위치 확인

```python
import numpy as np

def simple_rope(x, m, d=4, base=10000):
    """단순 버전: 2D 블록마다 회전"""
    theta = base ** (-np.arange(0, d, 2) / d)
    cos = np.cos(m * theta)
    sin = np.sin(m * theta)
    out = x.copy()
    for i in range(d // 2):
        x0, x1 = x[2*i], x[2*i+1]
        out[2*i]   = x0 * cos[i] - x1 * sin[i]
        out[2*i+1] = x0 * sin[i] + x1 * cos[i]
    return out

np.random.seed(0)
d = 4
q = np.random.randn(d)
k = np.random.randn(d)

# 다양한 위치 쌍의 내적
for (m, n) in [(0, 5), (3, 8), (10, 15), (100, 105)]:
    q_m = simple_rope(q, m, d)
    k_n = simple_rope(k, n, d)
    score = q_m @ k_n
    print(f"m={m:3d}, n={n:3d}, n-m={n-m}: score = {score:.6f}")
```

**관찰:** `n - m`이 같으면 (즉 $n - m = 5$인 케이스들) score가 일치해야 한다 — 상대 위치 불변성의 증거.

### 10.3 Long-range Decay

```python
import numpy as np

def rope_inner_product(q, k, m, n, d=64, base=10000):
    theta = base ** (-np.arange(0, d, 2) / d)
    delta = n - m
    # 내적 = Σ_i (q_{2i} k_{2i} + q_{2i+1} k_{2i+1}) cos(δθ_i) + cross terms sin
    s = 0
    for i in range(d // 2):
        s += (q[2*i]*k[2*i] + q[2*i+1]*k[2*i+1]) * np.cos(delta * theta[i])
        s += (q[2*i]*k[2*i+1] - q[2*i+1]*k[2*i]) * np.sin(delta * theta[i])
    return s

np.random.seed(0)
d = 64
q = np.random.randn(d)
k = np.random.randn(d)

print("상대 위치에 따른 내적 (랜덤 q, k 평균):")
for delta in [0, 1, 5, 10, 50, 100, 500, 1000]:
    scores = [rope_inner_product(q, k, 0, delta, d) for _ in range(1)]
    print(f"δ={delta:5d}: score = {np.mean(scores):+.4f}")
```

### 10.4 NTK-Aware Interpolation

```python
import numpy as np

def ntk_aware_freqs(d, scale_factor, base=10000):
    # base를 조정하여 고주파 유지, 저주파만 스케일
    new_base = base * (scale_factor ** (d / (d - 2)))
    return 1.0 / (new_base ** (np.arange(0, d, 2) / d))

d = 128
L_train = 2048
L_extend = 8192
scale = L_extend / L_train

orig_freqs = 1.0 / (10000 ** (np.arange(0, d, 2) / d))
ntk_freqs  = ntk_aware_freqs(d, scale)

print("Frequency 비교 (처음/끝 몇 개):")
print(f"원본 high freq: {orig_freqs[:3]}")
print(f"NTK  high freq: {ntk_freqs[:3]}  (거의 동일)")
print(f"원본 low freq:  {orig_freqs[-3:]}")
print(f"NTK  low freq:  {ntk_freqs[-3:]}  (스케일됨)")
```

### 10.5 RoPE Attention 적용

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def precompute_rope(d, max_len, base=10000):
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2) / d))
    t = np.arange(max_len)
    freqs = np.outer(t, inv_freq)
    cos = np.repeat(np.cos(freqs), 2, axis=-1)[:, :d]
    sin = np.repeat(np.sin(freqs), 2, axis=-1)[:, :d]
    return cos, sin

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return np.concatenate([-x2, x1], axis=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

np.random.seed(0)
S, d = 16, 32
Q = np.random.randn(S, d)
K = np.random.randn(S, d)
V = np.random.randn(S, d)

cos, sin = precompute_rope(d, S)
Q_rot = apply_rope(Q, cos, sin)
K_rot = apply_rope(K, cos, sin)

scores = Q_rot @ K_rot.T / np.sqrt(d)
A = softmax(scores, axis=-1)
O = A @ V

print(f"RoPE Attention 출력 shape: {O.shape}")

# 위치 5의 쿼리가 위치 3의 키와 맺는 점수
print(f"score[5][3] = {scores[5, 3]:.4f}")
```

---

## 11. 요약 및 다음 절 예고

### 핵심 공식

| 요소 | 공식 |
|---|---|
| 회전 행렬 | $R_m^{(i)} = \begin{pmatrix}\cos(m\theta_i) & -\sin(m\theta_i)\\\sin(m\theta_i) & \cos(m\theta_i)\end{pmatrix}$ |
| 주파수 | $\theta_i = 10000^{-2i/d}$ |
| RoPE | $\text{RoPE}(x, m) = R_m x$ |
| 점수 | $\langle R_m q, R_n k\rangle = q^T R_{n-m} k$ |
| 노름 | $\|R_m x\| = \|x\|$ (직교) |

### 한 줄 요약

> **RoPE는 쿼리와 키를 위치-의존 회전으로 변환하여 내적이 상대 위치만의 함수가 되도록 설계된 위치 부호화이다.**

### 다음 절 예고

Chapter 7과 Deep Dive 전체의 피날레는 **Random Matrix Theory** — 고차원 신경망의 스펙트럼 거동을 설명하는 이론으로, 초기화, 일반화, NTK 등 최근 딥러닝 이론의 중심.

---

[◀ 04. Spectral Normalization](./04-spectral-normalization.md) | [📚 README](../README.md) | [06. Random Matrix Theory ▶](./06-random-matrix-theory.md)
