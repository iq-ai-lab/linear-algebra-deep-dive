# 7.4 Spectral Normalization

> "가장 큰 특잇값을 1로 고정하는 것 — Lipschitz 상수를 제어하는 가장 직접적인 방법."

---

## 1. 학습 목표

- **Spectral norm** $\|W\|_2 = \sigma_{\max}(W)$가 행렬의 Lipschitz 상수임을 이해한다.
- **1-Lipschitz 제약**과 신경망의 안정성 관계를 유도한다.
- **Spectral Normalization** (Miyato et al. 2018)의 알고리즘 (power iteration)을 엄밀히 유도한다.
- GAN의 **Wasserstein 거리**와 관계를 본다.
- Lipschitz 제약의 대안(weight clipping, gradient penalty)과 비교한다.

---

## 2. Lipschitz 상수와 스펙트럼 노름

### 2.1 Lipschitz 상수 복습

함수 $f: \mathbb{R}^n \to \mathbb{R}^m$이 **$L$-Lipschitz**라는 것은

$$
\|f(x) - f(y)\| \le L \|x - y\|, \quad \forall x, y
$$

$L$의 최소값을 Lipschitz 상수 $\text{Lip}(f)$라 한다.

### 2.2 선형 함수의 Lipschitz 상수

$f(x) = Wx$에 대해

$$
\|Wx - Wy\| = \|W(x-y)\| \le \|W\|_2 \|x - y\|
$$

$\|W\|_2 = \sigma_{\max}(W)$는 연산자 노름 (제4장 참고). 따라서

$$
\text{Lip}(Wx) = \sigma_{\max}(W)
$$

등호를 얻는 $x, y$는 최대 특잇값의 특이벡터 방향으로 택하면 달성.

### 2.3 1-Lipschitz 활성함수

- ReLU: $|ReLU(x) - ReLU(y)| \le |x - y|$. 1-Lipschitz.
- LeakyReLU: 기울기 $\max(1, |\alpha|)$.
- Sigmoid: $\sup |\sigma'| = 1/4$, 1/4-Lipschitz.
- Tanh: 1-Lipschitz.

### 2.4 합성 함수의 Lipschitz

$$
\text{Lip}(g \circ f) \le \text{Lip}(g) \cdot \text{Lip}(f)
$$

따라서 신경망 $f = f_L \circ \cdots \circ f_1$의 Lipschitz 상수는 각 층의 곱:

$$
\text{Lip}(f) \le \prod_{l=1}^L \sigma_{\max}(W_l) \cdot \text{Lip}(\text{activation})
$$

---

## 3. Spectral Normalization의 정의

### 3.1 핵심 아이디어

각 층의 가중치를 **spectral norm으로 나누어** 1-Lipschitz로 만든다:

$$
W_{SN} = \frac{W}{\sigma_{\max}(W)}
$$

### 3.2 전체 Lipschitz

각 층이 1-Lipschitz이면 합성도 1-Lipschitz (activation도 1-Lipschitz라고 가정 시):

$$
\text{Lip}(f_{SN}) \le 1
$$

### 3.3 학습 중 계산

$\sigma_{\max}(W)$를 매 iteration마다 **SVD**로 계산하면 너무 비쌈. **Power iteration** (1번만으로 충분한 경우 많음).

---

## 4. Power Iteration의 재방문

### 4.1 알고리즘

$W \in \mathbb{R}^{m \times n}$에 대해 임의의 $u_0 \in \mathbb{R}^m$, $v_0 \in \mathbb{R}^n$으로 시작:

```
for step 1, 2, ...:
    v := W^T u / ||W^T u||
    u := W v / ||W v||
```

수렴 후 $\sigma_{\max}(W) \approx u^T W v$.

### 4.2 수렴 속도

비율 $(\sigma_2 / \sigma_1)^{2k}$로 수렴 (제3.5절). 딥러닝에서는 많은 경우 $\sigma_2/\sigma_1$이 크지 않아 **1회 반복**도 효과적 (Miyato 등의 관찰).

### 4.3 Warm-starting

이전 iteration의 $u, v$를 이번 iteration의 시작점으로. 가중치가 조금씩 변하므로 효율적.

---

## 5. Miyato et al.의 Spectral Normalization

### 5.1 알고리즘 (그대로)

Layer 매 호출 때:

1. $v \leftarrow W^T u / \|W^T u\|$
2. $u \leftarrow W v / \|W v\|$
3. $\sigma \leftarrow u^T W v$
4. $\hat W = W / \sigma$
5. 출력 = $\hat W x$

### 5.2 Backward 주의

$u, v$는 **gradient를 흘려보내지 않는다** (상수 취급). 그래야 $\sigma$가 학습 중에 안정적으로 추정.

파라미터 그래디언트:

$$
\frac{\partial L}{\partial W_{ij}} = \frac{1}{\sigma}\frac{\partial L}{\partial \hat W_{ij}} - \frac{1}{\sigma^2}\sum_{kl} \frac{\partial L}{\partial \hat W_{kl}} W_{kl} u_i v_j
$$

### 5.3 Conv에서의 SN

$W \in \mathbb{R}^{C_o \times C_i \times k \times k}$를 $W \in \mathbb{R}^{C_o \times (C_i k^2)}$로 reshape 후 SN 적용. 엄밀히는 이는 **채널-방향 linear operator**의 norm이며, 공간 컨볼루션의 정확한 Lipschitz와는 다르나 실용적 근사.

---

## 6. GAN에서의 활용

### 6.1 Wasserstein GAN (WGAN)

GAN의 판별자 $D$가 1-Lipschitz를 만족하면 Wasserstein-1 거리를 근사:

$$
W_1(p_{\text{real}}, p_{\text{gen}}) = \sup_{\|D\|_{\text{Lip}} \le 1} [\mathbb{E}_{x \sim p_{\text{real}}}[D(x)] - \mathbb{E}_{x \sim p_{\text{gen}}}[D(x)]]
$$

**Kantorovich-Rubinstein 쌍대**에 의해. 이 제약을 **어떻게 구현하는가**가 관건:

1. **Weight clipping** (WGAN 원 논문): $W$의 각 성분을 $[-c, c]$로 제한. 단순하지만 과도한 제약.
2. **Gradient Penalty** (WGAN-GP): $\mathbb{E}[(\|\nabla D(x)\| - 1)^2]$ 항 추가. 반쯤만 효과적.
3. **Spectral Normalization** (SN-GAN): 각 층의 spectral norm 정규화. **명시적**, 안정적.

### 6.2 SN의 GAN 안정화 효과

- 모드 붕괴(mode collapse) 감소
- 큰 학습률 사용 가능
- Adam과 호환 (GP는 종종 불안정)

---

## 7. 지도학습에서의 SN

### 7.1 적대적 강건성

Lipschitz 제약은 작은 입력 변화에 대한 출력 변화를 bound. 이는 **adversarial attack**에 저항성을 제공.

### 7.2 NTK와의 관계

Neural Tangent Kernel 분석에서 각 층의 spectrum이 NTK의 eigenvalue 구조를 결정. SN은 이 spectrum을 균일하게.

### 7.3 과적합 방지

Lipschitz 제약은 효과적으로 **용량(capacity)** 을 제한. 따라서 **정규화 역할**도.

---

## 8. 다른 Spectral 기법

### 8.1 Orthogonal Regularization

$\|W^T W - I\|_F^2$를 손실에 추가하여 $W$를 **직교 근사**로. 모든 특잇값이 1.

### 8.2 Weight Standardization

각 필터의 **평균을 빼고 분산을 1로**:

$$
W_{ij} \leftarrow (W_{ij} - \bar W_i) / \text{std}(W_i)
$$

BatchNorm과 결합하여 사용.

### 8.3 Beta Lipschitz

모든 특잇값을 $[\alpha, \beta]$ 구간으로 고정. $\sigma_{\min}$도 제약 → 그래디언트 소실 방지.

---

## 9. 계산 복잡도

### 9.1 SVD vs Power Iteration

- SVD: $O(mn^2)$ per layer per step
- Power iteration (1 step): $O(mn)$ per layer per step

큰 네트워크에서는 후자가 필수.

### 9.2 메모리

- SVD: $O(mn + n^2)$ (U, V, S)
- Power iteration: $O(m + n)$ (벡터만)

### 9.3 병렬화

Power iteration은 행렬-벡터곱 2개, GPU에서 매우 빠름.

---

## 10. Python 실험

### 10.1 Power Iteration 구현

```python
import numpy as np

def power_iteration(W, u_init=None, n_iter=10):
    m, n = W.shape
    u = np.random.randn(m) if u_init is None else u_init
    u = u / np.linalg.norm(u)
    for _ in range(n_iter):
        v = W.T @ u
        v = v / (np.linalg.norm(v) + 1e-12)
        u = W @ v
        u = u / (np.linalg.norm(u) + 1e-12)
    sigma = u @ W @ v
    return sigma, u, v

np.random.seed(0)
W = np.random.randn(20, 15)

sigma_pi, u, v = power_iteration(W, n_iter=50)
sigma_svd = np.linalg.svd(W, compute_uv=False)[0]

print(f"Power iteration: {sigma_pi:.6f}")
print(f"SVD:             {sigma_svd:.6f}")
print(f"오차: {abs(sigma_pi - sigma_svd):.2e}")
```

### 10.2 수렴 속도 측정

```python
import numpy as np

def power_iteration_track(W, n_iter=50):
    m, n = W.shape
    u = np.random.randn(m); u /= np.linalg.norm(u)
    sigmas = []
    for _ in range(n_iter):
        v = W.T @ u; v /= np.linalg.norm(v) + 1e-12
        u = W @ v; u /= np.linalg.norm(u) + 1e-12
        sigmas.append(u @ W @ v)
    return sigmas

np.random.seed(0)
W = np.random.randn(50, 50)
sigmas = power_iteration_track(W, n_iter=30)
sigma_true = np.linalg.svd(W, compute_uv=False)[0]

errors = [abs(s - sigma_true) for s in sigmas]
for i, e in enumerate(errors[:10]):
    print(f"iter {i+1}: error = {e:.2e}")
```

### 10.3 Spectral Normalization Layer

```python
import numpy as np

class SpectralNormLinear:
    def __init__(self, W):
        self.W = W
        self.u = np.random.randn(W.shape[0])
        self.u /= np.linalg.norm(self.u)

    def forward(self, x):
        # Power iteration 1 step
        v = self.W.T @ self.u
        v /= np.linalg.norm(v) + 1e-12
        u_new = self.W @ v
        u_new /= np.linalg.norm(u_new) + 1e-12
        sigma = u_new @ self.W @ v
        
        # Warm start
        self.u = u_new
        
        W_sn = self.W / sigma
        return W_sn @ x, sigma

np.random.seed(0)
W = np.random.randn(64, 32) * 2
layer = SpectralNormLinear(W)

# 여러 iteration 진행
for it in range(20):
    x = np.random.randn(32)
    y, sigma = layer.forward(x)
    if it % 5 == 0:
        svd_sigma = np.linalg.svd(W, compute_uv=False)[0]
        print(f"iter {it}: 추정 sigma={sigma:.4f}, 참값={svd_sigma:.4f}")
```

### 10.4 Lipschitz 상수 검증

```python
import numpy as np

def lipschitz_estimate(f, x_dim, n_samples=1000):
    """무작위 샘플링으로 Lipschitz 하한 추정"""
    max_ratio = 0
    for _ in range(n_samples):
        x = np.random.randn(x_dim)
        y = np.random.randn(x_dim)
        ratio = np.linalg.norm(f(x) - f(y)) / np.linalg.norm(x - y)
        max_ratio = max(max_ratio, ratio)
    return max_ratio

np.random.seed(0)
W = np.random.randn(10, 10)

# 일반
f_normal = lambda x: W @ x
L_normal = lipschitz_estimate(f_normal, 10)

# SN 적용
sigma = np.linalg.svd(W, compute_uv=False)[0]
W_sn = W / sigma
f_sn = lambda x: W_sn @ x
L_sn = lipschitz_estimate(f_sn, 10)

print(f"일반 W: Lipschitz 추정 = {L_normal:.4f} (true σ_max = {sigma:.4f})")
print(f"SN  W:  Lipschitz 추정 = {L_sn:.4f} (1이어야 함)")
```

### 10.5 GAN 안정성 (개념적)

```python
import numpy as np

def discriminator_without_sn(x, Ws):
    for W in Ws:
        x = np.tanh(W @ x)
    return x

def discriminator_with_sn(x, Ws):
    for W in Ws:
        sigma = np.linalg.svd(W, compute_uv=False)[0]
        W_sn = W / sigma
        x = np.tanh(W_sn @ x)
    return x

np.random.seed(0)
d = 10
L = 5

Ws_stable = [np.random.randn(d, d) * 0.5 for _ in range(L)]
Ws_large  = [np.random.randn(d, d) * 3.0 for _ in range(L)]  # 큰 가중치

x1 = np.random.randn(d)
x2 = x1 + np.random.randn(d) * 0.01  # 작은 perturbation

for label, Ws in [("small W", Ws_stable), ("large W", Ws_large)]:
    y1 = discriminator_without_sn(x1, Ws)
    y2 = discriminator_without_sn(x2, Ws)
    change_no_sn = np.linalg.norm(y1 - y2)
    
    y1_sn = discriminator_with_sn(x1, Ws)
    y2_sn = discriminator_with_sn(x2, Ws)
    change_sn = np.linalg.norm(y1_sn - y2_sn)
    
    print(f"{label}: no SN Δy = {change_no_sn:.4f}, with SN Δy = {change_sn:.4f}")
```

---

## 11. 요약 및 다음 절 예고

### 핵심 공식

| 요소 | 공식 |
|---|---|
| Lipschitz | $\text{Lip}(Wx) = \sigma_{\max}(W) = \|W\|_2$ |
| SN | $W_{SN} = W / \sigma_{\max}(W)$ |
| Power iter | $u \leftarrow Wv/\|Wv\|$, $v \leftarrow W^T u / \|W^T u\|$ |
| 신경망 bound | $\text{Lip}(f) \le \prod_l \sigma_{\max}(W_l)$ |

### 한 줄 요약

> **Spectral Normalization은 각 층의 최대 특잇값으로 가중치를 나눠 네트워크 전체를 1-Lipschitz로 제약하는 기법이다.**

### 다음 절 예고

다음은 **Rotary Position Embedding (RoPE)** — 선형대수의 회전 변환을 위치 부호화에 적용한 우아한 설계. 현대 LLM의 표준.

---

[◀ 03. BatchNorm](./03-batchnorm.md) | [📚 README](../README.md) | [05. RoPE ▶](./05-rope.md)
