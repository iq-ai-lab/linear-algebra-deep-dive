# 7.6 Random Matrix Theory (무작위 행렬 이론)

> "고차원 행렬의 스펙트럼은 '랜덤'임에도 불구하고 놀랍게 보편적인 분포를 따른다."

---

## 1. 학습 목표

- **Gaussian Ensemble** (GOE, GUE, GSE)과 **Wigner 행렬**의 정의를 이해한다.
- **Wigner 반원 법칙**과 **Marchenko-Pastur 분포**를 유도(스케치)하고 실험한다.
- 신경망 **Fisher 정보**, **Hessian**의 스펙트럼이 이들 분포를 따름을 본다.
- **Tracy-Widom** 분포와 최대 특잇값의 의미.
- **초기화 전략**과 spectrum의 관계 (Glorot, He 초기화)를 수식으로 유도한다.

---

## 2. 왜 Random Matrix Theory인가?

### 2.1 딥러닝의 현실

- 거대한 가중치 행렬 (예: $10^5 \times 10^5$)
- 초기화는 **무작위**
- 학습 과정도 본질적으로 **확률적(SGD)**

고차원, 랜덤, 중첩. 이 상황에서 **결정적 스펙트럼 법칙**이 나타난다는 것이 RMT의 핵심.

### 2.2 RMT가 답하는 질문

- 초기화 직후 신경망이 **학습 가능한가**?
- Hessian의 고유값 분포는 어떻게 생겼는가?
- 입력 차원 $n \to \infty$, 깊이 $L \to \infty$에서 어떤 극한?
- 일반화 오차의 전형적(typical) 크기는?

---

## 3. Gaussian Ensembles

### 3.1 GOE (Gaussian Orthogonal Ensemble)

$W \in \mathbb{R}^{N \times N}$, 성분 $W_{ij} \sim \mathcal{N}(0, 1)$ (iid). **대칭화**:

$$
S = \frac{W + W^T}{2}
$$

대각 성분 분산 1, 비대각 분산 1/2. **GOE** 분포.

### 3.2 GUE (Gaussian Unitary Ensemble)

Hermitian 복소수 행렬, 유니터리 불변.

### 3.3 GSE (Gaussian Symplectic Ensemble)

Quaternionic, symplectic 불변. 물리에 주로.

### 3.4 Wigner 행렬 (일반)

대칭이고 대각 위 성분이 iid (반드시 Gaussian은 아님), 평균 0, 분산 1인 행렬.

---

## 4. Wigner의 반원 법칙

### 4.1 정리 (Semicircle Law)

**정리 7.6.1 (Wigner, 1958).** $N \times N$ Wigner 행렬 $W_N$의 고유값을 $N^{1/2}$로 스케일:

$$
\lambda_i(W_N / \sqrt{N})
$$

이들의 경험적 분포는 $N \to \infty$일 때 다음 **반원 분포**로 수렴한다:

$$
\rho(\lambda) = \frac{1}{2\pi}\sqrt{4 - \lambda^2}, \quad \lambda \in [-2, 2]
$$

### 4.2 증명 스케치: Stieltjes 변환

Stieltjes 변환 $m(z) = \mathbb{E}[\frac{1}{N}\text{tr}((W - zI)^{-1})]$을 정의.

**자기-일관 방정식(self-consistent equation)**:

$$
m(z) = -\frac{1}{z + m(z)}
$$

즉 $m^2 + z m + 1 = 0$, $m(z) = \frac{-z + \sqrt{z^2 - 4}}{2}$.

역변환하면 반원 밀도 $\rho(\lambda) = \frac{1}{\pi} \text{Im}\, m(\lambda - i0^+)$가 나옴.

### 4.3 유니버셜리티

**놀라운 사실**: Wigner 분포는 성분의 개별 분포에 무관 (평균, 분산만). 베르누이든 가우시안이든 같은 극한.

---

## 5. Marchenko-Pastur 분포

### 5.1 Sample Covariance의 스펙트럼

$X \in \mathbb{R}^{p \times n}$, 성분 iid $\mathcal{N}(0, 1)$. Sample 공분산

$$
\hat \Sigma = \frac{1}{n} X X^T \in \mathbb{R}^{p \times p}
$$

### 5.2 Marchenko-Pastur 정리

**정리 7.6.2 (Marchenko-Pastur, 1967).** $p/n \to c \in (0, 1]$일 때 $\hat \Sigma$의 경험적 고유값 분포는

$$
\rho_{MP}(\lambda) = \frac{1}{2\pi c \lambda}\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}
$$

로 수렴. 여기서 $\lambda_\pm = (1 \pm \sqrt c)^2$.

$c > 1$이면 0에서 $(1 - 1/c)$의 질량을 가진 점 질량 추가.

### 5.3 해석

- $c = 0$: 모든 고유값이 1 (충분한 데이터)
- $c = 1$: 고유값 0부터 4까지 퍼짐
- 큰 $c$: 스펙트럼이 0과 큰 값으로 양극화

**실질적 시사**: 데이터가 충분치 않으면 공분산 추정이 심각히 왜곡.

### 5.4 응용: PCA와 Shrinkage

- **경험적 PCA**: 상위 고유값이 **과대추정**됨
- **해결**: eigenvalue shrinkage (Ledoit-Wolf 등)

---

## 6. 극값 분포: Tracy-Widom

### 6.1 최대 고유값

Wigner 행렬의 최대 고유값 $\lambda_{\max}$는 반원의 edge $\lambda = 2$ 근처에 집중. 그러나 **Edge fluctuation**은 Gaussian이 아님.

### 6.2 Tracy-Widom 정리

**정리 7.6.3 (Tracy-Widom, 1994).** GUE의 경우

$$
\text{Prob}[\lambda_{\max} \le 2 + s/N^{2/3}] \to F_2(s)
$$

$F_2$는 Painlevé II 방정식의 해로 표현되는 **Tracy-Widom 분포**. GOE, GSE에는 다른 $F_1, F_4$.

### 6.3 시사점

- $\lambda_{\max}$의 편차는 $N^{-2/3}$ 스케일 (Gaussian은 $N^{-1/2}$)
- **Phase transition**: Signal + noise 모델에서 신호가 임계치 위면 $\lambda_{\max}$가 반원 밖으로

---

## 7. 신경망에 적용

### 7.1 초기화: Glorot/He

**Glorot**: $W_{ij} \sim \mathcal{N}(0, 2/(n_{\text{in}} + n_{\text{out}}))$.

**He**: $W_{ij} \sim \mathcal{N}(0, 2/n_{\text{in}})$ (ReLU용).

이유: 각 층의 출력 분산이 1에 머물도록. Marchenko-Pastur 관점에서 **스펙트럼의 우측 edge가 고정**되도록.

### 7.2 Jacobian의 스펙트럼 (Pennington et al.)

ReLU-activated 신경망의 입력-출력 Jacobian:

$$
J = D_L W_L D_{L-1} W_{L-1} \cdots D_1 W_1
$$

각 $W_l$은 랜덤, $D_l$은 활성 마스크.

**정리 7.6.4.** $\sigma_{\max}(J) / \sigma_{\min}(J)$의 분포가 $L$에 따라 어떻게 변하는지 RMT로 분석. "**Dynamical isometry**"는 $\sigma_{\max} \approx \sigma_{\min}$를 보장하는 초기화.

### 7.3 Hessian의 스펙트럼

훈련된 신경망의 Hessian $H$의 고유값 분포:

- **Bulk**: 0 근처에 집중된 연속 분포 (MP와 유사)
- **Outliers**: 소수의 큰 고유값 — 학습된 방향

이는 **flat minima** 이론과 연결: 대부분 방향으로 손실이 평탄.

### 7.4 NTK 관점

NTK 행렬 $\Theta_{ij} = \nabla_\theta f(x_i, \theta)^T \nabla_\theta f(x_j, \theta)$는 무한 너비 극한에서 결정적. 유한 너비에서는 MP와 유사한 스펙트럼.

---

## 8. 딥러닝에서의 RMT 활용

### 8.1 Loss landscape의 곡률

Hessian spectrum으로 local minima의 **평탄성**, **안장점의 수** 추정. 학습률 선택에 영향.

### 8.2 Generalization

Double Descent 현상: 모델 크기가 보간 임계치 근처에서 test error가 급증 후 다시 감소. RMT로 설명 가능 (Belkin et al.).

### 8.3 Spectral Gap과 수렴 속도

$\lambda_2 - \lambda_1$의 크기가 **power iteration, Lanczos, Krylov 방법**의 수렴 속도 결정.

### 8.4 Compression 한계

$k$-랭크 근사의 오차가 RMT로 예측 가능. $\sigma_{k+1}^2 + \cdots + \sigma_N^2$ (Eckart-Young).

---

## 9. 최신 주제

### 9.1 Heavy-tailed Self-Regularization (Martin-Mahoney)

훈련된 심층 네트워크 가중치 행렬의 특잇값이 **power-law** (heavy-tailed)를 따름. MP가 아닌 다른 universality class. "**학습이 자기-정규화**"를 구현.

### 9.2 Free Probability

큰 랜덤 행렬들의 곱의 스펙트럼을 계산하는 대수적 framework. 딥 신경망 분석의 현대 도구.

### 9.3 Spin Glass와 Landscape

Loss landscape를 spin glass로 모델링. 낮은 loss의 local minima가 지수적으로 많지만 "비슷한" 가치.

---

## 10. Python 실험

### 10.1 Wigner 반원 법칙

```python
import numpy as np
import matplotlib.pyplot as plt

N = 2000
W = np.random.randn(N, N)
W = (W + W.T) / np.sqrt(2)  # 대칭화, 분산 조정
eigs = np.linalg.eigvalsh(W) / np.sqrt(N)

# 이론 PDF
x = np.linspace(-2, 2, 200)
pdf = np.sqrt(np.maximum(0, 4 - x**2)) / (2 * np.pi)

print(f"최소/최대 고유값: {eigs.min():.3f} / {eigs.max():.3f}")
print(f"이론 범위: [-2, 2]")

# 히스토그램 통계
bins = np.linspace(-2.5, 2.5, 50)
counts, _ = np.histogram(eigs, bins=bins, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# 이론과 비교
pdf_at_centers = np.sqrt(np.maximum(0, 4 - bin_centers**2)) / (2 * np.pi)
rel_error = np.abs(counts - pdf_at_centers) / (pdf_at_centers + 0.01)
print(f"평균 상대오차: {rel_error[pdf_at_centers > 0.05].mean():.3f}")
```

### 10.2 Marchenko-Pastur

```python
import numpy as np

def marchenko_pastur_test(p, n):
    X = np.random.randn(p, n)
    Sigma = X @ X.T / n
    eigs = np.linalg.eigvalsh(Sigma)
    
    c = p / n
    lam_plus = (1 + np.sqrt(c))**2
    lam_minus = (1 - np.sqrt(c))**2
    return eigs, lam_plus, lam_minus

for ratio in [0.1, 0.5, 0.9]:
    p = int(500 * ratio)
    n = 500
    eigs, lp, lm = marchenko_pastur_test(p, n)
    print(f"c = {ratio}: 고유값 범위 [{eigs.min():.3f}, {eigs.max():.3f}], 이론 [{lm:.3f}, {lp:.3f}]")
```

### 10.3 Tracy-Widom (최대 고유값)

```python
import numpy as np

def max_eig_distribution(N, n_trials=200):
    max_eigs = []
    for _ in range(n_trials):
        W = np.random.randn(N, N) / np.sqrt(2)
        W = (W + W.T) / np.sqrt(2)
        max_eigs.append(np.linalg.eigvalsh(W).max() / np.sqrt(N))
    return np.array(max_eigs)

N_values = [100, 400, 1600]
for N in N_values:
    max_eigs = max_eig_distribution(N)
    # Rescale: (λ_max - 2) * N^(2/3)
    rescaled = (max_eigs - 2) * N**(2/3)
    print(f"N={N:4d}: (λ_max - 2)*N^(2/3) 분포 평균={rescaled.mean():.3f}, std={rescaled.std():.3f}")
    # Tracy-Widom 분포의 평균은 약 -1.77 (F_1)
```

### 10.4 신경망 Jacobian의 스펙트럼

```python
import numpy as np

def compute_jacobian_spectrum(d, L, init_type='he'):
    """Feed-forward 네트워크 초기화 후 Jacobian의 조건수"""
    Ws = []
    for l in range(L):
        if init_type == 'he':
            W = np.random.randn(d, d) * np.sqrt(2 / d)
        elif init_type == 'glorot':
            W = np.random.randn(d, d) * np.sqrt(2 / (2 * d))
        elif init_type == 'naive':
            W = np.random.randn(d, d)
        Ws.append(W)
    
    x = np.random.randn(d)
    J = np.eye(d)
    for W in Ws:
        x = np.maximum(0, W @ x)
        D = np.diag((x > 0).astype(float))
        J = D @ W @ J
    
    s = np.linalg.svd(J, compute_uv=False)
    return s

np.random.seed(0)
d = 100
for L in [5, 10, 20]:
    for init in ['he', 'naive']:
        s = compute_jacobian_spectrum(d, L, init)
        print(f"L={L:2d}, init={init}: σ_max/σ_min = {s[0]/s[-1]:.2e}")
```

### 10.5 Outlier Analysis

```python
import numpy as np

def add_signal(N, signal_strength):
    """Wigner 행렬 + rank-1 신호"""
    W = np.random.randn(N, N) / np.sqrt(2)
    W = (W + W.T) / np.sqrt(2)
    u = np.random.randn(N)
    u /= np.linalg.norm(u)
    signal = signal_strength * np.outer(u, u)
    return (W + signal) / np.sqrt(N), signal_strength / np.sqrt(N)

N = 1000
for s in [0.5, 1.0, 2.0, 5.0]:  # Wigner는 [-2, 2], 임계는 s_norm > 1
    M, s_norm = add_signal(N, s * np.sqrt(N))
    eigs = np.linalg.eigvalsh(M)
    print(f"signal norm = {s_norm:.3f}: 최대 고유값 = {eigs[-1]:.3f} (Wigner edge = 2)")
```

### 10.6 Heavy-tailed in Trained Networks (모의)

```python
import numpy as np

def simulate_power_law_weights(N, alpha):
    """Pareto-like 분포로 가중치 생성 (heavy-tailed)"""
    W = np.random.pareto(alpha, size=(N, N)) * np.sign(np.random.randn(N, N))
    W = W / np.sqrt(N)
    return W

np.random.seed(0)
N = 500

# Gaussian (thin-tailed)
W_gauss = np.random.randn(N, N) / np.sqrt(N)
s_gauss = np.linalg.svd(W_gauss, compute_uv=False)

# Heavy-tailed
W_heavy = simulate_power_law_weights(N, alpha=2.5)
s_heavy = np.linalg.svd(W_heavy, compute_uv=False)

print(f"Gaussian: 상위 5개 σ = {s_gauss[:5]}")
print(f"          최소 σ = {s_gauss[-1]:.4f}")
print(f"Heavy:    상위 5개 σ = {s_heavy[:5]}")
print(f"          최소 σ = {s_heavy[-1]:.4f}")
```

---

## 11. 요약과 전체 Deep Dive 마무리

### 핵심 결과

| 법칙 | 분포 | 조건 |
|---|---|---|
| Wigner semicircle | $\frac{1}{2\pi}\sqrt{4 - \lambda^2}$ | 대칭 Wigner, $\lambda / \sqrt N$ |
| Marchenko-Pastur | $\frac{\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)}}{2\pi c \lambda}$ | Sample covariance |
| Tracy-Widom | $F_\beta(s)$ | Edge fluctuation $N^{2/3}$ |

### 한 줄 요약

> **Random Matrix Theory는 고차원 랜덤성 속에서 결정적인 스펙트럼 법칙을 발견하고, 이를 신경망의 초기화, 학습, 일반화에 적용한다.**

### Chapter 7 정리

우리는 Chapter 7에서

1. **Attention의 선형대수** — Transformer의 기반
2. **역전파와 VJP** — 자동미분의 수학
3. **BatchNorm** — 정규화의 선형대수
4. **Spectral Normalization** — Lipschitz 제약
5. **RoPE** — 위치 부호화의 회전 변환
6. **Random Matrix Theory** — 확률적 선형대수

를 통해 **현대 딥러닝의 수학적 기초**를 완성했다.

### Deep Dive 전체 마무리

7개 장, 41개 문서, 수천 개의 수식과 증명을 거쳐 우리는

- **제1장**: 벡터공간의 공리에서 출발
- **제2장**: 행렬 분해의 동물원 탐방
- **제3장**: 고유값 이론의 모든 것
- **제4장**: SVD의 완벽한 지배
- **제5장**: 내적 구조의 기하
- **제6장**: 텐서의 다중선형 대수
- **제7장**: AI/ML의 실전 응용

까지 완주했다. "공리에서 증명까지, 모든 것을 직접 유도한다"는 철학이 지켜졌다.

선형대수는 **수학의 가장 성공한 확장**이자 **AI의 모국어**다. 이 여정이 독자에게 선형대수를 **사용하는 도구**가 아닌, **사고하는 언어**로 만들어 주었기를 바란다.

---

[◀ 05. RoPE](./05-rope.md) | [📚 README](../README.md) | [🏁 Deep Dive 완주](../README.md)
