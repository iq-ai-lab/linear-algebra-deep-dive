# Ch4-06. Randomized SVD: 거대 행렬의 빠른 저랭크 분해

> "무작위 투영만으로도 대부분의 데이터 구조를 포착할 수 있다. (Johnson-Lindenstrauss의 행렬 버전.)"

## 📌 학습 목표

- Randomized SVD 알고리즘의 단계별 유도.
- 왜 무작위 Gaussian 투영이 **$A$의 range를 근사 포착**하는가.
- Halko-Martinsson-Tropp (2011) 오차 분석.
- **Power iteration 후처리**로 감쇠 속도를 향상하는 트릭.
- 실무: 대규모 추천시스템, topic modeling, Gaussian process 가속.

---

## 🎯 핵심 질문

> **질문 1**: 왜 Gaussian random matrix로 range를 근사할 수 있는가?
> **질문 2**: Oversampling parameter $p$의 역할은?
> **질문 3**: Power iteration이 왜 정확도를 지수적으로 개선하는가?

---

## 1. 동기

### 1.1 전통 SVD의 한계

$A \in \mathbb{R}^{m \times n}$의 full SVD: $O(\min(mn^2, m^2 n))$. $m, n \gg 10^4$에서 intractable.

**관찰**: 많은 실세계 행렬은 **저랭크 근사**로 잘 설명됨 (Eckart-Young의 상위 $k$ 특이값이 spectrum 대부분을 차지).

### 1.2 핵심 아이디어

1. $A$의 치역 $R(A)$을 **무작위 투영**으로 근사
2. 근사된 부분공간에서 소규모 SVD 수행
3. 원 공간으로 복원

**복잡도**: $O(mnk + k^3)$으로 $k \ll \min(m, n)$에서 **선형 시간**.

---

## 2. Stage A: Range Finder

### 알고리즘 A (기본 형태)

입력: $A \in \mathbb{R}^{m \times n}$, 목표 rank $k$, oversampling $p$ (기본 $p \geq 5$).

1. **Draw random test matrix**: $\Omega \in \mathbb{R}^{n \times (k+p)}$, 각 성분 $\sim \mathcal{N}(0, 1)$ iid
2. **Form sample matrix**: $Y = A\Omega \in \mathbb{R}^{m \times (k+p)}$
3. **QR 분해**: $Y = QR$, $Q \in \mathbb{R}^{m \times (k+p)}$ 정규직교

$Q$가 $R(A)$의 **근사 정규직교 기저**.

### 직관

$A$의 SVD $A = U\Sigma V^T$에서 $A\Omega = U\Sigma V^T \Omega$. $V^T \Omega$는 랜덤 직교 투영 (Gaussian → Gaussian, 회전 불변)이므로 $U\Sigma$의 성분을 랜덤 선형결합하여 **$U$ (range)의 sample** 생성. 상위 $k$ 특이값이 dominant하면 $A\Omega$는 $U_k$의 span을 잘 근사.

### 정리 2.1 (Halko-Martinsson-Tropp 2011, 기본 오차 한계)

$k + p \geq 2$일 때:

$$
\mathbb{E}\|A - Q Q^T A\|_F \leq \left( 1 + \frac{k}{p-1} \right)^{1/2} \left(\sum_{j > k} \sigma_j^2\right)^{1/2}
$$

### 해석

- 오차는 Eckart-Young 최적 오차 $\sqrt{\sum_{j>k}\sigma_j^2}$의 $\sqrt{1 + k/(p-1)}$ 배
- $p = 5$에서 $\sqrt{1 + k/4}$, $k = 50$이면 ≈ $\sqrt{13.5} ≈ 3.7$배
- Oversampling을 조금만 키워도 오차 급격히 감소

### 증명 아이디어

$Y = A\Omega$이 $A$의 row space의 "대표적 방향"을 샘플링. $\Omega$의 smallest singular value (Marchenko-Pastur 법칙으로 bound) → tail bound로 control.

---

## 3. Stage B: SVD on Small Matrix

### 알고리즘 B

1. $B = Q^T A \in \mathbb{R}^{(k+p) \times n}$
2. $B$의 SVD: $B = \tilde U \Sigma V^T$ (작은 행렬, 빠름)
3. $U = Q \tilde U$
4. Truncate to rank $k$: $U_k, \Sigma_k, V_k$

출력: $A \approx U_k \Sigma_k V_k^T$.

### 정당성

$A \approx QQ^T A = Q B = Q \tilde U \Sigma V^T = U \Sigma V^T$.

---

## 4. Power Iteration 개선

### 문제

$A$의 특이값이 완만히 감소 (slow decay)하면 basic randomized SVD의 오차가 크다. $q$회 power iteration:

$$
Y = (AA^T)^q A\Omega
$$

$(AA^T)^q$는 특이값을 $\sigma_i^{2q+1}$로 가속: 큰 $\sigma$는 더 크게, 작은 $\sigma$는 상대적으로 줄음.

### 정리 4.1 (Power Iteration Error)

$$
\mathbb{E}\|A - QQ^T A\|_2 \leq \left(1 + \sqrt{k/(p-1)} + \frac{e\sqrt{k+p}}{p}\right)^{1/(2q+1)} \sigma_{k+1}
$$

$q$ 증가 시 $(\cdot)^{1/(2q+1)} \to 1$ (지수 감소).

### 수치 안정화

Power iteration 시 각 단계마다 QR 재분해해야 수치 안정:

```
Y = A Ω
Q, _ = qr(Y)
for q in range(power_iter):
    Y = A^T Q
    Q, _ = qr(Y)
    Y = A Q
    Q, _ = qr(Y)
```

---

## 5. Truncated vs Randomized SVD 오차

### Deterministic (truncated, Eckart-Young)

$$
\|A - A_k\|_2 = \sigma_{k+1}
$$

### Randomized (p oversampling, q power iter)

$$
\|A - \tilde A_k\|_2 \leq C(k, p, q) \cdot \sigma_{k+1}
$$

$C \to 1$ as $p, q \to \infty$. 실무에서 $p = 10, q = 2$로 거의 optimal.

---

## 6. 응용

### 6.1 추천 시스템

Netflix 규모 ($10^5 \times 10^4$): randomized SVD로 $k = 50$ 분해. 단일 서버에서 몇 분.

### 6.2 LSA / Topic Modeling

Document-term 행렬 ($10^5$ 문서 × $10^5$ 단어). Randomized SVD로 latent semantic 공간 추출.

### 6.3 Neural Network 압축

Weight 행렬 $W \in \mathbb{R}^{d \times d}$를 $W \approx U V^T$ ($U \in \mathbb{R}^{d \times k}$, $V \in \mathbb{R}^{d \times k}$)로 분해 → $2dk$ 저장, 추론 시 $2dk$ 곱셈. 대형 모델 lightweight 추론.

### 6.4 Gaussian Process

$N \times N$ 커널 행렬 분해 가속. Nyström method, Random Fourier features와 결합.

---

## 7. 확장: CUR, Interpolative Decomposition

### CUR

$A \approx C U R$, $C$: $A$의 열 몇 개, $R$: $A$의 행 몇 개, $U$: 작은 중간. 해석 가능성 (실제 데이터 서브샘플링).

### Interpolative Decomposition (ID)

$A \approx A_J Z$, $A_J$: $A$의 선택된 열, $Z$: 계수 행렬. Skeletonization이라고도.

---

## 8. Python 실험

### 8.1 기본 Randomized SVD

```python
import numpy as np

def randomized_svd(A, k, p=10, q=2):
    m, n = A.shape
    # Stage A: range finder with power iteration
    Omega = np.random.randn(n, k + p)
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    for _ in range(q):
        Y = A.T @ Q
        Q, _ = np.linalg.qr(Y)
        Y = A @ Q
        Q, _ = np.linalg.qr(Y)
    # Stage B: small SVD
    B = Q.T @ A
    U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    return U[:, :k], s[:k], Vt[:k]

# Test
np.random.seed(0)
m, n = 1000, 800
A_low = np.random.randn(m, 20) @ np.random.randn(20, n)
A_noise = 0.01 * np.random.randn(m, n)
A = A_low + A_noise

import time
t0 = time.time()
U_r, s_r, Vt_r = randomized_svd(A, k=30, p=10, q=2)
print(f"Randomized SVD: {time.time()-t0:.3f} s")

t0 = time.time()
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(f"Full SVD:       {time.time()-t0:.3f} s")

# Compare top-30 singular values
print("Top-5 s (random):", s_r[:5])
print("Top-5 s (full):  ", s[:5])
```

### 8.2 Oversampling Effect

```python
errs_basic = []
errs_improved = []
ps = [0, 2, 5, 10, 20]
for p in ps:
    # Basic (no power iter)
    U_r, s_r, Vt_r = randomized_svd(A, k=20, p=p, q=0)
    A_k = U_r @ np.diag(s_r) @ Vt_r
    errs_basic.append(np.linalg.norm(A - A_k, 2))
    # Improved (q=2)
    U_r, s_r, Vt_r = randomized_svd(A, k=20, p=p, q=2)
    A_k = U_r @ np.diag(s_r) @ Vt_r
    errs_improved.append(np.linalg.norm(A - A_k, 2))

print("Basic    :", errs_basic)
print("q=2      :", errs_improved)
print("Optimal σ_21:", s[20])
```

### 8.3 Scikit-learn

```python
from sklearn.utils.extmath import randomized_svd
U_sk, s_sk, Vt_sk = randomized_svd(A, n_components=30, random_state=0)
print("sklearn s:", s_sk[:5])
# 거의 동일
```

### 8.4 Large Sparse Example

```python
from scipy.sparse import random as sp_random

M = sp_random(5000, 3000, density=0.01, format='csr')
# scipy는 sparse matmul 지원

def rsvd_sparse(A, k, p=10, q=2):
    m, n = A.shape
    Omega = np.random.randn(n, k + p)
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    for _ in range(q):
        Y = A.T @ Q
        Q, _ = np.linalg.qr(Y)
        Y = A @ Q
        Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)
    return Q @ U_tilde[:, :k], s[:k], Vt[:k]

import time
t0 = time.time()
U_r, s_r, Vt_r = rsvd_sparse(M, k=30)
print(f"Sparse rSVD: {time.time()-t0:.3f} s")
print("Top-5 s:", s_r[:5])
```

---

## 9. 요약

| 단계              | 연산                                        | 복잡도             |
| ----------------- | ------------------------------------------- | ------------------ |
| Random test matrix| $\Omega \in \mathbb{R}^{n \times (k+p)}$    | $O(n(k+p))$        |
| Sample            | $Y = A\Omega$                               | $O(mn(k+p))$       |
| Range QR          | $Y = QR$                                    | $O(m(k+p)^2)$      |
| Project           | $B = Q^T A$                                 | $O(mn(k+p))$       |
| Small SVD         | $B = \tilde U \Sigma V^T$                   | $O(n(k+p)^2)$      |
| Rotate back       | $U = Q\tilde U$                             | $O(m(k+p)^2)$      |
| **총**            |                                             | $O(mn(k+p) + (m+n)(k+p)^2)$ |

**실무 가이드**:
- $p = 5 \sim 20$ (보통 10)
- $q = 1 \sim 2$ (스펙트럼 감쇠 느리면 큰 $q$)
- 메모리가 주요 제약: out-of-core 버전도 존재 (한번 scan으로 $A\Omega$ 계산)

---

## 10. 참고 문헌

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). *Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions*. SIAM Review.
- Martinsson, P. G. (2019). *Randomized methods for matrix computations*. arXiv.
- Drineas & Mahoney (2016). *RandNLA: Randomized Numerical Linear Algebra*. CACM.

---

## 11. 챕터 4 총정리

본 챕터 (SVD):
1. 기하학적 유도 (단위구 → 타원체)
2. 존재성·유일성·Weyl 섭동
3. Moore-Penrose pseudoinverse
4. Eckart-Young 저랭크 근사
5. PCA = Truncated SVD
6. Randomized SVD 가속

→ Ch5 (내적공간·최소제곱)로 이어짐.

---

## 12. 내비게이션

[◀ 05. PCA](./05-pca.md) | [📚 README](../README.md) | [다음 챕터: 내적 공간 ▶](../ch5-inner-product/01-inner-product-cauchy-schwarz.md)
