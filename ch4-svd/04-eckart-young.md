# Ch4-04. Eckart-Young 정리: 최적의 저랭크 근사

> "SVD의 앞 $k$개 특이값을 남기면, 그것이 **모든** rank-$k$ 근사 중 최선이다."

## 📌 학습 목표

- Eckart-Young 정리의 엄밀한 진술과 증명.
- **Frobenius norm**과 **operator norm** (= 2-norm) 양쪽에서의 최소화.
- Mirsky의 일반화: 모든 유니타리 불변 노름.
- 데이터 압축, 노이즈 제거, 추천 시스템에서의 의의.

---

## 🎯 핵심 질문

> **질문 1**: 왜 상위 $k$ 특이값 유지가 다른 $k$-rank 근사보다 좋은가?
> **질문 2**: 오차의 **닫힌형** 공식은 무엇인가?
> **질문 3**: Frobenius와 2-norm 최적해가 **같은** 이유는?

---

## 1. Rank-$k$ 근사 문제

### 문제

$A \in \mathbb{R}^{m \times n}$, $k < \operatorname{rank}(A)$에 대해:

$$
\min_{B : \operatorname{rank}(B) \leq k} \|A - B\|
$$

노름 $\|\cdot\|$은 2-norm, Frobenius norm, 혹은 다른 유니타리 불변 노름.

### SVD-기반 근사 (Truncated SVD)

$A = U\Sigma V^T = \sum_i \sigma_i \mathbf{u}_i \mathbf{v}_i^T$. **Truncated SVD**:

$$
A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T
$$

여기서 $U_k = [\mathbf{u}_1 \mid \cdots \mid \mathbf{u}_k]$, $V_k = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_k]$, $\Sigma_k = \operatorname{diag}(\sigma_1, \ldots, \sigma_k)$.

---

## 2. Eckart-Young 정리

### 정리 2.1 (Eckart-Young, 2-norm)

$$
\min_{\operatorname{rank}(B) \leq k} \|A - B\|_2 = \sigma_{k+1}
$$

최적해: $B^* = A_k$.

### 정리 2.2 (Eckart-Young-Mirsky, Frobenius)

$$
\min_{\operatorname{rank}(B) \leq k} \|A - B\|_F = \sqrt{\sum_{i=k+1}^{\min(m,n)} \sigma_i^2}
$$

최적해: $B^* = A_k$ (동일).

---

## 3. 2-Norm 증명

### 증명 (하한)

$B = XY^T$ ($X \in \mathbb{R}^{m \times k}, Y \in \mathbb{R}^{n \times k}$)로 일반적 rank-$k$ 행렬. $\operatorname{rank}(B) \leq k$이므로 $\ker(B) \supset n - k$차원 부분공간, $\dim\ker(B) \geq n - k$.

Span($\mathbf{v}_1, \ldots, \mathbf{v}_{k+1}$)은 $k+1$차원 → $\dim\ker(B) + (k+1) \geq n + 1$, 즉 교집합 비영.

$\mathbf{w} \in \ker(B) \cap \operatorname{span}(\mathbf{v}_1, \ldots, \mathbf{v}_{k+1})$, $\|\mathbf{w}\| = 1$. $B\mathbf{w} = 0$:

$$
\|(A - B)\mathbf{w}\|^2 = \|A\mathbf{w}\|^2 = \sum_{i=1}^{k+1} c_i^2 \sigma_i^2 \geq \sigma_{k+1}^2 \sum c_i^2 = \sigma_{k+1}^2
$$

여기서 $\mathbf{w} = \sum_{i=1}^{k+1} c_i \mathbf{v}_i$, $\sum c_i^2 = 1$.

따라서 $\|A - B\|_2 \geq \sigma_{k+1}$.

### 증명 (상한 달성)

$B = A_k$:

$$
A - A_k = \sum_{i = k+1}^{\min(m,n)} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

이는 SVD의 "잔여" 부분, 최대 특이값이 $\sigma_{k+1}$:

$$
\|A - A_k\|_2 = \sigma_{k+1} \quad \blacksquare
$$

---

## 4. Frobenius 증명

### 증명

$\|A - B\|_F^2 = \operatorname{tr}((A - B)^T(A - B))$.

$A = U\Sigma V^T$로 좌표변환. $\tilde{B} = U^T B V$도 rank $\leq k$ (직교 변환은 rank 보존):

$$
\|A - B\|_F^2 = \|\Sigma - \tilde B\|_F^2
$$

$\Sigma$는 대각이므로 $\tilde B$ 중 대각 성분 선택이 최적. 대각 행렬 $\tilde B^* = \operatorname{diag}(\sigma_1, \ldots, \sigma_k, 0, \ldots)$이 rank $\leq k$ 제한 하에 최적 (대각 성분 선택으로 $\sigma_i^2$ 감소; 가장 큰 $k$개 유지).

최솟값:

$$
\sum_{i = k+1}^{\min(m,n)} \sigma_i^2 \quad \blacksquare
$$

**엄밀히**: 대각이 아닌 $\tilde B$가 더 낫지 않음을 보이려면 "대각화"가 $\|\cdot\|_F$를 감소시키지 않음을 증명 (Hoffman-Wielandt 부등식 계열).

---

## 5. Mirsky의 일반화

### 정의 5.1 (유니타리 불변 노름)

$\|\cdot\|$이 **유니타리 불변**이라 함은 임의 직교/유니타리 $U, V$에 대해 $\|UAV\| = \|A\|$.

예: 2-norm, Frobenius, nuclear norm ($\sum \sigma_i$), Schatten-$p$ norm.

### 정리 5.2 (Mirsky)

유니타리 불변 노름에서:

$$
\min_{\operatorname{rank}(B) \leq k} \|A - B\| = \left\| \operatorname{diag}(0, \ldots, 0, \sigma_{k+1}, \ldots, \sigma_p) \right\|
$$

최적해: $A_k$.

**증명 아이디어**: Weyl-type 불등식 "singular values are majorized" 활용.

---

## 6. 응용: 데이터 압축

### 6.1 이미지 압축

$m \times n$ 픽셀의 흑백 이미지를 rank-$k$ 근사:

- 원본: $mn$ 저장
- Rank $k$: $(m + n + 1)k$ 저장 (저장 공간 $\approx k/\min(m,n)$ 비율)

$k$가 작아도 인간 눈에 보이는 세부는 남김 (큰 특이값이 주요 특징 포착).

### 6.2 오차 공식

Rank-$k$ 근사의 **상대 오차**:

$$
\frac{\|A - A_k\|_F^2}{\|A\|_F^2} = \frac{\sum_{i > k} \sigma_i^2}{\sum_i \sigma_i^2}
$$

= **1 - energy retained**.

### 6.3 Noise 제거

작은 특이값이 노이즈 → 잘라냄으로써 **denoising**. Hyperspectral, 의학 영상, 추천 시스템에서 사용.

---

## 7. 주성분 분석 (PCA) 연결

$X \in \mathbb{R}^{n \times d}$ (n 샘플, d 특성), 중심화된 데이터. Truncated SVD $X \approx U_k \Sigma_k V_k^T$:

- $V_k$: 주성분 축 (특성 공간)
- $U_k \Sigma_k$: 저차원 표현 (scores)

**Eckart-Young**: PCA의 $k$-차원 투영이 **데이터 손실을 최소화**하는 선형 차원 축소. (Ch4-05에서 자세히.)

---

## 8. 희소/구조적 저랭크

### 8.1 Nuclear Norm

$\|A\|_* = \sum \sigma_i$: nuclear (trace) norm. L1 역할: 저랭크 행렬 복구 ("convex relaxation of rank").

**Robust PCA**: $\min \|L\|_* + \lambda \|S\|_1$ s.t. $A = L + S$ (L: 저랭크, S: 희소).

### 8.2 행렬 완성

Netflix Prize: 일부 항목만 관측된 행렬 $A$의 나머지 복원. Low-rank 가정 + nuclear norm 최소화:

$$
\min_X \|X\|_* \text{ s.t. } X_{ij} = A_{ij} \text{ for observed } (i,j)
$$

---

## 9. Python 실험

### 9.1 이미지 압축

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import face

img = face(gray=True).astype(float)
U, s, Vt = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(img, cmap='gray'); axes[0].set_title("Original")
for i, k in enumerate([5, 20, 50, 200]):
    img_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
    axes[i+1].imshow(img_k, cmap='gray')
    axes[i+1].set_title(f"k={k}")
    axes[i+1].axis('off')

# Relative error
for k in [5, 20, 50, 200]:
    err = np.sqrt((s[k:]**2).sum()) / np.linalg.norm(s)
    print(f"k={k}: rel err = {err:.4f}")
```

### 9.2 Eckart-Young 검증

```python
np.random.seed(0)
A = np.random.randn(20, 15)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

for k in [1, 3, 5, 10]:
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
    err_2 = np.linalg.norm(A - A_k, 2)
    err_F = np.linalg.norm(A - A_k, 'fro')
    
    pred_2 = s[k]
    pred_F = np.sqrt((s[k:]**2).sum())
    print(f"k={k}: ||·||_2 = {err_2:.4f} (pred {pred_2:.4f}), ||·||_F = {err_F:.4f} (pred {pred_F:.4f})")
```

### 9.3 PCA as Truncated SVD

```python
# Synthetic data: 3D Gaussian with elongated axis
n = 500
X = np.random.randn(n, 3) @ np.diag([5, 2, 0.1])
X -= X.mean(axis=0)  # center

U, s, Vt = np.linalg.svd(X, full_matrices=False)
print("Principal variances:", s**2 / (n-1))  # ≈ [25, 4, 0.01]
print("Principal directions:\n", Vt.T)

# Project to 2D
X_2d = U[:, :2] @ np.diag(s[:2])
print("2D score shape:", X_2d.shape)
```

### 9.4 Randomized Low-rank (미리보기)

```python
def rand_lowrank(A, k, p=5):
    """Halko et al. randomized SVD."""
    m, n = A.shape
    Omega = np.random.randn(n, k + p)
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    UB, sB, VBt = np.linalg.svd(B, full_matrices=False)
    U = Q @ UB
    return U[:, :k], sB[:k], VBt[:k]

A = np.random.randn(500, 300)
U_r, s_r, Vt_r = rand_lowrank(A, k=20)
A_k_approx = U_r @ np.diag(s_r) @ Vt_r

# Compare to true truncated SVD
s_true = np.linalg.svd(A, full_matrices=False)[1]
print("True top-20 σ:", s_true[:5], "...")
print("Random top-20 σ:", s_r[:5], "...")
```

---

## 10. 요약

| 노름              | 최적 오차                               |
| ----------------- | --------------------------------------- |
| $\|\cdot\|_2$     | $\sigma_{k+1}$                          |
| $\|\cdot\|_F$     | $\sqrt{\sum_{i > k} \sigma_i^2}$        |
| Schatten-$p$      | $\left(\sum_{i > k} \sigma_i^p\right)^{1/p}$ |
| Nuclear ($p = 1$) | $\sum_{i > k} \sigma_i$                 |

**핵심 교훈**:
- Truncated SVD **모든** 유니타리 불변 노름에서 동시에 최적
- 오차는 잘린 특이값들의 함수
- PCA, 추천 시스템, 압축, 노이즈 제거의 **통합 이론적 기반**

---

## 11. 참고 문헌

- Eckart, C., & Young, G. (1936). *The approximation of one matrix by another of lower rank*. Psychometrika.
- Mirsky, L. (1960). *Symmetric gauge functions and unitarily invariant norms*. Quart. J. Math.
- Golub & Van Loan, *Matrix Computations*, §2.4.
- Udell & Townsend (2019). *Why are big data matrices approximately low rank?*. SIAM J. Math. Data Sci.

---

## 12. 내비게이션

[◀ 03. Pseudoinverse](./03-pseudoinverse.md) | [📚 README](../README.md) | [05. PCA ▶](./05-pca.md)
