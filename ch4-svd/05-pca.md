# Ch4-05. 주성분 분석(PCA)의 선형대수적 유도

> "분산을 최대로 보존하는 방향 = 공분산 행렬의 top 고유벡터 = 데이터 행렬의 top 우특이벡터."

## 📌 학습 목표

- PCA를 두 가지 관점으로 유도: **분산 최대화**와 **재구성 오차 최소화**.
- 두 관점이 **동일한** 해를 주는 이유 (Pythagorean identity).
- PCA = 공분산 행렬의 스펙트럼 분해 = 데이터 행렬의 truncated SVD.
- **백색화(whitening)**, Mahalanobis 거리, Gaussian ML 연결.
- 한계: 선형성, 스케일 의존성 (→ Kernel PCA, NMF 확장).

---

## 🎯 핵심 질문

> **질문 1**: 왜 공분산 행렬의 **고유벡터**가 "중요한" 방향인가?
> **질문 2**: "분산 최대화"와 "재구성 오차 최소화"가 같은 해를 주는 이유?
> **질문 3**: 데이터 스케일 차이에서 PCA가 왜 실패하는가?

---

## 1. 설정

### 데이터

$N$개 샘플, $d$차원: $X \in \mathbb{R}^{N \times d}$. 각 행이 하나의 샘플 $\mathbf{x}_i^T$.

### 중심화

샘플 평균 $\bar{\mathbf{x}} = \frac{1}{N}\sum_i \mathbf{x}_i$. 중심화된 데이터: $\tilde{X}_{ij} = X_{ij} - \bar{x}_j$.

이후 논의는 $\tilde X$에 대한 것. (중심화하지 않으면 첫 주성분이 "평균 방향"을 그대로 포착.)

### 공분산 행렬

$$
\Sigma = \frac{1}{N - 1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{d \times d}
$$

대칭 PSD.

---

## 2. 관점 1: 분산 최대화

### 문제

방향 $\mathbf{w} \in \mathbb{R}^d$, $\|\mathbf{w}\| = 1$에 투영 $z_i = \mathbf{x}_i^T \mathbf{w}$. 투영된 분산:

$$
\operatorname{Var}(z) = \frac{1}{N - 1}\sum_i (z_i - \bar z)^2 = \frac{1}{N-1}\sum_i (\tilde{\mathbf{x}}_i^T \mathbf{w})^2 = \mathbf{w}^T \Sigma \mathbf{w}
$$

### 첫 주성분

$$
\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\| = 1} \mathbf{w}^T \Sigma \mathbf{w}
$$

Rayleigh 원리 (Ch3-03): $\mathbf{w}_1$은 $\Sigma$의 **최대 고유값 $\lambda_1$에 대응하는 고유벡터**. 이때 분산 $= \lambda_1$.

### $k$번째 주성분

$\mathbf{w}_1, \ldots, \mathbf{w}_{k-1}$에 직교하면서 분산 최대화 → $\lambda_k$-고유벡터.

---

## 3. 관점 2: 재구성 오차 최소화

### 문제

$d$차원 데이터를 $k$차원 부분공간 $W = \operatorname{span}(\mathbf{w}_1, \ldots, \mathbf{w}_k)$ (정규직교)에 투영한 후 복원:

$$
\hat{\mathbf{x}}_i = \sum_{j=1}^k (\mathbf{x}_i^T \mathbf{w}_j) \mathbf{w}_j
$$

### 목표

$$
\min_{W} \sum_i \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2
$$

### 정리 3.1 (동등성)

재구성 오차 최소화 = 분산 최대화. 둘 다 해: $W$ = $\Sigma$의 top-$k$ 고유공간.

### 증명

$\mathbf{x}_i = \hat{\mathbf{x}}_i + (\mathbf{x}_i - \hat{\mathbf{x}}_i)$, 직교:

$$
\|\mathbf{x}_i\|^2 = \|\hat{\mathbf{x}}_i\|^2 + \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2
$$

합:

$$
\underbrace{\sum_i \|\mathbf{x}_i\|^2}_{\text{고정}} = \sum_i \|\hat{\mathbf{x}}_i\|^2 + \sum_i \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2
$$

**재구성 오차 최소화** ⟺ **투영 노름(분산) 최대화**. 따라서 같은 해. $\blacksquare$

> **🔑 Pythagorean identity**: PCA의 두 "다른" 설명이 실제로는 하나.

---

## 4. PCA = Truncated SVD

### 정리 4.1

$\tilde{X} = U \Sigma_\text{SVD} V^T$ (SVD)이면:

- $V$의 열이 **주성분 축** (= 공분산 행렬의 고유벡터)
- $\sigma_i^2/(N-1)$이 분산 (= 공분산 행렬의 고유값)
- $U \Sigma_\text{SVD}$의 열이 **점수 (scores)**

### 증명

$\Sigma = \tilde X^T \tilde X / (N-1) = V \Sigma_\text{SVD}^2 V^T / (N-1)$. 즉 $V$가 $\Sigma$의 고유벡터 (= 주성분), 고유값 $\sigma_i^2 / (N-1)$. $\blacksquare$

### 실무 이점

$d$가 매우 크면 ($d \gg N$) $\Sigma$는 $d \times d$로 메모리 부족. 하지만 SVD는 $\tilde X$ ($N \times d$)에 직접 적용 가능, rank $\leq N$이므로 $N$개 큰 특이값만 계산.

---

## 5. 설명된 분산

### 정의 5.1

$k$ 주성분까지의 **설명된 분산 비율**:

$$
\text{EVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^d \sigma_i^2}
$$

### Scree Plot

$k$에 따른 $\lambda_k$ 감소 그래프. "elbow"가 자연스러운 $k$ 선택의 휴리스틱.

### Kaiser 기준

$\lambda_k > 1$ (상관행렬 PCA) 또는 $\lambda_k > \bar\lambda$ (공분산행렬 PCA).

---

## 6. 백색화(Whitening)

### 정의 6.1

$\mathbf{y} = \Lambda^{-1/2} V^T \mathbf{x}$로 변환하면:

$$
\operatorname{Cov}(\mathbf{y}) = \Lambda^{-1/2} V^T \Sigma V \Lambda^{-1/2} = I
$$

모든 방향의 분산 = 1, 서로 무상관 (독립 가정 시).

### 활용

- **ZCA whitening**: $W_{ZCA} = V\Lambda^{-1/2}V^T$. 백색화 후 원 좌표계로 회전 복원 (딥러닝에서 가끔 사용).
- ICA (독립성분 분석)에서 전처리.
- Mahalanobis 거리: $d_M(\mathbf{x}, \boldsymbol{\mu}) = (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) = \|\Lambda^{-1/2}V^T(\mathbf{x} - \boldsymbol{\mu})\|^2$.

---

## 7. 확률론적 PCA (PPCA)

### 생성모델

$$
\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_k), \quad \mathbf{x} = W\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_d)
$$

### 주변분포

$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, WW^T + \sigma^2 I)$.

### 정리 7.1 (Tipping & Bishop, 1999)

Log-likelihood 최대화 해:

$$
W_\text{ML} = U_k (\Lambda_k - \sigma^2 I)^{1/2} R
$$

$R$: 임의의 $k \times k$ 직교 (rotation).

즉 **PCA는 Gaussian + isotropic noise의 ML**.

### 장점

- 불확실성 모델링 가능
- Missing data 처리 (EM 알고리즘)
- Factor analysis, Mixture PPCA 등으로 확장

---

## 8. 한계와 확장

### 8.1 선형성

PCA는 **선형** 부분공간. 휘어진 manifold는 포착 못 함.

**해결**: Kernel PCA, t-SNE, UMAP, Autoencoder (비선형 차원축소).

### 8.2 스케일 의존성

특성의 스케일이 다르면 큰 스케일이 지배. 해결: **표준화(standardize)** $X \to (X - \mu)/\sigma$, 혹은 **상관 행렬 PCA**.

### 8.3 해석 가능성

주성분이 "원 특성의 가중합"이어서 의미 부여 어려움. **Sparse PCA**, **NMF**가 해석 가능.

### 8.4 음수 포함

NMF와 달리 PCA는 부호 제약 없음. "성분이 음수인 픽셀" 등 물리적으로 이상할 때 NMF.

---

## 9. Python 실험

### 9.1 기본 PCA (SVD 기반)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
# Synthetic 3D data: 평면 + noise
N = 500
t = np.random.randn(N)
u = np.random.randn(N)
X = np.column_stack([3*t + 0.1*np.random.randn(N),
                     2*u + 0.1*np.random.randn(N),
                     0.5*t - 0.3*u + 0.05*np.random.randn(N)])
X -= X.mean(axis=0)

U, s, Vt = np.linalg.svd(X, full_matrices=False)
var_explained = s**2 / (N - 1)
print("Principal variances:", var_explained)
print("Explained ratio:", var_explained / var_explained.sum())

# Project to 2D
X_2d = U[:, :2] @ np.diag(s[:2])
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5)
plt.axis('equal'); plt.title("PCA 2D projection")
```

### 9.2 이미지 얼굴 기저 (Eigenfaces)

```python
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces(shuffle=True, random_state=0)
X_img = faces.data  # (400, 4096)
X_img -= X_img.mean(axis=0)

U, s, Vt = np.linalg.svd(X_img, full_matrices=False)
# 상위 주성분 visualize
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(Vt[i].reshape(64, 64), cmap='gray')
    ax.set_title(f"PC {i+1}")
    ax.axis('off')
# "Eigenfaces": 평균 얼굴 + 변동 방향
```

### 9.3 Scree Plot

```python
cum_var = np.cumsum(s**2) / np.sum(s**2)
plt.plot(cum_var, 'o-')
plt.axhline(0.95, color='r', linestyle='--', label='95%')
plt.xlabel('Principal component')
plt.ylabel('Cumulative explained variance')
plt.legend(); plt.grid()
# 엘보우 찾기
```

### 9.4 Whitening

```python
X -= X.mean(axis=0)
Sigma = np.cov(X.T)
eigvals, V = np.linalg.eigh(Sigma)
Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + 1e-10))
W_pca = Lambda_inv_sqrt @ V.T
X_white = X @ W_pca.T
print("Cov after whitening:\n", np.cov(X_white.T).round(3))  # ≈ I
```

### 9.5 sklearn 비교

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d_sk = pca.fit_transform(X)

# 자체 구현과 일치? (부호/회전 제외)
print("sklearn components:\n", pca.components_)
print("SVD Vt[:2]:\n", Vt[:2])
print("sklearn variance:", pca.explained_variance_)
print("SVD variance:", var_explained[:2])
```

---

## 10. 요약

| 개념                         | 공식                                   |
| ---------------------------- | -------------------------------------- |
| 공분산 행렬                  | $\Sigma = \tilde X^T \tilde X / (N-1)$ |
| 주성분 방향                  | $\Sigma v_i = \lambda_i v_i$           |
| 분산                         | $\operatorname{Var}(z_i) = \lambda_i$  |
| SVD와의 관계                 | $\tilde X = U\Sigma V^T$, $v_i$ = 주성분 |
| 설명 분산 비율               | $\sum_{j \leq k} \lambda_j / \sum \lambda_j$ |
| Rank-$k$ 근사                | $\tilde X_k = U_k \Sigma_k V_k^T$      |
| 재구성 오차                  | $\sum_{j > k} \lambda_j$               |

**PCA의 세 얼굴**: 분산 최대화 = 재구성 오차 최소화 = Gaussian ML. 모두 같은 답.

---

## 11. 참고 문헌

- Pearson, K. (1901). *On lines and planes of closest fit to systems of points in space*. Philos. Mag.
- Tipping, M. E., & Bishop, C. M. (1999). *Probabilistic principal component analysis*. JRSS B.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Bishop, C. (2006). *Pattern Recognition and Machine Learning*, Ch 12.

---

## 12. 내비게이션

[◀ 04. Eckart-Young](./04-eckart-young.md) | [📚 README](../README.md) | [06. Randomized SVD ▶](./06-randomized-svd.md)
