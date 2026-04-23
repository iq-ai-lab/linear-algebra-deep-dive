# Ch5-03. 최소 제곱: 기하학적·대수적·통계적 유도

> "과결정 시스템의 최적해 = 정사영 = 정규방정식의 해 = Gauss-Markov 불편추정."

## 📌 학습 목표

- 최소 제곱 문제를 **기하학적(정사영)**, **대수적(정규방정식)**, **통계적(Gauss-Markov)** 세 관점에서 유도.
- **조건수 악화**와 QR/SVD 기반 해법의 이점.
- Weighted Least Squares, Tikhonov, LASSO의 연결.
- ML에서 linear regression, softmax regression의 기초.

---

## 🎯 핵심 질문

> **질문 1**: 정규방정식 $A^T A \mathbf{x} = A^T \mathbf{b}$의 기하적 의미는?
> **질문 2**: 왜 $A^T A$의 조건수가 $A$의 **제곱**이 되는가?
> **질문 3**: Gauss-Markov 정리가 주장하는 "최선"의 의미는?

---

## 1. 문제 정의

과결정 선형계 $A \mathbf{x} = \mathbf{b}$ ($A \in \mathbb{R}^{m \times n}$, $m \geq n$, 보통 해 없음):

$$
\min_{\mathbf{x} \in \mathbb{R}^n} \|A\mathbf{x} - \mathbf{b}\|_2^2
$$

---

## 2. 기하학적 유도 (정사영)

### 관점

$A\mathbf{x}$는 $R(A) = C(A)$의 임의 점. 목표: $\mathbf{b}$에 가장 가까운 $C(A)$ 점 = 정사영 $P\mathbf{b}$.

$A\mathbf{x}^* = P\mathbf{b}$를 풀면 $\mathbf{x}^*$가 최소 제곱 해.

### 정리 2.1 (기하적 조건)

$\mathbf{x}^*$가 최소 제곱 해 ⟺ $\mathbf{b} - A\mathbf{x}^* \perp C(A)$.

### 증명

$A\mathbf{x}$가 임의 $C(A)$ 원소이므로, Ch5-02 Best Approximation 정리 직접 적용. $\blacksquare$

---

## 3. 대수적 유도 (정규방정식)

### 정리 3.1 (Normal Equations)

$\mathbf{x}^*$가 최소 제곱 해 ⟺

$$
\boxed{A^T A \mathbf{x}^* = A^T \mathbf{b}}
$$

### 증명 1 (기하 직교 조건)

$\mathbf{b} - A\mathbf{x}^* \perp C(A) \iff A^T(\mathbf{b} - A\mathbf{x}^*) = \mathbf{0} \iff A^T A \mathbf{x}^* = A^T \mathbf{b}$.

### 증명 2 (미분)

$f(\mathbf{x}) = \|A\mathbf{x} - \mathbf{b}\|^2 = \mathbf{x}^T A^T A \mathbf{x} - 2\mathbf{b}^T A\mathbf{x} + \|\mathbf{b}\|^2$.

$\nabla f = 2A^T A \mathbf{x} - 2A^T \mathbf{b} = \mathbf{0} \iff A^T A \mathbf{x} = A^T \mathbf{b}$. $\blacksquare$

### 3.2 해 존재성

$A^T A$는 **항상** $A^T \mathbf{b} \in C(A^T)$의 존재에 의해 compatible.

- **Full column rank**: $A^T A$ 정칙, $\mathbf{x}^* = (A^T A)^{-1} A^T \mathbf{b}$ 유일
- **Rank-deficient**: 해 무한 개, pseudoinverse로 최소 노름 해 선택

---

## 4. QR 기반 해법

### 정리 4.1

$A$가 full column rank, thin QR $A = \hat Q \hat R$ ($\hat Q$ 열 정규직교, $\hat R$ 상삼각):

$$
\hat R \mathbf{x}^* = \hat Q^T \mathbf{b}
$$

### 증명

$A^T A = \hat R^T \hat Q^T \hat Q \hat R = \hat R^T \hat R$, $A^T \mathbf{b} = \hat R^T \hat Q^T \mathbf{b}$.

정규방정식: $\hat R^T \hat R \mathbf{x}^* = \hat R^T \hat Q^T \mathbf{b}$. $\hat R$ 정칙 → $\hat R^T$ 정칙:

$$
\hat R \mathbf{x}^* = \hat Q^T \mathbf{b}
$$

후방 대입으로 $O(n^2)$에 해결. $\blacksquare$

---

## 5. SVD 기반 해법

### 정리 5.1

$A = U\Sigma V^T$ (thin, $m \geq n$):

$$
\mathbf{x}^* = A^+ \mathbf{b} = V\Sigma^+ U^T \mathbf{b}
$$

### 최소 제곱 + 최소 노름

Rank-deficient 또는 조건이 나쁜 경우에도 SVD는 자동으로 **최소 제곱 + 최소 노름 해**를 준다 (Ch4-03).

### 정리 5.2 (조건수)

$$
\kappa_2(A^T A) = \kappa_2(A)^2
$$

### 증명

$A = U\Sigma V^T$, $A^T A = V\Sigma^2 V^T$. 특이값이 제곱. $\blacksquare$

### 의미

**정규방정식 직접 풀이는 조건수를 악화**시킨다. QR이나 SVD는 $A$의 조건수 그대로 유지.

---

## 6. Weighted Least Squares

### 정의 6.1

가중 $W \succ 0$ (대칭 PD):

$$
\min_\mathbf{x} (A\mathbf{x} - \mathbf{b})^T W (A\mathbf{x} - \mathbf{b})
$$

### 해

$$
(A^T W A) \mathbf{x}^* = A^T W \mathbf{b}
$$

### 유도 (변형)

$W = L L^T$ (Cholesky). $\tilde A = L^T A$, $\tilde{\mathbf{b}} = L^T \mathbf{b}$. 그러면:

$$
\|L^T(A\mathbf{x} - \mathbf{b})\|^2 = \|\tilde A \mathbf{x} - \tilde{\mathbf{b}}\|^2
$$

일반 최소 제곱 문제로 환원.

### 통계적 의미

$W = \Sigma^{-1}$ ($\Sigma$: 오차 공분산). 이것이 **GLS (Generalized Least Squares)**. 이질 분산 오차를 올바르게 가중.

---

## 7. Gauss-Markov 정리

### 설정

선형 회귀 모델: $\mathbf{b} = A\boldsymbol{\beta}^* + \boldsymbol{\epsilon}$, $\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$, $\operatorname{Cov}(\boldsymbol{\epsilon}) = \sigma^2 I$.

### 정리 7.1 (Gauss-Markov)

$\hat{\boldsymbol{\beta}} = (A^T A)^{-1} A^T \mathbf{b}$ (OLS)는 $\boldsymbol{\beta}^*$의 **BLUE** (Best Linear Unbiased Estimator): 모든 선형 불편 추정량 $\tilde{\boldsymbol{\beta}} = C\mathbf{b}$ 중 분산 최소.

### 증명

**불편성**: $\mathbb{E}[\hat{\boldsymbol{\beta}}] = (A^T A)^{-1} A^T (A\boldsymbol{\beta}^* + \mathbb{E}[\boldsymbol{\epsilon}]) = \boldsymbol{\beta}^*$.

**분산**: $\operatorname{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2 (A^T A)^{-1}$.

**최적성**: 임의 $C\mathbf{b}$ 불편이면 $CA = I$. $C = (A^T A)^{-1} A^T + D$로 쓰면 $DA = 0$.

$$
\operatorname{Cov}(C\mathbf{b}) = \sigma^2 C C^T = \sigma^2[(A^T A)^{-1} + DD^T]
$$

(교차항 소멸: $DA = 0 \implies D (A^T A)^{-1} A^T = 0 \implies \cdots$.)

$DD^T \succeq 0$이므로 OLS 분산 $\leq$ 기타. $\blacksquare$

### 의미

**가우시안 분포 가정 없이도** OLS가 "최선". 가우시안이면 ML과 일치.

---

## 8. Ridge / LASSO 정규화

### 8.1 Ridge (Tikhonov)

$$
\min \|A\mathbf{x} - \mathbf{b}\|^2 + \alpha \|\mathbf{x}\|^2
$$

해: $(A^T A + \alpha I) \mathbf{x} = A^T \mathbf{b}$.

**의미**: 조건수 개선, prior $\mathbf{x} \sim \mathcal{N}(0, \frac{\sigma^2}{\alpha} I)$의 MAP.

### 8.2 LASSO

$$
\min \|A\mathbf{x} - \mathbf{b}\|^2 + \alpha \|\mathbf{x}\|_1
$$

**의미**: Sparsity 유도, 변수 선택. Convex 최적화, LARS/coordinate descent로 해결.

---

## 9. 통계적 해석

### 잔차 분석

$\mathbf{r} = \mathbf{b} - A\hat{\boldsymbol{\beta}}$. 잔차는 $C(A)^\perp$에 속함: $A^T \mathbf{r} = \mathbf{0}$.

### 결정계수

$$
R^2 = 1 - \frac{\|\mathbf{r}\|^2}{\|\mathbf{b} - \bar b \mathbf{1}\|^2}
$$

모델이 설명하는 분산 비율.

### 가설 검정

$\hat{\boldsymbol{\beta}} \sim \mathcal{N}(\boldsymbol{\beta}^*, \sigma^2 (A^T A)^{-1})$ (정규 오차 가정) → t-test, F-test.

---

## 10. Python 실험

### 10.1 세 방법 비교

```python
import numpy as np
from scipy.linalg import lstsq, qr

np.random.seed(0)
m, n = 100, 5
A = np.random.randn(m, n)
beta_true = np.array([1, -2, 0.5, 3, -1])
b = A @ beta_true + 0.5 * np.random.randn(m)

# 1. Normal equations (dangerous for ill-conditioned)
x_ne = np.linalg.solve(A.T @ A, A.T @ b)
print("Normal eq:", x_ne)

# 2. QR
Q, R = qr(A, mode='economic')
x_qr = np.linalg.solve(R, Q.T @ b)
print("QR:       ", x_qr)

# 3. SVD / pinv
x_svd = np.linalg.pinv(A) @ b
print("SVD/pinv: ", x_svd)

# 4. np.linalg.lstsq
x_ls, _, _, _ = lstsq(A, b)
print("lstsq:    ", x_ls)

print("True:     ", beta_true)
```

### 10.2 조건수 비교

```python
# Ill-conditioned A
D = np.diag([1, 0.1, 0.001])
Q1 = np.linalg.qr(np.random.randn(100, 3))[0]
Q2 = np.linalg.qr(np.random.randn(3, 3))[0]
A_ill = Q1 @ D @ Q2.T  # 100x3, κ(A)=1000
beta_true = np.array([1, 2, 3])
b = A_ill @ beta_true + 1e-4 * np.random.randn(100)

print("κ(A):     ", np.linalg.cond(A_ill))
print("κ(A^T A): ", np.linalg.cond(A_ill.T @ A_ill))

x_ne = np.linalg.solve(A_ill.T @ A_ill, A_ill.T @ b)
x_qr = np.linalg.solve(*np.linalg.qr(A_ill, mode='reduced')[::-1]) @ b  # sketch
x_qr = np.linalg.lstsq(A_ill, b, rcond=None)[0]

print("NE err:   ", np.linalg.norm(x_ne - beta_true))
print("QR err:   ", np.linalg.norm(x_qr - beta_true))
```

### 10.3 Ridge 효과

```python
alphas = [0, 0.01, 1.0, 100.0]
for alpha in alphas:
    x = np.linalg.solve(A_ill.T @ A_ill + alpha * np.eye(3), A_ill.T @ b)
    print(f"α={alpha:6.3f}: β={x.round(3)}, err={np.linalg.norm(x - beta_true):.3e}")
# 적당한 α에서 오차 최소
```

### 10.4 Polynomial Regression

```python
import matplotlib.pyplot as plt

# Sample data
x_data = np.linspace(-2, 2, 20)
y_data = x_data**3 - 2*x_data + 1 + 0.3*np.random.randn(20)

# Design matrix
deg = 5
X = np.vander(x_data, deg+1, increasing=True)
beta = np.linalg.lstsq(X, y_data, rcond=None)[0]
print("Polynomial coeffs:", beta)

# Plot
x_fine = np.linspace(-2, 2, 200)
X_fine = np.vander(x_fine, deg+1, increasing=True)
plt.scatter(x_data, y_data)
plt.plot(x_fine, X_fine @ beta, 'r-')
plt.title(f"Polynomial LS (deg {deg})")
```

---

## 11. 요약

| 방법         | 수식                                    | 복잡도     | 조건수 |
| ------------ | --------------------------------------- | ---------- | ------ |
| Normal Eq    | $(A^T A)\mathbf{x} = A^T \mathbf{b}$    | $O(mn^2)$  | $\kappa^2$ |
| QR           | $R\mathbf{x} = Q^T \mathbf{b}$          | $O(mn^2)$  | $\kappa$ |
| SVD          | $\mathbf{x} = V\Sigma^+ U^T \mathbf{b}$ | $O(mn^2 + n^3)$ | $\kappa$ |
| Ridge        | $(A^TA + \alpha I)\mathbf{x} = A^T\mathbf{b}$ | $O(mn^2)$ | 개선됨 |

**실무 권장**: `np.linalg.lstsq` (SVD 기반, 기본적으로 안정). rank 부족 의심 시 pseudoinverse 사용.

---

## 12. 참고 문헌

- Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM.
- Golub & Van Loan, *Matrix Computations*, Ch 5.
- Hastie, Tibshirani, Friedman. *Elements of Statistical Learning*, Ch 3.

---

## 13. 내비게이션

[◀ 02. 정사영](./02-orthogonal-projection.md) | [📚 README](../README.md) | [04. Gram 행렬과 PSD ▶](./04-gram-matrix-psd.md)
