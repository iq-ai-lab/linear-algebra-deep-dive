# Ch3-05. Power Iteration과 QR 알고리즘

> "가장 큰 고윳값 하나만 필요한가? Power. 전부 필요한가? QR."

## 📌 학습 목표

- **Power iteration**의 수렴 속도와 수렴 조건을 증명한다.
- **Inverse iteration**, **shifted inverse iteration**의 가속 원리.
- **QR 알고리즘**의 정의와 수렴 증거 (Schur form으로 수렴).
- **Hessenberg 감소**와 **Wilkinson shift**를 통한 실무적 구현.

---

## 🎯 핵심 질문

> **질문 1**: Power iteration이 왜 $|\lambda_2/\lambda_1|^k$ 속도로 수렴하는가?
> **질문 2**: QR 알고리즘이 왜 Schur 분해로 수렴하는가?
> **질문 3**: 실무 LAPACK은 몇 개의 shift로 최적화되는가?

---

## 1. Power Iteration

### 알고리즘

```
x_0 ← 임의 벡터
repeat:
    y_k = A x_k
    x_{k+1} = y_k / ||y_k||
until 수렴
λ_approx = x^T A x  (Rayleigh)
```

### 정리 1.1 (수렴 정리)

대각화 가능한 $A$, 고윳값 $|\lambda_1| > |\lambda_2| \geq \cdots \geq |\lambda_n|$. 초기 벡터 $\mathbf{x}_0$가 $\mathbf{v}_1$ 성분을 가질 때 ($\langle \mathbf{x}_0, \mathbf{v}_1 \rangle \neq 0$):

$$
\mathbf{x}_k \to \pm \mathbf{v}_1, \quad \left\| \mathbf{x}_k - \pm\mathbf{v}_1 \right\| = O\left(\left|\frac{\lambda_2}{\lambda_1}\right|^k\right)
$$

### 증명

$\mathbf{x}_0 = \sum c_i \mathbf{v}_i$, $c_1 \neq 0$. 정규화를 무시한 $A^k \mathbf{x}_0$:

$$
A^k \mathbf{x}_0 = \sum c_i \lambda_i^k \mathbf{v}_i = \lambda_1^k \left( c_1 \mathbf{v}_1 + \sum_{i \geq 2} c_i \left(\frac{\lambda_i}{\lambda_1}\right)^k \mathbf{v}_i \right)
$$

$|\lambda_i/\lambda_1|^k \to 0$, $k$ 증가에 따라 $\mathbf{v}_1$ 방향이 우세. 정규화 후:

$$
\mathbf{x}_k = \frac{A^k \mathbf{x}_0}{\|A^k \mathbf{x}_0\|} \to \pm\frac{c_1 \mathbf{v}_1}{|c_1| \|\mathbf{v}_1\|} \quad \blacksquare
$$

### 한계

- $|\lambda_2/\lambda_1| \approx 1$이면 매우 느림
- Clustering이 심한 스펙트럼에서는 쓸모 없음
- 복소 고유값 경우 회전 + 확축 → 수렴하지 않을 수 있음

---

## 2. Inverse Iteration

### 핵심 아이디어

$A$ 대신 $A^{-1}$에 power iteration → **최소 절댓값** 고윳값 추출.

### Shifted Inverse Iteration

$A - \mu I$의 역행렬로 power iteration. $\mu$에 가장 가까운 고윳값의 고유벡터 빠르게 수렴.

$$
(A - \mu I) \mathbf{y}_{k+1} = \mathbf{x}_k, \qquad \mathbf{x}_{k+1} = \mathbf{y}_{k+1}/\|\mathbf{y}_{k+1}\|
$$

### 수렴률

$|\lambda_j - \mu|$가 가장 작은 $\lambda_j$로 수렴, 속도:

$$
\left| \frac{\lambda_j - \mu}{\lambda_{j'} - \mu} \right|^k
$$

여기서 $\lambda_{j'}$은 $\mu$ 기준 두 번째로 가까운 고윳값. $\mu \to \lambda_j$에 근접할수록 매우 빠름.

### Rayleigh Quotient Iteration (요약)

$\mu$를 Rayleigh 몫으로 매 단계 업데이트 → **3차 수렴** (대칭), **2차 수렴** (일반).

---

## 3. QR 알고리즘

### 기본 형태

```
A_0 = A
for k = 0, 1, 2, ...:
    A_k = Q_k R_k        (QR 분해)
    A_{k+1} = R_k Q_k
```

### 정리 3.1 (기본 QR의 수렴)

$A$의 고윳값들이 서로 다르고 $|\lambda_1| > |\lambda_2| > \cdots$이면 $A_k$는 **상삼각 행렬로 수렴**하고 대각 성분이 고윳값으로 수렴.

### 증명 스케치

QR 알고리즘이 **power iteration의 동시 버전**임을 보인다.

$A_{k+1} = R_k Q_k = Q_k^{-1} A_k Q_k$이므로 $A_k$는 모두 닮음 변환, **같은 고윳값**을 가진다.

전체 누적 $\tilde{Q}_k = Q_0 Q_1 \cdots Q_{k-1}$, $\tilde{R}_k = R_{k-1} \cdots R_1 R_0$에 대해:

$$
A^k = \tilde{Q}_k \tilde{R}_k
$$

(증명: 귀납법. $A^{k+1} = A \cdot A^k = A \tilde{Q}_k \tilde{R}_k = \tilde{Q}_k A_k \tilde{R}_k = \tilde{Q}_k Q_k R_k \tilde{R}_k = \tilde{Q}_{k+1} \tilde{R}_{k+1}$.)

즉, $A^k$의 QR 분해를 누적으로 계산. 첫 열은 power iteration, 둘째는 $\mathbf{v}_1$에 직교하며 power iteration, ... → 각 열이 고유벡터의 Schur 기저로 수렴. $A_k \to$ 상삼각. $\blacksquare$

### 복잡도

- 기본 QR: 각 iteration이 $O(n^3)$
- 수렴까지 $O(n)$ iterations → 총 $O(n^4)$ → 너무 느림

---

## 4. 실용적 QR: Hessenberg + Shifts

### 4.1 Hessenberg 감소

### 정의 4.1

$H$가 **상 Hessenberg**라 함은 $H_{ij} = 0$ ($i > j + 1$). 즉 첫 아래 대각 이외는 모두 0.

### 정리 4.2

모든 $A$는 Householder 반사로 Hessenberg 형태로 변환 가능: $A = Q H Q^T$. $O(n^3/3)$.

$QR$ 반복 시 Hessenberg 구조가 **보존**되므로 각 iteration $O(n^2)$로 감소. 전체 $O(n^3)$.

### 4.2 Shift 적용

$A_k - \mu_k I$의 QR을 수행:

```
A_k - μ_k I = Q_k R_k
A_{k+1} = R_k Q_k + μ_k I
```

$(A_{k+1} = Q_k^T A_k Q_k$, 동일 닮음.)

#### Rayleigh shift

$\mu_k = (A_k)_{nn}$. 우하단 코너의 2×2 블록 분석으로 선택. 대칭인 경우 **3차 수렴**.

#### Wilkinson shift

우하단 2×2 블록 $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$의 고윳값 중 $d$에 **더 가까운 것**을 $\mu$로. 복소 또는 실수 모두 처리 가능. 대칭일 때 항상 실수.

### 정리 4.3 (Wilkinson shift 수렴)

Wilkinson shift QR은 **항상 수렴**하며 (아직 개념 증명은 일반적이지 않음) 대칭 tridiagonal에 대해 **3차 수렴**.

### 4.3 Deflation

어떤 아래대각 성분 $H_{i+1, i}$가 작아지면 문제를 2개의 작은 문제로 분할:

```
[ H_11 |  *  ]
[------+-----]
[  0   | H_22]
```

→ 각 블록에서 QR 계속.

---

## 5. 대칭 / Tridiagonal 특화

### 정리 5.1

대칭 $A$의 Hessenberg 감소 결과는 **Tridiagonal** (삼대각).

### 증명

대칭성 유지: $Q^T A Q$도 대칭. 대칭 + 상 Hessenberg = 삼대각. $\blacksquare$

### 복잡도 이점

- Tridiagonal QR step: $O(n)$
- 전체 $O(n^2)$ (고유값만) / $O(n^3)$ (고유벡터 포함)

### 현대 대체법

- **Divide and conquer** (Cuppen 1981): $O(n^2)$, 병렬화 잘됨
- **MRRR** (Dhillon & Parlett 2000): $O(n^2)$, 선택적 고유벡터 가능

---

## 6. 대규모 희소: Arnoldi / Lanczos

### 6.1 Krylov 부분공간

$\mathcal{K}_k(A, \mathbf{v}) = \operatorname{span}(\mathbf{v}, A\mathbf{v}, A^2\mathbf{v}, \ldots, A^{k-1}\mathbf{v})$.

Power iteration의 **히스토리 전체**.

### 6.2 Arnoldi (일반)

Krylov 기저를 Gram-Schmidt로 정규직교화:

$$
A V_k = V_{k+1} \tilde{H}_k
$$

$V_k$ 정규직교, $\tilde{H}_k$는 $(k+1) \times k$ 상 Hessenberg. **Ritz values** $\operatorname{eig}(H_k)$가 $A$의 극단 고윳값 근사.

### 6.3 Lanczos (대칭)

Arnoldi + 대칭성 → $T_k$는 tridiagonal, 3-term recurrence:

$$
\beta_{k+1} \mathbf{v}_{k+1} = A\mathbf{v}_k - \alpha_k \mathbf{v}_k - \beta_k \mathbf{v}_{k-1}
$$

**$O(kn)$ 연산 + $k$ 벡터 저장**. 대규모 희소 대칭에 이상적.

---

## 7. Python 실험

### 7.1 Power Iteration

```python
import numpy as np

def power_iter(A, n_iter=100):
    n = A.shape[0]
    x = np.random.randn(n)
    x /= np.linalg.norm(x)
    for k in range(n_iter):
        x_new = A @ x
        x_new /= np.linalg.norm(x_new)
        x = x_new
    return x @ A @ x, x

A = np.array([[4.0, 1.0, 2.0],
              [1.0, 3.0, 0.0],
              [2.0, 0.0, 5.0]])

lam, v = power_iter(A)
print("Power:", lam, "True:", np.linalg.eigvalsh(A).max())

# 수렴 시각화
def power_convergence(A, x0, true_eigvec, n_iter=50):
    x = x0 / np.linalg.norm(x0)
    errs = []
    for k in range(n_iter):
        x = A @ x; x /= np.linalg.norm(x)
        # 사인 거리
        err = min(np.linalg.norm(x - true_eigvec), np.linalg.norm(x + true_eigvec))
        errs.append(err)
    return errs

eigvals, eigvecs = np.linalg.eigh(A)
errs = power_convergence(A, np.random.randn(3), eigvecs[:, -1])
ratio = np.abs(eigvals[-2] / eigvals[-1])
print(f"Ratio |λ2/λ1| = {ratio:.3f}")
print("Errors:", errs[::10])
```

### 7.2 Shifted Inverse Iteration

```python
def inv_iter(A, mu, n_iter=10):
    n = A.shape[0]
    x = np.random.randn(n); x /= np.linalg.norm(x)
    M = A - mu * np.eye(n)
    for _ in range(n_iter):
        y = np.linalg.solve(M, x)
        x = y / np.linalg.norm(y)
    lam = x @ A @ x
    return lam, x

# μ=3에 가장 가까운 고유값 찾기
lam, v = inv_iter(A, mu=3.0, n_iter=5)
print("Near μ=3:", lam)
# 실제 eigvals: [2.44, 3.67, 5.89]
```

### 7.3 단순한 QR 알고리즘

```python
def simple_qr_algorithm(A, n_iter=200):
    A = A.copy().astype(float)
    for _ in range(n_iter):
        Q, R = np.linalg.qr(A)
        A = R @ Q
    return np.diag(A)  # 대각 성분 = 고유값

eigs_qr = simple_qr_algorithm(A.copy())
print("QR algorithm:", sorted(eigs_qr))
print("Ground truth:", sorted(np.linalg.eigvalsh(A)))
```

### 7.4 Hessenberg 감소

```python
from scipy.linalg import hessenberg
H = hessenberg(np.random.randn(5, 5))
print("Hessenberg H (upper):\n", H.round(2))
# 첫 아래 대각 외에는 모두 0
```

### 7.5 Arnoldi (sparse)

```python
from scipy.sparse.linalg import eigs
from scipy.sparse import random as sp_random

n = 500
A_sparse = sp_random(n, n, density=0.01, format='csr')

# 최대 절댓값 고유값 5개
vals, vecs = eigs(A_sparse, k=5, which='LM')
print("Top-5 eigenvalues (LM):", vals)
```

---

## 8. 요약

| 알고리즘        | 용도                      | 복잡도/iter | 수렴률         | 비고              |
| --------------- | ------------------------- | ----------- | -------------- | ----------------- |
| Power           | 최대 $|\lambda|$          | $O(n^2)$    | $|\lambda_2/\lambda_1|^k$ | 가장 간단 |
| Inverse         | 최소 $|\lambda|$          | $O(n^3)$ once + $O(n^2)$ | — | LU 재활용 |
| Shifted Inverse | $\mu$ 근처 고유값          | $O(n^3)$ once + $O(n^2)$ | 매우 빠름 | |
| RQI             | 대칭 임의 eigenvalue       | $O(n^3)$/iter | 3차 | shift 변경 |
| 기본 QR         | 전체                      | $O(n^3)$/iter | linear | 느림 |
| Hessenberg QR   | 전체                      | $O(n^2)$/iter + $O(n^3)$ 초기화 | cubic (shift) | 실무용 |
| Tridiagonal QR  | 대칭 전체                  | $O(n)$/iter  | cubic | LAPACK `dsyev` |
| Arnoldi/Lanczos | 희소 대규모 극단 eigenvalue | $O(nk)$/iter  | — | `ARPACK` |

---

## 9. 참고 문헌

- Golub & Van Loan, *Matrix Computations*, Ch 7–9.
- Trefethen & Bau, *Numerical Linear Algebra*, Lectures 25–32.
- Parlett, *The Symmetric Eigenvalue Problem*.
- Saad, Y. (2011). *Numerical Methods for Large Eigenvalue Problems* (revised).

---

## 10. 다음 문서

- **[06. 조건수](./06-condition-number.md)**: 고유값 계산 오차와 eigenvalue condition number.

---

## 11. 내비게이션

[◀ 04. Perron-Frobenius](./04-perron-frobenius.md) | [📚 README](../README.md) | [06. 조건수 ▶](./06-condition-number.md)
