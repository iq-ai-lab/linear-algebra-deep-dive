# Ch2-07. 분해의 복잡도와 수치 안정성

> "이론에서는 모두 같은 답을 주지만, 유한 정밀도에서는 **어떤 분해를 선택하는가**가 결과를 결정한다."

## 📌 학습 목표

- 각 분해(LU, QR, Cholesky, Eigen, SVD)의 **flops** 복잡도를 통합 정리한다.
- **후방 오차(backward error)**와 **전방 오차(forward error)** 개념을 정의한다.
- **조건수 $\kappa(A)$**가 오차 증폭의 상한을 주는 부등식을 증명한다.
- 각 분해의 **안정성 등급**을 비교한다.

---

## 🎯 핵심 질문

> **질문 1**: 왜 전방 오차는 조건수 $\kappa(A)$에 비례하는가?
> **질문 2**: 왜 QR이 LU보다 **수치적으로 안전**하지만 **느린가**?
> **질문 3**: 실무에서 어떤 기준으로 분해를 선택하는가?

---

## 1. 복잡도 비교표

$A \in \mathbb{R}^{m \times n}$ ($m \geq n$)에 대해:

| 분해              | 조건              | Leading-order flops           | 메모리    |
| ----------------- | ----------------- | ----------------------------- | --------- |
| **LU** (no pivot) | 모든 leading minor 비특이 | $\tfrac{2}{3}n^3$       | $n^2$     |
| **PLU** (partial pivot) | 비특이            | $\tfrac{2}{3}n^3$       | $n^2$     |
| **Cholesky**      | 대칭 PD           | $\tfrac{1}{3}n^3$             | $n^2/2$   |
| **QR (Gram-Schmidt)** | full column rank | $2mn^2$                    | $mn$      |
| **QR (Householder)** | 일반             | $2mn^2 - \tfrac{2}{3}n^3$    | $mn$      |
| **Eigendecomposition** (symmetric) | 대칭   | $\approx 9n^3$             | $n^2$     |
| **Schur**         | 일반              | $\approx 25n^3$               | $n^2$     |
| **SVD**           | 일반              | $4mn^2 + 8n^3$ (reduced)      | $mn$      |

**관찰**:
- 대칭/PD일수록 빠르다 (Cholesky < LU)
- 직교성을 요구하면 느려진다 (QR > LU)
- 고윳값 계산은 반복적이며 $\sim 10-25 n^3$

---

## 2. 조건수

### 정의 2.1

정칙 $A \in \mathbb{R}^{n \times n}$의 **2-norm 조건수**:

$$
\kappa_2(A) = \|A\|_2 \|A^{-1}\|_2 = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
$$

### 정의 2.2 (일반 유도 노름)

$$
\kappa_p(A) = \|A\|_p \|A^{-1}\|_p
$$

### 성질

- $\kappa(A) \geq 1$ (단위행렬에서 1)
- $\kappa(AB) \leq \kappa(A)\kappa(B)$
- $\kappa(\alpha A) = \kappa(A)$
- 직교 $Q$: $\kappa_2(Q) = 1$

---

## 3. 전방 오차 분석

### 정리 3.1 (데이터 섭동 민감도)

$A \mathbf{x} = \mathbf{b}$가 $(A + \delta A)(\mathbf{x} + \delta \mathbf{x}) = \mathbf{b} + \delta \mathbf{b}$로 섭동되면 (단, $\|A^{-1}\| \|\delta A\| < 1$):

$$
\frac{\|\delta \mathbf{x}\|}{\|\mathbf{x}\|} \leq \frac{\kappa(A)}{1 - \kappa(A) \frac{\|\delta A\|}{\|A\|}} \left( \frac{\|\delta A\|}{\|A\|} + \frac{\|\delta \mathbf{b}\|}{\|\mathbf{b}\|} \right)
$$

### 증명 스케치

$A \delta \mathbf{x} = \delta \mathbf{b} - \delta A \cdot \mathbf{x} - \delta A \cdot \delta \mathbf{x}$, 재배열:

$$
(I + A^{-1} \delta A) \delta \mathbf{x} = A^{-1}(\delta \mathbf{b} - \delta A \cdot \mathbf{x})
$$

$\|A^{-1} \delta A\| < 1$이면 Neumann 급수: $\|(I + A^{-1}\delta A)^{-1}\| \leq 1/(1 - \|A^{-1}\delta A\|)$.

양변 노름, $\|\mathbf{x}\| \geq \|\mathbf{b}\|/\|A\|$로 나누어 정리. $\blacksquare$

### 해석

**전방 오차 ≈ 조건수 × 데이터 오차**. 조건수가 $10^6$이면 5자리 정밀도의 데이터에서 해답은 *유효숫자 없음*.

---

## 4. 후방 오차

### 정의 4.1

알고리즘의 수치 해 $\hat{\mathbf{x}}$가 $(A + \Delta A)\hat{\mathbf{x}} = \mathbf{b}$를 **정확히** 만족하는 작은 $\Delta A$가 존재하면, $\|\Delta A\|/\|A\|$를 **후방 오차**라 한다.

### 정의 4.2 (후방 안정)

$\|\Delta A\| = O(\epsilon_M)$이면 알고리즘이 **후방 안정(backward stable)**.

### 정리 4.3

후방 안정 알고리즘의 전방 오차:

$$
\frac{\|\hat{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq C \cdot \kappa(A) \cdot \epsilon_M
$$

**의미**: 알고리즘이 최선이라도 **문제의 조건수 한계까지는** 오차를 허용해야 한다.

---

## 5. 각 분해의 안정성

### 5.1 Gaussian Elimination (LU without pivoting)

**후방 오차**: $\|\Delta A\| \leq c \cdot \rho_n \cdot \epsilon_M \|A\|$

$\rho_n$: **growth factor** = $\max_{i,j,k} |u_{ij}^{(k)}| / \max |a_{ij}|$.

- No pivoting: $\rho_n$ 무제한 증가 가능 → **불안정**
- Partial pivoting: $\rho_n \leq 2^{n-1}$ 이론, 실전 $\rho_n \sim n^{1/2}$ → **대체로 안정**
- Complete pivoting: $\rho_n \leq O(n^{1/2} \log n)$ → **항상 안정**

### 5.2 Cholesky

**후방 안정** ($A$ PD). Pivoting 불필요.

$$
\|\Delta A\| \leq O(n \epsilon_M \|A\|)
$$

### 5.3 QR (Householder)

**가장 안정한 분해 중 하나**. $\|\Delta A\| \leq O(mn \epsilon_M \|A\|)$.

Classical Gram-Schmidt는 직교성 손실이 $\kappa^2 \epsilon_M$로 매우 나쁨.

### 5.4 SVD

**최고로 안정**. 모든 singular value가 최대 $O(\epsilon_M \|A\|)$의 절대오차로 계산됨. 조건수 계산의 황금 표준.

---

## 6. 실무적 분해 선택 가이드

### 6.1 의사결정 트리

```
A가 정칙인가?
├── Yes:
│   ├── 대칭인가?
│   │   ├── Yes: PD인가?
│   │   │   ├── Yes: → Cholesky (가장 빠름)
│   │   │   └── No: → LDL^T (Bunch-Kaufman)
│   │   └── No: → PLU (partial pivoting)
│   └── 고윳값/고유벡터 필요?
│       ├── Yes: → eigh (대칭) or eig (일반)
│       └── No: → LU or QR
└── No (또는 비정사각):
    ├── 최소 제곱: → QR (또는 SVD)
    ├── rank 결정, pseudoinverse: → SVD
    └── rank-k 근사: → truncated SVD
```

### 6.2 "큰 데이터" 변형

- $m \gg n$ (tall): QR이 자연스럽다. SVD도 $4mn^2 + 8n^3$ (reduced).
- $m \approx n$, 대칭 PD: Cholesky
- Sparse: 분해는 fill-in 발생. iterative methods (CG, GMRES) 선호.

### 6.3 조건수 진단

```python
# np.linalg.cond 사용, SVD 기반
kappa = np.linalg.cond(A)
if kappa > 1e10:
    print("⚠️ 악조건. SVD나 regularization 필요")
```

---

## 7. Python 실험: 조건수와 오차

### 7.1 Hilbert 행렬 (악명 높은 악조건)

```python
import numpy as np
import scipy.linalg as la

for n in [4, 8, 12, 16]:
    H = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
    x_true = np.ones(n)
    b = H @ x_true
    x_num = la.solve(H, b)
    err = np.linalg.norm(x_num - x_true) / np.linalg.norm(x_true)
    print(f"n={n}: kappa={np.linalg.cond(H):.2e}, err={err:.2e}")
# n=16: kappa ~ 10^18, err ~ O(1)   # 해가 완전히 망가짐
```

### 7.2 분해별 속도 비교

```python
import time
np.random.seed(0)
n = 2000
M = np.random.randn(n, n)
A_sym = (M + M.T) / 2 + n * np.eye(n)   # 대칭 PD
A_gen = np.random.randn(n, n)
b = np.random.randn(n)

def time_it(fn, trials=3):
    ts = []
    for _ in range(trials):
        t = time.time(); fn(); ts.append(time.time() - t)
    return min(ts)

print("Cholesky solve:  %.3f s" % time_it(lambda: la.cho_solve(la.cho_factor(A_sym), b)))
print("PLU solve:       %.3f s" % time_it(lambda: la.lu_solve(la.lu_factor(A_gen), b)))
print("QR solve:        %.3f s" % time_it(lambda: la.solve(A_gen, b, assume_a='gen')))
print("eigh (sym):      %.3f s" % time_it(lambda: np.linalg.eigh(A_sym)))
print("eig (gen):       %.3f s" % time_it(lambda: np.linalg.eig(A_gen)))
print("SVD:             %.3f s" % time_it(lambda: np.linalg.svd(A_gen)))
```

전형적 비율 (한 노트북에서):
- Cholesky : PLU : eigh : SVD ≈ 1 : 2 : 10 : 15

### 7.3 QR vs Normal Equations (Least Squares)

```python
# y = A x + noise, A: 100 x 10, 악조건
np.random.seed(1)
m, n = 100, 10
A = np.random.randn(m, n)
# 마지막 열을 거의 0으로 → 거의 rank-deficient
A[:, -1] = A[:, 0] * 1e-7 + np.random.randn(m) * 1e-9
x_true = np.ones(n)
b = A @ x_true + 1e-10 * np.random.randn(m)

# QR
Q, R = np.linalg.qr(A, mode='reduced')
x_qr = la.solve_triangular(R, Q.T @ b)

# 정규방정식
AtA = A.T @ A
Atb = A.T @ b
x_normal = la.solve(AtA, Atb)

print("QR error:    ", np.linalg.norm(x_qr - x_true))
print("Normal error:", np.linalg.norm(x_normal - x_true))
print("kappa(A):   ", np.linalg.cond(A))
print("kappa(A^TA):", np.linalg.cond(AtA))   # kappa(A)^2 → 매우 큼
```

---

## 8. 요약 체크리스트

| 질문 | 추천 분해 |
| ---- | --------- |
| 단순 $Ax=b$, 일반 행렬 | PLU |
| $A^T A \mathbf{x} = A^T \mathbf{b}$ (최소제곱, full rank) | Thin QR (Householder) |
| $A$ 대칭 PD | Cholesky |
| $A$ 대칭, 부정부호 | Bunch-Kaufman ($LDL^T$) |
| 고유값/고유벡터 필요 | `eigh` (symmetric) / `eig` (일반) |
| rank 또는 pseudoinverse 필요 | SVD |
| $A$ 매우 희소 | CG / GMRES / sparse LU |
| $A$ 거대 (rank-$k$ 근사) | Truncated / Randomized SVD |

---

## 9. 참고 문헌

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*.
- Demmel, J. W. (1997). *Applied Numerical Linear Algebra*. SIAM.
- Wilkinson, J. H. (1965). *The Algebraic Eigenvalue Problem*. Oxford. (classic)

---

## 10. 챕터 2 총정리

본 챕터에서 다룬 분해들을 관계도로 정리하면:

```
                  ┌────────────┐
                  │   A ∈ ℝ^n×n│
                  └──────┬─────┘
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    대칭 PD?         일반 정방?       비정사각 or 랭크?
       │                │                │
       ▼                ▼                ▼
    Cholesky          PLU / LU          QR / SVD
       │                │                │
       ▼                ▼                ▼
     고윳값?         고윳값?           최소제곱?
       │                │                │
       ▼                ▼                ▼
  Spectral (Q Λ Q^T)  Jordan / Schur   QR or SVD
```

→ Ch 3 (고윳값 심화)로 이어짐.

---

## 11. 내비게이션

[◀ 06. Jordan Form](./06-jordan-form.md) | [📚 README](../README.md) | [다음 챕터: 고윳값 심화 ▶](../ch3-eigenvalue-theory/01-characteristic-polynomial.md)
