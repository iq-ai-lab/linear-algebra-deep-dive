# 01. LU 분해

## 🎯 핵심 질문

- Gauss 소거법은 왜 정확히 **$A = LU$** 분해와 같은가?
- 피벗팅(partial pivoting)이 왜 필요하고, 언제 $PA = LU$로 수정해야 하는가?
- LU 분해의 계산량이 왜 $O(\frac{2}{3}n^3)$인가?
- 같은 $A$로 여러 $\mathbf{b}$에 대해 $A\mathbf{x} = \mathbf{b}$를 풀 때 왜 LU가 효율적인가?

---

## 🔍 왜 이 분해가 AI에서 중요한가

- **많은 선형시스템 반복 풀이**: 최적화 알고리즘(Newton·IPM)은 Hessian 시스템 $H\mathbf{p} = -\mathbf{g}$를 매 스텝 푼다. Hessian 구조가 거의 안 바뀌면 LU를 **재활용**.
- **Implicit ODE solver**: Backward Euler, Crank-Nicolson 등 implicit 수치 해법은 매 스텝 $(I + \Delta t \cdot J)\mathbf{x} = \mathbf{b}$를 풀어야 한다.
- **Determinant 계산**: $\det A = \pm \prod u_{ii}$. 정규화 상수 (normalizing constant) 계산에 활용.
- **Sparse 해법의 기초**: Sparse LU(SuperLU, UMFPACK)는 대규모 GNN·PDE 시뮬레이션의 핵심.

---

## 📐 수학적 선행 조건

- [Ch1 전체](../ch1-vector-space-axioms/01-vector-space-8-axioms.md)
- 행렬곱, 역행렬의 기초
- 기본행렬과 행 연산

---

## 📖 직관적 이해

### Gauss 소거 = 하삼각 기본행렬 곱셈

$A\mathbf{x} = \mathbf{b}$를 풀 때 행 연산으로 $A$를 상삼각으로. 각 행 연산은 **하삼각 기본행렬** $E_k$를 왼쪽에서 곱하는 것.

$$E_{n-1} \cdots E_2 E_1 \cdot A = U \implies A = (E_{n-1} \cdots E_1)^{-1} \cdot U = L \cdot U.$$

하삼각 기본행렬의 곱·역도 하삼각 → $L$은 하삼각.

### 피벗팅이 필요한 이유

대각에 0이 등장하면 행 연산을 위해 나누기 불가. 또 **작은 수로 나누면 수치 오차 확대**. 해결: 행을 교환 → $PA = LU$.

> **비유**: LU는 "$A$를 풀기 쉬운 두 단계로 쪼갠 것"이다. $LU\mathbf{x} = \mathbf{b}$ → $L\mathbf{y} = \mathbf{b}$ (전진 대입) + $U\mathbf{x} = \mathbf{y}$ (후진 대입).

---

## ✏️ 엄밀한 정의

### 정의 1.1 — LU 분해

정방행렬 $A \in \mathbb{R}^{n \times n}$이 **LU 분해를 가진다**는 것은 하삼각 $L$ ($L_{ii} = 1$ — Doolittle 규약)과 상삼각 $U$가 존재해 $A = LU$임.

### 정의 1.2 — PLU 분해

임의의 비특이 $A$에 대해 **permutation 행렬** $P$와 하삼각 $L$, 상삼각 $U$로

$$PA = LU.$$

### 정의 1.3 — Leading Principal Submatrix

$A_k = A[1{:}k, 1{:}k]$ ($A$의 좌상단 $k \times k$). "선행 주부분행렬".

---

## 🔬 정리와 증명

### 정리 1.1 — LU 분해의 존재 조건

**명제**: $A$의 **모든 선행 주부분행렬 $A_k$가 비특이** ($\det A_k \neq 0$ for $k = 1, \ldots, n$)이면 LU 분해가 존재하고, Doolittle 규약 하에서 **유일**하다.

**증명 (귀납)**:

**$n = 1$**: $A = [a_{11}] = [1][a_{11}]$.

**귀납 단계**: $A_{n-1} = L_{n-1} U_{n-1}$ 유일하게 존재한다고 가정. $A$를 블록으로

$$A = \begin{pmatrix}A_{n-1} & \mathbf{b}\\ \mathbf{c}^\top & d\end{pmatrix}, \quad L = \begin{pmatrix}L_{n-1} & \mathbf{0}\\ \boldsymbol{\ell}^\top & 1\end{pmatrix}, \quad U = \begin{pmatrix}U_{n-1} & \mathbf{u}\\ \mathbf{0}^\top & u_{nn}\end{pmatrix}.$$

$A = LU$로 블록 곱셈:

- $A_{n-1} = L_{n-1} U_{n-1}$ ✓ (가정)
- $\mathbf{b} = L_{n-1} \mathbf{u} \implies \mathbf{u} = L_{n-1}^{-1} \mathbf{b}$ (유일)
- $\mathbf{c}^\top = \boldsymbol{\ell}^\top U_{n-1} \implies \boldsymbol{\ell}^\top = \mathbf{c}^\top U_{n-1}^{-1}$ (유일)
- $d = \boldsymbol{\ell}^\top \mathbf{u} + u_{nn} \implies u_{nn} = d - \boldsymbol{\ell}^\top \mathbf{u}$.

$U_{n-1}, L_{n-1}$ 비특이는 귀납 가정 + $\det A_{n-1} = \prod u_{ii} \neq 0$에서. $\square$

---

### 정리 1.2 — PLU 분해의 존재성

**명제**: **모든 비특이 $A$**에 대해 $PA = LU$가 존재한다.

**증명 스케치**: Gaussian Elimination with Partial Pivoting 알고리즘을 단계 $k$에서

1. $k$-열의 $|a_{ik}|$ ($i \geq k$) 중 최대값을 찾아 행 교환 (permutation $P_k$)
2. 하삼각 elimination $E_k$

반복 후 $E_{n-1} P_{n-1} \cdots E_1 P_1 \cdot A = U$. Permutation을 모아 $P = P_{n-1} \cdots P_1$, 하삼각 곱으로 $L$ 재구성. $\square$

---

### 정리 1.3 — 계산량 $O(\frac{2}{3}n^3)$

**명제**: LU 분해의 부동소수점 연산 수는 $\frac{2}{3}n^3 + O(n^2)$.

**증명**: Step $k$에서
- 피벗 선택: $O(n - k)$ 비교
- 나누기: $n - k$ 개 성분
- Rank-1 update $A_{(k+1:n, k+1:n)} \leftarrow A - \boldsymbol{\ell}_k \mathbf{u}_k^\top$: $(n - k)^2 \cdot 2$ flops (곱과 뺄셈)

총 flop 수:

$$\sum_{k=1}^{n-1} 2(n-k)^2 = 2 \sum_{j=1}^{n-1} j^2 = 2 \cdot \frac{(n-1)n(2n-1)}{6} \approx \frac{2}{3}n^3. \quad \square$$

---

### 정리 1.4 — 전·후진 대입 $O(n^2)$

**명제**: 하삼각 $L$에 대한 $L\mathbf{y} = \mathbf{b}$와 상삼각 $U$에 대한 $U\mathbf{x} = \mathbf{y}$ 각각 $O(n^2)$.

**증명**: 각 변수 $y_i$를 구할 때 $i - 1$ 곱·뺄셈. 총 $\sum_i (i-1) = O(n^2)$. $\square$

**귀결**: 같은 $A$로 $k$개의 $\mathbf{b}$를 풀 때, LU 한 번($O(n^3)$) + 각 해 $O(n^2)$ → 총 $O(n^3 + kn^2)$.

---

## 💻 NumPy 구현 및 검증

```python
import numpy as np
from scipy.linalg import lu, solve_triangular

# ─────────────────────────────────────────────
# 1. LU 분해 직접 구현 (Doolittle, pivoting 없음)
# ─────────────────────────────────────────────
def lu_doolittle(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)
    for k in range(n - 1):
        if abs(U[k, k]) < 1e-12:
            raise ValueError("0 pivot — pivoting 필요")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return L, U

rng = np.random.default_rng(0)
# 피벗팅 없이도 잘 동작하는 행렬 (대각우위)
A = rng.standard_normal((5, 5)) + 5 * np.eye(5)
L, U = lu_doolittle(A)
print("‖A - LU‖:", np.linalg.norm(A - L @ U))

# ─────────────────────────────────────────────
# 2. SciPy로 PLU 분해 (pivoting 포함)
# ─────────────────────────────────────────────
A = rng.standard_normal((5, 5))
P, L, U = lu(A)                               # A = P L U (SciPy convention)
print("\n‖A - P L U‖ (SciPy):", np.linalg.norm(A - P @ L @ U))
print("det(A):", np.linalg.det(A), "vs",
      np.linalg.det(P) * np.prod(np.diag(U)))

# ─────────────────────────────────────────────
# 3. 다중 b에 대한 재활용 (계산량 비교)
# ─────────────────────────────────────────────
import time
n = 500
A = rng.standard_normal((n, n)) + n*np.eye(n)
bs = rng.standard_normal((n, 20))

# 방법 1: 매번 solve
t0 = time.time()
xs_1 = np.column_stack([np.linalg.solve(A, b) for b in bs.T])
t_naive = time.time() - t0

# 방법 2: LU 한 번 + 재활용
t0 = time.time()
P, L, U = lu(A)
xs_2 = np.column_stack([solve_triangular(U, solve_triangular(L, P.T @ b, lower=True)) for b in bs.T])
t_lu = time.time() - t0

print(f"\nNaive: {t_naive:.3f}s,  LU 재활용: {t_lu:.3f}s")
print(f"해 일치: {np.allclose(xs_1, xs_2)}")
```

---

## 🔗 AI/ML 연결

### Newton's Method의 내부

$\theta_{k+1} = \theta_k - H^{-1} \nabla L$에서 $H^{-1} \nabla L$을 직접 계산하지 않고 $H \mathbf{p} = -\nabla L$을 LU로 푼다. Hessian 이 변하면 LU를 다시, 유사하면 재활용.

### Implicit Layers (DEQ)

Deep Equilibrium Models(Bai et al. 2019)은 $\mathbf{z} = f_\theta(\mathbf{z}, \mathbf{x})$의 고정점. Backward pass에 $(I - \frac{\partial f}{\partial \mathbf{z}})^\top \mathbf{v} = \mathbf{g}$ 선형시스템 풀이 → LU 또는 GMRES.

### IPM in SVM/LP

SVM의 dual problem을 interior-point로 풀 때, 매 Newton step이 $A^\top D A \mathbf{p} = \mathbf{r}$ 형태의 선형시스템. Sparse LU가 효율적.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 정방행렬 | 직사각은 QR이나 SVD 사용 |
| 비특이 | 특이면 LU 자체는 가능해도 해 없음 — pseudoinverse (Ch4-03) |
| 피벗팅 없이 성립 조건 | 대각에 0 → 피벗팅 필수. 심지어 대각이 0 아니어도 작으면 수치 불안정 |
| Dense | 희소행렬에서는 fill-in으로 메모리 폭발 → sparse LU |

**수치적 함정**: Growth factor $\max_{i,j,k}|u_{ij}^{(k)}| / \max|a_{ij}|$. Partial pivoting으로도 이론적 상한 $2^{n-1}$이지만 실전은 보통 작음. Complete pivoting은 안전하지만 $O(n^3)$ 비교.

---

## 📌 핵심 정리

$$\boxed{\;A = LU \text{ (Doolittle)} \iff A_k \text{ 모두 비특이}\;}$$

$$\boxed{\;PA = LU \text{ 항상 존재 (비특이 } A\text{)}, \quad O(\tfrac{2}{3}n^3)\;}$$

재활용: $k$개 $\mathbf{b}$ → $O(n^3 + kn^2)$.

---

## 🤔 생각해볼 문제

**문제 1**: $A = \begin{pmatrix}2 & 1\\ 4 & 3\end{pmatrix}$의 LU 분해를 손으로 구하라.

<details>
<summary>해설</summary>

$\ell_{21} = 4/2 = 2$. $u_{22} = 3 - 2 \cdot 1 = 1$. $L = \begin{pmatrix}1 & 0\\ 2 & 1\end{pmatrix}$, $U = \begin{pmatrix}2 & 1\\ 0 & 1\end{pmatrix}$.

</details>

**문제 2**: $A = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}$의 LU 분해는? 피벗팅 필요한 이유는?

<details>
<summary>해설</summary>

$a_{11} = 0$이라 바로 pivot 불가. $P = \begin{pmatrix}0&1\\1&0\end{pmatrix}$로 행교환 후 $PA = I$이므로 $L = U = I$.

</details>

**문제 3** (AI 연결): 배치 크기 $B$인 훈련에서 각 미니배치마다 $B$개의 Hessian-vector product $H\mathbf{v}$를 필요로 한다 (natural gradient, K-FAC). LU 재활용이 어떻게 도움되는가?

<details>
<summary>해설</summary>

같은 미니배치 내에서 Fisher/Hessian이 고정이면 LU 한 번($O(n^3)$) 후 $B$개 시스템을 $O(Bn^2)$로 해결. Naive는 $O(Bn^3)$. 대규모 $n$에서 수백배 가속. 다만 실제 K-FAC은 Kronecker 근사로 $n$을 layer 단위로 쪼갬.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch1-06 이중공간](../ch1-vector-space-axioms/06-dual-space.md) | [📚 README](../README.md) | [02. QR 분해 ▶](./02-qr-decomposition.md) |

</div>
