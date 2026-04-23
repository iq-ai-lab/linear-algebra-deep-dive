# 6.2 Kronecker 곱과 Vec 연산

> "Kronecker 곱은 텐서를 행렬에 가두는 '언어 변환 장치'다."

---

## 1. 학습 목표

- **Kronecker 곱 $A \otimes B$** 의 정의와 블록 구조를 이해한다.
- Kronecker 곱의 기본 성질 (혼합곱, 전치, 역, 고유값, 특잇값)을 유도한다.
- **Vec 연산자**와 Kronecker 곱의 관계 $\text{vec}(AXB) = (B^T \otimes A)\text{vec}(X)$를 증명한다.
- Sylvester 방정식 $AX + XB = C$를 Kronecker 곱으로 변환해 푸는 방법을 본다.
- 텐서곱 공간의 **구체적 계산 모델**로서의 역할을 정리한다.

---

## 2. Kronecker 곱의 정의

### 2.1 정의

$A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times q}$일 때 **Kronecker 곱** $A \otimes B \in \mathbb{R}^{mp \times nq}$는

$$
A \otimes B = \begin{pmatrix}
a_{11} B & a_{12} B & \cdots & a_{1n} B \\
a_{21} B & a_{22} B & \cdots & a_{2n} B \\
\vdots & & & \vdots \\
a_{m1} B & a_{m2} B & \cdots & a_{mn} B
\end{pmatrix}
$$

즉 $A$의 각 성분을 $B$로 대체하는 블록 행렬이다.

### 2.2 지표 표기

$(A \otimes B)_{(i-1)p + k,\ (j-1)q + l} = a_{ij} b_{kl}$.

더 간결하게 (이중 지표로):

$$
(A \otimes B)_{(ik),(jl)} = a_{ij} b_{kl}
$$

### 2.3 예시

$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 5 \\ 6 & 7 \end{pmatrix}$이면

$$
A \otimes B = \begin{pmatrix}
0 & 5 & 0 & 10 \\
6 & 7 & 12 & 14 \\
0 & 15 & 0 & 20 \\
18 & 21 & 24 & 28
\end{pmatrix}
$$

**주의**: $A \otimes B \ne B \otimes A$ (일반적으로). 다만 **치환 행렬에 의한 유사행렬**이다 (후술).

---

## 3. 기본 성질

### 3.1 결합성, 분배성

- $(A \otimes B) \otimes C = A \otimes (B \otimes C)$
- $A \otimes (B + C) = A \otimes B + A \otimes C$
- $(kA) \otimes B = A \otimes (kB) = k(A \otimes B)$

### 3.2 전치와 에르미트 전치

$$
(A \otimes B)^T = A^T \otimes B^T, \quad (A \otimes B)^H = A^H \otimes B^H
$$

### 3.3 혼합곱 공식 (Mixed-Product Property)

**정리 6.2.1 (Mixed-Product).** 차원이 맞는 $A, B, C, D$에 대해

$$
\boxed{(A \otimes B)(C \otimes D) = (AC) \otimes (BD)}
$$

**증명.**

$(i, k), (j, l)$-성분:

$$
\sum_{(a, b)} (A \otimes B)_{(ik),(ab)} (C \otimes D)_{(ab),(jl)} = \sum_{a, b} a_{ia} b_{kb} c_{aj} d_{bl}
$$

$$
= \left(\sum_a a_{ia} c_{aj}\right)\left(\sum_b b_{kb} d_{bl}\right) = (AC)_{ij} (BD)_{kl} = ((AC) \otimes (BD))_{(ik),(jl)}. \qquad \blacksquare
$$

### 3.4 역행렬

$A, B$ 가역이면

$$
(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}
$$

**증명.** $(A \otimes B)(A^{-1} \otimes B^{-1}) = (AA^{-1}) \otimes (BB^{-1}) = I \otimes I = I$. $\blacksquare$

### 3.5 행렬식과 대각합

**정리 6.2.2.** $A \in \mathbb{R}^{m \times m}$, $B \in \mathbb{R}^{n \times n}$에 대해

- $\text{tr}(A \otimes B) = \text{tr}(A) \cdot \text{tr}(B)$
- $\det(A \otimes B) = (\det A)^n (\det B)^m$

**증명 (det).**

$A$의 Schur 분해 $A = U_A T_A U_A^*$, $B$의 Schur 분해 $B = U_B T_B U_B^*$. 혼합곱 공식으로

$$
A \otimes B = (U_A \otimes U_B)(T_A \otimes T_B)(U_A^* \otimes U_B^*) = (U_A \otimes U_B)(T_A \otimes T_B)(U_A \otimes U_B)^*
$$

$T_A$의 대각이 $\alpha_i$, $T_B$의 대각이 $\beta_j$라면 $T_A \otimes T_B$의 대각은 $\alpha_i \beta_j$ ($i = 1, \ldots, m$, $j = 1, \ldots, n$).

$$
\det(A \otimes B) = \prod_{i, j} \alpha_i \beta_j = \left(\prod_i \alpha_i\right)^n \left(\prod_j \beta_j\right)^m = (\det A)^n (\det B)^m. \qquad \blacksquare
$$

**대각합 증명**은 간단한 합 계산으로 생략.

### 3.6 고유값과 특잇값

**정리 6.2.3.** $A$의 고유값을 $\alpha_1, \ldots, \alpha_m$, $B$의 고유값을 $\beta_1, \ldots, \beta_n$이라 하자.

- $A \otimes B$의 고유값: $\alpha_i \beta_j$ ($mn$개, 중복 허용)
- $A \oplus B := A \otimes I_n + I_m \otimes B$의 고유값: $\alpha_i + \beta_j$ (**Kronecker 합**)

마찬가지로 특잇값에 대해서도

- $A \otimes B$의 특잇값: $\sigma_i(A) \sigma_j(B)$

**대응 고유벡터.** $A u_i = \alpha_i u_i$, $B v_j = \beta_j v_j$이면

$$
(A \otimes B)(u_i \otimes v_j) = (A u_i) \otimes (B v_j) = (\alpha_i \beta_j)(u_i \otimes v_j)
$$

---

## 4. Vec 연산자

### 4.1 정의

$X \in \mathbb{R}^{m \times n}$에 대해 **vec** 연산은 **열을 차례로 쌓아** 긴 벡터로 만든다:

$$
\text{vec}(X) = \begin{pmatrix} X_{:,1} \\ X_{:,2} \\ \vdots \\ X_{:,n} \end{pmatrix} \in \mathbb{R}^{mn}
$$

지표 표기: $\text{vec}(X)_{(j-1)m + i} = X_{ij}$.

### 4.2 핵심 등식

**정리 6.2.4 (vec-Kronecker 관계).** 차원이 맞는 행렬 $A, X, B$에 대해

$$
\boxed{\text{vec}(A X B) = (B^T \otimes A)\, \text{vec}(X)}
$$

**증명.**

$(AXB)_{ij} = \sum_{k, l} A_{ik} X_{kl} B_{lj}$. $\text{vec}$의 $(j-1)m + i$번째 성분:

$$
[\text{vec}(AXB)]_{(j-1)m + i} = \sum_{k, l} A_{ik} B_{lj} X_{kl} = \sum_{k, l} (A_{ik} B_{lj}) [\text{vec}(X)]_{(l-1)n + k}
$$

여기서 $B_{lj}$가 $B^T$의 $(j, l)$ 성분임에 주의. $B^T \otimes A$의 $((j-1)m + i, (l-1)n + k)$ 성분은

$$
(B^T \otimes A)_{(ji),(lk)} = (B^T)_{jl} A_{ik} = B_{lj} A_{ik}
$$

따라서 일치. $\blacksquare$

### 4.3 파생 공식

$$
\text{vec}(AB) = (I \otimes A)\text{vec}(B) = (B^T \otimes I)\text{vec}(A)
$$

$$
\text{tr}(A^T B) = \text{vec}(A)^T \text{vec}(B)
$$

$$
\text{tr}(ABCD) = \text{vec}(D)^T (C^T \otimes A)\text{vec}(B) = \text{vec}(A^T)^T (D^T \otimes B^T)\text{vec}(C^T)
$$

---

## 5. 응용 1: Sylvester 방정식

### 5.1 문제

주어진 $A, B, C$에 대해 다음을 푸는 $X$를 찾는다:

$$
AX + XB = C
$$

### 5.2 Kronecker 곱으로 변환

양변에 vec를 적용:

$$
\text{vec}(AX) + \text{vec}(XB) = \text{vec}(C)
$$

$$
(I \otimes A)\text{vec}(X) + (B^T \otimes I)\text{vec}(X) = \text{vec}(C)
$$

$$
\boxed{(I_n \otimes A + B^T \otimes I_m)\, \text{vec}(X) = \text{vec}(C)}
$$

이는 선형계. 해가 유일하게 존재할 조건은 좌측 계수행렬이 가역, 즉

$$
\alpha_i + \beta_j \ne 0 \quad \forall i, j
$$

즉 $A$와 $-B$가 **고유값을 공유하지 않음**.

### 5.3 Lyapunov 방정식

특수 경우 $B = A^T$, $C = -Q$ (대칭)이면

$$
AX + XA^T = -Q
$$

이는 제어 이론의 **Lyapunov 방정식**. $A$가 안정(모든 고유값이 좌반평면)이면 유일 해가 존재하고, $Q$가 PSD이면 $X$도 PSD.

### 5.4 실무 해법

Kronecker 곱 시스템은 $mn \times mn$ 크기가 되어 $m = n = 100$이면 $10^8$ 성분. 직접 풀지 않고 **Bartels-Stewart 알고리즘** (Schur 분해 후 순차 풀이, $O(m^3 + n^3)$)을 사용.

---

## 6. 응용 2: 양자역학의 합성 시스템

### 6.1 힐베르트 공간 텐서곱

두 양자계 $\mathcal{H}_A$, $\mathcal{H}_B$의 합성계는 $\mathcal{H}_A \otimes \mathcal{H}_B$로 기술된다. 연산자도 $O_A \otimes O_B$ 형태.

### 6.2 얽힘(Entanglement)

$|\psi\rangle \in \mathcal{H}_A \otimes \mathcal{H}_B$이 분해되지 않을 때, 즉

$$
|\psi\rangle \ne |\phi_A\rangle \otimes |\phi_B\rangle\ \text{(어떤 } |\phi_A\rangle, |\phi_B\rangle \text{에 대해서도)}
$$

이를 **얽힘 상태**라 한다. 예: $|\psi\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ (Bell 상태).

### 6.3 얽힘 진단: Schmidt 분해

$|\psi\rangle = \sum_{ij} c_{ij} |i\rangle_A \otimes |j\rangle_B$를 행렬 $C = [c_{ij}]$로 보면, **SVD** $C = U \Sigma V^T$로부터

$$
|\psi\rangle = \sum_k \sigma_k |\tilde u_k\rangle_A \otimes |\tilde v_k\rangle_B
$$

(Schmidt 분해). 비영 특잇값의 개수 = **Schmidt rank** = 얽힘의 척도.

---

## 7. 응용 3: 그래프 곱과 동적 시스템

### 7.1 그래프의 Kronecker 곱

그래프 $G, H$의 **텐서 곱(Kronecker 곱)** 의 인접행렬은 $A_G \otimes A_H$. 고유값이 $\lambda_i \mu_j$이므로 spectrum이 결정적이다.

### 7.2 헤로난 매트릭스, Hadamard 곱

- 카르테시안 곱: $A_G \square A_H = A_G \otimes I + I \otimes A_H$ (Kronecker 합)
- 강한 곱: $A_G \boxtimes A_H = (A_G + I) \otimes (A_H + I) - I \otimes I$

대규모 네트워크의 모델링에 활용 (예: 스토캐스틱 Kronecker 그래프).

---

## 8. 교환자 행렬과 vec의 변환

### 8.1 교환자 행렬

$K_{m, n} \in \mathbb{R}^{mn \times mn}$은 다음을 만족하는 치환 행렬:

$$
K_{m, n}\, \text{vec}(X) = \text{vec}(X^T), \quad X \in \mathbb{R}^{m \times n}
$$

### 8.2 주요 성질

- $K_{m, n}^{-1} = K_{m, n}^T = K_{n, m}$
- $K_{n, m}(A \otimes B) K_{p, q} = B \otimes A$ ($A \in \mathbb{R}^{m \times p}, B \in \mathbb{R}^{n \times q}$)

즉 $A \otimes B$와 $B \otimes A$는 "행/열의 치환" 하에서 같다.

---

## 9. 특수 Kronecker 구조

### 9.1 Kronecker-factored 행렬 (K-FAC)

딥러닝 2차 최적화에서 Fisher 정보행렬 $F$를 $F \approx A \otimes B$로 근사. 역행렬 $F^{-1} \approx A^{-1} \otimes B^{-1}$을 쉽게 계산 (Natural Gradient K-FAC).

### 9.2 Hadamard 곱과의 혼합

$\text{diag}(v)(A)\text{diag}(w) = A \odot (vw^T)$ ($\odot$: 성분별 곱). Kronecker, Hadamard, 일반 행렬곱이 얽힌 공식은 "행렬 매직"의 주요 원천.

### 9.3 텐서-트레인 (Tensor Train) 분해

고차 텐서를 $T = T_1 \otimes T_2 \otimes \cdots \otimes T_d$의 "사슬" 형태로 분해 (후속 절 6.4 참고).

---

## 10. Python 실험

### 10.1 Kronecker 곱의 기본

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 5], [6, 7]])

K = np.kron(A, B)
print(K)
print(f"Shape: {K.shape}")

# Mixed-product 검증
C = np.random.randn(2, 3)
D = np.random.randn(2, 3)
lhs = np.kron(A, B) @ np.kron(C, D)
rhs = np.kron(A @ C, B @ D)
print(f"(A⊗B)(C⊗D) = (AC)⊗(BD): {np.allclose(lhs, rhs)}")
```

### 10.2 고유값 확인

```python
import numpy as np

np.random.seed(0)
A = np.random.randn(3, 3)
B = np.random.randn(4, 4)

# Kronecker 곱의 고유값 = α_i β_j
K = np.kron(A, B)
eigs_K  = np.sort(np.linalg.eigvals(K))
eigs_AB = np.sort(np.outer(np.linalg.eigvals(A), np.linalg.eigvals(B)).ravel())

print(f"최대 오차: {np.max(np.abs(eigs_K - eigs_AB)):.2e}")
```

### 10.3 vec와 Kronecker 공식

```python
import numpy as np

np.random.seed(0)
m, n, p, q = 3, 4, 5, 2

A = np.random.randn(m, n)
X = np.random.randn(n, p)
B = np.random.randn(p, q)

# 직접 계산
Y_direct = A @ X @ B
vec_Y_direct = Y_direct.reshape(-1, order='F')  # column-major

# vec(AXB) = (B^T ⊗ A) vec(X)
vec_X = X.reshape(-1, order='F')
vec_Y_kron = np.kron(B.T, A) @ vec_X

print(f"일치: {np.allclose(vec_Y_direct, vec_Y_kron)}")
```

### 10.4 Sylvester 방정식

```python
import numpy as np
from scipy.linalg import solve_sylvester

np.random.seed(0)
m, n = 5, 4
A = np.random.randn(m, m)
B = np.random.randn(n, n)
C = np.random.randn(m, n)

# Method 1: SciPy
X_scipy = solve_sylvester(A, B, C)

# Method 2: Kronecker 곱 (소규모에서만)
I_m = np.eye(m)
I_n = np.eye(n)
M = np.kron(I_n, A) + np.kron(B.T, I_m)
vec_C = C.reshape(-1, order='F')
vec_X = np.linalg.solve(M, vec_C)
X_kron = vec_X.reshape(m, n, order='F')

print(f"잔차 (SciPy): {np.linalg.norm(A @ X_scipy + X_scipy @ B - C):.2e}")
print(f"잔차 (Kron):  {np.linalg.norm(A @ X_kron  + X_kron  @ B - C):.2e}")
print(f"두 해의 차이: {np.linalg.norm(X_scipy - X_kron):.2e}")
```

### 10.5 Schmidt 분해로 얽힘 측정

```python
import numpy as np

def schmidt_rank(psi, dim_A, dim_B, tol=1e-10):
    C = psi.reshape(dim_A, dim_B)
    s = np.linalg.svd(C, compute_uv=False)
    return np.sum(s > tol), s

# Bell 상태: (|00> + |11>) / sqrt(2)
psi_bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
rank, s = schmidt_rank(psi_bell, 2, 2)
print(f"Bell 상태: rank = {rank}, 특잇값 = {s}")

# 분리 가능: |0> ⊗ |+> = (|00> + |01>)/sqrt(2)
psi_sep = np.array([1, 1, 0, 0]) / np.sqrt(2)
rank, s = schmidt_rank(psi_sep, 2, 2)
print(f"분리상태: rank = {rank}, 특잇값 = {s}")
```

---

## 11. 요약 및 다음 절 예고

### 핵심 결과

| 결과 | 공식 |
|---|---|
| 정의 | $(A \otimes B)_{(ik),(jl)} = a_{ij} b_{kl}$ |
| 혼합곱 | $(A \otimes B)(C \otimes D) = AC \otimes BD$ |
| 역, 전치 | $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$, $(A \otimes B)^T = A^T \otimes B^T$ |
| 고유값 | $\alpha_i \beta_j$ (Kronecker), $\alpha_i + \beta_j$ (Kronecker 합) |
| vec 공식 | $\text{vec}(AXB) = (B^T \otimes A)\text{vec}(X)$ |
| Sylvester | $(I \otimes A + B^T \otimes I) \text{vec}(X) = \text{vec}(C)$ |

### 한 줄 요약

> **Kronecker 곱은 텐서곱을 블록 행렬로 구현한 것이고, vec 연산자와 함께 텐서 방정식을 선형계로 환원한다.**

### 다음 절 예고

다음 절에서는 **Einstein 합 규약**과 `np.einsum`을 다룬다. 텐서 네트워크 계산의 언어이자, 수식과 코드 사이의 다리 역할을 하는 표기법이다.

---

[◀ 01. 텐서의 정의](./01-tensor-definition.md) | [📚 README](../README.md) | [03. einsum ▶](./03-einsum.md)
