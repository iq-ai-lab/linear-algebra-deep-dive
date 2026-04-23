# 5.5 QR 분해의 내적 관점 재해석

> "Gram-Schmidt는 기하적 절차이고, QR은 그 기하를 행렬 방정식으로 쓴 것이다."

---

## 1. 학습 목표

- **QR 분해**를 내적과 정사영의 관점에서 재해석한다.
- **Gram-Schmidt 과정**이 어떻게 QR 분해를 생성하는지 수식으로 유도한다.
- **수정된 Gram-Schmidt(MGS)** 와 **Householder 반사**의 내적 기반 해석을 본다.
- QR과 **Cholesky**, **Gram 행렬**의 삼각관계를 완성한다.
- QR을 활용한 최소제곱, Krylov 부분공간, 고유값 알고리즘의 연결을 조망한다.

---

## 2. QR 분해의 세 가지 시각

### 2.1 대수적 시각 (제2장)

$A \in \mathbb{R}^{m \times n}$ ($m \ge n$, 열이 선형독립)에 대해

$$
A = QR, \quad Q \in \mathbb{R}^{m \times n},\ Q^T Q = I_n,\ R \in \mathbb{R}^{n \times n}\ \text{상삼각, 양의 대각}
$$

이것이 QR 분해. 제2장에서는 Gram-Schmidt와 Householder를 통해 **구성적으로** 존재성과 유일성을 증명했다.

### 2.2 기하적 시각

$A$의 열공간 $\mathcal{R}(A) \subseteq \mathbb{R}^m$에 대한 **정규직교기저를 구성**하는 절차.

- $Q$의 열: $\mathcal{R}(A)$의 정규직교기저
- $R$: 원래 열을 이 기저로 표현한 계수들

### 2.3 내적 관점 시각 (본 절)

QR은 **Gram 행렬의 Cholesky 분해와 대응**한다:

$$
G = A^T A = R^T R
$$

즉 QR은 Gram 행렬을 "루트"로 취한 것에 해당한다. 이 관점에서 QR의 모든 성질이 내적 구조로부터 자연스럽게 유도된다.

---

## 3. Gram-Schmidt 재유도

### 3.1 문제 설정

$a_1, \ldots, a_n \in \mathbb{R}^m$이 선형독립이라 가정. 목표: 정규직교기저 $q_1, \ldots, q_n$을 구성하되

$$
\text{span}(q_1, \ldots, q_k) = \text{span}(a_1, \ldots, a_k), \quad \forall k
$$

를 만족하도록.

### 3.2 반복 절차

$q_k$가 $\text{span}(a_1, \ldots, a_k)$에 속해야 하므로 $a_k$에서 $\text{span}(q_1, \ldots, q_{k-1})$로의 정사영을 빼자:

$$
\tilde q_k = a_k - \sum_{i=1}^{k-1} \langle a_k, q_i \rangle\, q_i
$$

그 다음 정규화

$$
q_k = \frac{\tilde q_k}{\|\tilde q_k\|}
$$

역으로 $a_k$를 $q_i$들로 쓰면

$$
a_k = \sum_{i=1}^{k-1} \langle a_k, q_i \rangle\, q_i + \|\tilde q_k\|\, q_k
$$

### 3.3 QR로의 환원

$r_{ik} = \langle a_k, q_i \rangle$ ($i < k$), $r_{kk} = \|\tilde q_k\|$로 정의하면

$$
a_k = \sum_{i=1}^k r_{ik} q_i
$$

이것이 바로 $A = QR$의 제 $k$열. $R$이 상삼각인 것은 합의 상한 $i \le k$에서 즉시 보인다.

**유일성.** $r_{kk} > 0$으로 고정하면 (즉 $\tilde q_k$를 방향 그대로 정규화하면) QR은 유일하게 결정된다.

### 3.4 내적 공식

**내적의 정의가 Gram-Schmidt의 모든 것을 결정한다**:

| 양 | 내적 공식 |
|---|---|
| $r_{ik}\ (i < k)$ | $\langle a_k, q_i \rangle$ |
| $\tilde q_k$ | $a_k - \sum r_{ik} q_i$ |
| $r_{kk}$ | $\|\tilde q_k\|$ |
| $q_k$ | $\tilde q_k / r_{kk}$ |

---

## 4. QR = Gram의 Cholesky

### 4.1 핵심 등식

QR $A = QR$이 있다고 하자. 그러면

$$
A^T A = (QR)^T (QR) = R^T Q^T Q R = R^T R
$$

$R$은 상삼각 양의 대각, 따라서 $R^T R$은 **Cholesky 분해**이다!

$$
\boxed{A^T A = R^T R \quad (\text{Cholesky of } G)}
$$

역방향: Gram 행렬 $G = A^T A$의 Cholesky $G = R^T R$이 주어지면 $Q = A R^{-1}$로 $QR$이 복원된다.

### 4.2 수치적 관점의 교훈

이 대응 관계는 수치적으로 다음을 시사한다:

- 정규방정식 $A^T A x = A^T b$를 Cholesky로 풀면, $A^T A$를 **명시적으로 구성**하기 때문에 조건수가 $\kappa(A)^2$로 증가.
- QR로 풀면 $R x = Q^T b$만 풀면 되고, $A^T A$를 절대 만들지 않기 때문에 조건수가 $\kappa(A)$로 유지.

따라서 **최소제곱에는 QR이 Cholesky보다 수치적으로 우수하다**. QR과 Cholesky의 대수적 동치에도 불구하고 이 차이가 발생하는 이유는 **중간 단계의 rounding error 증폭**이다.

---

## 5. 수정된 Gram-Schmidt (MGS)

### 5.1 문제: 고전 GS의 수치적 취약성

고전 GS:

$$
\tilde q_k = a_k - \sum_{i=1}^{k-1} \langle a_k, q_i \rangle q_i
$$

모든 $\langle a_k, q_i \rangle$를 **$a_k$와 계산**한다. 반올림 오차로 $q_i$들이 완벽히 직교하지 않으면 $\tilde q_k$의 오차가 누적되어 결과가 **직교성을 잃는다**.

### 5.2 MGS: 한 번에 하나씩 빼기

$v := a_k$로 시작해서

```
for i = 1, ..., k-1:
    r_{ik} = <v, q_i>
    v := v - r_{ik} * q_i
r_{kk} = ||v||
q_k = v / r_{kk}
```

### 5.3 왜 MGS가 더 안정한가?

매번 내적을 계산할 때 **이미 직교한 부분이 제거된 $v$**와 $q_i$의 내적을 취한다. 따라서 $q_i$의 수치적 불완전성이 덜 영향을 준다.

**정량적 결과.** $Q$의 직교성 손실 $\|Q^T Q - I\|$는

- 고전 GS: $O(\varepsilon \cdot \kappa(A)^2)$
- MGS: $O(\varepsilon \cdot \kappa(A))$

### 5.4 내적 관점의 해석

MGS는 "$a_k$를 $q_1, \ldots, q_{k-1}$에 대해 순차적으로 정사영 제거"한다. 고전 GS는 "한꺼번에 정사영"한다. 둘은 **이론적으로 같지만** 수치 연산 순서가 다르다.

---

## 6. Householder 반사와 내적

### 6.1 Householder 정의 복습

단위벡터 $u$에 대한 Householder 반사:

$$
H = I - 2uu^T
$$

이는 $u$에 직교하는 초평면에 대한 반사이다.

### 6.2 반사의 내적 해석

임의의 $x$에 대해

$$
Hx = x - 2(u^T x) u = x - 2\langle x, u \rangle u
$$

즉 "**$x$에서 $u$ 방향 성분을 반대로 뒤집는다**". 이는 정사영의 **반대 방향 두 배**에 해당한다.

### 6.3 반사를 통한 삼각화

$x \in \mathbb{R}^m$을 $\pm \|x\| e_1$으로 보내는 Householder 벡터:

$$
u = \frac{x - \sigma \|x\| e_1}{\|x - \sigma \|x\| e_1\|}, \quad \sigma = -\text{sign}(x_1)
$$

이렇게 하면 $Hx = \sigma \|x\| e_1$. 이 연산을 차례로 적용하면 $A$의 첫 번째 열부터 아래가 0이 되며, 결국 상삼각 $R$을 얻는다:

$$
H_n \cdots H_2 H_1 A = R \quad \Rightarrow \quad A = (H_1 H_2 \cdots H_n) R = Q R
$$

### 6.4 내적-기반 해석

Householder는 **$x - \sigma \|x\| e_1$ 방향으로의 정사영 대칭**이다. Gram-Schmidt가 "덧셈형" (정사영 빼기)이라면 Householder는 "반사형"으로, 수치적으로 더 안정하다 (직교성 손실이 $O(\varepsilon)$).

---

## 7. 최소제곱법 (복습과 통합)

### 7.1 QR 기반 해법

$\min_x \|Ax - b\|$의 해는 $A = QR$을 이용해

$$
\|Ax - b\|^2 = \|QRx - b\|^2 = \|Rx - Q^T b\|^2 + \|(I - QQ^T) b\|^2
$$

(두 번째 항은 $x$와 무관). 따라서 최적해는

$$
R x = Q^T b \quad (\text{상삼각 system, back-sub})
$$

### 7.2 해석

- $Q^T b$: $b$를 $\mathcal{R}(A)$의 정규직교기저로 표현한 계수
- $R x = Q^T b$: 원래 열 기저의 계수로 환산
- 내적 관점: $\hat b = QQ^T b$는 $b$의 $\mathcal{R}(A)$로의 정사영, $x$는 그 정사영을 $a_i$ 기저로 표현한 것

### 7.3 QR이 왜 수치적으로 뛰어난가?

| 방법 | 해 | 조건수 |
|---|---|---|
| 정규방정식 | $G x = A^T b$ | $\kappa(A)^2$ |
| QR | $R x = Q^T b$ | $\kappa(A)$ |
| SVD | $x = V\Sigma^{-1} U^T b$ | $\kappa(A)$, 가장 안정 |

---

## 8. QR의 확장: Krylov 부분공간과 Arnoldi

### 8.1 Krylov 부분공간

행렬 $A$와 벡터 $b$에 대해

$$
\mathcal{K}_k(A, b) = \text{span}(b, Ab, A^2b, \ldots, A^{k-1}b)
$$

이것이 **Krylov 부분공간**. 큰 희소 행렬의 고유값/선형계 해를 반복적으로 근사하는 알고리즘의 토대.

### 8.2 Arnoldi 과정

$b, Ab, A^2b, \ldots$는 일반적으로 거의 선형종속(power iteration과 같이 $\lambda_1$-고유벡터로 수렴). 따라서 **Gram-Schmidt로 직교화**해야 수치적으로 유의미한 기저가 된다.

Arnoldi:

```
q_1 = b / ||b||
for k = 1, 2, ..., m:
    v = A q_k
    for i = 1, ..., k:
        h_{ik} = <v, q_i>
        v = v - h_{ik} q_i
    h_{k+1, k} = ||v||
    q_{k+1} = v / h_{k+1, k}
```

결과: 상Hessenberg 행렬 $H$와 $Q$에 대해

$$
A Q_k = Q_{k+1} \tilde H_k
$$

### 8.3 대칭 경우: Lanczos

$A$가 대칭이면 Hessenberg가 삼중대각이 되고, Arnoldi는 **Lanczos**로 단순화된다:

$$
\beta_{k+1} q_{k+1} = A q_k - \alpha_k q_k - \beta_k q_{k-1}
$$

3항 점화식만으로 직교 기저를 생성한다 (이론적으로). 이는 내적 구조가 매우 강할 때 QR의 구조가 단순화되는 전형적 예시.

---

## 9. QR 알고리즘 (고유값)

### 9.1 알고리즘

$A_0 = A$에서 시작:

```
for k = 0, 1, 2, ...:
    A_k = Q_k R_k   (QR 분해)
    A_{k+1} = R_k Q_k
```

### 9.2 왜 수렴하는가 (직관)

$A_{k+1} = R_k Q_k = Q_k^T (Q_k R_k) Q_k = Q_k^T A_k Q_k$

즉 $A_{k+1}$은 $A_k$의 닮음변환(similarity), 고유값 동일. 그리고 $A_k$는 **동시 거듭제곱(simultaneous power iteration)** 과 동치라는 것을 보일 수 있다:

$$
A^k = (Q_0 Q_1 \cdots Q_{k-1})(R_{k-1} \cdots R_1 R_0)
$$

즉 $Q_0 \cdots Q_{k-1}$은 $A^k$의 QR 분해의 $Q$. 이 $Q$는 점차 $A$의 고유벡터 기저로 수렴한다.

### 9.3 Hessenberg 축소와 shift

- Hessenberg 형태로 만들어 계산량을 $O(n^2)$로 감소 (Hessenberg QR은 한 단계당 $O(n^2)$).
- Wilkinson shift로 수렴을 **입방적**으로 가속.

---

## 10. Python 실험

### 10.1 QR과 Cholesky의 일치 확인

```python
import numpy as np

np.random.seed(0)
m, n = 8, 5
A = np.random.randn(m, n)

# QR 분해
Q, R = np.linalg.qr(A)

# Cholesky
G = A.T @ A
L = np.linalg.cholesky(G)  # G = L L^T, L 하삼각

# R^T = L? (sign convention에 따라 ±)
print("QR의 R (상삼각, 양의 대각 조정):")
print(np.diag(np.sign(np.diag(R))) @ R)
print()
print("Cholesky의 L^T:")
print(L.T)
```

### 10.2 고전 GS vs. MGS 수치 안정성

```python
import numpy as np

def classical_gs(A):
    m, n = A.shape
    Q = np.zeros_like(A)
    R = np.zeros((n, n))
    for k in range(n):
        Q[:, k] = A[:, k].copy()
        for i in range(k):
            R[i, k] = Q[:, i] @ A[:, k]
            Q[:, k] -= R[i, k] * Q[:, i]
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] /= R[k, k]
    return Q, R

def modified_gs(A):
    m, n = A.shape
    Q = A.copy()
    R = np.zeros((n, n))
    for k in range(n):
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] /= R[k, k]
        for j in range(k+1, n):
            R[k, j] = Q[:, k] @ Q[:, j]
            Q[:, j] -= R[k, j] * Q[:, k]
    return Q, R

# 조건수가 나쁜 행렬
np.random.seed(1)
n = 10
A = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])

Q_cgs, _ = classical_gs(A)
Q_mgs, _ = modified_gs(A)
Q_np, _  = np.linalg.qr(A)

I = np.eye(n)
print(f"직교성 손실:")
print(f"  Classical GS: {np.linalg.norm(Q_cgs.T @ Q_cgs - I):.2e}")
print(f"  Modified GS:  {np.linalg.norm(Q_mgs.T @ Q_mgs - I):.2e}")
print(f"  NumPy (HH):   {np.linalg.norm(Q_np.T  @ Q_np  - I):.2e}")
```

### 10.3 QR 알고리즘 직접 구현

```python
import numpy as np

def qr_algorithm(A, num_iter=100):
    A = A.copy().astype(float)
    for _ in range(num_iter):
        Q, R = np.linalg.qr(A)
        A = R @ Q
    return A

np.random.seed(0)
n = 5
M = np.random.randn(n, n)
A = M + M.T  # 대칭이면 수렴 빠름

A_final = qr_algorithm(A)
eigs_qr = np.sort(np.diag(A_final))
eigs_np = np.sort(np.linalg.eigvalsh(A))

print("QR 알고리즘:", eigs_qr)
print("NumPy:     ", eigs_np)
print("오차:       ", np.max(np.abs(eigs_qr - eigs_np)))
```

### 10.4 Householder QR 직접 구현

```python
import numpy as np

def householder_qr(A):
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
        v = v / np.linalg.norm(v)
        R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2 * np.outer(Q[:, k:] @ v, v)
    return Q, R

np.random.seed(0)
A = np.random.randn(5, 3)
Q, R = householder_qr(A)

print(f"A = QR error: {np.linalg.norm(A - Q @ R[:len(A[0]), :]):.2e}")
print(f"Q orthogonal: {np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0])):.2e}")
```

### 10.5 Arnoldi 과정

```python
import numpy as np

def arnoldi(A, b, k):
    n = A.shape[0]
    Q = np.zeros((n, k+1))
    H = np.zeros((k+1, k))
    Q[:, 0] = b / np.linalg.norm(b)
    for j in range(k):
        v = A @ Q[:, j]
        for i in range(j+1):
            H[i, j] = Q[:, i] @ v
            v -= H[i, j] * Q[:, i]
        H[j+1, j] = np.linalg.norm(v)
        if H[j+1, j] < 1e-12:
            break
        Q[:, j+1] = v / H[j+1, j]
    return Q, H

np.random.seed(0)
n = 50
A = np.random.randn(n, n)
b = np.random.randn(n)

Q, H = arnoldi(A, b, k=10)

# Hessenberg 행렬의 고유값은 A의 고유값에 수렴 (Ritz values)
ritz = np.linalg.eigvals(H[:10, :])
eigs_A = np.linalg.eigvals(A)

print("Ritz 고유값:     ", np.sort(np.abs(ritz))[-5:])
print("A의 상위 고유값:", np.sort(np.abs(eigs_A))[-5:])
```

---

## 11. 요약 및 다음 장으로

### 핵심 결과

| 결과 | 공식/의미 |
|---|---|
| QR = GS | $q_k = (a_k - \sum \langle a_k, q_i\rangle q_i) / \|\cdot\|$ |
| QR ↔ Cholesky | $A^T A = R^T R$ |
| Householder | $H = I - 2uu^T$, 반사 대칭 |
| 수치 안정성 | Householder > MGS > CGS |
| 최소제곱 | $Rx = Q^T b$ |
| Krylov | Arnoldi = GS in $\mathcal{K}_k$ |
| QR 알고리즘 | $A_{k+1} = R_k Q_k$ $\to$ Schur |

### 한 줄 요약

> **QR 분해는 "Gram-Schmidt를 행렬 방정식으로 쓴 것"이며, 내적 구조의 Cholesky 루트이다.**

### Chapter 5 정리

제5장에서는 내적공간의 네 가지 기둥을 순서대로 쌓아올렸다:

1. **내적 & Cauchy-Schwarz**: 기하의 기초 재료
2. **정사영**: 최적 근사의 기하 원리
3. **최소제곱**: 정사영의 대수적 구체화
4. **Gram 행렬 & PSD**: 내적 구조의 행렬 압축
5. **QR 재해석**: 모든 조각의 통합

이로써 내적공간의 이론-계산-응용이 **하나의 관점**으로 통합되었다.

### 다음 장 예고

제6장에서는 **텐서**로 확장한다. 스칼라(0-tensor), 벡터(1-tensor), 행렬(2-tensor)을 일반화하여 다차원 배열과 다중선형 사상의 대수를 전개하며, Kronecker 곱, Einstein 합, CP/Tucker 분해를 다룬다.

---

[◀ 04. Gram 행렬과 PSD](./04-gram-matrix-psd.md) | [📚 README](../README.md) | [6장: 텐서와 다중선형 대수 ▶](../ch6-tensor/01-tensor-definition.md)
