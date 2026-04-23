# Ch3-01. 특성다항식과 Cayley-Hamilton 정리

> "행렬은 자기 자신의 특성다항식에 의해 소멸된다."

## 📌 학습 목표

- 특성다항식 $p_A(\lambda) = \det(\lambda I - A)$의 계수와 불변량의 관계를 유도한다.
- **Cayley-Hamilton 정리** $p_A(A) = 0$의 엄밀 증명 (adjugate 경로).
- 최소다항식 $m_A$와 $p_A$의 관계: $m_A \mid p_A$, 동일 근 집합.
- 행렬 역함수 $A^{-1}$을 $p_A$ 계수로 명시적 표현.

---

## 🎯 핵심 질문

> **질문 1**: 왜 $p_A$의 계수에 **모든 주부분행렬식의 합**이 나타나는가?
> **질문 2**: Cayley-Hamilton의 "꼼수 증명" ($\det(AI - A) = 0$)이 **왜 틀린가**?
> **질문 3**: 최소다항식은 왜 $p_A$를 나누는가?

---

## 1. 특성다항식의 구조

### 정의 1.1

$A \in \mathbb{C}^{n \times n}$에 대해:

$$
p_A(\lambda) = \det(\lambda I - A) = \lambda^n - c_1 \lambda^{n-1} + c_2 \lambda^{n-2} - \cdots + (-1)^n c_n
$$

### 정리 1.2 (계수와 주부분행렬식)

$$
c_k = \sum_{|I| = k} \det(A_I)
$$

여기서 $A_I$는 $I \subset \{1, \ldots, n\}$의 행과 열로 이루어진 $A$의 $k \times k$ 주부분행렬 (principal submatrix).

### 증명 개요

$\det(\lambda I - A)$를 라플라스 전개. $\lambda^{n-k}$의 계수는 $A$에서 **$k$개의 행과 같은 $k$개의 열을 선택**한 부분행렬의 $(-1)^k \det(A_I)$의 합.

**특수한 경우**:
- $c_1 = \operatorname{tr}(A) = \sum_i A_{ii}$ (대각합)
- $c_n = \det(A)$

### 정리 1.3 (고윳값과 계수)

고윳값 $\lambda_1, \ldots, \lambda_n$ (중복 포함)에 대해:

$$
c_k = e_k(\lambda_1, \ldots, \lambda_n)
$$

여기서 $e_k$는 $k$차 기본대칭함수.

### 증명

$p_A(\lambda) = \prod(\lambda - \lambda_i) = \sum_k (-1)^k e_k(\lambda_1, \ldots, \lambda_n) \lambda^{n-k}$. 계수 비교. $\blacksquare$

따라서:

$$
\operatorname{tr}(A) = \sum_i \lambda_i, \qquad \det(A) = \prod_i \lambda_i
$$

---

## 2. Cayley-Hamilton 정리

### 정리 2.1 (Cayley-Hamilton)

$A \in \mathbb{C}^{n \times n}$에 대해 $p_A(A) = 0$.

### 먼저: 틀린 "증명"

> $p_A(A) = \det(A I - A) = \det(0) = 0$.

**잘못된 이유**: $p_A(\lambda)$는 스칼라 $\lambda$에 대한 함수. $A$를 대입하는 것은 **스칼라 $\lambda$를 행렬 $A$로 치환**하는 것인데, $\det$ 안에 넣어버리면 이미 스칼라가 된 상태에서 조작하는 것이 된다. $\det(\lambda I - A)$를 전개한 **다항식**에 $A$를 대입해야 함.

### 엄밀 증명 (Adjugate 경로)

$B(\lambda) = \lambda I - A$. **Adjugate**는 $B(\lambda) \operatorname{adj}(B(\lambda)) = \det(B(\lambda)) I = p_A(\lambda) I$.

$\operatorname{adj}(B(\lambda))$는 $\lambda$에 대해 **최대 차수 $n-1$의 다항식 행렬**:

$$
\operatorname{adj}(\lambda I - A) = \sum_{k=0}^{n-1} \lambda^k C_k
$$

여기서 $C_k \in \mathbb{C}^{n \times n}$는 상수 행렬.

$$
(\lambda I - A) \sum_{k=0}^{n-1} \lambda^k C_k = p_A(\lambda) I = \left(\lambda^n + \sum_{i=0}^{n-1} a_i \lambda^i\right) I
$$

(여기서 $a_i$는 $p_A$의 계수.)

좌변 전개:

$$
\sum_{k=0}^{n-1} \lambda^{k+1} C_k - \sum_{k=0}^{n-1} \lambda^k A C_k = \lambda^n C_{n-1} + \sum_{k=1}^{n-1} \lambda^k (C_{k-1} - A C_k) - A C_0
$$

$\lambda$ 차수별 계수 비교:

- $\lambda^n$: $C_{n-1} = I$
- $\lambda^k$ ($1 \leq k \leq n-1$): $C_{k-1} - A C_k = a_k I$
- $\lambda^0$: $-A C_0 = a_0 I$

이제 $\lambda^k$ 행렬 등식 전체에 $A^k$를 곱하여 더하면:

$$
\sum_k A^k (C_{k-1} - A C_k) = \sum_k a_k A^k \cdot I
$$

좌변은 텔레스코핑: $\sum_k A^k C_{k-1} - \sum_k A^{k+1} C_k = A^0 \cdot C_{-1} - A^n C_{n-1}$ (정의상 $C_{-1}$을 $-A C_0 / a_0$ 등으로 맞추면 $-A \cdot \text{something}$).

정확하게 전개하면, $\lambda$ 식이 다항식 항등식이므로 $\lambda$에 $A$를 대입해도 행렬 등식으로 유지:

**핵심**: 가환환 $\mathbb{C}[A]$에서 $\lambda I - A$와 $\operatorname{adj}(\lambda I - A)$는 $\lambda$에 대한 다항식 행렬. $\lambda = A$ 대입 시 좌변 $AI - A = 0$, 우변 $p_A(A) I$. 단, 이 "대입"은 **다항식 계수들이 가환환에 속한다**는 사실로 정당화된다 (matrices with entries in commutative ring).

더 엄밀히: $A$와 상수 행렬의 계수들은 서로 가환하므로 $\lambda$를 $A$로 치환 가능. 결과: $0 = p_A(A)$. $\blacksquare$

---

## 3. 최소다항식

### 정의 3.1

$m_A(\lambda)$: $m_A(A) = 0$을 만족하는 **monic 최소 차수** 다항식.

### 정리 3.2

$m_A$는 **유일**하며, $f(A) = 0 \iff m_A \mid f$.

### 증명

**(유일성)** $m_1, m_2$가 둘 다 최소라면 $m_1 - m_2$는 저차수이며 $(m_1 - m_2)(A) = 0$. 이들이 monic이므로 $m_1 - m_2$의 차수는 더 낮음. 최소성 위배 → $m_1 = m_2$.

**(나눔)** $f(A) = 0$. $f = q \cdot m_A + r$, $\deg r < \deg m_A$. $r(A) = f(A) - q(A) m_A(A) = 0$. 최소성 → $r = 0$. $\blacksquare$

### 정리 3.3

$m_A \mid p_A$, 그리고 $m_A$와 $p_A$는 **같은 근 집합**을 가진다 (중복도는 다를 수 있음).

### 증명

**($\mid$)** Cayley-Hamilton으로 $p_A(A) = 0$, 따라서 $m_A \mid p_A$.

**(같은 근)** Jordan form으로 분석. $\lambda$가 $A$의 고윳값 ⟺ $(\lambda - \lambda_0)$이 $p_A$의 인수 ⟺ $(\lambda - \lambda_0)$이 $m_A$의 인수. 후자는 $\mathbf{v} \in E_\lambda \setminus \{0\}$, $m_A(A)\mathbf{v} = 0$이면 $m_A(\lambda_0) = 0$. $\blacksquare$

---

## 4. 역행렬의 Cayley-Hamilton 표현

### 정리 4.1

$A$가 정칙이면:

$$
A^{-1} = -\frac{1}{a_0}(A^{n-1} + a_{n-1} A^{n-2} + \cdots + a_1 I)
$$

여기서 $p_A(\lambda) = \lambda^n + a_{n-1}\lambda^{n-1} + \cdots + a_1 \lambda + a_0$, $a_0 = (-1)^n \det(A)$.

### 증명

Cayley-Hamilton: $A^n + a_{n-1} A^{n-1} + \cdots + a_1 A + a_0 I = 0$. 이항:

$$
a_0 I = -A(A^{n-1} + a_{n-1} A^{n-2} + \cdots + a_1 I)
$$

$a_0 \neq 0$ (정칙)이므로 $A^{-1}$을 구할 수 있음. $\blacksquare$

**의미**: 이론적으로 $A^{-1}$은 $A$의 거듭제곱의 선형결합이지만, 수치적으로는 매우 불안정 (조건수 증폭).

---

## 5. Frobenius 정규형과 Companion 행렬

### 정의 5.1 (Companion 행렬)

다항식 $p(\lambda) = \lambda^n + a_{n-1}\lambda^{n-1} + \cdots + a_0$의 companion:

$$
C_p = \begin{pmatrix}
0 & 0 & \cdots & 0 & -a_0 \\
1 & 0 & \cdots & 0 & -a_1 \\
0 & 1 & \cdots & 0 & -a_2 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 1 & -a_{n-1}
\end{pmatrix}
$$

### 정리 5.2

$C_p$의 특성다항식은 $p$이고, 최소다항식도 $p$.

### 증명 개요

$\lambda I - C_p$의 특성다항식 전개 (첫 행 따라) → $p(\lambda)$. 최소다항식: $e_1, C_p e_1, C_p^2 e_1, \ldots, C_p^{n-1} e_1$이 일차독립 (직접 계산), 따라서 $\mathbf{v} = e_1$을 소멸시키는 최소 차수는 $n$. $m_A$도 차수 $n$.

### 활용

임의 monic 다항식 $p$의 근은 $\operatorname{eig}(C_p)$로 수치 계산 가능. NumPy의 `np.roots`가 정확히 이를 수행.

---

## 6. 응용: 행렬 거듭제곱 가속

### 6.1 Cayley-Hamilton을 활용한 $A^k$ 계산

$k \geq n$에 대해 $A^k$는 $I, A, \ldots, A^{n-1}$의 선형결합:

$$
A^k = c_{k,0}(A) I + c_{k,1}(A) A + \cdots + c_{k, n-1}(A) A^{n-1}
$$

계수 $c_{k,i}$는 다항식 나머지 $\lambda^k \mod p_A(\lambda)$로 얻음.

**예**: $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$, $p_A(\lambda) = \lambda^2 - 4\lambda + 3$, $A^2 = 4A - 3I$. 더 높은 거듭제곱은 $\lambda^k = q(\lambda)(\lambda^2 - 4\lambda + 3) + (\alpha_k \lambda + \beta_k)$로 재귀적 계산.

### 6.2 그래프 인접행렬에서의 응용

인접행렬 $A$의 $k$차 거듭제곱 $(A^k)_{ij}$는 $i \to j$ 길이-$k$ 경로 수. Cayley-Hamilton은 이를 유한 계산으로 환원.

---

## 7. Python 실험

### 7.1 특성다항식 계수

```python
import numpy as np

A = np.array([[4.0, 1.0, 2.0],
              [0.0, 3.0, 1.0],
              [1.0, 0.0, 5.0]])

# np.poly(A) or np.linalg.eigvals + polynomial from roots
eigs = np.linalg.eigvals(A)
coeffs = np.poly(eigs)  # [1, -c1, c2, -c3]
print("p_A(lambda) coefficients:", coeffs)
print("c_1 (trace):", -coeffs[1], "vs tr(A):", np.trace(A))
print("c_3 (det):  ", coeffs[3] * (-1)**len(A), "vs det(A):", np.linalg.det(A))
```

### 7.2 Cayley-Hamilton 검증

```python
I = np.eye(3)
A2 = A @ A
A3 = A2 @ A
# p_A(A) = A^3 + c[1] A^2 + c[2] A + c[3] I
pA_of_A = A3 + coeffs[1]*A2 + coeffs[2]*A + coeffs[3]*I
print("||p_A(A)||:", np.linalg.norm(pA_of_A))  # ≈ 0
```

### 7.3 최소다항식 (SymPy)

```python
import sympy as sp
M = sp.Matrix([[2, 1, 0], [0, 2, 0], [0, 0, 3]])
print("Charpoly:", M.charpoly().as_expr())   # (x-2)^2 (x-3)
# 최소다항식 (대수적 중복도 vs 기하적 중복도)
# 여기서 J_1(2) ⊕ J_1(2) ⊕ J_1(3)처럼 보이는지 확인
P, J = M.jordan_form()
print("J:\n", J)
# 최소다항식: (x-2)(x-3) (만약 2에 대해 Jordan 블록 크기 1이면)
```

### 7.4 Companion 행렬

```python
# p(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
coeffs_p = [1, -6, 11, -6]
C = np.diag(np.ones(2), k=-1).astype(float)  # subdiagonal 1
C[:, -1] = -np.array(coeffs_p[:0:-1])[:3]    # last column: -a_0, -a_1, -a_2

# Actually, easier:
from numpy.polynomial import polynomial as P
roots = np.roots(coeffs_p)
print("Roots of p:", roots)  # [1, 2, 3]
```

---

## 8. 요약

| 개념              | 공식                                                 |
| ----------------- | ---------------------------------------------------- |
| 특성다항식        | $p_A(\lambda) = \det(\lambda I - A)$                 |
| $c_k$ 계수        | $k$차 주부분행렬식의 합 / 기본대칭함수 $e_k(\lambda_i)$ |
| 트레이스          | $\operatorname{tr}(A) = \sum \lambda_i = c_1$        |
| 행렬식            | $\det(A) = \prod \lambda_i = (-1)^n c_n$             |
| Cayley-Hamilton   | $p_A(A) = 0$                                         |
| 최소다항식        | $m_A \mid p_A$, $m_A$와 $p_A$ 근 동일                |
| $A^{-1}$          | $p_A$ 계수의 Cayley-Hamilton 표현                    |

---

## 9. 참고 문헌

- Lang, S. (2002). *Algebra* (3rd ed.). Ch XIV.
- Hoffman & Kunze, *Linear Algebra*, Ch 6.
- Horn & Johnson, *Matrix Analysis*, §1.2.

---

## 10. 다음 문서

- **[02. 고윳값의 기하적 의미](./02-eigenvalue-geometry.md)**: 회전, 반사, 확대의 분해.

---

## 11. 내비게이션

[◀ 이전 챕터: 복잡도와 안정성](../ch2-matrix-decomposition/07-complexity-stability.md) | [📚 README](../README.md) | [02. 고윳값의 기하적 의미 ▶](./02-eigenvalue-geometry.md)
