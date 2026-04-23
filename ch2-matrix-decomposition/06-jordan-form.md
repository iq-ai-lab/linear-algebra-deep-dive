# Ch2-06. Jordan 표준형: 결함 행렬의 구조

> "대각화가 불가능할 때, **Jordan 블록**이 그 다음으로 가장 단순한 구조이다."

## 📌 학습 목표

- 일반화된 고유벡터(generalized eigenvector)의 정의와 성질을 유도한다.
- Jordan 블록 $J_k(\lambda)$의 대수적 구조를 파악한다.
- Jordan 표준형 $A = P J P^{-1}$의 **존재와 유일성**(permutation 제외)을 증명한다.
- 블록 크기가 $\dim \ker(A - \lambda I)^j$의 차이로 결정됨을 보인다.

---

## 🎯 핵심 질문

> **질문 1**: 대각화 불가 행렬이 존재하는데, **왜 Jordan 형태까지는 보장되는가**?
> **질문 2**: Jordan 블록의 크기는 무엇으로 결정되는가?
> **질문 3**: Jordan 형태는 수치적으로 왜 불안정한가?

---

## 1. 일반화된 고유공간

### 정의 1.1

$A \in \mathbb{C}^{n \times n}$, 고윳값 $\lambda$에 대해 **일반화된 고유공간**은:

$$
K_\lambda = \ker(A - \lambda I)^n
$$

### 정리 1.2

$K_\lambda$는 $A$-불변이고 $\dim K_\lambda = m_a(\lambda)$ (대수적 중복도).

### 증명 개요

$A - \lambda I$ 제한을 생각하면 $K_\lambda$ 위에서 niltpotent. $K_\lambda$의 존재는 **Cayley-Hamilton 정리**로부터 ($(A - \lambda I)^{m_a}$가 $K_\lambda$ 위에서 영).

차원 부분은 §3의 블록 구조 구축 후 명확해진다.

---

## 2. 일반화된 고유벡터 사슬

### 정의 2.1

$\mathbf{v} \in K_\lambda$가 **Jordan 사슬(chain)의 꼭대기**라 함은 $(A - \lambda I)^k \mathbf{v} = \mathbf{0}$이지만 $(A - \lambda I)^{k-1} \mathbf{v} \neq \mathbf{0}$인 $k$가 있을 때.

이 $\mathbf{v}$에서 출발하여:

$$
\mathbf{v}_k = \mathbf{v}, \quad \mathbf{v}_{k-1} = (A - \lambda I) \mathbf{v}_k, \quad \ldots, \quad \mathbf{v}_1 = (A - \lambda I) \mathbf{v}_2
$$

를 얻으면 $\mathbf{v}_1$은 (일반) 고유벡터: $(A - \lambda I) \mathbf{v}_1 = \mathbf{0}$.

### Jordan 블록의 등장

기저 $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$에서 $A$는:

$$
A \mathbf{v}_j = \lambda \mathbf{v}_j + \mathbf{v}_{j-1} \quad (j \geq 2), \qquad A \mathbf{v}_1 = \lambda \mathbf{v}_1
$$

행렬로:

$$
[A]_{\{\mathbf{v}_i\}} = J_k(\lambda) = \begin{pmatrix} \lambda & 1 & & \\ & \lambda & \ddots & \\ & & \ddots & 1 \\ & & & \lambda \end{pmatrix}_{k \times k}
$$

이것이 **Jordan 블록**.

---

## 3. Jordan 표준형 정리

### 정리 3.1 (Jordan Normal Form)

$A \in \mathbb{C}^{n \times n}$에 대해 가역 $P$가 존재하여:

$$
A = P J P^{-1}, \quad J = \operatorname{diag}(J_{k_1}(\lambda_1), J_{k_2}(\lambda_2), \ldots, J_{k_r}(\lambda_r))
$$

블록의 크기·개수·고윳값은 **순서 제외 유일**하게 결정된다.

### 증명 스케치 (존재성)

**단계 1**: $\mathbb{C}^n = K_{\lambda_1} \oplus K_{\lambda_2} \oplus \cdots \oplus K_{\lambda_p}$ (서로 다른 고윳값에 대한 일반화된 고유공간의 직합).

이는 primary decomposition theorem: $p_A(\lambda) = \prod (\lambda - \lambda_i)^{m_a(\lambda_i)}$의 인수들이 서로소이므로 $\mathbb{C}[A]$-모듈 관점에서 직합.

**단계 2**: 각 $K_\lambda$ 위에서 $N = (A - \lambda I)$는 nilpotent. Nilpotent $N$에 대한 Jordan 기저를 구성.

**단계 3 (Nilpotent Jordan 기저 구성)**: $N^k = 0$, $N^{k-1} \neq 0$이라 하자. 다음을 정의:

- $V_j = \ker N^j$, $V_0 = \{0\} \subset V_1 \subset \cdots \subset V_k = K_\lambda$

$V_{k}$에서 $V_{k-1}$ 보수 공간의 기저 ↔ 최대 길이 사슬의 꼭대기. 그 사슬들을 계산하여 $V_{k-1} \setminus V_{k-2}$에 대해 반복. 이 과정이 Young diagram 구조를 만든다.

### 유일성 증명

블록 수 (길이 $\geq j$)는 다음과 같이 결정:

$$
\#\{\text{길이} \geq j \text{ Jordan 블록}\} = \dim \ker(A - \lambda I)^j - \dim \ker(A - \lambda I)^{j-1}
$$

이 차원 수열이 $A$의 불변량이므로 Jordan 구조가 유일.

---

## 4. Jordan 블록의 대수

### 정리 4.1 ($J_k(\lambda)$의 성질)

- 고윳값: $\lambda$ (유일, 중복도 $k$)
- 기하적 중복도: 1
- $(J_k(\lambda) - \lambda I)^k = 0$, $(J_k(\lambda) - \lambda I)^{k-1} \neq 0$
- 최소다항식: $(x - \lambda)^k$

### 증명

$N = J_k(\lambda) - \lambda I$는 **초대각(superdiagonal)이 1인 nilpotent**. $N^j$는 초대각을 $j$칸 위로 이동.

$\ker N = \operatorname{span}(\mathbf{e}_1)$ (1차원), $\ker N^j = \operatorname{span}(\mathbf{e}_1, \ldots, \mathbf{e}_j)$ ($j$차원). 따라서 $\dim \ker N = 1$ 즉 기하적 중복도 1. $\blacksquare$

### 정리 4.2 ($J_k(\lambda)$의 거듭제곱)

$$
J_k(\lambda)^n = \sum_{j=0}^{k-1} \binom{n}{j} \lambda^{n-j} N^j
$$

(이항정리, $\lambda I$와 $N$ 가환성 활용.)

---

## 5. 최소다항식

### 정의 5.1 (최소다항식)

$A$의 **최소다항식** $m_A(x)$는 $m_A(A) = 0$을 만족하는 **monic 최소 차수** 다항식.

### 정리 5.2

$m_A(x) = \prod_\lambda (x - \lambda)^{s(\lambda)}$, 여기서 $s(\lambda)$는 $\lambda$-블록 중 **최대** 크기.

### 정리 5.3 (Cayley-Hamilton)

$p_A(A) = 0$. 즉 $m_A \mid p_A$.

### 증명

Jordan form $A = P J P^{-1}$에서 $p_A(\lambda) = \prod (\lambda - \lambda_i)^{m_a(\lambda_i)}$. $p_A(J) = \operatorname{blockdiag}(p_A(J_{k_i}(\lambda_i)))$. 각 블록에서 $(J - \lambda_i I)^{m_a(\lambda_i)} = 0$ (nilpotent, exponent $\leq m_a$).

$p_A(A) = P p_A(J) P^{-1} = 0$. $\blacksquare$

---

## 6. 행렬 함수

### 정리 6.1 (Jordan 블록의 함수)

$f$가 $\lambda$ 근방에서 해석적이면:

$$
f(J_k(\lambda)) = \sum_{j=0}^{k-1} \frac{f^{(j)}(\lambda)}{j!} N^j = \begin{pmatrix}
f(\lambda) & f'(\lambda) & \frac{f''(\lambda)}{2!} & \cdots & \frac{f^{(k-1)}(\lambda)}{(k-1)!} \\
 & f(\lambda) & f'(\lambda) & \cdots & \frac{f^{(k-2)}(\lambda)}{(k-2)!} \\
 & & \ddots & \ddots & \vdots \\
 & & & f(\lambda) & f'(\lambda) \\
 & & & & f(\lambda)
\end{pmatrix}
$$

### 응용 6.2 (행렬 지수)

$$
e^{J_k(\lambda) t} = e^{\lambda t} \sum_{j=0}^{k-1} \frac{(Nt)^j}{j!}
$$

선형미분방정식 $\dot{\mathbf{x}} = A\mathbf{x}$의 해 $\mathbf{x}(t) = e^{At} \mathbf{x}_0$에서 결함 행렬의 해는 $t^j e^{\lambda t}$ 꼴의 다항 × 지수.

---

## 7. 수치적 불안정성

### 관찰 7.1

Jordan form은 **좌표 불변 연속함수로 계산 불가**: 임의의 작은 섭동 $A + \epsilon E$는 보통 단순 고윳값을 가지며(대각화 가능), 원래의 Jordan 구조와 완전히 다르다.

### 예시

$A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ ($J_2(0)$). $A + \epsilon \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ \epsilon & 0 \end{pmatrix}$의 고윳값 $\pm\sqrt{\epsilon}$. $\epsilon \to 0$에서 고윳값은 0으로 수렴하지만 **대각화 가능**. 고유벡터는 극단적으로 병진.

### 실무적 처방

- **Jordan은 이론 도구**, 실무 수치 계산에서는 쓰지 않는다.
- **Schur 분해** $A = U T U^H$ (상삼각)이 수치 안정한 대체.
- 결함 근처에서는 Schur + pseudo-eigendecomposition / Arnoldi 사용.

---

## 8. Python 실험

### 8.1 Jordan 블록 만들기

```python
import numpy as np
import sympy as sp

# NumPy에는 Jordan 계산기가 없음 (수치적으로 불안정)
# 심볼릭: SymPy
A = sp.Matrix([[5, 4, 2, 1],
               [0, 1, -1, -1],
               [-1, -1, 3, 0],
               [1, 1, -1, 2]])
P, J = A.jordan_form()
print("J =\n", J)
print("P =\n", P)
```

### 8.2 Nilpotent 거듭제곱

```python
k = 4
N = np.diag(np.ones(k-1), k=1)  # J_k(0) - 0*I
print("N:\n", N)
print("N^2:\n", np.linalg.matrix_power(N, 2))
print("N^3:\n", np.linalg.matrix_power(N, 3))
print("N^4:\n", np.linalg.matrix_power(N, 4))  # zero
```

### 8.3 수치 불안정성

```python
eps = 1e-8
A0 = np.array([[0.0, 1.0], [0.0, 0.0]])
A_pert = A0 + eps * np.array([[0.0, 0.0], [1.0, 0.0]])

print("Eigvals original:", np.linalg.eigvals(A0))       # [0, 0]
print("Eigvals perturbed:", np.linalg.eigvals(A_pert))  # [+sqrt(eps), -sqrt(eps)]
# 1e-8 섭동이 고윳값을 1e-4만큼 움직임 → 매우 민감
```

### 8.4 Schur 분해 (수치 안정 대체)

```python
from scipy.linalg import schur
A = np.random.randn(5, 5)
T, Z = schur(A)
print("T (upper triangular):\n", np.triu(T, -1))  # 0 below first subdiag (complex case → real block Schur)
print("||A - Z T Z^T||:", np.linalg.norm(A - Z @ T @ Z.T))
```

---

## 9. 요약

| 개념                    | 정의/의미                                       |
| ----------------------- | ----------------------------------------------- |
| 일반화된 고유공간       | $K_\lambda = \ker(A - \lambda I)^n$             |
| Jordan 사슬             | $(A - \lambda I)$-chain: $\mathbf{v}_k \to \mathbf{v}_{k-1} \to \cdots$ |
| Jordan 블록 $J_k(\lambda)$ | 대각 $\lambda$, 초대각 1                     |
| Jordan form 유일성      | 블록 개수 = $\dim\ker N^j - \dim\ker N^{j-1}$   |
| 최소다항식              | $(x-\lambda)^{s(\lambda)}$, $s$ = 최대 블록 크기 |
| Cayley-Hamilton         | $p_A(A) = 0$                                    |

**핵심**: 모든 $n \times n$ 복소 행렬은 **Jordan form까지는 도달 가능**, 그러나 **수치적으로 매우 민감**.

---

## 10. 참고 문헌

- Horn & Johnson, *Matrix Analysis*, Ch 3.
- Axler, *Linear Algebra Done Right*, Ch 8.
- Golub & Van Loan, *Matrix Computations*, §7.6 (수치 이슈).

---

## 11. 내비게이션

[◀ 05. 스펙트럼 정리](./05-spectral-theorem.md) | [📚 README](../README.md) | [07. 복잡도와 안정성 ▶](./07-complexity-stability.md)
