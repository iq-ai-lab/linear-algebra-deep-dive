# Ch2-04. 고유값 분해: 대각화의 조건

> "고유벡터가 충분히 많으면, 행렬은 그 방향에 대해 단순한 스칼라 곱이 된다."

## 📌 학습 목표

- 고윳값과 고유벡터의 대수적/기하적 정의를 공리부터 유도한다.
- 특성다항식과 **대수적/기하적 중복도**의 관계를 증명한다.
- **대각화 가능성**의 필요충분조건: 기하적 중복도 합 $= n$.
- 서로 다른 고윳값에 대응하는 고유벡터의 **일차독립성**을 증명한다.

---

## 🎯 핵심 질문

> **질문 1**: 왜 $\det(A - \lambda I) = 0$인가?
> **질문 2**: 기하적 중복도가 대수적 중복도보다 **항상 작거나 같은** 이유는?
> **질문 3**: 비대각화 가능한 행렬(defective)이 존재하는 이유는?

---

## 1. 정의: 고윳값과 고유벡터

### 정의 1.1

$A \in \mathbb{C}^{n \times n}$에 대해 $\mathbf{v} \in \mathbb{C}^n \setminus \{\mathbf{0}\}$과 $\lambda \in \mathbb{C}$가 $A \mathbf{v} = \lambda \mathbf{v}$를 만족하면 $\lambda$를 **고윳값(eigenvalue)**, $\mathbf{v}$를 **고유벡터(eigenvector)**라 한다.

### 동치 조건

$A\mathbf{v} = \lambda \mathbf{v} \iff (A - \lambda I) \mathbf{v} = \mathbf{0} \iff \mathbf{v} \in \ker(A - \lambda I)$

따라서 $\lambda$가 고윳값이려면 $\ker(A - \lambda I) \neq \{\mathbf{0}\}$ 즉 $A - \lambda I$가 비가역:

$$
\boxed{\det(A - \lambda I) = 0}
$$

### 정의 1.2 (특성다항식)

$$
p_A(\lambda) = \det(\lambda I - A)
$$

$p_A$는 $\lambda$에 대해 **차수 $n$**의 monic 다항식.

---

## 2. 고유공간과 중복도

### 정의 2.1 (고유공간)

$$
E_\lambda = \ker(A - \lambda I) = \{\mathbf{v} : A\mathbf{v} = \lambda \mathbf{v}\}
$$

($\mathbf{0}$ 포함된 벡터 공간)

### 정의 2.2 (중복도)

- **대수적 중복도** $m_a(\lambda)$: $p_A(\lambda)$에서 $(\lambda - \lambda_0)$의 중근 차수
- **기하적 중복도** $m_g(\lambda)$: $\dim E_\lambda$

### 정리 2.3

$$
1 \leq m_g(\lambda) \leq m_a(\lambda)
$$

### 증명

**(하한)** $\lambda$가 고윳값이면 정의상 $E_\lambda \neq \{\mathbf{0}\}$, $m_g \geq 1$.

**(상한)** $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$를 $E_\lambda$의 기저라 하자 ($k = m_g$). $\mathbb{C}^n$의 기저로 확장: $B = \{\mathbf{v}_1, \ldots, \mathbf{v}_k, \mathbf{w}_1, \ldots, \mathbf{w}_{n-k}\}$.

$P = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_k \mid \mathbf{w}_1 \mid \cdots \mid \mathbf{w}_{n-k}]$에 대해:

$$
P^{-1} A P = \begin{pmatrix} \lambda I_k & B_{12} \\ 0 & B_{22} \end{pmatrix}
$$

(첫 $k$열 $A \mathbf{v}_i = \lambda \mathbf{v}_i$이므로 $P^{-1} A \mathbf{v}_i = \lambda \mathbf{e}_i$.)

$\det(\mu I - P^{-1}AP) = \det(\mu I - A)$이므로:

$$
p_A(\mu) = (\mu - \lambda)^k \det(\mu I_{n-k} - B_{22})
$$

따라서 $p_A$에서 $(\mu - \lambda)$의 중근 차수는 **적어도** $k$, 즉 $m_a(\lambda) \geq k = m_g(\lambda)$. $\blacksquare$

---

## 3. 서로 다른 고윳값과 일차독립

### 정리 3.1

$\lambda_1, \ldots, \lambda_k$가 **서로 다른** 고윳값이고 $\mathbf{v}_i$가 각각의 고유벡터이면 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$는 일차독립.

### 증명 (귀납법)

$k = 1$: $\mathbf{v}_1 \neq \mathbf{0}$.

$k - 1$까지 성립한다고 가정. $c_1 \mathbf{v}_1 + \cdots + c_k \mathbf{v}_k = \mathbf{0}$에 $A$를 곱:

$$
c_1 \lambda_1 \mathbf{v}_1 + \cdots + c_k \lambda_k \mathbf{v}_k = \mathbf{0}
$$

원식에 $\lambda_k$를 곱해서 빼면:

$$
c_1 (\lambda_1 - \lambda_k) \mathbf{v}_1 + \cdots + c_{k-1}(\lambda_{k-1} - \lambda_k) \mathbf{v}_{k-1} = \mathbf{0}
$$

귀납가정으로 $\{\mathbf{v}_1, \ldots, \mathbf{v}_{k-1}\}$ 독립, 또 $\lambda_i \neq \lambda_k$이므로 $c_i = 0$ ($i < k$). 따라서 $c_k \mathbf{v}_k = \mathbf{0}$, $\mathbf{v}_k \neq \mathbf{0}$이므로 $c_k = 0$. $\blacksquare$

---

## 4. 대각화 가능성

### 정의 4.1 (대각화)

$A$가 대각화 가능(diagonalizable)하다는 것은 가역 $P$와 대각 $D$가 존재하여:

$$
A = P D P^{-1}
$$

### 정리 4.2 (대각화 가능 동치 조건)

다음은 동치:

1. $A$가 대각화 가능
2. $A$의 고유벡터가 $\mathbb{C}^n$의 **기저**를 이룬다
3. $\sum_\lambda m_g(\lambda) = n$
4. 모든 $\lambda$에 대해 $m_g(\lambda) = m_a(\lambda)$

### 증명

**(1 ⇒ 2)** $AP = PD$에서 $P$의 $i$번째 열 $\mathbf{p}_i$는 $A \mathbf{p}_i = d_i \mathbf{p}_i$, 즉 고유벡터. $P$ 가역이므로 기저.

**(2 ⇒ 1)** $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$이 기저이고 $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$이면 $P = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_n]$, $D = \operatorname{diag}(\lambda_i)$.

**(2 ⇔ 3)** 정리 3.1에 의해 서로 다른 고윳값의 고유공간의 합은 직합: $E_{\lambda_1} \oplus \cdots \oplus E_{\lambda_r}$. 이 공간의 차원은 $\sum m_g(\lambda_i)$. 이것이 $n$과 같아야 $\mathbb{C}^n$ 전체를 생성.

**(3 ⇔ 4)** $\sum m_a = n$ (특성다항식 차수), $m_g \leq m_a$이고 $\sum m_g = n$이어야 모든 $\lambda$에서 등호. $\blacksquare$

---

## 5. 대각화 공식

### 정리 5.1 (고윳값 분해)

$A$가 대각화 가능하면:

$$
A = P D P^{-1} = \sum_{i=1}^n \lambda_i \mathbf{p}_i \mathbf{q}_i^T
$$

여기서 $\mathbf{p}_i$는 $P$의 열, $\mathbf{q}_i^T$는 $P^{-1}$의 행.

### 증명

$P = [\mathbf{p}_1 \mid \cdots \mid \mathbf{p}_n]$, $P^{-1} = \begin{pmatrix} \mathbf{q}_1^T \\ \vdots \\ \mathbf{q}_n^T \end{pmatrix}$.

$$
PDP^{-1} = \sum_i \mathbf{p}_i (D P^{-1})_{i,:} = \sum_i \mathbf{p}_i \lambda_i \mathbf{q}_i^T \quad \blacksquare
$$

> **🔑 의미**: $A$는 서로 다른 $n$개 방향(rank-1 연산)의 중첩. 각 방향에서 스칼라 $\lambda_i$로 늘림.

### 정리 5.2 (행렬 거듭제곱)

$A^k = P D^k P^{-1}$.

### 증명

$A^k = (PDP^{-1})(PDP^{-1}) \cdots = P D^k P^{-1}$ (중간의 $P^{-1}P$가 상쇄). $\blacksquare$

### 활용: 피보나치

$F_n = F_{n-1} + F_{n-2}$는 $\mathbf{u}_n = (F_n, F_{n-1})^T$에 대해 $\mathbf{u}_n = A \mathbf{u}_{n-1}$, $A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$. 고윳값 $\lambda_\pm = (1 \pm \sqrt{5})/2$ (황금비). 대각화로 **Binet's formula** 유도.

---

## 6. 비대각화 예시

### 예시 6.1 (결함 행렬, defective)

$$
A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
$$

$p_A(\lambda) = (\lambda - 1)^2$, $m_a(1) = 2$.

$A - I = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$의 커널은 $\{(c, 0)^T : c \in \mathbb{R}\}$, $m_g(1) = 1 < 2$.

→ 비대각화. Jordan Form (Ch2-06) 필요.

### 정리 6.2

**거의 모든 행렬은 대각화 가능**: Generic (measure-zero 집합 제외) 행렬은 고윳값이 서로 다르고, 정리 3.1에 의해 대각화 가능.

---

## 7. 스펙트럼과 고윳값의 성질

### 정의 7.1 (스펙트럼)

$\sigma(A) = \{\lambda : \lambda \text{는 } A \text{의 고윳값}\}$

### 정리 7.2

- $\operatorname{tr}(A) = \sum_i \lambda_i$
- $\det(A) = \prod_i \lambda_i$

### 증명

$p_A(\lambda) = \det(\lambda I - A) = \prod_i (\lambda - \lambda_i)$. 계수 비교:

- $\lambda^{n-1}$ 계수: $-\sum \lambda_i$. 또 $\det(\lambda I - A)$를 전개하면 $\lambda^{n-1}$ 계수는 $-\operatorname{tr}(A)$.
- 상수항 ($\lambda = 0$): $\det(-A) = (-1)^n \det(A)$. 또 $\prod (\lambda - \lambda_i)|_{\lambda=0} = (-1)^n \prod \lambda_i$. $\blacksquare$

### 정리 7.3 (닮은 행렬은 같은 특성다항식)

$B = P^{-1} A P$이면 $p_A = p_B$.

### 증명

$$
p_B(\lambda) = \det(\lambda I - P^{-1} A P) = \det(P^{-1}(\lambda I - A) P) = \det(\lambda I - A) = p_A(\lambda) \quad \blacksquare
$$

> **🔑 의미**: 특성다항식, 고윳값, 트레이스, 행렬식은 **좌표계와 무관한 불변량**.

---

## 8. Python 실험

### 8.1 고유값 분해

```python
import numpy as np

A = np.array([[4.0, 1.0, 0.0],
              [2.0, 3.0, 0.0],
              [0.0, 0.0, 5.0]])

eigvals, eigvecs = np.linalg.eig(A)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)

# 대각화 확인: A = P D P^-1
D = np.diag(eigvals)
P = eigvecs
print("||A - P D P^-1||:", np.linalg.norm(A - P @ D @ np.linalg.inv(P)))

# trace = sum(eigvals)
print("tr(A):", np.trace(A), "sum eigvals:", eigvals.sum())
# det = prod(eigvals)
print("det(A):", np.linalg.det(A), "prod:", np.prod(eigvals))
```

### 8.2 비대각화 예제

```python
A = np.array([[1.0, 1.0],
              [0.0, 1.0]])
eigvals, eigvecs = np.linalg.eig(A)
print("Eigenvalues:", eigvals)  # [1, 1]
print("Eigenvectors:\n", eigvecs)  # 수치적으로 독립처럼 보일 수 있음
# 하지만 rank(eigvecs) = 1
print("rank(eigvecs):", np.linalg.matrix_rank(eigvecs))
```

### 8.3 거듭제곱으로 피보나치

```python
A = np.array([[1.0, 1.0], [1.0, 0.0]])
u0 = np.array([1.0, 0.0])  # F_1, F_0

# Direct
u_k = u0.copy()
for _ in range(10):
    u_k = A @ u_k
print("F_11:", u_k[0])

# Eigendecomposition
eigvals, P = np.linalg.eig(A)
D_10 = np.diag(eigvals**10)
P_inv = np.linalg.inv(P)
u_10_eig = P @ D_10 @ P_inv @ u0
print("F_11 (eig):", u_10_eig[0])
```

---

## 9. 요약

| 개념                   | 정의                            |
| ---------------------- | ------------------------------- |
| 고윳값 $\lambda$       | $\det(\lambda I - A) = 0$ 의 근 |
| 고유공간 $E_\lambda$   | $\ker(A - \lambda I)$           |
| 대수적 중복도 $m_a$    | $p_A$에서 $(\lambda - \lambda_0)$ 중근 차수 |
| 기하적 중복도 $m_g$    | $\dim E_\lambda$                |
| 대각화 조건            | $\sum m_g = n$ ⟺ $m_g = m_a$ (모든 $\lambda$) |

**핵심 부등식**: $1 \leq m_g(\lambda) \leq m_a(\lambda)$.

---

## 10. 참고 문헌

- Axler, S. (2015). *Linear Algebra Done Right* (3rd ed.). Ch 5.
- Horn, R. A., & Johnson, C. R. (2013). *Matrix Analysis* (2nd ed.). Ch 1.
- Lax, P. D. (2007). *Linear Algebra and Its Applications* (2nd ed.). Ch 6.

---

## 11. 내비게이션

[◀ 03. 촐레스키 분해](./03-cholesky-decomposition.md) | [📚 README](../README.md) | [05. 스펙트럼 정리 ▶](./05-spectral-theorem.md)
