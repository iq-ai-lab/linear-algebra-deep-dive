# Ch4-02. SVD의 존재성·유일성·변분적 특성

> "SVD는 존재하며, 특이값은 유일하다. 그러나 특이벡터는 부호를 제외하고도 여러 가능성이 있다."

## 📌 학습 목표

- 변분적(variational) 접근: 단위구 위에서 $\|A\mathbf{v}\|$의 최대화 반복으로 SVD를 구성.
- 특이값의 **유일성**을 증명한다.
- 특이벡터의 **비유일성** 조건과 범위를 규명한다.
- **Courant-Fischer / min-max**의 특이값 버전.
- **Weyl-type 부등식**: 특이값은 섭동에 Lipschitz.

---

## 🎯 핵심 질문

> **질문 1**: 왜 특이값은 유일한데 특이벡터는 아닌가?
> **질문 2**: 중복 특이값에서 특이벡터는 **어떤 범위**까지 유일한가?
> **질문 3**: $\sigma_k$의 min-max 표현은 무엇인가?

---

## 1. 변분적 존재 증명

기존 스펙트럼 정리 기반의 존재 증명(Ch4-01)과 달리, **독립적으로** 단위구 최적화로 SVD를 구성한다.

### 정리 1.1 (SVD 변분 구성)

$A \in \mathbb{R}^{m \times n}$, rank $r$. SVD $A = U \Sigma V^T$은 다음 반복 절차로 존재한다.

#### Step 1: $\sigma_1$

$$
\sigma_1 = \max_{\|\mathbf{v}\| = 1} \|A\mathbf{v}\|, \qquad \mathbf{v}_1 \in \arg\max
$$

단위구가 컴팩트, 함수 $\mathbf{v} \mapsto \|A\mathbf{v}\|$가 연속 → 최댓값 달성.

$\mathbf{u}_1 = A\mathbf{v}_1 / \sigma_1$ (만약 $\sigma_1 = 0$이면 $A = 0$으로 끝).

#### Step 2: 귀납

$k - 1$까지 $\mathbf{v}_1, \ldots, \mathbf{v}_{k-1}$, $\mathbf{u}_1, \ldots, \mathbf{u}_{k-1}$ 구성했다고 하자. 제약된 최대화:

$$
\sigma_k = \max_{\substack{\|\mathbf{v}\| = 1 \\ \mathbf{v} \perp \mathbf{v}_1, \ldots, \mathbf{v}_{k-1}}} \|A\mathbf{v}\|, \qquad \mathbf{v}_k \in \arg\max
$$

$\sigma_k > 0$이면 $\mathbf{u}_k = A\mathbf{v}_k / \sigma_k$. $\sigma_k = 0$이면 과정 종료; 남은 $\mathbf{v}_{k+1}, \ldots$는 $N(A)$의 정규직교 기저로, $\mathbf{u}_{k+1}, \ldots$는 $R(A)^\perp$의 정규직교 기저로.

#### Step 3: 직교성 자동 보장

**Lemma 1.2**: 위 구성에서 $\mathbf{u}_k \perp \mathbf{u}_j$ ($j < k$).

**증명**: $\mathbf{v}(t) = \cos t \cdot \mathbf{v}_k + \sin t \cdot \mathbf{v}_j$ ($\|\mathbf{v}(t)\| = 1$). $f(t) = \|A\mathbf{v}(t)\|^2$의 $t = 0$에서 미분 $= 0$ (제약된 최적):

$$
\frac{d}{dt}\Big|_{t=0} \|A\mathbf{v}(t)\|^2 = 2 (A\mathbf{v}_k)^T (A\mathbf{v}_j) = 2\sigma_k \sigma_j \mathbf{u}_k^T \mathbf{u}_j = 0
$$

$\sigma_k, \sigma_j > 0$이므로 $\mathbf{u}_k^T \mathbf{u}_j = 0$. $\blacksquare$

#### Step 4: $AV = U\Sigma$

첫 열부터 차례로: $A\mathbf{v}_k = \sigma_k \mathbf{u}_k$. 전체: $AV = U\Sigma$. $V$ 직교 → $A = U\Sigma V^T$.

---

## 2. 특이값의 유일성

### 정리 2.1

$A \in \mathbb{R}^{m \times n}$의 특이값 $\sigma_1 \geq \cdots \geq \sigma_p$ ($p = \min(m, n)$)는 **유일**하다.

### 증명

$\{\sigma_i^2\}$는 $A^T A$의 고유값 집합 (중복 포함). 고유값은 특성다항식의 근으로 유일 결정. 음이 아니므로 $\sigma_i \geq 0$으로 유일. $\blacksquare$

---

## 3. 특이벡터의 비유일성

### 정리 3.1

$\sigma_k$가 **단순 특이값** ($\sigma_{k-1} > \sigma_k > \sigma_{k+1}$)이면 $\mathbf{v}_k$는 부호를 제외하고 유일.

### 정리 3.2

$\sigma_k = \sigma_{k+1} = \cdots = \sigma_{k+\ell - 1}$ (중복도 $\ell$)이면 대응하는 특이벡터 부분공간은 $\ell$차원이며, 그 안에서 **임의의 정규직교 기저**가 유효.

### 증명

중복 고유값에 대한 고유공간 $E_\sigma = \ker(A^T A - \sigma^2 I)$는 $\ell$차원. 임의의 정규직교 기저 $\{\mathbf{v}_k, \ldots\}$ 선택 가능. 이에 따라 $\mathbf{u}_i = A\mathbf{v}_i / \sigma$도 달라짐. $\blacksquare$

### 예시

$A = I_2$. SVD: $\Sigma = I_2$, $\sigma_1 = \sigma_2 = 1$. $V$는 임의의 $2 \times 2$ 직교 행렬이면 됨.

---

## 4. Courant-Fischer for Singular Values

### 정리 4.1 (특이값의 min-max)

$$
\sigma_k(A) = \max_{\substack{S \subset \mathbb{R}^n \\ \dim S = k}} \min_{\substack{\mathbf{x} \in S \\ \|\mathbf{x}\| = 1}} \|A\mathbf{x}\|
$$

$$
\sigma_k(A) = \min_{\substack{T \subset \mathbb{R}^n \\ \dim T = n - k + 1}} \max_{\substack{\mathbf{x} \in T \\ \|\mathbf{x}\| = 1}} \|A\mathbf{x}\|
$$

### 증명

$\sigma_k^2$는 $A^T A$의 $k$번째 큰 고유값. $R_{A^T A}(\mathbf{x}) = \|A\mathbf{x}\|^2/\|\mathbf{x}\|^2$에 Ch3-03의 Courant-Fischer 적용 후 제곱근. $\blacksquare$

---

## 5. 특이값 섭동 (Weyl-type)

### 정리 5.1

$A, E \in \mathbb{R}^{m \times n}$. $\tilde{A} = A + E$:

$$
|\sigma_k(\tilde{A}) - \sigma_k(A)| \leq \|E\|_2
$$

### 증명

Courant-Fischer:

$$
\sigma_k(\tilde{A}) = \max_{\dim S = k} \min_{\mathbf{x} \in S, \|\mathbf{x}\|=1} \|(A + E)\mathbf{x}\|
$$

삼각부등식: $\|A\mathbf{x}\| - \|E\|_2 \leq \|(A+E)\mathbf{x}\| \leq \|A\mathbf{x}\| + \|E\|_2$.

따라서:

$$
\sigma_k(A) - \|E\|_2 \leq \sigma_k(\tilde{A}) \leq \sigma_k(A) + \|E\|_2 \quad \blacksquare
$$

### 의미

고유값과 달리 **비대칭**도 Lipschitz-1. 이것이 SVD가 최고로 안정적인 이유.

### 정리 5.2 (Mirsky)

$$
\sum_k |\sigma_k(\tilde{A}) - \sigma_k(A)|^2 \leq \|E\|_F^2
$$

(프로베니우스 노름에서의 Weyl 버전.)

---

## 6. 특이값과 행렬 랭크

### 정리 6.1

$\operatorname{rank}(A) = r \iff \sigma_r > 0 = \sigma_{r+1} = \cdots$.

### 의미

SVD로 랭크를 "정확히" 파악. 수치적으로는 "작은" 특이값을 **수치적 rank**로 간주:

$$
\operatorname{rank}_\epsilon(A) = \#\{i : \sigma_i > \epsilon\}
$$

### 응용

- 원래 랭크는 떨어질 수 있음 (컴퓨터의 float 오차로)
- $\sigma_r / \sigma_1 < \text{기계엡실론}$ 정도면 수치적으로 rank deficient

---

## 7. Polar Decomposition

### 정리 7.1 (극분해)

정사각 $A \in \mathbb{R}^{n \times n}$는 다음과 같이 분해:

$$
A = Q P, \quad Q \text{ 직교}, \quad P \succeq 0 \text{ 대칭 PSD}
$$

### 증명

SVD $A = U\Sigma V^T$. $Q = UV^T$ (직교), $P = V\Sigma V^T$ (대칭 PSD):

$$
QP = UV^T V \Sigma V^T = U\Sigma V^T = A \quad \blacksquare
$$

### 기하적 의미

임의 선형 변환 = 회전/반사($Q$) + 축별 확축($P$). 복소수 $z = re^{i\theta}$의 **행렬 버전**.

---

## 8. Python 실험

### 8.1 Variational Construction

```python
import numpy as np
from scipy.optimize import minimize

np.random.seed(0)
A = np.random.randn(5, 3)

# σ_1 by maximizing ||A v|| on unit sphere
def neg_norm(v):
    return -np.linalg.norm(A @ v)

res = minimize(neg_norm, np.random.randn(3), 
               constraints={'type': 'eq', 'fun': lambda v: v@v - 1})
v1 = res.x / np.linalg.norm(res.x)
sigma_1 = np.linalg.norm(A @ v1)
print(f"σ_1 (variational): {sigma_1:.4f}")
print(f"σ_1 (numpy):      {np.linalg.svd(A)[1][0]:.4f}")
```

### 8.2 Weyl Perturbation

```python
A = np.random.randn(4, 4)
E = 0.01 * np.random.randn(4, 4)
A_pert = A + E

s = np.linalg.svd(A)[1]
s_pert = np.linalg.svd(A_pert)[1]

diffs = np.abs(s_pert - s)
print(f"max |σ_k(A+E) - σ_k(A)|: {diffs.max():.4f}")
print(f"||E||_2: {np.linalg.norm(E, 2):.4f}")
# max diff ≤ ||E||_2
```

### 8.3 중복 특이값의 비유일성

```python
# 고의로 σ_1 = σ_2 만드는 행렬
A = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 0.5]])

U, s, Vt = np.linalg.svd(A)
print("σ:", s)
print("V^T:\n", Vt)

# V는 (1,1) 부분공간에서 임의의 회전을 선택할 수 있음
# NumPy는 일관된 결과를 주지만 다른 라이브러리는 다를 수 있음
```

### 8.4 Rank via SVD

```python
# Rank-deficient matrix
A = np.random.randn(10, 5)
B = np.random.randn(5, 10)
C = A @ B   # rank ≤ 5

s = np.linalg.svd(C)[1]
print("Singular values:", s[:7], "...")
print(f"Numerical rank (eps=1e-10): {(s > 1e-10).sum()}")
print(f"np.linalg.matrix_rank(C): {np.linalg.matrix_rank(C)}")
```

### 8.5 Polar Decomposition

```python
from scipy.linalg import polar
A = np.random.randn(3, 3)
Q, P = polar(A)
print("||Q^T Q - I||:", np.linalg.norm(Q.T @ Q - np.eye(3)))  # 직교
print("||P - P^T||:  ", np.linalg.norm(P - P.T))               # 대칭
print("Eigvals of P:", np.linalg.eigvalsh(P))                  # PSD (non-negative)
print("||A - Q P||: ", np.linalg.norm(A - Q @ P))
```

---

## 9. 요약

| 개념                   | 공식/의미                                        |
| ---------------------- | ------------------------------------------------ |
| 존재성                 | 모든 $A$에 대해 SVD 존재 (변분 또는 $A^T A$ 경유) |
| 특이값 유일성          | $\{\sigma_i\}$ 유일 (집합)                       |
| 특이벡터 유일성        | 단순값: 부호 제외 유일; 중복값: 부분공간 기저 자유 |
| min-max                | $\sigma_k = \max_{S^k} \min \|A\mathbf{x}\|$     |
| Weyl 섭동              | $\|\sigma_k(\tilde A) - \sigma_k(A)\| \leq \|E\|_2$ |
| Rank                   | $\operatorname{rank}(A) = \#\{\sigma_i > 0\}$    |
| Polar                  | $A = QP$ (회전 × PSD)                            |

---

## 10. 참고 문헌

- Horn & Johnson, *Topics in Matrix Analysis*, Ch 3.
- Stewart & Sun, *Matrix Perturbation Theory*, Ch 3.
- Stewart, G. W. (1993). *On the early history of the singular value decomposition*. SIAM Review.

---

## 11. 내비게이션

[◀ 01. SVD 기하](./01-svd-geometric.md) | [📚 README](../README.md) | [03. Pseudoinverse ▶](./03-pseudoinverse.md)
