# Ch2-05. 스펙트럼 정리: 대칭 행렬의 완벽한 대각화

> "실대칭 행렬은 **반드시** 실 고윳값을 가지며, 고유벡터로 이루어진 **정규직교 기저**를 가진다."

## 📌 학습 목표

- 대칭 행렬의 고윳값이 실수임을 증명한다.
- 서로 다른 고윳값에 대응하는 고유벡터가 **직교**함을 증명한다.
- 스펙트럼 정리 $A = Q \Lambda Q^T$를 **유도적으로** 증명한다 (Schur 분해 경유).
- 정규 행렬(normal), 에르미트(Hermitian)로의 확장을 이해한다.
- PCA, 양자역학 관측량, 그래프 Laplacian 등 응용의 기초.

---

## 🎯 핵심 질문

> **질문 1**: 왜 **대칭** 조건이 고윳값의 **실수성**을 보장하는가?
> **질문 2**: 서로 다른 고윳값의 고유벡터가 직교하는 이유는?
> **질문 3**: Jordan Form이 아닌 **완전 대각화**가 가능한 구조적 이유는?

---

## 1. 스펙트럼 정리

### 정리 1.1 (실대칭 행렬의 스펙트럼 정리)

$A \in \mathbb{R}^{n \times n}$이 대칭($A = A^T$)이면, **직교 행렬** $Q \in \mathbb{R}^{n \times n}$ ($Q^T Q = I$)와 실수 대각 행렬 $\Lambda$가 존재하여:

$$
\boxed{A = Q \Lambda Q^T = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T}
$$

$\lambda_i \in \mathbb{R}$은 $A$의 고윳값, $\mathbf{q}_i$는 서로 **정규직교**인 고유벡터.

---

## 2. 보조정리들

### 보조정리 2.1 (대칭이면 고윳값 실수)

$A \in \mathbb{R}^{n \times n}$이 대칭이면 모든 (복소) 고윳값은 실수.

### 증명

$A \mathbf{v} = \lambda \mathbf{v}$ ($\mathbf{v} \in \mathbb{C}^n \setminus \{\mathbf{0}\}$, $\lambda \in \mathbb{C}$). 복소 켤레 전치 $\bar{\mathbf{v}}^T = \mathbf{v}^H$:

$$
\mathbf{v}^H A \mathbf{v} = \lambda \mathbf{v}^H \mathbf{v} = \lambda \|\mathbf{v}\|^2
$$

한편 $A$가 실수 대칭이므로 $A^H = A^T = A$. 따라서:

$$
\mathbf{v}^H A \mathbf{v} = (A^H \mathbf{v})^H \mathbf{v} = (A \mathbf{v})^H \mathbf{v} = (\lambda \mathbf{v})^H \mathbf{v} = \bar{\lambda} \|\mathbf{v}\|^2
$$

비교: $\lambda \|\mathbf{v}\|^2 = \bar{\lambda} \|\mathbf{v}\|^2$, $\mathbf{v} \neq 0$이므로 $\lambda = \bar{\lambda}$, 즉 $\lambda \in \mathbb{R}$. $\blacksquare$

### 보조정리 2.2 (실수 고윳값이면 실수 고유벡터 존재)

$\lambda \in \mathbb{R}$이 $A$의 고윳값이면 실수 고유벡터 존재.

### 증명

$A - \lambda I$가 실계수 특이 행렬이므로 $\ker(A - \lambda I)$는 $\mathbb{R}$에서 trivial하지 않다 (즉 $\mathbb{R}^n$ 안에서 비자명한 커널 존재). $\blacksquare$

### 보조정리 2.3 (서로 다른 고윳값 → 직교)

$A$가 대칭, $A \mathbf{v}_1 = \lambda_1 \mathbf{v}_1$, $A \mathbf{v}_2 = \lambda_2 \mathbf{v}_2$, $\lambda_1 \neq \lambda_2$이면 $\mathbf{v}_1 \perp \mathbf{v}_2$.

### 증명

$$
\lambda_1 \mathbf{v}_1^T \mathbf{v}_2 = (A\mathbf{v}_1)^T \mathbf{v}_2 = \mathbf{v}_1^T A^T \mathbf{v}_2 = \mathbf{v}_1^T A \mathbf{v}_2 = \lambda_2 \mathbf{v}_1^T \mathbf{v}_2
$$

$(\lambda_1 - \lambda_2) \mathbf{v}_1^T \mathbf{v}_2 = 0$, $\lambda_1 \neq \lambda_2$이므로 $\mathbf{v}_1^T \mathbf{v}_2 = 0$. $\blacksquare$

---

## 3. 스펙트럼 정리 증명 (Schur 경유)

### 보조정리 3.1 (실 Schur 정리, 대칭 버전)

$A \in \mathbb{R}^{n \times n}$가 대칭이면 직교 $U$와 **상삼각** $T$가 존재하여 $A = U T U^T$.

### 증명 (귀납법)

$n = 1$: trivial.

$n - 1$까지 성립한다고 가정. $A$의 실수 고윳값 $\lambda_1$과 정규 고유벡터 $\mathbf{q}_1$ 선택 (보조정리 2.1, 2.2). $\mathbf{q}_1$을 확장하여 정규직교 기저 $\{\mathbf{q}_1, \mathbf{u}_2, \ldots, \mathbf{u}_n\}$ (Gram-Schmidt).

$U_1 = [\mathbf{q}_1 \mid \mathbf{u}_2 \mid \cdots \mid \mathbf{u}_n]$ 직교, 첫 열 $A$-작용:

$$
U_1^T A U_1 = \begin{pmatrix} \lambda_1 & \mathbf{w}^T \\ \mathbf{0} & B \end{pmatrix}
$$

(첫 열은 $U_1^T A \mathbf{q}_1 = \lambda_1 U_1^T \mathbf{q}_1 = \lambda_1 \mathbf{e}_1$.)

**여기서 $A$의 대칭성**: $U_1^T A U_1$도 대칭이므로:

$$
\begin{pmatrix} \lambda_1 & \mathbf{w}^T \\ \mathbf{0} & B \end{pmatrix} = \begin{pmatrix} \lambda_1 & \mathbf{0}^T \\ \mathbf{w} & B^T \end{pmatrix}
$$

비교: $\mathbf{w} = \mathbf{0}$, $B = B^T$ (대칭). 즉:

$$
U_1^T A U_1 = \begin{pmatrix} \lambda_1 & 0 \\ 0 & B \end{pmatrix}
$$

귀납가정으로 $B = U_2 \Lambda_B U_2^T$ ($U_2$ 직교, $\Lambda_B$ 대각). 따라서:

$$
A = U_1 \begin{pmatrix} 1 & 0 \\ 0 & U_2 \end{pmatrix} \begin{pmatrix} \lambda_1 & 0 \\ 0 & \Lambda_B \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & U_2 \end{pmatrix}^T U_1^T = Q \Lambda Q^T \quad \blacksquare
$$

이 증명은 **Schur 분해를 거치지 않고** 곧장 스펙트럼 정리를 주는 직관적 경로다.

---

## 4. 복소 에르미트로의 확장

### 정의 4.1

$A \in \mathbb{C}^{n \times n}$가 **에르미트(Hermitian)**라 함은 $A^H = A$ (켤레전치 = 자기 자신).

### 정리 4.2 (복소 스펙트럼 정리)

에르미트 행렬은 유니타리 $U$ ($U^H U = I$)와 실 대각 $\Lambda$로 $A = U \Lambda U^H$.

증명은 §3과 동일하게 진행, $U^T$를 $U^H$로만 바꿈.

### 정리 4.3 (정규 행렬의 스펙트럼 정리)

$A \in \mathbb{C}^{n \times n}$가 **정규(normal)**, 즉 $A A^H = A^H A$이면 유니타리 대각화 가능: $A = U D U^H$.

**Normal의 예**: Hermitian, skew-Hermitian, unitary 모두 normal.

**증명 핵심**: Schur 분해 $A = U T U^H$ ($T$ 상삼각). $A$ normal ⟺ $T$ normal. 상삼각 + normal ⟹ 대각.

---

## 5. 스펙트럼 분해의 기하적 의미

### 정리 5.1 (Rayleigh 원리)

대칭 $A$의 고윳값을 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$이라 하자.

$$
\lambda_1 = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}, \qquad \lambda_n = \min_{\mathbf{x} \neq \mathbf{0}} \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}
$$

### 증명

$A = Q \Lambda Q^T$. $\mathbf{y} = Q^T \mathbf{x}$:

$$
\frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}} = \frac{\mathbf{y}^T \Lambda \mathbf{y}}{\mathbf{y}^T \mathbf{y}} = \frac{\sum \lambda_i y_i^2}{\sum y_i^2}
$$

$\lambda_i \in [\lambda_n, \lambda_1]$이므로 $\frac{\sum \lambda_i y_i^2}{\sum y_i^2} \in [\lambda_n, \lambda_1]$. 상한은 $\mathbf{y} = \mathbf{e}_1$ ($\mathbf{x} = \mathbf{q}_1$)에서 달성. $\blacksquare$

### 정리 5.2 (Courant-Fischer / min-max)

$$
\lambda_k = \min_{\substack{S \subset \mathbb{R}^n \\ \dim S = n - k + 1}} \max_{\mathbf{x} \in S \setminus \{0\}} \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}
$$

**의미**: 각 고윳값은 부분공간 제한에서 Rayleigh 극값.

---

## 6. 응용

### 6.1 PCA (주성분 분석)

데이터 행렬 $X \in \mathbb{R}^{n \times d}$의 공분산 $\Sigma = \frac{1}{n} X^T X$는 대칭 PSD. 스펙트럼 정리로 $\Sigma = Q \Lambda Q^T$, $\mathbf{q}_1$이 분산 최대 방향 (Rayleigh 원리).

### 6.2 양자역학

관측량(observable)은 에르미트 연산자. 고윳값이 **관측 가능 값**, 고유벡터가 **상태**.

### 6.3 그래프 Laplacian

$L = D - W$ ($D$는 degree 대각, $W$는 가중치). 대칭 PSD, 고윳값 $0 = \lambda_1 \leq \lambda_2 \leq \cdots$. $\lambda_2$ (Fiedler value)가 그래프 연결성/clustering의 핵심.

### 6.4 이차형식 분류

$q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$의 부호 유형: $A$의 고윳값 부호로 분류. 모두 $> 0$이면 PD (타원형), $<0$면 ND (역타원), 혼합이면 안장.

---

## 7. Python 실험

### 7.1 기본 스펙트럼 분해

```python
import numpy as np

# 대칭 행렬
A = np.array([[4.0, 1.0, 2.0],
              [1.0, 3.0, 0.0],
              [2.0, 0.0, 5.0]])
print("Symmetric?", np.allclose(A, A.T))

# np.linalg.eigh: 에르미트/대칭 전용 (실수 고윳값 보장)
eigvals, Q = np.linalg.eigh(A)
print("Eigenvalues:", eigvals)
print("Q^T Q:\n", Q.T @ Q)  # I
print("||A - Q Lambda Q^T||:", np.linalg.norm(A - Q @ np.diag(eigvals) @ Q.T))
```

### 7.2 Rayleigh 원리 확인

```python
n = 5
np.random.seed(0)
M = np.random.randn(n, n)
A = (M + M.T) / 2  # 대칭화

eigvals, Q = np.linalg.eigh(A)
lam_max = eigvals.max()

# 임의 방향
vals = []
for _ in range(10000):
    x = np.random.randn(n)
    vals.append(x @ A @ x / (x @ x))
print("max Rayleigh:", max(vals), "true:", lam_max)
# max Rayleigh ≤ lam_max
```

### 7.3 PCA 기본

```python
np.random.seed(1)
X = np.random.randn(100, 3) @ np.diag([5, 2, 0.5])  # 방향별 분산 다름
Sigma = np.cov(X.T)
eigvals, Q = np.linalg.eigh(Sigma)
idx = np.argsort(eigvals)[::-1]
eigvals, Q = eigvals[idx], Q[:, idx]
print("Principal variances:", eigvals)  # ≈ [25, 4, 0.25]
print("First component:", Q[:, 0])      # 가장 분산 큰 방향
```

### 7.4 Jordan 없는 대칭

```python
# 결함 행렬 (비대칭)
A_def = np.array([[1.0, 1.0], [0.0, 1.0]])
print("Symmetric?", np.allclose(A_def, A_def.T))  # False
# eigh를 쓰면 오류 혹은 잘못된 결과 → eig 사용 필요

# 대칭이면 항상 완전 대각화
A_sym = np.array([[1.0, 2.0], [2.0, 1.0]])
eigvals, Q = np.linalg.eigh(A_sym)
print("No Jordan needed:", np.allclose(A_sym, Q @ np.diag(eigvals) @ Q.T))
```

---

## 8. 요약

| 행렬 종류          | 분해                     | 특성                       |
| ------------------ | ------------------------ | -------------------------- |
| 실대칭             | $A = Q\Lambda Q^T$       | $\lambda_i \in \mathbb{R}$, $Q$ 직교 |
| 에르미트           | $A = U\Lambda U^H$       | $\lambda_i \in \mathbb{R}$, $U$ 유니타리 |
| 정규 $A A^H = A^H A$| $A = U D U^H$            | $D$ 복소 대각              |
| 일반 실수          | $A = P D P^{-1}$ (가능시) | 대각화 불가시 Jordan       |
| 임의 복소          | $A = U T U^H$ (Schur)    | 상삼각                     |

**핵심**: $A = A^T$ (대칭) ⟹ 완전 대각화 + 실 고윳값 + 직교 고유벡터.

---

## 9. 참고 문헌

- Horn & Johnson, *Matrix Analysis*, Ch 2.
- Strang, *Linear Algebra and Its Applications*, Ch 6.
- Reed & Simon, *Functional Analysis I*, Ch VII (확장).

---

## 10. 다음 문서 예고

- **[06. Jordan Form](./06-jordan-form.md)**: 대각화 불가능한 경우 generalized eigenvector와 블록 구조로 대체.

---

## 11. 내비게이션

[◀ 04. 고유값 분해](./04-eigendecomposition.md) | [📚 README](../README.md) | [06. Jordan Form ▶](./06-jordan-form.md)
