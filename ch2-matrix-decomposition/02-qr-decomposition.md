# Ch2-02. QR 분해: 그람-슈미트부터 하우스홀더까지

> "모든 행렬은 직교 행렬과 상삼각 행렬의 곱으로 분해된다."

## 📌 학습 목표

- QR 분해의 존재성을 **그람-슈미트 과정**으로 증명한다.
- Classical Gram-Schmidt(CGS)와 Modified Gram-Schmidt(MGS)의 **수치적 차이**를 유도한다.
- **하우스홀더 반사(Householder Reflection)**를 통한 QR 분해가 왜 수치적으로 안정한지 보인다.
- QR 분해의 계산 복잡도가 $O(2mn^2 - \frac{2}{3}n^3)$ 임을 보인다.
- 최소 제곱 문제(Least Squares)에서 QR의 역할을 이해한다.

---

## 🎯 핵심 질문

> **질문 1**: 왜 LU 분해와 달리 QR 분해는 **모든** 행렬에 대해 항상 존재하는가?
> **질문 2**: Classical GS는 왜 유한 정밀도에서 직교성을 잃는가?
> **질문 3**: 하우스홀더 변환의 기하학적 의미는 무엇인가?

---

## 1. 정의: QR 분해

### 정의 1.1 (QR 분해)

$A \in \mathbb{R}^{m \times n}$ ($m \geq n$)에 대해 다음을 만족하는 **직교 행렬** $Q \in \mathbb{R}^{m \times m}$와 **상삼각 행렬** $R \in \mathbb{R}^{m \times n}$이 존재하면 이를 **QR 분해**라 한다:

$$
A = QR, \qquad Q^T Q = I_m
$$

### 정의 1.2 (Thin QR / Reduced QR)

$A$가 full column rank이면 $m \geq n$ 행렬에 대해 다음 **축약형 QR**이 존재한다:

$$
A = \hat{Q} \hat{R}, \qquad \hat{Q} \in \mathbb{R}^{m \times n},\ \hat{Q}^T \hat{Q} = I_n,\ \hat{R} \in \mathbb{R}^{n \times n}
$$

여기서 $\hat{R}$은 상삼각이며 대각 성분이 양수이도록 할 수 있다.

---

## 2. 그람-슈미트 직교화

### 정리 2.1 (Gram-Schmidt Orthogonalization)

$\mathbf{a}_1, \ldots, \mathbf{a}_n \in \mathbb{R}^m$이 일차독립이면, 다음으로 정의된 $\{\mathbf{q}_1, \ldots, \mathbf{q}_n\}$은 정규직교집합이며 $\operatorname{span}\{\mathbf{q}_1, \ldots, \mathbf{q}_k\} = \operatorname{span}\{\mathbf{a}_1, \ldots, \mathbf{a}_k\}$이 모든 $k$에 대해 성립한다.

$$
\tilde{\mathbf{q}}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} (\mathbf{q}_j^T \mathbf{a}_k) \mathbf{q}_j, \qquad \mathbf{q}_k = \frac{\tilde{\mathbf{q}}_k}{\|\tilde{\mathbf{q}}_k\|}
$$

### 증명

**(귀납법)** $k = 1$: $\mathbf{q}_1 = \mathbf{a}_1 / \|\mathbf{a}_1\|$이므로 $\|\mathbf{q}_1\| = 1$, span 동일.

**(귀납 단계)** $k - 1$까지 성립한다고 가정. $\tilde{\mathbf{q}}_k$의 $\mathbf{q}_i$ 방향 성분 ($i < k$):

$$
\mathbf{q}_i^T \tilde{\mathbf{q}}_k = \mathbf{q}_i^T \mathbf{a}_k - \sum_{j=1}^{k-1} (\mathbf{q}_j^T \mathbf{a}_k)\underbrace{\mathbf{q}_i^T \mathbf{q}_j}_{\delta_{ij}} = \mathbf{q}_i^T \mathbf{a}_k - \mathbf{q}_i^T \mathbf{a}_k = 0
$$

따라서 $\tilde{\mathbf{q}}_k \perp \mathbf{q}_i$. 또한 $\mathbf{a}_k$가 일차독립이므로 $\tilde{\mathbf{q}}_k \neq \mathbf{0}$이고 $\mathbf{q}_k$는 well-defined. $\blacksquare$

### 따름정리 2.2 (Thin QR 존재성)

$A = [\mathbf{a}_1 \mid \cdots \mid \mathbf{a}_n]$이 full column rank이면 $A = \hat{Q}\hat{R}$이 존재하며, $R_{ik} = \mathbf{q}_i^T \mathbf{a}_k$ ($i < k$), $R_{kk} = \|\tilde{\mathbf{q}}_k\|$이다.

$$
\mathbf{a}_k = \|\tilde{\mathbf{q}}_k\| \mathbf{q}_k + \sum_{j=1}^{k-1}(\mathbf{q}_j^T \mathbf{a}_k) \mathbf{q}_j = \sum_{j=1}^{k} R_{jk} \mathbf{q}_j
$$

---

## 3. Classical vs Modified Gram-Schmidt

### 3.1 CGS의 불안정성

Classical GS(CGS)는 **모든 투영을 동시에** 계산한다:

$$
r_{jk} = \mathbf{q}_j^T \mathbf{a}_k \quad (j = 1, \ldots, k-1), \qquad \tilde{\mathbf{q}}_k = \mathbf{a}_k - \sum_j r_{jk} \mathbf{q}_j
$$

유한 정밀도에서 $\mathbf{q}_1, \ldots, \mathbf{q}_{k-1}$이 이미 **완벽한 직교가 아닌** 상태로 계산되어 있다. 이 오차가 누적되어 $\mathbf{q}_k$의 직교성이 크게 훼손된다.

### 3.2 MGS의 안정화

Modified GS(MGS)는 **각 투영을 순차적으로** 적용한다:

$$
\mathbf{v}^{(0)} = \mathbf{a}_k, \qquad \mathbf{v}^{(j)} = \mathbf{v}^{(j-1)} - (\mathbf{q}_j^T \mathbf{v}^{(j-1)}) \mathbf{q}_j, \qquad \tilde{\mathbf{q}}_k = \mathbf{v}^{(k-1)}
$$

정확 산술에서는 CGS와 동일하나, 유한 정밀도에서는 **이전 단계에서 제거된 성분이 다음 투영 계수에 반영되지 않아** 오차 누적이 감소한다.

### 정리 3.1 (MGS 직교성 손실 한계)

머신 엡실론 $\epsilon_M$, 조건수 $\kappa(A)$에 대해:

$$
\|\hat{Q}^T \hat{Q} - I\| \lesssim \epsilon_M \kappa(A)
$$

(CGS는 $\epsilon_M \kappa(A)^2$로 훨씬 나쁘다.)

**증명 개요**: MGS는 각 단계에서 rank-1 업데이트만 수행하므로, Björck(1967)의 후방 오차 해석에 따라 누적 오차가 $\kappa(A)$에 선형으로 의존한다.

---

## 4. 하우스홀더 반사

### 정의 4.1 (하우스홀더 행렬)

$\mathbf{v} \in \mathbb{R}^m \setminus \{\mathbf{0}\}$에 대해:

$$
H_{\mathbf{v}} = I - \frac{2 \mathbf{v} \mathbf{v}^T}{\mathbf{v}^T \mathbf{v}}
$$

### 정리 4.2 (하우스홀더 행렬의 성질)

$H_{\mathbf{v}}$는 **대칭 직교 행렬**이며, 기하학적으로 $\mathbf{v}$에 수직인 초평면에 대한 **반사**이다.

### 증명

**(대칭)** $H^T = I - 2\frac{(\mathbf{v}\mathbf{v}^T)^T}{\mathbf{v}^T \mathbf{v}} = I - 2\frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}} = H$.

**(직교)**

$$
H^T H = H^2 = \left(I - \frac{2\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}}\right)^2 = I - \frac{4\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}} + \frac{4\mathbf{v}(\mathbf{v}^T\mathbf{v})\mathbf{v}^T}{(\mathbf{v}^T\mathbf{v})^2} = I
$$

**(반사)** $\mathbf{x} = \alpha \mathbf{v} + \mathbf{w}$ ($\mathbf{w} \perp \mathbf{v}$)에 대해 $H\mathbf{x} = -\alpha \mathbf{v} + \mathbf{w}$. $\blacksquare$

### 정리 4.3 (하우스홀더 QR의 핵심 단계)

$\mathbf{x} \in \mathbb{R}^m$에 대해 $\mathbf{v} = \mathbf{x} - \sigma \|\mathbf{x}\| \mathbf{e}_1$ ($\sigma = \operatorname{sign}(x_1)$)로 놓으면:

$$
H_{\mathbf{v}} \mathbf{x} = \sigma \|\mathbf{x}\| \mathbf{e}_1
$$

즉, $\mathbf{x}$의 첫 성분 이후를 모두 **0으로** 만들 수 있다.

### 증명

$\mathbf{v}^T \mathbf{v} = \|\mathbf{x}\|^2 - 2\sigma\|\mathbf{x}\| x_1 + \|\mathbf{x}\|^2 = 2\|\mathbf{x}\|(\|\mathbf{x}\| - \sigma x_1) = -2\sigma \|\mathbf{x}\| v_1$.

$\mathbf{v}^T \mathbf{x} = \|\mathbf{x}\|^2 - \sigma \|\mathbf{x}\| x_1 = \|\mathbf{x}\|(\|\mathbf{x}\| - \sigma x_1) = -\sigma \|\mathbf{x}\| v_1$. (마지막 등식은 $v_1 = x_1 - \sigma\|\mathbf{x}\|$로부터.)

따라서:

$$
H_{\mathbf{v}} \mathbf{x} = \mathbf{x} - \frac{2 \mathbf{v}^T \mathbf{x}}{\mathbf{v}^T \mathbf{v}} \mathbf{v} = \mathbf{x} - \frac{-2\sigma\|\mathbf{x}\| v_1}{-2\sigma\|\mathbf{x}\| v_1} \mathbf{v} = \mathbf{x} - \mathbf{v} = \sigma \|\mathbf{x}\| \mathbf{e}_1 \quad \blacksquare
$$

> **🔑 수치적 안정성**: $\sigma = \operatorname{sign}(x_1)$로 선택하는 이유는 $\mathbf{v} = \mathbf{x} - \sigma\|\mathbf{x}\|\mathbf{e}_1$의 성분이 부호 상쇄(catastrophic cancellation)를 일으키지 않도록 하기 위함이다.

### 정리 4.4 (하우스홀더 QR)

$A \in \mathbb{R}^{m \times n}$ ($m \geq n$)에 대해 $n$개의 하우스홀더 반사 $H_1, \ldots, H_n$이 존재하여:

$$
H_n \cdots H_2 H_1 A = R \quad (\text{상삼각})
$$

이때 $Q = H_1 H_2 \cdots H_n$ ($Q^T = Q^{-1}$)로 $A = QR$.

---

## 5. 계산 복잡도

### 정리 5.1

- **Gram-Schmidt (Classical/Modified)**: $2mn^2$ flops
- **Householder QR**: $2mn^2 - \frac{2}{3}n^3$ flops

### 증명 (Householder)

$k$번째 단계에서 $(m-k+1) \times (n-k+1)$ 부분행렬에 하우스홀더 반사 $H_k$를 적용한다. $H_k = I - \beta \mathbf{v}\mathbf{v}^T$이므로 명시적 곱셈 없이:

$$
H_k A_{sub} = A_{sub} - \beta \mathbf{v}(\mathbf{v}^T A_{sub})
$$

이 계산은 $\mathbf{v}^T A_{sub}$ ($2(m-k+1)(n-k+1)$ flops) + rank-1 업데이트 ($2(m-k+1)(n-k+1)$ flops) = $4(m-k+1)(n-k+1)$ flops.

전체:

$$
\sum_{k=1}^{n} 4(m-k+1)(n-k+1) \approx 4 \int_0^n (m-k)(n-k)\,dk = 2mn^2 - \frac{2}{3}n^3 \quad \blacksquare
$$

---

## 6. QR과 최소 제곱

### 정리 6.1 (QR로 Least Squares 풀기)

$A \mathbf{x} = \mathbf{b}$ ($A \in \mathbb{R}^{m \times n}$, $m \geq n$)의 최소 제곱 해는 thin QR $A = \hat{Q}\hat{R}$을 이용해:

$$
\hat{R} \mathbf{x} = \hat{Q}^T \mathbf{b}
$$

로 구해진다. $\hat{R}$이 상삼각이므로 **후방 대입**으로 $O(n^2)$에 해결.

### 증명

최소 제곱은 $\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|^2$. 정규방정식 $A^T A \mathbf{x} = A^T \mathbf{b}$에서 $A^T A = \hat{R}^T \hat{Q}^T \hat{Q} \hat{R} = \hat{R}^T \hat{R}$, $A^T \mathbf{b} = \hat{R}^T \hat{Q}^T \mathbf{b}$. $\hat{R}$이 가역(full rank)이므로 $\hat{R}^T$도 가역, 양변에 $\hat{R}^{-T}$ 곱하면 $\hat{R}\mathbf{x} = \hat{Q}^T \mathbf{b}$. $\blacksquare$

> **⚠️ 정규방정식 vs QR**: 정규방정식은 조건수가 $\kappa(A^T A) = \kappa(A)^2$로 악화되지만, QR은 $\kappa(\hat{R}) = \kappa(A)$ 그대로 유지된다.

---

## 7. Python 실험

### 7.1 CGS vs MGS 직교성 비교

```python
import numpy as np

def classical_gs(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for k in range(n):
        R[:k, k] = Q[:, :k].T @ A[:, k]        # 동시 투영
        v = A[:, k] - Q[:, :k] @ R[:k, k]
        R[k, k] = np.linalg.norm(v)
        Q[:, k] = v / R[k, k]
    return Q, R

def modified_gs(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy().astype(float)
    for k in range(n):
        R[k, k] = np.linalg.norm(V[:, k])
        Q[:, k] = V[:, k] / R[k, k]
        for j in range(k+1, n):
            R[k, j] = Q[:, k] @ V[:, j]
            V[:, j] = V[:, j] - R[k, j] * Q[:, k]  # 순차 투영
    return Q, R

# Hilbert 행렬 (악조건)
n = 8
H = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])

Q_cgs, _ = classical_gs(H)
Q_mgs, _ = modified_gs(H)

print("CGS ||Q^T Q - I||:", np.linalg.norm(Q_cgs.T @ Q_cgs - np.eye(n)))
print("MGS ||Q^T Q - I||:", np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(n)))
# CGS: ~10^-2, MGS: ~10^-8
```

### 7.2 하우스홀더 QR

```python
def householder_qr(A):
    A = A.copy().astype(float)
    m, n = A.shape
    Q = np.eye(m)
    for k in range(n):
        x = A[k:, k]
        sigma = -np.sign(x[0]) if x[0] != 0 else 1.0
        v = x.copy()
        v[0] -= sigma * np.linalg.norm(x)
        v /= np.linalg.norm(v)
        # 부분행렬에 반사 적용
        A[k:, k:] -= 2.0 * np.outer(v, v @ A[k:, k:])
        H_full = np.eye(m)
        H_full[k:, k:] -= 2.0 * np.outer(v, v)
        Q = Q @ H_full
    return Q, np.triu(A)

A = np.random.randn(5, 3)
Q, R = householder_qr(A)
print("||A - QR||:", np.linalg.norm(A - Q @ R))
print("||Q^T Q - I||:", np.linalg.norm(Q.T @ Q - np.eye(5)))
```

### 7.3 SciPy와 비교

```python
from scipy.linalg import qr
Q_sp, R_sp = qr(A)
print("||A - Q_sp @ R_sp||:", np.linalg.norm(A - Q_sp @ R_sp))
```

---

## 8. 요약

| 방법             | 복잡도                      | 직교성 손실             | 메모리    |
| ---------------- | --------------------------- | ----------------------- | --------- |
| Classical GS     | $2mn^2$                     | $\epsilon_M \kappa^2$   | $O(mn)$   |
| Modified GS      | $2mn^2$                     | $\epsilon_M \kappa$     | $O(mn)$   |
| Householder      | $2mn^2 - \tfrac{2}{3}n^3$   | $\epsilon_M$            | $O(mn)$   |
| Givens           | $3mn^2 - n^3$               | $\epsilon_M$            | $O(mn)$   |

- 이론/교육: Gram-Schmidt
- 실무 (dense): **Householder**
- Sparse: Givens

---

## 9. 다음 문서 예고

- **[03. 촐레스키 분해](./03-cholesky-decomposition.md)**: 양의 정부호 행렬 $A = LL^T$, 공분산 행렬, 수치 안정성.

---

## 10. 참고 문헌

- Björck, Å. (1967). *Solving linear least squares problems by Gram-Schmidt orthogonalization*. BIT.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Ch 5.
- Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. Lectures 7–10.
- Householder, A. S. (1958). *Unitary triangularization of a nonsymmetric matrix*. JACM.

---

## 11. 내비게이션

[◀ 01. LU 분해](./01-lu-decomposition.md) | [📚 README](../README.md) | [03. 촐레스키 분해 ▶](./03-cholesky-decomposition.md)
