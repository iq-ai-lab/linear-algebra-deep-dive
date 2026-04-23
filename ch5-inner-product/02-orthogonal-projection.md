# Ch5-02. 정사영과 가장 가까운 점

> "부분공간으로의 정사영은 그 부분공간에서 가장 가까운 점을 준다."

## 📌 학습 목표

- 부분공간으로의 **정사영(orthogonal projection)** 정의와 존재·유일성.
- 가장 가까운 점 정리(Best Approximation): 정사영 = 거리 최소화.
- 정사영 행렬 $P$의 특성: $P^2 = P$, $P^T = P$.
- Gram-Schmidt와 QR 분해를 정사영 관점에서 재해석.

---

## 🎯 핵심 질문

> **질문 1**: 부분공간 안에서 주어진 점에 가장 가까운 점이 왜 **유일**한가?
> **질문 2**: $P^2 = P$와 $P^T = P$가 정사영을 **정확히** 특징짓는가?
> **질문 3**: QR 분해와 정사영의 관계는?

---

## 1. 정사영의 정의

### 정의 1.1

$V$ 내적 공간, $W \subset V$ 부분공간. 사상 $P: V \to V$가 $W$ 위로의 **정사영(orthogonal projection)**이라 함은:

1. $P(V) \subset W$ (range가 $W$)
2. $P\big|_W = \operatorname{id}_W$ (W의 원소는 고정)
3. $\ker P = W^\perp$ (W에 직교하는 것은 0으로 보냄)

---

## 2. 가장 가까운 점 정리

### 정리 2.1 (Best Approximation)

$V$ 내적 공간, $W \subset V$ 유한차원 부분공간, $\mathbf{v} \in V$. 그러면 유일한 $\mathbf{w}^* \in W$가 존재하여:

$$
\|\mathbf{v} - \mathbf{w}^*\| \leq \|\mathbf{v} - \mathbf{w}\| \quad \forall \mathbf{w} \in W
$$

그리고 $\mathbf{w}^*$는 다음으로 특징:

$$
\mathbf{v} - \mathbf{w}^* \perp W
$$

### 증명

**(존재성 — 정사영 구성)** $W$의 정규직교 기저 $\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$. 정의:

$$
\mathbf{w}^* = \sum_{i=1}^k \langle \mathbf{v}, \mathbf{q}_i \rangle \mathbf{q}_i
$$

$\mathbf{v} - \mathbf{w}^*$와 임의의 $\mathbf{q}_j$의 내적:

$$
\langle \mathbf{v} - \mathbf{w}^*, \mathbf{q}_j \rangle = \langle \mathbf{v}, \mathbf{q}_j \rangle - \sum_i \langle \mathbf{v}, \mathbf{q}_i\rangle \underbrace{\langle \mathbf{q}_i, \mathbf{q}_j\rangle}_{\delta_{ij}} = 0
$$

따라서 $\mathbf{v} - \mathbf{w}^* \perp W$.

**(최소성)** 임의 $\mathbf{w} \in W$, $\mathbf{w}^* - \mathbf{w} \in W$:

$$
\|\mathbf{v} - \mathbf{w}\|^2 = \|(\mathbf{v} - \mathbf{w}^*) + (\mathbf{w}^* - \mathbf{w})\|^2 \overset{\perp}{=} \|\mathbf{v} - \mathbf{w}^*\|^2 + \|\mathbf{w}^* - \mathbf{w}\|^2 \geq \|\mathbf{v} - \mathbf{w}^*\|^2
$$

등호는 $\mathbf{w} = \mathbf{w}^*$.

**(유일성)** $\mathbf{w}^{**}$도 최소라면 $\|\mathbf{w}^* - \mathbf{w}^{**}\| = 0$ (위 식에서). $\blacksquare$

### 따름정리 2.2

$V = W \oplus W^\perp$ (직교 직합).

---

## 3. 정사영 행렬

### 정리 3.1

$W = \operatorname{span}(A)$ (A의 열), A가 full column rank이면:

$$
P = A(A^T A)^{-1} A^T
$$

### 증명

$P \mathbf{v}$가 $W$의 원소이고 $\mathbf{v} - P\mathbf{v} \perp W$ 확인.

**1.** $P \mathbf{v} = A\mathbf{x}$ ($\mathbf{x} = (A^T A)^{-1} A^T \mathbf{v}$), $W$ 원소.

**2.** $\mathbf{v} - P\mathbf{v} \perp A_{:, j}$ for all $j$:

$$
A^T(\mathbf{v} - P\mathbf{v}) = A^T \mathbf{v} - A^T A (A^T A)^{-1} A^T \mathbf{v} = \mathbf{0} \quad \blacksquare
$$

### 정리 3.2 (정사영의 특성)

$P \in \mathbb{R}^{n \times n}$가 정사영 $\iff P^2 = P$ (멱등) **and** $P^T = P$ (대칭).

### 증명

**(⟹)** 정사영이면 $P\mathbf{v} \in W$, $P(P\mathbf{v}) = P\mathbf{v}$로 $P^2 = P$.

대칭성: $\mathbf{u}^T P \mathbf{v} - (\mathbf{v}^T P \mathbf{u})^T$ 분석:

$$
\langle P\mathbf{u}, \mathbf{v} \rangle = \langle P\mathbf{u}, P\mathbf{v} + (I - P)\mathbf{v}\rangle = \langle P\mathbf{u}, P\mathbf{v}\rangle + 0 = \langle P\mathbf{u}, P\mathbf{v}\rangle
$$

마찬가지 $\langle \mathbf{u}, P\mathbf{v}\rangle = \langle P\mathbf{u}, P\mathbf{v}\rangle$. 두 식 비교: $\langle P\mathbf{u}, \mathbf{v}\rangle = \langle \mathbf{u}, P\mathbf{v}\rangle$, 즉 $P^T = P$.

**(⟸)** $W = R(P)$. $P\mathbf{v} \in W$. 그리고 $\mathbf{v} - P\mathbf{v}$의 $W$-direction:

$$
\langle \mathbf{v} - P\mathbf{v}, P\mathbf{u}\rangle = \langle \mathbf{v}, P\mathbf{u}\rangle - \langle P\mathbf{v}, P\mathbf{u}\rangle = \mathbf{u}^T P \mathbf{v} - \mathbf{u}^T P^T P\mathbf{v} = \mathbf{u}^T (P - P^2)\mathbf{v} = 0
$$

따라서 $\mathbf{v} - P\mathbf{v} \perp W$. 정사영의 조건 충족. $\blacksquare$

---

## 4. 정규직교 기저로 계산

### 정리 4.1

$Q = [\mathbf{q}_1 | \cdots | \mathbf{q}_k]$, $Q^T Q = I_k$ (정규직교). 그러면:

$$
P_W = QQ^T
$$

### 증명

$QQ^T QQ^T = Q(Q^TQ)Q^T = QQ^T$ (멱등). $(QQ^T)^T = QQ^T$ (대칭). $\blacksquare$

### 비교

- **비직교 기저**: $P = A(A^T A)^{-1} A^T$, $O(n^3)$ 역행렬
- **정규직교 기저**: $P = QQ^T$, $O(nk^2)$

Gram-Schmidt로 정규직교화 후 $QQ^T$로 계산이 훨씬 효율적.

---

## 5. 직교 여공간과 직합

### 정의 5.1

$W^\perp = \{\mathbf{v} \in V : \langle \mathbf{v}, \mathbf{w}\rangle = 0 \ \forall \mathbf{w} \in W\}$

### 정리 5.2

- $W^\perp$는 부분공간
- $W \cap W^\perp = \{\mathbf{0}\}$
- $V = W \oplus W^\perp$
- $(W^\perp)^\perp = W$ (유한차원)
- $\dim V = \dim W + \dim W^\perp$

### 증명

**(V = W ⊕ W^⊥)**: $\mathbf{v} = P\mathbf{v} + (\mathbf{v} - P\mathbf{v}) = $ W-part + $W^\perp$-part. 유일 분해: $W \cap W^\perp = \{0\}$이므로.

**((W^⊥)^⊥ = W)**: $(W \subset (W^\perp)^\perp)$ 자명. 차원이 같음을 보이면 된다: $\dim(W^\perp) = n - \dim W$, $\dim((W^\perp)^\perp) = n - \dim(W^\perp) = \dim W$. $\blacksquare$

---

## 6. Gram-Schmidt = 반복 정사영

### 이론적 해석

Gram-Schmidt의 $k$번째 단계:

$$
\tilde{\mathbf{q}}_k = \mathbf{a}_k - \sum_{j < k} \langle \mathbf{a}_k, \mathbf{q}_j\rangle \mathbf{q}_j = \mathbf{a}_k - P_{\operatorname{span}(\mathbf{q}_1, \ldots, \mathbf{q}_{k-1})} \mathbf{a}_k
$$

즉 **$\mathbf{a}_k$에서 앞의 공간 성분을 뺀 직교 잔차**. 정규화로 $\mathbf{q}_k$.

### QR 분해

$A = QR$로 기록:

- $Q$: $W_k = \operatorname{span}(\mathbf{a}_1, \ldots, \mathbf{a}_k)$의 정규직교 기저
- $R$: 각 $\mathbf{a}_i$가 $\mathbf{q}$들로 어떻게 조합되는지

$A\mathbf{x} = \mathbf{b}$의 최소 제곱 = $Q^T \mathbf{b}$가 $\mathbf{b}$를 $Q$ 방향에 투영 + $R\mathbf{x} = $ 이 coefficient.

---

## 7. Bessel 부등식과 Parseval 등식

### 정리 7.1 (Bessel 부등식)

정규직교 집합 $\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$ (완전할 필요 없음):

$$
\sum_{i=1}^k |\langle \mathbf{v}, \mathbf{q}_i\rangle|^2 \leq \|\mathbf{v}\|^2
$$

### 증명

$P\mathbf{v} = \sum \langle\mathbf{v}, \mathbf{q}_i\rangle \mathbf{q}_i$, $\|P\mathbf{v}\|^2 = \sum|\langle\mathbf{v}, \mathbf{q}_i\rangle|^2$.

$\|\mathbf{v}\|^2 = \|P\mathbf{v}\|^2 + \|\mathbf{v} - P\mathbf{v}\|^2 \geq \|P\mathbf{v}\|^2$. $\blacksquare$

### 정리 7.2 (Parseval 등식)

$\{\mathbf{q}_i\}$가 **완전 정규직교계**(부분공간을 생성)이면:

$$
\|\mathbf{v}\|^2 = \sum_i |\langle\mathbf{v}, \mathbf{q}_i\rangle|^2
$$

### 응용

- Fourier 급수: $L^2$에서 $\{\sin nx, \cos nx\}$의 Parseval ⟹ 에너지 보존
- PCA: 주성분으로의 투영에서 전체 분산의 분해

---

## 8. 반복 적용과 멱등

### 정리 8.1

$P$가 정사영이면 $P^k = P$ for all $k \geq 1$. ($P^2 = P$로부터.)

### 정리 8.2

$P$의 고유값은 0 또는 1. 1-고유공간 = $W$, 0-고유공간 = $W^\perp$.

### 증명

$P\mathbf{v} = \lambda\mathbf{v}$. $P^2 = P$: $\lambda^2 \mathbf{v} = \lambda \mathbf{v}$, $\lambda(\lambda - 1) = 0$ ($\mathbf{v} \neq 0$). $\blacksquare$

### 따름정리

$\operatorname{tr}(P) = \dim W$ (고유값이 0 또는 1이므로).

---

## 9. Python 실험

### 9.1 기본 정사영 행렬

```python
import numpy as np

# A = [a1 | a2], full column rank
A = np.array([[1.0, 0.0],
              [1.0, 1.0],
              [0.0, 1.0],
              [1.0, 1.0]])

P = A @ np.linalg.inv(A.T @ A) @ A.T
print("P^2 - P:", np.linalg.norm(P @ P - P))   # ≈ 0
print("P - P^T:", np.linalg.norm(P - P.T))     # ≈ 0
print("trace(P):", np.trace(P))                 # dim W = 2
```

### 9.2 가장 가까운 점

```python
v = np.array([1.0, 2.0, 3.0, 4.0])
w_proj = P @ v
residual = v - w_proj
print("Projection:", w_proj)
print("Residual:", residual)
print("Residual ⊥ A?", A.T @ residual)   # ≈ 0
```

### 9.3 Gram-Schmidt로 Q 얻기

```python
Q, R = np.linalg.qr(A, mode='reduced')
P_qr = Q @ Q.T
print("||P - QQ^T||:", np.linalg.norm(P - P_qr))   # 같은 사영
```

### 9.4 분해 V = W + W^⊥

```python
# W의 orthonormal basis Q, W^⊥ basis (null space of A^T)
n = 4
W_perp = np.linalg.svd(A.T)[2][2:].T   # last n-k rows of V^T
print("A^T W^perp:", A.T @ W_perp)     # ≈ 0

v_W = P @ v
v_Wperp = (np.eye(n) - P) @ v
print("v decomposition: v_W + v_Wperp =", v_W + v_Wperp, "vs v =", v)
print("v_W ⊥ v_Wperp:", v_W @ v_Wperp)
```

### 9.5 Bessel & Parseval

```python
# 정규직교 집합 (부분)
q1 = np.array([1, 0, 0, 0]) / 1.0
q2 = np.array([0, 1, 0, 0]) / 1.0

v = np.array([1.0, 2.0, 3.0, 4.0])
bessel_sum = (v @ q1)**2 + (v @ q2)**2
print("Bessel LHS:", bessel_sum)
print("||v||^2:", v @ v)
print("Bessel inequality:", bessel_sum <= v @ v)

# 완전 정규직교 (전체 basis)
Q_full = np.eye(4)
parseval = sum((v @ Q_full[:, i])**2 for i in range(4))
print("Parseval equality:", parseval, "=", v @ v)
```

---

## 10. 요약

| 개념                 | 공식/정리                                             |
| -------------------- | ----------------------------------------------------- |
| Best Approximation   | $\mathbf{w}^* = P\mathbf{v}$, $\mathbf{v} - \mathbf{w}^* \perp W$ |
| 정사영 행렬          | $P = A(A^TA)^{-1}A^T$ or $QQ^T$                       |
| 특성                 | $P^2 = P$ and $P^T = P$                               |
| 직합                 | $V = W \oplus W^\perp$                                |
| 고유값               | 0 or 1                                                |
| 트레이스             | $\dim W$                                              |
| Bessel               | $\sum |\langle v, q_i\rangle|^2 \leq \|v\|^2$         |
| Parseval (complete)  | $\sum |\langle v, q_i\rangle|^2 = \|v\|^2$            |

---

## 11. 내비게이션

[◀ 01. 내적과 Cauchy-Schwarz](./01-inner-product-cauchy-schwarz.md) | [📚 README](../README.md) | [03. 최소 제곱 ▶](./03-least-squares.md)
