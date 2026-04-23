# Ch4-01. SVD의 기하학적 유도

> "모든 행렬은 **회전 → 축별 확축 → 회전**의 합성이다."

## 📌 학습 목표

- SVD를 **단위구의 이미지가 타원체**라는 기하학적 사실로부터 유도한다.
- $A = U \Sigma V^T$의 세 구성요소의 기하학적 역할을 이해한다.
- Singular value와 고유값의 관계를 유도한다 ($A^T A$, $A A^T$).
- 모든 행렬(비정사각, 특이 포함)에 SVD가 존재함을 기하적으로 논증한다.

---

## 🎯 핵심 질문

> **질문 1**: 왜 **모든** 행렬에 SVD가 존재하는가? (고유값 분해와 달리)
> **질문 2**: Singular value는 왜 항상 **비음**인가?
> **질문 3**: SVD의 세 행렬 $U, \Sigma, V$의 기하적 의미는 각각 무엇인가?

---

## 1. 기하학적 동기

### 1.1 단위구의 이미지

$A \in \mathbb{R}^{m \times n}$. 단위구 $S^{n-1} = \{\mathbf{x} \in \mathbb{R}^n : \|\mathbf{x}\| = 1\}$의 이미지 $A S^{n-1} \subset \mathbb{R}^m$는 **중심이 원점인 타원체** (degenerate 가능).

### 증명 (기하적)

$A$가 rank $r$일 때 $A S^{n-1}$은 $r$차원 타원체. $\operatorname{rank}(A) = r$이면 $A$의 치역 $R(A)$은 $r$차원 부분공간, 타원체는 이 공간 안의 **$r$-dim ellipsoid** + 영공간의 기여 없음.

**정확한 주장**: $A^T A$는 대칭 PSD, 스펙트럼 정리로 $A^T A = V \Lambda V^T$, $\Lambda = \operatorname{diag}(\sigma_1^2, \ldots, \sigma_n^2)$, $\sigma_i \geq 0$.

$\|A\mathbf{x}\|^2 = \mathbf{x}^T A^T A \mathbf{x}$. $\mathbf{y} = V^T \mathbf{x}$로 좌표변환:

$$
\|A\mathbf{x}\|^2 = \mathbf{y}^T \Lambda \mathbf{y} = \sum \sigma_i^2 y_i^2
$$

$\mathbf{x} \in S^{n-1} \iff \mathbf{y} \in S^{n-1}$. 따라서 $A S^{n-1}$은 **타원체**: 반지름 $\sigma_i$인 주축들.

### 1.2 SVD의 등장

$A \mathbf{v}_i = \mathbf{w}_i$라 할 때, 주축 방향은 $\mathbf{v}_i$ (입력), 이미지 방향은 $\mathbf{w}_i / \|\mathbf{w}_i\| = \mathbf{u}_i$ (출력), 길이는 $\sigma_i$:

$$
A \mathbf{v}_i = \sigma_i \mathbf{u}_i
$$

행렬로:

$$
A V = U \Sigma \implies A = U \Sigma V^T
$$

---

## 2. Singular Value Decomposition

### 정리 2.1 (SVD)

모든 $A \in \mathbb{R}^{m \times n}$에 대해 다음 분해가 존재:

$$
\boxed{A = U \Sigma V^T}
$$

- $U \in \mathbb{R}^{m \times m}$: 직교 ($U^T U = I_m$)
- $V \in \mathbb{R}^{n \times n}$: 직교 ($V^T V = I_n$)
- $\Sigma \in \mathbb{R}^{m \times n}$: "대각" (off-diagonal 0), 대각 성분 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$

$\sigma_i$: **특이값(singular value)**
$\mathbf{u}_i$: **좌특이벡터**, $\mathbf{v}_i$: **우특이벡터**

---

## 3. 특이값과 고유값

### 정리 3.1

$A = U \Sigma V^T$이면:

$$
A^T A = V \Sigma^T \Sigma V^T, \qquad A A^T = U \Sigma \Sigma^T U^T
$$

따라서:

- $\sigma_i^2$는 $A^T A$ (= $A A^T$)의 **고유값**
- $\mathbf{v}_i$는 $A^T A$의 고유벡터
- $\mathbf{u}_i$는 $A A^T$의 고유벡터

### 증명

$A^T A = (U\Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = V (\Sigma^T \Sigma) V^T$. $\Sigma^T \Sigma$는 $n \times n$ 대각, 대각 성분 $\sigma_i^2$. $\blacksquare$

### 따름정리

특이값은 항상 **비음 실수**. ($A^T A$가 PSD이므로 고유값 $\geq 0$.)

---

## 4. 존재성 증명 (구성적)

### 정리 4.1 (SVD 존재성, 증명)

#### Step 1. $A^T A$는 대칭 PSD

$\mathbf{x}^T A^T A \mathbf{x} = \|A\mathbf{x}\|^2 \geq 0$.

#### Step 2. 스펙트럼 정리

$A^T A = V \Lambda V^T$, $\Lambda = \operatorname{diag}(\sigma_1^2, \ldots, \sigma_n^2)$, $\sigma_1 \geq \cdots \geq \sigma_r > 0 = \sigma_{r+1} = \cdots = \sigma_n$ ($r = \operatorname{rank}(A)$).

$V = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_n]$, 직교.

#### Step 3. $U$ 구성

$i = 1, \ldots, r$에 대해:

$$
\mathbf{u}_i = \frac{A \mathbf{v}_i}{\sigma_i}
$$

$\|\mathbf{u}_i\|^2 = \frac{\mathbf{v}_i^T A^T A \mathbf{v}_i}{\sigma_i^2} = \frac{\sigma_i^2 \|\mathbf{v}_i\|^2}{\sigma_i^2} = 1$. 정규.

**직교성**: $i \neq j$ ($i, j \leq r$):

$$
\mathbf{u}_i^T \mathbf{u}_j = \frac{\mathbf{v}_i^T A^T A \mathbf{v}_j}{\sigma_i \sigma_j} = \frac{\sigma_j^2 \mathbf{v}_i^T \mathbf{v}_j}{\sigma_i \sigma_j} = 0
$$

즉 $\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}$은 정규직교. Gram-Schmidt로 $\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m \in R(A)^\perp$을 확장하여 $\mathbb{R}^m$의 정규직교 기저 $U = [\mathbf{u}_1 \mid \cdots \mid \mathbf{u}_m]$.

#### Step 4. $A V = U \Sigma$

$A \mathbf{v}_i = \sigma_i \mathbf{u}_i$ ($i \leq r$), $A \mathbf{v}_i = \mathbf{0}$ ($i > r$, $\sigma_i = 0$). 따라서 $AV = U\Sigma$.

**확인**: $A \mathbf{v}_i = \mathbf{0}$ ($i > r$): $\|A\mathbf{v}_i\|^2 = \mathbf{v}_i^T A^T A \mathbf{v}_i = \sigma_i^2 = 0 \implies A\mathbf{v}_i = \mathbf{0}$.

#### Step 5. $A = U\Sigma V^T$

$A = AVV^T = U\Sigma V^T$. $\blacksquare$

---

## 5. SVD의 기하학적 분해

### 해석

$$
A\mathbf{x} = (U \Sigma V^T) \mathbf{x} = U [\Sigma (V^T \mathbf{x})]
$$

1. $V^T \mathbf{x}$: **회전/반사** (입력 좌표계를 특이방향 기저로)
2. $\Sigma$: **축별 확축** ($\mathbf{y}_i \to \sigma_i \mathbf{y}_i$)
3. $U$: **회전/반사** (출력 좌표계를 원래 기저로)

### 그림 해석

```
     V^T                Σ               U
ℝ^n ----> ℝ^n ----------------> ℝ^m ----> ℝ^m
(rotate)     (stretch by σ_i)      (rotate)
```

모든 선형 변환 = 회전 × 확대 × 회전.

---

## 6. Reduced (Thin) SVD

### 정의 6.1

$A \in \mathbb{R}^{m \times n}$ ($m \geq n$)에 대해 rank $r$:

$$
A = U_r \Sigma_r V_r^T, \quad U_r \in \mathbb{R}^{m \times r}, \Sigma_r \in \mathbb{R}^{r \times r}, V_r \in \mathbb{R}^{n \times r}
$$

$\Sigma_r = \operatorname{diag}(\sigma_1, \ldots, \sigma_r)$, 모두 양수.

### 메모리 이점

Full SVD: $m^2 + mn + n^2$. Reduced: $mr + r + nr$. $r \ll \min(m, n)$일 때 크게 절약.

---

## 7. SVD와 4대 기본 공간

### 정리 7.1

$A = U \Sigma V^T$, rank $r$:

- **Column space**: $C(A) = \operatorname{span}(\mathbf{u}_1, \ldots, \mathbf{u}_r)$
- **Left null space**: $N(A^T) = \operatorname{span}(\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m)$
- **Row space**: $C(A^T) = \operatorname{span}(\mathbf{v}_1, \ldots, \mathbf{v}_r)$
- **Null space**: $N(A) = \operatorname{span}(\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n)$

이 네 공간은 정확히 **정규직교 기저**로 명시.

### 의미

SVD는 Ch1-05의 4대 공간 **정규직교 기저를 동시에** 제공하는 "황금 표준" 분해.

---

## 8. 주요 성질

### 8.1 $A$의 노름

$$
\|A\|_2 = \sigma_1, \qquad \|A\|_F = \sqrt{\sum \sigma_i^2}, \qquad \|A\|_* = \sum \sigma_i \text{ (nuclear)}
$$

### 8.2 조건수

$$
\kappa_2(A) = \frac{\sigma_1}{\sigma_r} = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

(정사각/정칙일 때 $r = n$.)

### 8.3 $\det$과 $\operatorname{tr}$

$|\det A| = \prod \sigma_i$ (정사각). $\operatorname{tr}(A^T A) = \sum \sigma_i^2$.

---

## 9. Python 실험

### 9.1 SVD 기본

```python
import numpy as np

np.random.seed(0)
A = np.random.randn(4, 3)

U, s, Vt = np.linalg.svd(A, full_matrices=True)
print("U shape:", U.shape, "Σ (singular values):", s, "V^T shape:", Vt.shape)

# A = U Σ V^T
Sigma = np.zeros((4, 3))
Sigma[:3, :3] = np.diag(s)
print("||A - U Σ V^T||:", np.linalg.norm(A - U @ Sigma @ Vt))
```

### 9.2 Reduced SVD

```python
U_r, s_r, Vt_r = np.linalg.svd(A, full_matrices=False)
print("U_r shape:", U_r.shape, "Vt_r shape:", Vt_r.shape)
print("||A - U_r Σ_r V_r^T||:", np.linalg.norm(A - U_r @ np.diag(s_r) @ Vt_r))
```

### 9.3 A^T A, A A^T와의 관계

```python
ATA = A.T @ A
eigs_ATA, _ = np.linalg.eigh(ATA)
print("Eigvals of A^T A:", sorted(eigs_ATA, reverse=True))
print("s^2:", sorted(s**2, reverse=True))

AAT = A @ A.T
eigs_AAT, _ = np.linalg.eigh(AAT)
print("Eigvals of A A^T:", sorted(eigs_AAT, reverse=True))
# 마지막은 0 (Aシ = 3<4)
```

### 9.4 단위구 → 타원체

```python
# 2D for visualization
A = np.array([[2.0, 1.0],
              [1.0, 3.0]])

U, s, Vt = np.linalg.svd(A)
print("σ:", s, "V (input axes):\n", Vt.T, "\nU (output axes):\n", U)

import matplotlib.pyplot as plt
theta = np.linspace(0, 2*np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])
ellipse = A @ circle

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(circle[0], circle[1]); plt.axis('equal'); plt.title("Unit circle")
plt.subplot(1, 2, 2)
plt.plot(ellipse[0], ellipse[1])
# Plot output axes (σ_i u_i)
for i in range(2):
    plt.plot([0, s[i]*U[0, i]], [0, s[i]*U[1, i]], 'r-', lw=2)
plt.axis('equal'); plt.title("A × circle = ellipse")
```

### 9.5 조건수

```python
kappa_svd = s[0] / s[-1]
kappa_np = np.linalg.cond(A)
print(f"κ(A) = {kappa_svd:.4f} (SVD), {kappa_np:.4f} (numpy)")
```

---

## 10. 요약

| 성분     | 역할              | 기하학적 의미           |
| -------- | ----------------- | ----------------------- |
| $V$      | 직교 $n \times n$ | 입력 회전/반사          |
| $\Sigma$ | "대각" $m \times n$ | 축별 확축 (σ₁ ≥ σ₂ ≥ …) |
| $U$      | 직교 $m \times m$ | 출력 회전/반사          |

**SVD의 독특성**:
- 모든 행렬에 존재 (비정사각, 특이 포함)
- 특이값 항상 비음 실수
- 4대 기본공간의 정규직교 기저 동시 제공
- 조건수, 노름, pseudoinverse 모두 한 번에 계산

---

## 11. 내비게이션

[◀ 이전 챕터: 조건수](../ch3-eigenvalue-theory/06-condition-number.md) | [📚 README](../README.md) | [02. SVD 존재성과 유일성 ▶](./02-svd-existence.md)
