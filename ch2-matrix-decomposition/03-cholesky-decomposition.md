# Ch2-03. 촐레스키 분해: 양의 정부호 행렬의 제곱근

> "대칭 양의 정부호 행렬은 하삼각 행렬의 제곱이다."

## 📌 학습 목표

- 촐레스키 분해 $A = LL^T$의 **존재성과 유일성**을 양의 정부호(PD) 조건 아래 증명한다.
- LU 분해와의 관계: $A = LL^T$이면 $A = LDL^T$의 특수 경우임을 보인다.
- 촐레스키의 **복잡도 $\frac{1}{3}n^3$**이 일반 LU의 절반인 이유를 유도한다.
- 공분산 행렬, 가우스 샘플링, 칼만 필터에서의 활용을 본다.

---

## 🎯 핵심 질문

> **질문 1**: 왜 촐레스키는 pivoting이 필요 없는가?
> **질문 2**: $A$가 PD가 아니면 왜 $L_{kk} \leq 0$이 나오는가?
> **질문 3**: 공분산 행렬을 왜 **촐레스키**로 샘플링에 사용하는가?

---

## 1. 정의와 주요 정리

### 정의 1.1 (양의 정부호 행렬)

대칭 행렬 $A \in \mathbb{R}^{n \times n}$ ($A = A^T$)가 **양의 정부호(Positive Definite, PD)**라 함은 모든 $\mathbf{x} \neq \mathbf{0}$에 대해 $\mathbf{x}^T A \mathbf{x} > 0$이라는 뜻이다.

### 정의 1.2 (촐레스키 분해)

PD 행렬 $A$에 대해 대각 성분이 **양수**인 하삼각 행렬 $L$이 존재하여 $A = LL^T$로 쓰일 수 있으면 이를 **촐레스키 분해**라 한다.

### 정리 1.3 (존재성과 유일성)

$A$가 PD이면 $A = LL^T$이 되는 **양의 대각** 하삼각 $L$이 **유일하게** 존재한다.

---

## 2. 존재성 증명 (구성적)

$A$의 주요 소행렬식을 $A_k = A_{1:k, 1:k}$라 하자.

### 보조정리 2.1

$A$가 PD이면 모든 $k$에 대해 $A_k$도 PD이다.

### 증명

$\mathbf{x} \in \mathbb{R}^k \setminus \{\mathbf{0}\}$에 대해 $\tilde{\mathbf{x}} = (\mathbf{x}^T, \mathbf{0})^T \in \mathbb{R}^n$. 그러면:

$$
\mathbf{x}^T A_k \mathbf{x} = \tilde{\mathbf{x}}^T A \tilde{\mathbf{x}} > 0 \quad \blacksquare
$$

### 정리 2.2 (촐레스키 귀납 구성)

$n = 1$: $A = [a_{11}]$이 PD이면 $a_{11} > 0$, $L = [\sqrt{a_{11}}]$.

$n \geq 2$: 블록 분할

$$
A = \begin{pmatrix} \alpha & \mathbf{b}^T \\ \mathbf{b} & A_{22} \end{pmatrix}, \quad \alpha = a_{11} > 0
$$

로 놓으면:

$$
A = \begin{pmatrix} \sqrt{\alpha} & 0 \\ \mathbf{b}/\sqrt{\alpha} & I \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & A_{22} - \mathbf{b}\mathbf{b}^T/\alpha \end{pmatrix} \begin{pmatrix} \sqrt{\alpha} & \mathbf{b}^T/\sqrt{\alpha} \\ 0 & I \end{pmatrix}
$$

**Schur complement** $S = A_{22} - \mathbf{b}\mathbf{b}^T/\alpha$도 PD임을 보이면 귀납가정에 의해 $S = L_2 L_2^T$.

**Schur가 PD인 증명**: $\mathbf{y} \in \mathbb{R}^{n-1}$, $\mathbf{y} \neq \mathbf{0}$. $\mathbf{z} = (-\mathbf{b}^T\mathbf{y}/\alpha, \mathbf{y}^T)^T$.

$$
\mathbf{z}^T A \mathbf{z} = \frac{(\mathbf{b}^T\mathbf{y})^2}{\alpha} - 2\frac{(\mathbf{b}^T\mathbf{y})^2}{\alpha} + \mathbf{y}^T A_{22} \mathbf{y} = \mathbf{y}^T S \mathbf{y}
$$

$\mathbf{z} \neq \mathbf{0}$이고 $A$가 PD이므로 $\mathbf{y}^T S \mathbf{y} > 0$. $\blacksquare$

따라서:

$$
L = \begin{pmatrix} \sqrt{\alpha} & 0 \\ \mathbf{b}/\sqrt{\alpha} & L_2 \end{pmatrix}
$$

---

## 3. 유일성 증명

### 정리 3.1

$A = L_1 L_1^T = L_2 L_2^T$이고 $L_1, L_2$의 대각이 양수이면 $L_1 = L_2$.

### 증명

$L_1 L_1^T = L_2 L_2^T$에서 $L_2^{-1} L_1 = L_2^T L_1^{-T}$. 좌변은 하삼각, 우변은 상삼각이므로 **대각 행렬** $D$:

$$
L_2^{-1} L_1 = D \implies L_1 = L_2 D
$$

$L_1 L_1^T = L_2 D (L_2 D)^T = L_2 D D^T L_2^T = L_2 L_2^T$이므로 $D D^T = I$. $D$가 대각이고 대각 성분 $d_i$는 $L_{1, ii}/L_{2, ii} > 0$이므로 $d_i = 1$, 즉 $D = I$. $\blacksquare$

---

## 4. 촐레스키 알고리즘

$A_{ij} = \sum_{k=1}^{\min(i,j)} L_{ik} L_{jk}$에서 $i \geq j$라 하면:

$$
A_{ij} = L_{ij} L_{jj} + \sum_{k=1}^{j-1} L_{ik} L_{jk}
$$

### $j$열을 $j-1$열까지 이미 계산했다고 할 때

- **$i = j$**: $L_{jj} = \sqrt{A_{jj} - \sum_{k=1}^{j-1} L_{jk}^2}$
- **$i > j$**: $L_{ij} = \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk}\right)$

### 정리 4.1 (복잡도)

촐레스키 분해는 $\frac{1}{3}n^3 + O(n^2)$ flops이다.

### 증명

$j$번째 열 계산 flops:

- $L_{jj}$: 곱 $j-1$ + 제곱근 1 ≈ $2(j-1)$
- $L_{ij}$ ($i = j+1, \ldots, n$): 각 $2(j-1) + 1$ flops, 총 $(n-j)(2j-1)$

전체:

$$
\sum_{j=1}^n [2(j-1) + (n-j)(2j-1)] \approx \sum_{j=1}^n 2j(n-j) \approx 2 \int_0^n j(n-j) dj = \frac{n^3}{3} \quad \blacksquare
$$

LU($\frac{2}{3}n^3$)의 **절반**. 대칭성 때문이다.

---

## 5. LDL^T 분해와의 관계

### 정의 5.1

$A$가 대칭이면 $A = LDL^T$ ($L$은 단위 하삼각, $D$는 대각)로도 분해 가능. 이때 $A$가 PD이면 $D_{ii} > 0$.

### 정리 5.2

$A = LDL^T$와 $A = \tilde{L}\tilde{L}^T$의 관계: $\tilde{L} = L D^{1/2}$, 즉 촐레스키의 $L$에서 대각을 꺼내면 $LDL^T$이다.

**장점**: $LDL^T$는 **제곱근 계산이 없어** 수치적으로 더 안정적이고, **준정부호(PSD)** 또는 **부정부호** 대칭에도 확장된다 (block $LDL^T$).

---

## 6. PD 판정법

### 정리 6.1 (여러 PD 동치 조건)

대칭 $A \in \mathbb{R}^{n \times n}$에 대해 다음은 동치:

1. $A$가 PD ($\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$)
2. 모든 고윳값 $\lambda_i > 0$
3. 모든 leading principal minor $\det(A_k) > 0$ (Sylvester's criterion)
4. 촐레스키 분해 $A = LL^T$ ($L$ 대각 양수) 존재

### 증명 스케치

(1 ⇔ 2): 스펙트럼 정리 $A = Q\Lambda Q^T$에서 $\mathbf{x}^T A \mathbf{x} = \sum \lambda_i y_i^2$ ($\mathbf{y} = Q^T\mathbf{x}$).

(1 ⇔ 4): 본 문서 §2, §3.

(1 ⇒ 3): $\mathbf{x}^T A_k \mathbf{x} > 0$로 $A_k$ PD, 모든 고윳값 양수로 $\det(A_k) > 0$.

(3 ⇒ 4): $A_k$의 $\det > 0$이면 LU 분해 존재 (Ch2-01), 대칭성으로 $A = LDL^T$, $D_{kk} = \det(A_k)/\det(A_{k-1}) > 0$로 $\sqrt{D}$ 가능. $\blacksquare$

---

## 7. 응용: 공분산 행렬과 샘플링

### 문제

평균 $\boldsymbol{\mu}$, 공분산 $\Sigma$ ($\Sigma$는 PD)인 다변량 정규분포 $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$에서 샘플링.

### 방법

$\Sigma = LL^T$. $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$이면:

$$
\mathbf{x} = \boldsymbol{\mu} + L\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)
$$

**확인**: $\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}$, $\operatorname{Cov}(\mathbf{x}) = L \operatorname{Cov}(\mathbf{z}) L^T = LL^T = \Sigma$.

### 왜 촐레스키인가?

- 대안: 스펙트럼 분해 $\Sigma = Q\Lambda Q^T$, $\mathbf{x} = \boldsymbol{\mu} + Q\Lambda^{1/2}\mathbf{z}$
- 스펙트럼은 $O(n^3)$, 촐레스키는 $\frac{1}{3}n^3$: **3배 빠름**
- GP, 베이지안 최적화, VAE의 reparameterization trick에서 필수

---

## 8. Python 실험

### 8.1 기본 촐레스키

```python
import numpy as np
from scipy.linalg import cholesky, cho_solve, cho_factor

# PD 행렬 생성
np.random.seed(42)
M = np.random.randn(100, 100)
A = M @ M.T + 0.1 * np.eye(100)  # 양의 정부호

# NumPy (하삼각)
L = np.linalg.cholesky(A)
print("||A - L L^T||:", np.linalg.norm(A - L @ L.T))

# SciPy로 선형계 풀기 (LU보다 2배 빠름)
c, low = cho_factor(A)
x = cho_solve((c, low), np.ones(100))
print("|| A x - 1 ||:", np.linalg.norm(A @ x - 1))
```

### 8.2 가우스 샘플링

```python
# 2차원 공분산
Sigma = np.array([[2.0, 0.8],
                  [0.8, 0.5]])
mu = np.array([1.0, -2.0])
L = np.linalg.cholesky(Sigma)

n = 10000
z = np.random.randn(n, 2)
x = mu + z @ L.T

print("샘플 평균:", x.mean(axis=0))          # ≈ mu
print("샘플 공분산:", np.cov(x.T))            # ≈ Sigma
```

### 8.3 LU vs 촐레스키 속도

```python
import time
from scipy.linalg import lu_factor, lu_solve

n = 2000
M = np.random.randn(n, n)
A = M @ M.T + n * np.eye(n)
b = np.random.randn(n)

t0 = time.time()
lu, piv = lu_factor(A)
lu_solve((lu, piv), b)
print("LU time:", time.time() - t0)

t0 = time.time()
c, low = cho_factor(A)
cho_solve((c, low), b)
print("Cholesky time:", time.time() - t0)
# Cholesky는 LU의 약 1/2
```

---

## 9. 촐레스키의 한계

- **대칭 PD에만 적용**: 비대칭이면 LU, 준정부호면 블록 $LDL^T$
- **수치 민감성**: $A$가 거의 특이(near-singular)하면 $L_{kk}$가 0에 가까워 불안정. 이때 **pivoted Cholesky** 사용 (LAPACK `?pstrf`)
- **업데이트**: rank-1 업데이트 $A + \mathbf{v}\mathbf{v}^T$의 촐레스키는 $O(n^2)$에 갱신 가능 (칼만 필터에서 활용)

---

## 10. 요약

| 분해        | 조건               | 복잡도            | 안정성     |
| ----------- | ------------------ | ----------------- | ---------- |
| LU          | 일반               | $\frac{2}{3}n^3$  | pivoting 필요 |
| PLU         | 모든 비특이        | $\frac{2}{3}n^3$  | 안정       |
| QR          | 모든 full rank     | $2mn^2$           | 매우 안정  |
| **Cholesky**| **대칭 PD**        | $\frac{1}{3}n^3$  | **매우 안정, pivot 불필요** |
| $LDL^T$     | 대칭 (PD/PSD/부정) | $\frac{1}{3}n^3$  | 제곱근 없음 |

---

## 11. 내비게이션

[◀ 02. QR 분해](./02-qr-decomposition.md) | [📚 README](../README.md) | [04. 고유값 분해 ▶](./04-eigendecomposition.md)
