# 5.4 Gram 행렬과 양의 준정부호성 (PSD)

> "내적 구조를 행렬 한 장에 담으면, 기하는 선형대수가 된다."

---

## 1. 학습 목표

- **Gram 행렬** $G_{ij} = \langle v_i, v_j \rangle$의 정의와 기본 성질을 이해한다.
- Gram 행렬이 **항상 양의 준정부호(PSD)** 임을 증명하고, 언제 양의 정부호(PD)가 되는지 특징짓는다.
- **Gram 행렬식(Gram determinant)** 이 평행육면체의 부피의 제곱임을 유도한다.
- **선형독립성의 대수적 판정** (Gram 행렬의 가역성)을 확립한다.
- Gram 행렬과 **Cholesky 분해**, **커널 방법(Kernel Method)** 의 연결을 본다.

---

## 2. Gram 행렬의 정의

### 2.1 정의

내적공간 $(V, \langle \cdot, \cdot \rangle)$에서 벡터들 $v_1, v_2, \ldots, v_k \in V$가 주어졌을 때, **Gram 행렬(Gram matrix)** $G \in \mathbb{R}^{k \times k}$는 다음과 같이 정의된다:

$$
G_{ij} = \langle v_i, v_j \rangle, \quad 1 \le i, j \le k
$$

행렬 형태로 $A = [v_1 \mid v_2 \mid \cdots \mid v_k] \in \mathbb{R}^{n \times k}$ (열벡터가 $v_i$인 행렬)로 쓰면,

$$
\boxed{G = A^T A}
$$

이 간단한 식이 Gram 행렬의 **모든 성질을 결정**한다.

### 2.2 복소수 내적공간의 경우

복소수 내적공간에서는 켤레를 취하므로

$$
G_{ij} = \langle v_i, v_j \rangle, \quad G = A^* A \quad (A^* = \overline{A}^T)
$$

이 경우 $G$는 **에르미트 행렬(Hermitian)** 이며 PSD이다. 이하에서는 주로 실수 경우를 다룬다.

---

## 3. 기본 성질

### 3.1 대칭성

$$
G_{ji} = \langle v_j, v_i \rangle = \langle v_i, v_j \rangle = G_{ij}
$$

따라서 $G = G^T$ (대칭).

### 3.2 양의 준정부호성 (Positive Semi-Definite)

**정리 5.4.1 (Gram 행렬의 PSD성).** 임의의 벡터 $v_1, \ldots, v_k$에 대해 Gram 행렬 $G = A^T A$는 PSD이다. 즉,

$$
x^T G x \ge 0 \quad \forall x \in \mathbb{R}^k
$$

**증명.**

$x = (x_1, \ldots, x_k)^T \in \mathbb{R}^k$를 임의로 택하자.

$$
x^T G x = x^T (A^T A) x = (Ax)^T (Ax) = \|Ax\|_2^2 \ge 0. \qquad \blacksquare
$$

이 증명은 **세 줄**로 끝나지만, 핵심을 관통한다: $G$의 이차형식은 $A$의 열의 선형결합의 **노름의 제곱**이다.

### 3.3 양의 정부호성 (Positive Definite)

**정리 5.4.2.** Gram 행렬 $G$가 양의 정부호(PD)일 필요충분조건은 $v_1, \ldots, v_k$가 **선형독립**인 것이다.

**증명.**

$G$가 PD라는 것은 $x^T G x = 0 \iff x = 0$을 의미한다. 위 3.2의 계산에서

$$
x^T G x = \|Ax\|_2^2 = \left\|\sum_{i=1}^k x_i v_i\right\|^2
$$

따라서

$$
x^T G x = 0 \iff \sum x_i v_i = 0
$$

$(\Rightarrow)$ $G$ PD라 가정하자. $\sum x_i v_i = 0$이면 $x^T G x = 0$, 즉 $x = 0$. 따라서 $v_i$들은 선형독립.

$(\Leftarrow)$ $v_i$들이 선형독립이라 가정하자. $x^T G x = 0$이면 $\sum x_i v_i = 0$이고, 선형독립에 의해 $x = 0$. 따라서 $G$는 PD. $\blacksquare$

---

## 4. Gram 행렬식과 부피

### 4.1 2차원: 평행사변형 넓이

두 벡터 $v_1, v_2 \in \mathbb{R}^n$이 만드는 평행사변형의 넓이를 계산해 보자.

$v_1$과 $v_2$ 사이의 각을 $\theta$라 하면

$$
\text{Area} = \|v_1\| \|v_2\| \sin\theta
$$

넓이의 제곱은

$$
\text{Area}^2 = \|v_1\|^2 \|v_2\|^2 \sin^2\theta = \|v_1\|^2 \|v_2\|^2 (1 - \cos^2\theta) = \|v_1\|^2\|v_2\|^2 - \langle v_1, v_2 \rangle^2
$$

한편 Gram 행렬은

$$
G = \begin{pmatrix} \langle v_1, v_1 \rangle & \langle v_1, v_2 \rangle \\ \langle v_2, v_1 \rangle & \langle v_2, v_2 \rangle \end{pmatrix}, \quad \det G = \|v_1\|^2\|v_2\|^2 - \langle v_1, v_2 \rangle^2
$$

따라서

$$
\boxed{\text{Area}^2 = \det G}
$$

### 4.2 일반화: Gram 행렬식 = 부피의 제곱

**정리 5.4.3 (Gram 행렬식과 평행육면체의 부피).**

$v_1, \ldots, v_k \in \mathbb{R}^n$ ($k \le n$)이 만드는 $k$차원 평행육면체의 부피를 $\text{Vol}_k(v_1, \ldots, v_k)$라 하면,

$$
\text{Vol}_k(v_1, \ldots, v_k)^2 = \det G
$$

여기서 $G$는 Gram 행렬이다.

**증명.**

$A = [v_1 \mid \cdots \mid v_k] \in \mathbb{R}^{n \times k}$. $A$의 QR 분해

$$
A = QR, \quad Q \in \mathbb{R}^{n \times k},\ Q^T Q = I_k,\ R \in \mathbb{R}^{k \times k}\ \text{(상삼각)}
$$

평행육면체의 부피는 Gram-Schmidt로 정규화하면서 각 축의 길이를 곱한 것이므로

$$
\text{Vol}_k = |r_{11} r_{22} \cdots r_{kk}| = |\det R|
$$

한편

$$
G = A^T A = (QR)^T (QR) = R^T Q^T Q R = R^T R
$$

따라서

$$
\det G = \det(R^T R) = (\det R)^2 = \text{Vol}_k^2. \qquad \blacksquare
$$

### 4.3 선형독립성의 기하적 해석

$v_1, \ldots, v_k$가 선형**종속** $\iff$ 평행육면체가 납작(부피 = 0) $\iff \det G = 0$ $\iff G$ 특이 행렬

**정리 5.4.4.**

$$
\boxed{v_1, \ldots, v_k\ \text{선형독립} \iff \det G \ne 0 \iff G\ \text{가역}}
$$

**증명.** 위의 4.2와 정리 5.4.2를 조합하면 즉시. $\blacksquare$

---

## 5. Gram 행렬과 Cholesky 분해

### 5.1 역방향 문제

지금까지는 "벡터 $\to$ Gram 행렬"의 방향이었다. 반대로 물어보자:

> 임의의 대칭 PSD 행렬 $G \in \mathbb{R}^{k \times k}$가 주어졌을 때, $G = A^T A$를 만족하는 $A$가 존재하는가?

**정리 5.4.5.** 대칭 행렬 $G$가 PSD일 필요충분조건은 $G = A^T A$로 표현되는 $A$가 존재하는 것이다.

**증명.**

$(\Leftarrow)$ 정리 5.4.1에 의해 $A^T A$는 항상 PSD.

$(\Rightarrow)$ $G$가 PSD라 가정. 스펙트럼 정리(제2장)에 의해 직교 대각화 가능:

$$
G = U \Lambda U^T, \quad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_k),\ \lambda_i \ge 0
$$

$\Lambda^{1/2} = \text{diag}(\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_k})$를 정의하고 $A = \Lambda^{1/2} U^T$라 두면

$$
A^T A = U \Lambda^{1/2} \Lambda^{1/2} U^T = U \Lambda U^T = G. \qquad \blacksquare
$$

### 5.2 Cholesky를 통한 표현

$G$가 PD(strictly positive)이면 제2장의 Cholesky 분해에 의해

$$
G = L L^T, \quad L\ \text{하삼각, 양의 대각}
$$

따라서 $A = L^T$가 원하는 상삼각 행렬이 된다.

### 5.3 수치적 의의

- Cholesky는 $G = A^T A$의 "루트"를 구해주는 것과 같다.
- $A^T A$를 직접 구성하면 조건수가 제곱이 되는 반면 ($\kappa(A^T A) = \kappa(A)^2$), Cholesky는 대칭 구조를 활용하여 복잡도를 $\frac{1}{3}k^3$로 줄인다.
- 최소제곱의 **정규방정식**이 $A^T A x = A^T b$이므로 Cholesky가 자연스럽게 적용된다.

---

## 6. 일반 내적에서의 Gram 행렬

### 6.1 가중 내적

$M \in \mathbb{R}^{n \times n}$이 PD일 때 $\langle u, v \rangle_M = u^T M v$라는 가중 내적을 생각하자.

$M = L L^T$ (Cholesky)라 하면

$$
\langle u, v \rangle_M = u^T L L^T v = (L^T u)^T (L^T v) = \langle L^T u, L^T v \rangle
$$

즉 $M$-내적은 좌표변환 $u \mapsto L^T u$ 후의 표준 내적이다. Gram 행렬은

$$
G_{ij} = \langle v_i, v_j \rangle_M = v_i^T M v_j
$$

벡터 행렬 $A = [v_1 \mid \cdots \mid v_k]$에 대해

$$
G = A^T M A
$$

이 역시 PSD이다 ($x^T G x = x^T A^T M A x = (Ax)^T M (Ax) \ge 0$).

### 6.2 함수공간의 Gram 행렬

$L^2([a, b])$에서 함수 $f_1, \ldots, f_k$의 Gram 행렬은

$$
G_{ij} = \int_a^b f_i(t) f_j(t)\, dt
$$

이를 통해 **다항식 근사(Legendre, Chebyshev)**, **푸리에 급수** 등이 Gram 행렬 관점에서 균일하게 다뤄진다.

---

## 7. 커널 방법 (Kernel Trick)

### 7.1 아이디어

데이터 $x_1, \ldots, x_n \in \mathbb{R}^d$를 고차원 특징공간으로 매핑하는 함수 $\phi: \mathbb{R}^d \to \mathcal{H}$가 있다고 하자.

많은 알고리즘(SVM, Kernel PCA, Gaussian Process)은 데이터를 **내적 $\langle \phi(x_i), \phi(x_j) \rangle$ 를 통해서만 사용**한다. 따라서 매핑 $\phi$를 직접 계산하지 않고

$$
K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle
$$

만 정의해도 알고리즘이 작동한다.

### 7.2 커널 행렬 = Gram 행렬

$K_{ij} = K(x_i, x_j)$로 이루어진 행렬을 **커널 행렬(Kernel matrix)** 혹은 **Gram 행렬**이라 부른다. 대응되는 $\phi$가 존재하려면 $K$가 PSD이어야 한다.

**Mercer 정리.** 연속 대칭 커널 $K(x, y)$가 PSD이면 (즉, 모든 유한 샘플 커널 행렬이 PSD이면) Hilbert 공간 $\mathcal{H}$와 매핑 $\phi$가 존재하여 $K(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}$이다.

### 7.3 예시

- **선형 커널**: $K(x, y) = x^T y$
- **다항식 커널**: $K(x, y) = (x^T y + c)^d$
- **RBF (Gaussian) 커널**: $K(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$
- **라플라스 커널**: $K(x, y) = \exp(-\alpha \|x - y\|)$

이 모든 경우 커널 행렬은 PSD이며, 암묵적 특징공간이 존재한다.

---

## 8. 응용: Gram 행렬식 부등식

### 8.1 Hadamard 부등식

**정리 5.4.6 (Hadamard).** $A \in \mathbb{R}^{n \times n}$의 열을 $a_1, \ldots, a_n$이라 할 때

$$
|\det A| \le \prod_{i=1}^n \|a_i\|_2
$$

등호는 열들이 서로 직교할 때 성립한다.

**증명.**

$G = A^T A$. 그러면 $(\det A)^2 = \det G$. $G$는 PSD이고 대각성분은 $G_{ii} = \|a_i\|^2$이다. 스펙트럼 정리에 의해 $G$의 고유값 $\lambda_i \ge 0$. AM-GM 부등식:

$$
\det G = \prod \lambda_i \le \left(\frac{\sum \lambda_i}{n}\right)^n = \left(\frac{\text{tr}(G)}{n}\right)^n
$$

그러나 이 bound는 Hadamard보다 약하다. 정확한 증명은 **Gram-Schmidt**로:

$a_1, \ldots, a_n$에 GS를 적용하면 $a_i = \sum_{j \le i} r_{ji} q_j$이고

$$
\|a_i\|^2 = \sum_{j \le i} r_{ji}^2 \ge r_{ii}^2
$$

한편 $|\det A| = |\det R| = \prod |r_{ii}|$이므로

$$
|\det A|^2 = \prod r_{ii}^2 \le \prod \|a_i\|^2. \qquad \blacksquare
$$

### 8.2 Fischer 부등식

$G = \begin{pmatrix} G_{11} & G_{12} \\ G_{12}^T & G_{22} \end{pmatrix}$가 PSD 분할이면

$$
\det G \le \det G_{11} \cdot \det G_{22}
$$

증명은 Schur 보수 $G_{22} - G_{12}^T G_{11}^{-1} G_{12}$가 PSD라는 성질을 이용한다 (제2장 참고).

---

## 9. Gram 행렬의 역행렬 (있을 때)

### 9.1 역행렬의 쓸모

**최적 계수 공식.** 선형독립 $v_1, \ldots, v_k$에 대해 표적 벡터 $b$의 $\text{span}(v_1, \ldots, v_k)$ 위 정사영은

$$
P b = \sum_{i=1}^k c_i v_i, \quad c = G^{-1} A^T b
$$

유도: $A^T b = A^T A c = G c$ (정규방정식), $G$ PD이면 $c = G^{-1} A^T b$.

### 9.2 조건수

$$
\kappa_2(G) = \kappa_2(A^T A) = \kappa_2(A)^2
$$

$A$의 조건수가 $10^4$이면 $G$의 조건수는 $10^8$. 이는 **정규방정식을 직접 푸는 것이 수치적으로 불리**한 이유다 (제5.3절 참고).

---

## 10. Python 실험

### 10.1 Gram 행렬의 PSD성 검증

```python
import numpy as np

np.random.seed(0)

# 무작위 벡터
n, k = 10, 5
A = np.random.randn(n, k)

# Gram 행렬
G = A.T @ A

# 대칭성
print("대칭성:", np.allclose(G, G.T))

# 고유값 모두 비음수?
eigs = np.linalg.eigvalsh(G)
print("고유값:", eigs)
print("모두 비음수:", np.all(eigs >= -1e-10))
```

### 10.2 Gram 행렬식 = 부피의 제곱

```python
import numpy as np

# 3차원 평행육면체
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([1.0, 2.0, 0.0])
v3 = np.array([0.0, 1.0, 3.0])

A = np.column_stack([v1, v2, v3])
G = A.T @ A

vol_from_gram = np.sqrt(np.linalg.det(G))
vol_from_det  = abs(np.linalg.det(A))

print(f"√det(G) = {vol_from_gram:.6f}")
print(f"|det(A)| = {vol_from_det:.6f}")
print(f"일치: {np.isclose(vol_from_gram, vol_from_det)}")
```

### 10.3 선형종속 검출

```python
import numpy as np

# 선형종속 벡터들 (v3 = v1 + v2)
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
v3 = v1 + v2

A = np.column_stack([v1, v2, v3])
G = A.T @ A

print(f"det(G) = {np.linalg.det(G):.2e}")
print(f"G의 rank = {np.linalg.matrix_rank(G)}")
```

### 10.4 커널 행렬의 PSD성 (RBF 커널)

```python
import numpy as np

def rbf_kernel(X, sigma=1.0):
    n = X.shape[0]
    sq_dist = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T
    return np.exp(-sq_dist / (2 * sigma**2))

np.random.seed(0)
X = np.random.randn(20, 3)
K = rbf_kernel(X, sigma=1.0)

eigs = np.linalg.eigvalsh(K)
print(f"최소 고유값: {eigs.min():.2e}")
print(f"모두 비음수: {np.all(eigs >= -1e-10)}")
```

### 10.5 정규방정식 vs. QR via Gram

```python
import numpy as np

np.random.seed(0)
m, n = 100, 20

# Hilbert-like ill-conditioned matrix
A = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(m)])
b = np.random.randn(m)

# 방법 1: 정규방정식 (Gram 행렬 이용)
G = A.T @ A
x_normal = np.linalg.solve(G, A.T @ b)

# 방법 2: QR
Q, R = np.linalg.qr(A)
x_qr = np.linalg.solve(R, Q.T @ b)

# 방법 3: SVD (가장 안정)
x_svd = np.linalg.lstsq(A, b, rcond=None)[0]

print(f"κ(A)   = {np.linalg.cond(A):.2e}")
print(f"κ(A^TA) = {np.linalg.cond(G):.2e}")
print(f"‖x_normal - x_svd‖ = {np.linalg.norm(x_normal - x_svd):.2e}")
print(f"‖x_qr - x_svd‖     = {np.linalg.norm(x_qr - x_svd):.2e}")
```

---

## 11. 요약 및 다음 장으로

### 핵심 결과

| 주제 | 결과 |
|---|---|
| Gram 행렬 정의 | $G = A^T A$, $G_{ij} = \langle v_i, v_j \rangle$ |
| 기본 성질 | $G$ 대칭, 항상 PSD |
| PD 특징 | $G$ PD $\iff$ $\{v_i\}$ 선형독립 |
| 부피 공식 | $\text{Vol}_k^2 = \det G$ |
| Hadamard | $|\det A|^2 \le \prod \|a_i\|^2$ |
| 조건수 | $\kappa(G) = \kappa(A)^2$ |
| 커널 행렬 | $K_{ij} = K(x_i, x_j)$, PSD 필요 |

### 한 줄 요약

> **Gram 행렬은 내적 구조를 한 장의 PSD 행렬로 압축하며, 그 행렬식은 평행육면체의 부피의 제곱이다.**

### 다음 장 예고

Ch5의 마지막 절에서는 지금까지의 모든 내용을 **QR 분해를 내적 관점에서 재해석**하는 것으로 마무리한다. Gram-Schmidt는 정사영의 반복이고, Householder는 반사 대칭이며, 이 둘이 만나 Gram 행렬의 Cholesky 인수분해를 구성한다.

---

[◀ 03. 최소제곱법](./03-least-squares.md) | [📚 README](../README.md) | [05. QR 재해석 ▶](./05-qr-reinterpretation.md)
