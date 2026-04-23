# 6.3 Einstein 합 규약과 `einsum`

> "einsum은 모든 텐서 연산의 공통언어다."

---

## 1. 학습 목표

- **Einstein 합 규약**의 규칙과 암묵적 합의 원리를 정확히 이해한다.
- 행렬곱, 내적, 외적, 대각합, 전치 등 기본 연산을 Einstein 표기로 작성한다.
- `numpy.einsum`의 문법이 Einstein 규약의 직접적 구현임을 본다.
- 텐서 네트워크(**tensor network**)의 그래프 표기를 익힌다.
- 다양한 신경망 연산 (Attention, Convolution, Multi-head)을 einsum으로 통일적으로 표현한다.

---

## 2. Einstein 합 규약의 기원

### 2.1 문제의식

일반상대론을 쓰면서 아인슈타인은 "$\sum$ 기호가 너무 많아 성가시다"고 생각했다. 그래서:

> 반복되는 지표 하나는 위, 하나는 아래에 나타나면 합을 취한다.

예:

$$
a_i b^i \equiv \sum_i a_i b^i
$$

### 2.2 수학적 해석

위/아래 지표의 구분은 공변/반변을 구별한다 (6.1 절). 이 규칙은 텐서 계산을 **불변량(invariant)** 중심으로 조직한다.

### 2.3 단순화된 편의 규약

수치 계산이나 일반 선형대수에서는 지표 위치를 구별하지 않고

> 반복되는 지표는 합을 취한다.

로 단순화하기도 한다. NumPy의 `einsum`이 이를 따른다.

---

## 3. 기본 연산의 Einstein 표기

### 3.1 벡터 · 행렬 연산

| 연산 | 일반 표기 | Einstein 표기 |
|---|---|---|
| 내적 | $c = \sum_i a_i b_i$ | $c = a_i b_i$ |
| 외적 | $C_{ij} = a_i b_j$ | $C_{ij} = a_i b_j$ |
| 행렬-벡터 | $y_i = \sum_j A_{ij} x_j$ | $y_i = A_{ij} x_j$ |
| 행렬-행렬 | $C_{ij} = \sum_k A_{ik} B_{kj}$ | $C_{ij} = A_{ik} B_{kj}$ |
| 전치 | $B_{ij} = A_{ji}$ | $B_{ij} = A_{ji}$ |
| 대각합 | $\text{tr}(A) = \sum_i A_{ii}$ | $A_{ii}$ |

### 3.2 고차 텐서

| 연산 | Einstein |
|---|---|
| 3-텐서 축약 | $B_{ij} = T_{ijk} v_k$ |
| 이중 축약 | $C_{i} = T_{ijk} A_{jk}$ |
| 완전 축약 | $s = T_{ijk} S_{ijk}$ |
| Kronecker 곱 | $(A \otimes B)_{(ij),(kl)} = A_{ik} B_{jl}$ |

### 3.3 중요한 단순 표기

- **Kronecker 델타**: $\delta_{ij} = [i = j]$. 이것은 $I$의 성분.
- **Levi-Civita 기호**: $\epsilon_{ijk}$. $\mathbb{R}^3$의 외적 $\mathbf{a} \times \mathbf{b}$의 $i$-성분:
  $$
  (\mathbf{a} \times \mathbf{b})_i = \epsilon_{ijk} a_j b_k
  $$
- **행렬식** ($n \times n$): $\det A = \epsilon_{i_1 \cdots i_n} A_{1 i_1} A_{2 i_2} \cdots A_{n i_n}$

---

## 4. `np.einsum` 문법

### 4.1 기본 규칙

```python
np.einsum("입력spec -> 출력spec", 텐서1, 텐서2, ...)
```

- 반복되는 지표 중 **출력 spec에 없는 것**은 합을 취한다.
- 출력 spec에 있으면 **그대로 유지**.

### 4.2 기본 예시

```python
import numpy as np

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# 행렬곱
C = np.einsum("ik,kj->ij", A, B)

# 원소별 곱 (Hadamard)
X = np.random.randn(3, 3)
Y = np.random.randn(3, 3)
Z = np.einsum("ij,ij->ij", X, Y)

# 내적
u = np.random.randn(5)
v = np.random.randn(5)
s = np.einsum("i,i->", u, v)

# 외적
M = np.einsum("i,j->ij", u, v)

# 대각합
T = np.random.randn(4, 4)
t = np.einsum("ii->", T)  # tr(T)

# 전치
AT = np.einsum("ij->ji", A)
```

### 4.3 축약 여러 개

```python
# 3-텐서와 벡터 축약
T = np.random.randn(3, 4, 5)
v = np.random.randn(5)
M = np.einsum("ijk,k->ij", T, v)  # 모드-3 축소

# 배치 행렬곱
A = np.random.randn(10, 3, 4)
B = np.random.randn(10, 4, 5)
C = np.einsum("bij,bjk->bik", A, B)  # batched matmul

# 이중 축약
T = np.random.randn(3, 4, 5)
A = np.random.randn(4, 5)
B = np.einsum("ijk,jk->i", T, A)
```

### 4.4 출력 순서 지정

```python
# 축의 순서 바꾸기
T = np.random.randn(2, 3, 4, 5)
T_perm = np.einsum("ijkl->jilk", T)  # (3, 2, 5, 4)
```

### 4.5 생략 기호 `...`

배치 차원을 잘 모를 때:

```python
# 마지막 두 축만 행렬곱, 나머지는 브로드캐스트
A = np.random.randn(7, 3, 4)
B = np.random.randn(7, 4, 5)
C = np.einsum("...ij,...jk->...ik", A, B)
```

---

## 5. 텐서 네트워크 다이어그램

### 5.1 그래프 표기법

텐서를 **노드**로, 각 지표를 **간선**으로 표기:

- **벡터** $v_i$: 하나의 간선을 가진 점.
- **행렬** $A_{ij}$: 두 간선을 가진 점.
- **3-텐서** $T_{ijk}$: 세 간선.
- **축약**: 공유 지표는 두 노드를 **잇는 선**.
- **출력 지표**: 밖으로 나가는 자유 간선.

### 5.2 예시

행렬곱 $C_{ij} = A_{ik} B_{kj}$:

```
    i          j
    |          |
  [ A ]──k──[ B ]
```

두 간선(자유)이 외부로 나가고, 하나(k)는 내부로 축약됨.

### 5.3 계산 순서의 최적화

$$
(A B) C = A (B C)
$$

는 수학적으로 같지만 **계산량이 다를 수 있다**.

예: $A \in \mathbb{R}^{1000 \times 10}$, $B \in \mathbb{R}^{10 \times 1000}$, $C \in \mathbb{R}^{1000 \times 5}$.

- $(AB)C$: $AB$는 $1000 \times 1000$ (계산 $10^7$), 그 다음 $C$ 곱 $5 \times 10^6$. 총 $1.5 \times 10^7$.
- $A(BC)$: $BC$는 $10 \times 5$ (계산 $5 \times 10^4$), 그 다음 $A$ 곱 $5 \times 10^4$. 총 $10^5$.

$100$배 이상 차이! `einsum(optimize=True)`는 이러한 최적 순서를 자동으로 선택한다.

---

## 6. 응용: Attention 메커니즘

### 6.1 Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

### 6.2 Einsum으로

```python
# Q, K, V: (batch, seq, d_k)
# 점수: S[b, i, j] = Σ_d Q[b, i, d] * K[b, j, d]
S = np.einsum("bid,bjd->bij", Q, K) / np.sqrt(d_k)
A = softmax(S, axis=-1)
# 결과: O[b, i, d] = Σ_j A[b, i, j] * V[b, j, d]
O = np.einsum("bij,bjd->bid", A, V)
```

### 6.3 Multi-Head Attention

$h$개 헤드 버전:

```python
# Q, K, V: (batch, seq, heads, d_k)
S = np.einsum("bihd,bjhd->bhij", Q, K) / np.sqrt(d_k)
A = softmax(S, axis=-1)
O = np.einsum("bhij,bjhd->bihd", A, V)
```

한 줄에 헤드 분리와 행렬곱이 담겨있다.

---

## 7. 응용: Convolution

### 7.1 2D 컨볼루션

$$
O[b, c_o, y, x] = \sum_{c_i, k_y, k_x} I[b, c_i, y + k_y, x + k_x] \cdot W[c_o, c_i, k_y, k_x]
$$

(stride = 1, no padding).

### 7.2 Einsum으로 (unfold 이후)

컨볼루션을 **im2col** 연산으로 변환하면

- $I$를 $(B, C_i, K_y, K_x, H_o, W_o)$로 풀어 `I_unf`
- $W$는 $(C_o, C_i, K_y, K_x)$

그러면

```python
O = np.einsum("bcyxhw,ocyx->bohw", I_unf, W)
```

한 줄로 모든 배치, 모든 채널, 모든 공간 위치의 합이 표현된다.

---

## 8. 응용: 물리와 양자 정보

### 8.1 텐서 네트워크 상태 (MPS, PEPS)

양자 다체계 상태

$$
|\psi\rangle = \sum_{i_1 \cdots i_N} c_{i_1 \cdots i_N} |i_1 \cdots i_N\rangle
$$

의 계수 텐서 $c$를 작은 텐서들의 곱으로 분해. **Matrix Product State**:

$$
c_{i_1 i_2 \cdots i_N} = \sum_{\alpha_1 \cdots \alpha_{N-1}} A^{[1]}_{i_1, \alpha_1} A^{[2]}_{\alpha_1, i_2, \alpha_2} \cdots A^{[N]}_{\alpha_{N-1}, i_N}
$$

Einstein 표기: $c = A^{[1]} A^{[2]} \cdots A^{[N]}$ (공유 지표 축약).

### 8.2 기대값 계산

$\langle \psi | O | \psi \rangle$는 텐서 네트워크의 **수축(contraction)**. 순서에 따라 계산 복잡도가 기하급수적으로 달라진다.

---

## 9. 응용: NN 학습에서의 einsum

### 9.1 Bilinear Layer

$y_i = \sum_{jk} W_{ijk} x_j z_k$ → `np.einsum("ijk,j,k->i", W, x, z)`.

### 9.2 Batched Gradient Computation

Forward: $Y = XW$, $X \in \mathbb{R}^{B \times D_{in}}$.

Gradient: $\partial L / \partial W = X^T \partial L / \partial Y$.

Einsum:

```python
grad_W = np.einsum("bi,bj->ij", X, grad_Y)
```

### 9.3 Outer Product in Attention 점수 행렬

$QK^T$ 대신 einsum을 쓰면 batch와 head 차원을 명시적으로 다룰 수 있어 디버깅이 쉽다.

---

## 10. Python 실험

### 10.1 기본 연산 대조

```python
import numpy as np

np.random.seed(0)
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# 행렬곱 3가지 방법
C1 = A @ B
C2 = np.matmul(A, B)
C3 = np.einsum("ik,kj->ij", A, B)

print(f"max diff: {np.max(np.abs(C1 - C3)):.2e}")
```

### 10.2 Batched matmul

```python
import numpy as np

A = np.random.randn(32, 10, 20)
B = np.random.randn(32, 20, 30)

C_einsum = np.einsum("bij,bjk->bik", A, B)
C_matmul = np.matmul(A, B)

print(f"일치: {np.allclose(C_einsum, C_matmul)}")
```

### 10.3 Levi-Civita로 외적 구현

```python
import numpy as np

def levi_civita(n=3):
    eps = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                perm = (i, j, k)
                if len(set(perm)) < n:
                    continue
                sign = 1
                sorted_perm = list(perm)
                for a in range(n):
                    for b in range(a+1, n):
                        if sorted_perm[a] > sorted_perm[b]:
                            sorted_perm[a], sorted_perm[b] = sorted_perm[b], sorted_perm[a]
                            sign *= -1
                eps[i, j, k] = sign
    return eps

eps = levi_civita(3)
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

cross_einsum = np.einsum("ijk,j,k->i", eps, a, b)
cross_np = np.cross(a, b)

print(f"einsum 외적: {cross_einsum}")
print(f"NumPy 외적:  {cross_np}")
```

### 10.4 Attention 구현

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

np.random.seed(0)
B, S, D = 2, 5, 8
Q = np.random.randn(B, S, D)
K = np.random.randn(B, S, D)
V = np.random.randn(B, S, D)

scores = np.einsum("bid,bjd->bij", Q, K) / np.sqrt(D)
A = softmax(scores, axis=-1)
O = np.einsum("bij,bjd->bid", A, V)

print(f"Attention 출력 shape: {O.shape}")
print(f"확률 합이 1인지: {np.allclose(A.sum(axis=-1), 1)}")
```

### 10.5 경로 최적화 효과

```python
import numpy as np
import time

np.random.seed(0)
A = np.random.randn(100, 10)
B = np.random.randn(10, 100)
C = np.random.randn(100, 5)

# optimize=False: 왼쪽부터
t0 = time.time()
for _ in range(100):
    _ = np.einsum("ij,jk,kl->il", A, B, C, optimize=False)
t_noopt = time.time() - t0

# optimize=True: 자동 최적
t0 = time.time()
for _ in range(100):
    _ = np.einsum("ij,jk,kl->il", A, B, C, optimize=True)
t_opt = time.time() - t0

print(f"no opt: {t_noopt*1000:.2f} ms")
print(f"opt:    {t_opt*1000:.2f} ms")
print(f"가속:    {t_noopt / t_opt:.2f}x")
```

### 10.6 텐서 네트워크 수축 예제

```python
import numpy as np

# 1D 이징 모형의 전달 행렬과 비슷한 형태
# chain: T1[i1, a1] T2[a1, i2, a2] T3[a2, i3, a3] ... T_N[a_{N-1}, i_N]
L = 5
d = 2
D = 3  # bond dimension

np.random.seed(0)
Ts = [np.random.randn(d, D)]
for _ in range(L-2):
    Ts.append(np.random.randn(D, d, D))
Ts.append(np.random.randn(D, d))

# 수축: 모든 스핀 지표 자유, 결합 지표 축약
result = Ts[0]  # (d, D)
for t in Ts[1:-1]:
    result = np.einsum("...a,abc->...bc", result, t)  # 끝 축과 축약
# 마지막
result = np.einsum("...a,ab->...b", result, Ts[-1])

print(f"MPS 전체 텐서 shape: {result.shape}")
print(f"요소 개수: {result.size} (= {d}^{L} = {d**L})")
```

---

## 11. 요약 및 다음 절 예고

### 핵심 규칙

| 규칙 | 내용 |
|---|---|
| 반복 지표 = 합 | $a_i b_i = \sum_i a_i b_i$ |
| 출력 spec 없는 지표 | 축약 (합) |
| 출력 spec 있는 지표 | 유지 |
| `...` | 나머지 축 브로드캐스트 |
| `optimize=True` | 자동 경로 최적 |

### 주요 공식

| 연산 | einsum |
|---|---|
| 행렬곱 | `"ik,kj->ij"` |
| batched matmul | `"bij,bjk->bik"` |
| Hadamard | `"ij,ij->ij"` |
| 대각합 | `"ii->"` |
| 외적 | `"i,j->ij"` |
| Attention | `"bid,bjd->bij"` |

### 한 줄 요약

> **Einstein 합 규약은 수식과 코드 사이의 다리이며, einsum은 모든 텐서 연산을 한 줄로 쓰게 해준다.**

### 다음 절 예고

다음 절에서는 **텐서 분해** (CP, Tucker, Tensor Train)를 다룬다. 고차 텐서에서 SVD의 일반화에 해당하며, 딥러닝 압축과 추천 시스템에 널리 쓰인다.

---

[◀ 02. Kronecker 곱](./02-kronecker-product.md) | [📚 README](../README.md) | [04. 텐서 분해 ▶](./04-tensor-decomposition.md)
