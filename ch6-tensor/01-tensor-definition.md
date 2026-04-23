# 6.1 텐서의 정의: 다중선형 사상의 관점

> "텐서는 '다차원 배열'이 아니라, 좌표계와 무관하게 정의되는 다중선형 사상이다."

---

## 1. 학습 목표

- **텐서**를 다중선형 사상(multilinear map)으로 엄밀히 정의한다.
- 벡터, 쌍대벡터(covector), 행렬, 이중선형형식이 텐서의 특수 경우임을 본다.
- **공변(covariant), 반변(contravariant)** 지표의 의미와 변환 법칙을 유도한다.
- 텐서곱 공간 $V \otimes W$의 구성(보편성, 기저)을 제시한다.
- 실제 계산에 쓰이는 **다차원 배열 표현**과 추상 텐서의 관계를 명확히 한다.

---

## 2. 동기: 왜 텐서인가?

### 2.1 단일 선형 사상의 한계

선형대수에서 우리는 $f: V \to W$ 형태의 **일변수** 선형 사상을 공부했다. 행렬은 이런 선형 사상을 유한 차원에서 표현한 것이다.

그러나 많은 문제에서 **여러 벡터를 동시에 입력받는 선형성**이 필요하다:

- **이중선형형식** $B: V \times V \to \mathbb{R}$: 내적, 이차형식, 편미분 행렬
- **결정자** $\det: \underbrace{V \times \cdots \times V}_{n\text{번}} \to \mathbb{R}$: $n$-선형 교대형식
- **벡터곱(cross product)** $V \times V \to V$
- **곡률텐서** (일반상대론), **관성 텐서** (고전역학)
- **신경망 가중치** (여러 축에 대해 동시에 합을 취하는 연산)

이들은 모두 "**여러 입력 슬롯에 대해 각각 선형**"이라는 공통점이 있다.

### 2.2 "다차원 배열"이 아니다

프로그래밍에서는 텐서를 `np.ndarray` 같은 다차원 배열로 생각하지만, 수학적 정의는 **좌표계에 독립적**이다. 배열은 특정 기저에서의 성분 표현일 뿐이다.

**핵심 인사이트**: 좌표변환 하에서 성분이 **정해진 규칙으로 변환**되는 대상이 텐서다.

---

## 3. 다중선형 사상으로서의 텐서

### 3.1 이중선형 사상부터

$V, W, U$가 벡터공간일 때, 함수 $B: V \times W \to U$가 **이중선형(bilinear)** 이라는 것은

$$
B(\alpha v_1 + \beta v_2, w) = \alpha B(v_1, w) + \beta B(v_2, w)
$$
$$
B(v, \alpha w_1 + \beta w_2) = \alpha B(v, w_1) + \beta B(v, w_2)
$$

를 의미한다. 즉 **각 변수에 대해 개별적으로 선형**.

### 3.2 다중선형 사상

일반적으로 $k$-선형(multilinear) 사상 $T: V_1 \times V_2 \times \cdots \times V_k \to U$는 각 변수에 대해 선형인 함수이다.

### 3.3 텐서의 정의 1: 쌍대공간과 섞인 형

**정의 6.1.1 (텐서).** $V$의 쌍대공간을 $V^* = \{\phi: V \to \mathbb{R}\ |\ \phi\ \text{선형}\}$라 하자. **$(p, q)$-형 텐서**는 다음 다중선형 사상이다:

$$
T: \underbrace{V^* \times \cdots \times V^*}_{p\text{번}} \times \underbrace{V \times \cdots \times V}_{q\text{번}} \to \mathbb{R}
$$

- $p$: **반변(contravariant)** 차수
- $q$: **공변(covariant)** 차수
- $p + q$: **랭크(rank)** 또는 차수(order)

### 3.4 특수 경우

| 타입 | $(p, q)$ | 정체 |
|---|---|---|
| 스칼라 | $(0, 0)$ | $\mathbb{R}$ |
| 벡터 | $(1, 0)$ | $V^{**} \simeq V$ |
| 쌍대벡터 | $(0, 1)$ | $V^*$ |
| 선형연산자 | $(1, 1)$ | $V \to V$ (행렬) |
| 이중선형형식 | $(0, 2)$ | $V \times V \to \mathbb{R}$ |
| 이중선형연산자 | $(1, 2)$ | $V \times V \to V$ |
| 내적 | $(0, 2)$ | 대칭 PSD 이중선형형식 |

### 3.5 왜 $V^{**} \simeq V$인가?

**자연 동형사상.** $v \in V$에 대해 $\hat v: V^* \to \mathbb{R}$을 $\hat v(\phi) = \phi(v)$로 정의. 이는 $v \mapsto \hat v$ 형태의 $V \to V^{**}$ 선형사상이며, 유한차원에서 **단사·전사**이다.

- 단사성: $\hat v = 0$이면 모든 $\phi$에 대해 $\phi(v) = 0$, 특히 기저 쌍대에 대해 $v = 0$.
- 전사성: 차원의 동일성 $\dim V = \dim V^* = \dim V^{**}$로부터.

따라서 "벡터 = (1,0)-텐서"는 엄밀히 의미를 가진다.

---

## 4. 텐서곱 공간

### 4.1 구성의 동기

두 벡터공간의 텐서곱 $V \otimes W$는 "이중선형 사상 $V \times W \to U$가 **선형 사상** $V \otimes W \to U$로 환원**되도록** 만든 공간이다.

### 4.2 보편성(Universal Property)

**정의 6.1.2 (텐서곱의 보편성).** $V \otimes W$와 이중선형 사상 $\otimes: V \times W \to V \otimes W$가 다음을 만족한다:

> 임의의 벡터공간 $U$와 이중선형 사상 $B: V \times W \to U$에 대해, 유일한 선형 사상 $\tilde B: V \otimes W \to U$가 존재하여 $B(v, w) = \tilde B(v \otimes w)$.

$$
\begin{array}{ccc}
V \times W & \xrightarrow{\otimes} & V \otimes W \\
& \searrow^{B} & \downarrow^{\tilde B} \\
& & U
\end{array}
$$

### 4.3 유한차원 구성

$V$의 기저 $\{e_i\}_{i=1}^n$, $W$의 기저 $\{f_j\}_{j=1}^m$가 있으면 $V \otimes W$의 기저는

$$
\{e_i \otimes f_j\ :\ 1 \le i \le n,\ 1 \le j \le m\}
$$

따라서 $\dim(V \otimes W) = nm$.

임의의 원소는

$$
T = \sum_{i, j} T^{ij}\, e_i \otimes f_j
$$

$T^{ij}$가 **성분**이다. 텐서곱의 원소 중 일부는 $v \otimes w$ 형태로 쓰이지만, 대부분은 그렇지 않다 (이를 **비분해(entangled)** 상태라 한다).

### 4.4 일반 텐서곱 공간

$(p, q)$-형 텐서는

$$
T \in \underbrace{V \otimes \cdots \otimes V}_{p} \otimes \underbrace{V^* \otimes \cdots \otimes V^*}_{q}
$$

차원 $n = \dim V$이면 $\dim = n^{p+q}$.

---

## 5. 성분 표현과 지표 표기법

### 5.1 기저 선택

$V$의 기저 $\{e_i\}$, 쌍대기저 $\{e^i\}$ ($e^i(e_j) = \delta^i_j$). 그러면 $(p, q)$-텐서 $T$의 성분은

$$
T^{i_1 \cdots i_p}_{j_1 \cdots j_q} = T(e^{i_1}, \ldots, e^{i_p}; e_{j_1}, \ldots, e_{j_q})
$$

### 5.2 Einstein 합 규약

반복되는 지표 하나는 위(반변), 하나는 아래(공변)에 나타나면 **암묵적으로 합**을 취한다.

$$
v = v^i e_i \quad \text{은}\quad v = \sum_{i=1}^n v^i e_i
$$

$$
T^i_j v^j \quad \text{은}\quad \sum_j T^i_j v^j = (Tv)^i
$$

이 규약으로 지표가 수없이 반복되는 식을 축약한다 (제6.3절에서 자세히).

### 5.3 좌표변환 법칙

기저를 $\tilde e_i = S^j_i e_j$로 바꾸면 ($S$: 가역):

- **쌍대기저**: $\tilde e^i = (S^{-1})^i_j e^j$
- **벡터 성분**: $\tilde v^i = (S^{-1})^i_j v^j$ ← 기저와 **반대** 방향 변환
- **쌍대벡터 성분**: $\tilde \phi_i = \phi_j S^j_i$ ← 기저와 **같은** 방향 변환

이것이 "반변(contra-)"와 "공변(co-)"의 어원이다:

- **반변(contravariant) 지표**: 기저와 반대로 변환 $\to$ 위첨자 $v^i$
- **공변(covariant) 지표**: 기저와 같이 변환 $\to$ 아래첨자 $\phi_i$

### 5.4 일반 텐서의 변환

$(p, q)$-텐서의 성분은

$$
\tilde T^{i_1 \cdots i_p}_{j_1 \cdots j_q} = (S^{-1})^{i_1}_{k_1} \cdots (S^{-1})^{i_p}_{k_p}\, S^{l_1}_{j_1} \cdots S^{l_q}_{j_q}\, T^{k_1 \cdots k_p}_{l_1 \cdots l_q}
$$

이 **변환 법칙이 텐서의 본질**이다 (고전적 "텐서의 정의").

---

## 6. 기본 연산

### 6.1 덧셈과 스칼라곱

같은 $(p, q)$-형 텐서끼리 성분별 덧셈, 스칼라곱.

### 6.2 텐서곱

$(p_1, q_1)$-텐서 $T$와 $(p_2, q_2)$-텐서 $S$의 텐서곱 $T \otimes S$는 $(p_1 + p_2, q_1 + q_2)$-텐서:

$$
(T \otimes S)^{i_1 \cdots i_{p_1}\ k_1 \cdots k_{p_2}}_{j_1 \cdots j_{q_1}\ l_1 \cdots l_{q_2}} = T^{i_1 \cdots i_{p_1}}_{j_1 \cdots j_{q_1}} S^{k_1 \cdots k_{p_2}}_{l_1 \cdots l_{q_2}}
$$

### 6.3 축약 (Contraction)

$(p, q)$-텐서의 반변 지표 하나와 공변 지표 하나를 맞추어 합을 취하면 $(p-1, q-1)$-텐서가 된다.

$$
(C T)^{i_1 \cdots \hat i_k \cdots i_p}_{j_1 \cdots \hat j_l \cdots j_q} = \sum_{\alpha} T^{i_1 \cdots \alpha \cdots i_p}_{j_1 \cdots \alpha \cdots j_q}
$$

대표 예: **행렬의 대각합(trace)** 은 $(1,1)$-텐서의 축약 $\text{tr}(A) = A^i_i$ (Einstein 규약).

### 6.4 지표 올리기/내리기 (Metric)

내적 $g_{ij} = \langle e_i, e_j \rangle$(메트릭)이 있으면 지표를 자유롭게 올리고 내릴 수 있다:

$$
v_i = g_{ij} v^j, \quad v^i = g^{ij} v_j \quad (g^{ij}: g_{ij}\ \text{의 역행렬})
$$

유클리드 공간에서 $g_{ij} = \delta_{ij}$이므로 구분이 사라진다. 하지만 일반 (위상)공간에서는 매우 중요하다.

---

## 7. 대칭과 반대칭

### 7.1 정의

$(0, k)$-텐서 $T$가 **대칭(symmetric)** 이라는 것은 임의의 치환 $\sigma$에 대해

$$
T(v_{\sigma(1)}, \ldots, v_{\sigma(k)}) = T(v_1, \ldots, v_k)
$$

**반대칭(alternating/skew-symmetric)** 은

$$
T(v_{\sigma(1)}, \ldots, v_{\sigma(k)}) = \text{sign}(\sigma) T(v_1, \ldots, v_k)
$$

### 7.2 대표 예

- **이차형식** $q(v) = B(v, v)$: 대칭 (0,2)-텐서로부터
- **결정자**: 반대칭 $(0, n)$-텐서
- **외적 $v \times w$** ($\mathbb{R}^3$): 반대칭 $(1, 2)$-텐서 $\epsilon^i_{jk}$로부터

### 7.3 대칭화와 반대칭화 연산자

임의의 텐서를 대칭/반대칭 부분으로 분해할 수 있다:

$$
T_{\text{sym}} = \frac{1}{k!}\sum_{\sigma} T_\sigma, \quad T_{\text{alt}} = \frac{1}{k!} \sum_\sigma \text{sign}(\sigma) T_\sigma
$$

---

## 8. 예시: 관성 텐서와 스트레스 텐서

### 8.1 관성 텐서 (Inertia Tensor)

강체의 각운동량 $L^i$와 각속도 $\omega^j$의 관계

$$
L^i = I^i_j \omega^j
$$

$I$는 $(1, 1)$-텐서 (선형연산자). 메트릭을 가진 $\mathbb{R}^3$에서는 $(0, 2)$ 또는 $(2, 0)$으로 변환 가능.

$$
I_{ij} = \int_V \rho(x) (\delta_{ij} \|x\|^2 - x_i x_j)\, dV
$$

대각화하면 **주관성모멘트**.

### 8.2 코시 응력 텐서 (Cauchy Stress Tensor)

연속체의 한 점에서 단위법선 $n$에 작용하는 응력 $t$:

$$
t^i = \sigma^{ij} n_j
$$

$\sigma$는 대칭 $(2, 0)$-텐서. 주응력은 $\sigma$의 고유값.

### 8.3 곡률 텐서 (Riemann Curvature)

$(1, 3)$-텐서 $R^i_{jkl}$로, 일반상대론의 아인슈타인 방정식의 핵심.

---

## 9. 텐서의 유한차원 표현: 다차원 배열

### 9.1 프로그래밍에서의 텐서

NumPy의 `ndarray`, PyTorch/TensorFlow의 `Tensor`는 $n_1 \times n_2 \times \cdots \times n_k$ 크기의 배열. 수학적 텐서의 **성분 표현**에 해당.

### 9.2 모드(mode)와 축(axis)

- **모드 1**: 첫 번째 지표. `np.ndarray`에서 `axis=0`.
- **행렬(2D)**: 행 = 모드 1, 열 = 모드 2.
- **3D 텐서**: (frontal slice, lateral slice, horizontal slice)

### 9.3 Reshape와 전개(Unfolding)

3D 텐서 $T \in \mathbb{R}^{I \times J \times K}$를 2D로 풀어내는 방법:

- **mode-1 unfolding**: $T_{(1)} \in \mathbb{R}^{I \times JK}$
- **mode-2 unfolding**: $T_{(2)} \in \mathbb{R}^{J \times IK}$
- **mode-3 unfolding**: $T_{(3)} \in \mathbb{R}^{K \times IJ}$

이는 텐서 분해 알고리즘에서 핵심 (6.4절 참고).

---

## 10. Python 실험

### 10.1 기본 텐서 연산

```python
import numpy as np

# 3차 텐서
T = np.random.randn(3, 4, 5)

print(f"Shape: {T.shape}")
print(f"ndim:  {T.ndim}")
print(f"size:  {T.size}")

# 성분 접근
print(f"T[1, 2, 3] = {T[1, 2, 3]:.4f}")

# 축약 (contraction): 두 번째 축을 따라 합
contracted = T.sum(axis=1)
print(f"After contraction: {contracted.shape}")
```

### 10.2 텐서곱 (outer product)

```python
import numpy as np

u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0])
w = np.array([6.0, 7.0, 8.0, 9.0])

# u ⊗ v ⊗ w: 3D rank-1 텐서
T = np.einsum('i,j,k->ijk', u, v, w)
print(f"Shape: {T.shape}")       # (3, 2, 4)
print(f"T[1, 0, 2] = {T[1, 0, 2]}")
print(f"u[1]*v[0]*w[2] = {u[1]*v[0]*w[2]}")
```

### 10.3 Mode unfolding

```python
import numpy as np

def unfold(T, mode):
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)

T = np.random.randn(3, 4, 5)

T1 = unfold(T, 0)  # 3 x 20
T2 = unfold(T, 1)  # 4 x 15
T3 = unfold(T, 2)  # 5 x 12

print(f"Mode-1 unfolding: {T1.shape}")
print(f"Mode-2 unfolding: {T2.shape}")
print(f"Mode-3 unfolding: {T3.shape}")
```

### 10.4 좌표변환 법칙 검증

```python
import numpy as np

np.random.seed(0)

# 3차원 벡터와 (0, 2) 이중선형형식 (대칭)
v = np.random.randn(3)
B = np.random.randn(3, 3)
B = (B + B.T) / 2

# 기저 변환
S = np.random.randn(3, 3)
S_inv = np.linalg.inv(S)

# 변환된 성분
# 벡터 (반변): v_tilde = S^{-1} v
v_tilde = S_inv @ v

# (0, 2) 텐서 (공변): B_tilde_{ij} = S^k_i S^l_j B_{kl} = S^T B S
B_tilde = S.T @ B @ S

# 불변량: v^T B v = (S^{-1} v)^T B_tilde (S^{-1} v)
invariant_before = v @ B @ v
invariant_after  = v_tilde @ B_tilde @ v_tilde

print(f"원래:   v^T B v = {invariant_before:.6f}")
print(f"변환후: v'^T B' v' = {invariant_after:.6f}")
print(f"일치: {np.isclose(invariant_before, invariant_after)}")
```

### 10.5 대칭·반대칭 분해

```python
import numpy as np

# 임의의 2-텐서
T = np.random.randn(4, 4)

T_sym = (T + T.T) / 2
T_alt = (T - T.T) / 2

print(f"대칭성: {np.allclose(T_sym, T_sym.T)}")
print(f"반대칭성: {np.allclose(T_alt, -T_alt.T)}")
print(f"복원: {np.allclose(T_sym + T_alt, T)}")
```

---

## 11. 요약 및 다음 절 예고

### 핵심 결과

| 개념 | 정의/공식 |
|---|---|
| $(p, q)$-텐서 | $\underbrace{V^* \cdots}_{p} \times \underbrace{V \cdots}_{q} \to \mathbb{R}$, 다중선형 |
| 보편성 | $B: V \times W \to U$ $\to$ 유일한 $\tilde B: V \otimes W \to U$ |
| 차원 | $\dim(V \otimes W) = \dim V \cdot \dim W$ |
| 변환 법칙 | 반변 ↔ $S^{-1}$, 공변 ↔ $S$ |
| Einstein 규약 | 반복 지표 = 합 |
| 축약 | $(p, q) \to (p-1, q-1)$ |

### 한 줄 요약

> **텐서란 좌표계 변환 하에서 정해진 규칙에 따라 변환되는 다중선형 대상이다.**

### 다음 절 예고

다음 절에서는 텐서곱의 **행렬 표현**인 Kronecker 곱을 다룬다. 이는 텐서의 대수를 행렬 연산으로 환원하며, 양자역학, 그래프 이론, 수치 선형대수 전반에 등장한다.

---

[◀ Ch5 마무리](../ch5-inner-product/05-qr-reinterpretation.md) | [📚 README](../README.md) | [02. Kronecker 곱 ▶](./02-kronecker-product.md)
