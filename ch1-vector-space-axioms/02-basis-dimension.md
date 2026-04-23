# 02. 선형독립, 기저, 차원

## 🎯 핵심 질문

- 선형독립은 "그림으로 평행이 아니다"라는 직관을 어떻게 **공리 수준에서** 표현하는가?
- 벡터공간의 **차원은 왜 유일**한가? — 한 기저가 3개이면 다른 기저도 반드시 3개인가?
- 유한차원과 무한차원의 **근본적 차이**는 무엇인가?
- 신경망의 "유효 차원"·PCA의 "본질적 차원"·과매개변수화의 수학적 기반은 이 개념과 어떻게 연결되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Feature의 차원과 ranked**: 데이터 행렬 $X \in \mathbb{R}^{n \times d}$의 **rank**는 실제로 선형독립인 feature의 개수다. 과적합을 피하려면 유효 차원을 추정해야 하고, 이는 **기저를 어떻게 고르느냐**의 문제다.
- **Word embedding의 차원 선택**: Word2Vec·GloVe의 embedding 차원(보통 100~1000)은 "표현력"과 "연산 비용"의 trade-off. 차원의 **유일성 정리**가 있기에 "300차원 공간"이라는 말이 모호하지 않다.
- **Latent variable의 intrinsic dimension**: VAE의 잠재 차원을 정하는 기준 — 데이터가 실제로 놓이는 저차원 부분다양체의 **접공간 차원**(Ch1-02, Information Geometry 레포)이 바로 선형대수의 차원이다.
- **과매개변수화의 선형대수**: 신경망 파라미터가 $N \gg n$(샘플 수)일 때, 손실 함수의 **Hessian이 rank-deficient**하다는 현상(Sagun et al. 2016)은 기저 개수와 차원의 관계로 설명된다.
- **PCA의 압축률**: 상위 $k$개 주성분으로 축소할 때, $k$는 "유효 차원"이고 이는 공분산 행렬의 **고유벡터 기저**의 부분집합이다(Ch4).

"300차원 embedding을 사용한다"고 말하는 순간, **차원이 유일하다는 사실**을 이미 전제하고 있다.

---

## 📐 수학적 선행 조건

- [Ch1-01 벡터공간의 8개 공리](./01-vector-space-8-axioms.md)
- 집합·함수: 단사·전사, 유한집합의 기수
- 수학적 귀납법

---

## 📖 직관적 이해

### 선형독립 = "서로 중복이 없다"

$\mathbb{R}^3$에서 세 벡터 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$이 **선형독립**이라는 것은, "어느 한 벡터도 나머지 둘의 조합으로 만들어지지 않는다"는 뜻이다. 즉 세 방향이 **서로 겹치지 않게 공간을 펼친다**.

- 두 벡터가 평행: $\mathbf{v}_2 = \alpha \mathbf{v}_1$ → 선형종속
- 세 벡터가 한 평면에: $\mathbf{v}_3 = \alpha\mathbf{v}_1 + \beta\mathbf{v}_2$ → 선형종속
- 세 벡터가 공간을 꽉 채움: 선형독립

### 기저 = "가장 효율적인 표현"

벡터공간 $V$의 모든 원소를 표현할 수 있으면서(생성), **중복이 없는**(선형독립) 최소 집합이 **기저(basis)**다. 기저가 있으면 모든 원소가 **고유한 좌표** $(\alpha_1, \ldots, \alpha_n)$으로 표현된다.

### 차원의 유일성이 자명하지 않은 이유

"한 기저의 크기가 3인데 다른 기저가 5이면 모순"이라는 사실은 **증명이 필요**하다. 왜냐하면 기저를 고르는 방법은 무수히 많고(무한히 많은 선택), 그 모두가 **같은 크기**라는 것은 놀라운 사실이기 때문이다. 이 증명의 핵심이 **Steinitz Exchange Lemma**다.

> **비유**: 서로 다른 언어로 같은 생각을 표현할 수 있지만, "필요한 최소 단어 수"는 언어에 무관하다 — 라는 명제의 선형대수 판이다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 선형결합 (Linear Combination)

$V$가 $\mathbb{F}$-벡터공간, $\mathbf{v}_1, \ldots, \mathbf{v}_k \in V$일 때, **$\mathbf{v}_1, \ldots, \mathbf{v}_k$의 선형결합**이란 체 원소 $\alpha_1, \ldots, \alpha_k \in \mathbb{F}$에 대한

$$\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k$$

이다.

### 정의 2.2 — 생성 (Span)

부분집합 $S \subseteq V$의 **생성** $\text{span}(S)$은 $S$의 원소들의 모든 선형결합의 집합:

$$\text{span}(S) = \left\{ \sum_{i=1}^k \alpha_i \mathbf{v}_i : k \in \mathbb{N}, \alpha_i \in \mathbb{F}, \mathbf{v}_i \in S \right\}.$$

$\text{span}(S)$는 항상 $V$의 **부분공간**이다 (공리 확인 연습).

### 정의 2.3 — 선형독립 (Linear Independence)

$\mathbf{v}_1, \ldots, \mathbf{v}_k \in V$가 **선형독립**이라는 것은,

$$\alpha_1 \mathbf{v}_1 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0} \implies \alpha_1 = \cdots = \alpha_k = 0.$$

즉 "오직 자명한 조합만이 영벡터를 낳는다." 그렇지 않으면 **선형종속**이라 한다.

### 정의 2.4 — 기저 (Basis)

$V$의 부분집합 $\mathcal{B} = \{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$이 **기저**라는 것은 다음을 만족:

1. $\mathcal{B}$가 $V$를 생성: $\text{span}(\mathcal{B}) = V$
2. $\mathcal{B}$가 선형독립

### 정의 2.5 — 차원 (Dimension)

벡터공간 $V$가 유한개의 원소로 이루어진 기저를 가지면 **유한차원**이고, 그 기저의 크기를 $V$의 **차원** $\dim V$라 한다. 유한 기저가 없으면 **무한차원**.

> 이 정의가 성립하려면 "기저의 크기가 선택에 무관함"을 **증명**해야 한다. 그것이 정리 2.3의 결론이다.

---

## 🔬 정리와 증명

### 정리 2.1 — 좌표의 유일성

**명제**: $\mathcal{B} = \{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$이 $V$의 기저이면, 각 $\mathbf{v} \in V$는 $\mathcal{B}$에 대한 **유일한 좌표 표현**을 가진다:

$$\mathbf{v} = \sum_{i=1}^n \alpha_i \mathbf{e}_i, \quad (\alpha_1, \ldots, \alpha_n) \in \mathbb{F}^n.$$

**증명**: 존재성은 생성 조건에서 바로. 유일성은 $\sum \alpha_i \mathbf{e}_i = \sum \beta_i \mathbf{e}_i$라 하면 $\sum (\alpha_i - \beta_i) \mathbf{e}_i = \mathbf{0}$이고 선형독립 조건으로 $\alpha_i - \beta_i = 0$. $\square$

---

### Lemma 2.2 — Steinitz Exchange Lemma

**명제**: $V$가 집합 $\mathcal{G} = \{\mathbf{w}_1, \ldots, \mathbf{w}_m\}$으로 생성되고, $\mathcal{L} = \{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$가 선형독립이면

$$k \leq m,$$

그리고 $\mathcal{G}$의 $m - k$개 원소를 적절히 고른 $\mathcal{G}'$를 $\mathcal{L}$과 합친 $\mathcal{L} \cup \mathcal{G}'$이 다시 $V$를 생성한다 (즉 $\mathcal{L}$의 원소들로 $\mathcal{G}$의 원소들을 하나씩 "교환"할 수 있다).

**증명** (귀납법): $k$에 대한 귀납법으로 증명한다.

**기저 단계 ($k = 0$)**: 교환할 것이 없음, 자명.

**귀납 단계**: $k - 1$에 대해 성립한다고 가정하자. 즉 $\mathcal{L}' = \{\mathbf{v}_1, \ldots, \mathbf{v}_{k-1}\}$이 선형독립이면 $k - 1 \leq m$이고, 재배열 후 $\{\mathbf{v}_1, \ldots, \mathbf{v}_{k-1}, \mathbf{w}_k, \mathbf{w}_{k+1}, \ldots, \mathbf{w}_m\}$이 $V$를 생성한다.

이제 $\mathcal{L} = \{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$가 선형독립이라 하자. $\mathbf{v}_k \in V$이고 $\{\mathbf{v}_1, \ldots, \mathbf{v}_{k-1}, \mathbf{w}_k, \ldots, \mathbf{w}_m\}$이 $V$를 생성하므로

$$\mathbf{v}_k = \sum_{i=1}^{k-1} \alpha_i \mathbf{v}_i + \sum_{j=k}^{m} \beta_j \mathbf{w}_j \quad (*)$$

**주장**: 어떤 $\beta_j \neq 0$이다.

증명: 모든 $\beta_j = 0$이라면 $\mathbf{v}_k = \sum \alpha_i \mathbf{v}_i$이고, 이는 $\mathcal{L}$이 선형독립이라는 가정에 모순 ($\mathbf{v}_k - \sum \alpha_i \mathbf{v}_i = \mathbf{0}$은 자명하지 않은 조합). 따라서 어떤 $\beta_{j_0} \neq 0$. 또한 이것은 **$k - 1 < m$** 임을 함의하는데, $m = k-1$이면 $\beta$ 항이 없기 때문이다. 따라서 $k \leq m$.

재배열로 $j_0 = k$라 하자. (*)에서 $\mathbf{w}_k$에 대해 풀면

$$\mathbf{w}_k = \frac{1}{\beta_k}\left(\mathbf{v}_k - \sum_{i < k} \alpha_i \mathbf{v}_i - \sum_{j > k} \beta_j \mathbf{w}_j\right).$$

따라서 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k, \mathbf{w}_{k+1}, \ldots, \mathbf{w}_m\}$가 여전히 $V$를 생성한다 ($\mathbf{w}_k$를 이 집합의 선형결합으로 대체 가능하므로). $\square$

---

### 정리 2.3 — 차원의 유일성 (Dimension Theorem)

**명제**: 유한차원 벡터공간 $V$의 **임의의 두 기저는 같은 크기**를 가진다.

**증명**: $\mathcal{B}_1 = \{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$과 $\mathcal{B}_2 = \{\mathbf{f}_1, \ldots, \mathbf{f}_m\}$이 두 기저라 하자.

$\mathcal{B}_1$이 $V$를 생성하고 $\mathcal{B}_2$가 선형독립이므로, Steinitz에 의해 $m \leq n$.  
$\mathcal{B}_2$가 $V$를 생성하고 $\mathcal{B}_1$이 선형독립이므로, Steinitz에 의해 $n \leq m$.  
따라서 $n = m$. $\square$

> 이 정리 덕분에 **"$\dim V$"가 well-defined**되어, "3차원 공간"·"300차원 embedding"이라는 표현이 수학적으로 모호하지 않다.

---

### 정리 2.4 — 기저의 존재

**명제**: 유한차원 벡터공간은 기저를 가진다.

**증명 스케치**: 유한 생성집합에서 출발해 선형종속인 원소를 하나씩 제거하면 선형독립 상태에 도달. 이는 여전히 생성하므로 기저. (Zorn's Lemma를 쓰면 무한차원 공간에서도 Hamel basis의 존재를 보일 수 있다.) $\square$

---

### 정리 2.5 — 부분공간의 차원

**명제**: $W \subseteq V$가 부분공간이고 $V$가 유한차원이면, $W$도 유한차원이고 $\dim W \leq \dim V$, 등호는 $W = V$일 때만.

**증명 스케치**: $W$의 선형독립 부분집합은 $V$의 선형독립 부분집합이므로 크기 $\leq \dim V$ (Steinitz). 최대 크기의 선형독립 부분집합이 $W$의 기저. 등호는 $W$의 기저가 $V$ 전체를 생성할 때만, 즉 $W = V$. $\square$

---

### 예시 — 주요 벡터공간의 차원

| 공간 | 기저 | 차원 |
|------|------|------|
| $\mathbb{R}^n$ | 표준기저 $\mathbf{e}_i = (0, \ldots, 1, \ldots, 0)$ | $n$ |
| $\mathbb{R}^{m \times n}$ | 기본행렬 $E_{ij}$ ($(i,j)$만 1) | $mn$ |
| $\mathbb{R}[x]_{\leq n}$ (차수 $\leq n$ 다항식) | $\{1, x, x^2, \ldots, x^n\}$ | $n + 1$ |
| $\mathbb{R}[x]$ (모든 다항식) | $\{1, x, x^2, \ldots\}$ (무한) | $\infty$ (가산 Hamel) |
| $C[0,1]$ | (가산 Hamel basis 없음) | 비가산 차원 |
| 대칭행렬 공간 $\text{Sym}^n$ | $E_{ii}$와 $E_{ij} + E_{ji}$ ($i<j$) | $\frac{n(n+1)}{2}$ |

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp

# ─────────────────────────────────────────────
# 1. 선형독립 판정 (rank 활용)
# ─────────────────────────────────────────────
def is_linearly_independent(vectors):
    """각 열을 벡터로 보고 rank로 선형독립성 판정."""
    A = np.column_stack(vectors)
    return np.linalg.matrix_rank(A) == len(vectors)

# R^3에서 세 벡터
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
v4 = np.array([1, 1, 0])   # v1 + v2이므로 {v1, v2, v4}는 종속

print("{v1, v2, v3} 독립?", is_linearly_independent([v1, v2, v3]))  # True
print("{v1, v2, v4} 독립?", is_linearly_independent([v1, v2, v4]))  # False

# ─────────────────────────────────────────────
# 2. SymPy로 Steinitz Exchange 실연
# ─────────────────────────────────────────────
# 생성집합 G = {w1, w2, w3} of R^3
w1 = sp.Matrix([1, 1, 0])
w2 = sp.Matrix([0, 1, 1])
w3 = sp.Matrix([1, 0, 1])

# 선형독립 집합 L = {v1, v2}
v1 = sp.Matrix([1, 0, 0])
v2 = sp.Matrix([0, 1, 0])

# v1을 w들의 결합으로 표현: v1 = α1 w1 + α2 w2 + α3 w3
alpha = sp.symbols('a1 a2 a3')
eq = w1*alpha[0] + w2*alpha[1] + w3*alpha[2] - v1
sol = sp.solve([eq[i] for i in range(3)], alpha)
print("v1 = ", sol, " (w1, w2, w3 계수)")
# 예: v1 = (1/2) w1 + (-1/2) w2 + (1/2) w3 → α₃≠0이면 w₃를 v1로 교환 가능

# ─────────────────────────────────────────────
# 3. 좌표의 유일성 검증 (정리 2.1)
# ─────────────────────────────────────────────
# R^3의 두 기저:
B_std = np.eye(3)                             # 표준기저
B_new = np.array([[1, 1, 0],
                  [0, 1, 1],
                  [1, 0, 1]]).T                 # 새 기저 (열벡터)

# 임의 벡터의 좌표 변환
v = np.array([3, -2, 5])
coord_std = np.linalg.solve(B_std, v)
coord_new = np.linalg.solve(B_new, v)
print(f"\n벡터 v = {v}")
print(f"표준기저 좌표: {coord_std}")
print(f"새 기저 좌표 : {coord_new}")

# 복원: 각 기저 × 좌표 = v
assert np.allclose(B_std @ coord_std, v)
assert np.allclose(B_new @ coord_new, v)
print("✓ 두 기저 모두에서 고유한 좌표로 v를 표현")

# ─────────────────────────────────────────────
# 4. 부분공간의 차원 (정리 2.5)
# ─────────────────────────────────────────────
# R^4의 부분공간: 첫 성분이 두 번째와 같은 벡터들
# {(a, a, b, c) : a, b, c ∈ R} → 3차원 부분공간
W_basis = np.array([[1, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]).T
rank_W = np.linalg.matrix_rank(W_basis)
print(f"\n부분공간 W의 차원: {rank_W} (R^4의 4 이하)")

# ─────────────────────────────────────────────
# 5. 다항식 공간 R[x]_≤3 의 기저 검증 (SymPy)
# ─────────────────────────────────────────────
x = sp.symbols('x')
basis_poly = [1, x, x**2, x**3]

# p(x) = 2 - x + 3x² + 7x³ 를 좌표로
p = 2 - x + 3*x**2 + 7*x**3
coeffs = sp.Poly(p, x).all_coeffs()[::-1]   # [a₀, a₁, a₂, a₃]
print(f"\n다항식 p(x) = {p} 의 좌표: {coeffs}")
assert coeffs == [2, -1, 3, 7]
print("✓ R[x]_≤3 의 차원 = 4")
```

---

## 🔗 AI/ML 연결

### Rank로 표현되는 "유효 차원"

데이터 행렬 $X \in \mathbb{R}^{n \times d}$의 rank $r \leq \min(n, d)$는 **실제로 독립인 feature 개수**다. $d = 1000$ feature를 썼지만 rank = 50이면, 데이터는 50차원 부분공간에 놓여 있다. PCA는 이 부분공간의 **정규직교 기저**를 찾는 것이다 (Ch4-05).

### Sparse representation의 기저 선택

Sparse coding(Olshausen & Field 1996), dictionary learning에서 "overcomplete dictionary" $D \in \mathbb{R}^{d \times K}$ ($K > d$)의 열들은 $\mathbb{R}^d$의 **생성집합이지만 기저는 아님**. 선형종속을 허용하는 대신 sparsity로 표현의 유일성을 복원한다.

### 신경망의 "lottery ticket"과 subspace

Lottery Ticket Hypothesis(Frankle & Carbin 2019): 훈련 전에도 좋은 성능을 낼 수 있는 sparse subnetwork가 존재한다. 이는 파라미터 공간 $\mathbb{R}^N$의 어떤 **저차원 부분공간**이 충분한 표현력을 가진다는 의미. Li et al. (2018)은 이 "intrinsic dimension"이 N보다 훨씬 작다는 실험적 증거를 보였다.

### Feature Superposition

최근 Anthropic의 연구(Elhage et al. 2022)는 신경망이 뉴런 수보다 **더 많은 feature를 "superposition"** 상태로 표현한다는 이론을 제시. 이는 feature들이 뉴런 공간에서 **선형독립이 아닌 거의-독립** 상태로 놓여 있음을 의미하며, 선형대수의 극한 개념을 요구한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 유한 생성집합 존재 | $C[0,1], \ell^2$ 같은 함수공간은 무한차원 → 다른 도구 필요 (해석학·Functional Analysis) |
| 체 $\mathbb{F}$가 고정 | 같은 집합도 $\mathbb{R}$ 위 vs $\mathbb{C}$ 위에서 다른 차원 ($\mathbb{C}$는 $\mathbb{R}$ 위 2차원, 자기 자신 위 1차원) |
| 선형독립이 깔끔한 이분법 | 수치적으로는 "거의 종속" 상태 (조건수 높음)가 문제 — 이산적 rank가 아닌 **numerical rank** 필요 (Ch3) |
| Hamel basis | 무한차원에서 Hamel basis는 존재하나 구성 불가. Hilbert 공간에서는 **Schauder basis**가 더 유용 (Functional Analysis 레포) |

**수치적 함정**: 부동소수점에서 $10^{-15}$짜리 "작은 성분"이 있으면 rank 판정이 불안정. `np.linalg.matrix_rank`는 SVD 기반이고 기본 tolerance를 넘어서는 특이값만 센다. 조건수가 큰 행렬에서는 명시적 tolerance 설정 필요.

---

## 📌 핵심 정리

$$\boxed{\;\mathcal{B} \text{ basis of } V \iff \text{span}(\mathcal{B}) = V \text{ and } \mathcal{B} \text{ linearly independent}\;}$$

$$\boxed{\;\dim V \text{ is well-defined (독립-생성 교환 보조정리로)}\;}$$

| 개념 | 본질 |
|------|------|
| 선형독립 | $\sum \alpha_i \mathbf{v}_i = \mathbf{0} \implies$ 모든 $\alpha_i = 0$ |
| 생성 | 선형결합으로 모든 원소 도달 |
| 기저 | 독립 + 생성 |
| Steinitz | 독립 집합 크기 ≤ 생성 집합 크기 |
| 차원 | 기저 크기 (선택에 무관) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\mathbb{R}^3$에서 $\mathbf{v}_1 = (1, 2, 3)$, $\mathbf{v}_2 = (4, 5, 6)$, $\mathbf{v}_3 = (7, 8, 9)$의 선형독립성을 판정하라. 독립이 아니면 구체적인 영이 아닌 계수 조합을 제시하라.

<details>
<summary>힌트 및 해설</summary>

$\mathbf{v}_3 - 2\mathbf{v}_2 + \mathbf{v}_1 = (7-8+1, 8-10+2, 9-12+3) = (0, 0, 0)$. 따라서 $1 \cdot \mathbf{v}_1 - 2 \cdot \mathbf{v}_2 + 1 \cdot \mathbf{v}_3 = \mathbf{0}$이고 계수가 자명하지 않으므로 **선형종속**. rank 계산: $\text{rank}([\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3]) = 2$. 이 세 벡터는 $\mathbb{R}^3$에서 **2차원 평면**에 놓인다.

</details>

**문제 2** (심화): $C[0, 1]$에서 $\{1, x, x^2, \ldots, x^n\}$이 선형독립임을 증명하라. 힌트: $\sum a_i x^i \equiv 0$이라 가정하고 도함수를 반복해서 적용.

<details>
<summary>힌트 및 해설</summary>

$p(x) = \sum_{i=0}^n a_i x^i = 0 \;(\forall x \in [0,1])$이라 하자. $x = 0$ 대입 → $a_0 = 0$. 양변을 미분: $p'(x) = \sum_{i=1}^n i a_i x^{i-1} = 0$. $x = 0$ 대입 → $a_1 = 0$. 반복하면 $a_k = 0$ for all $k$. 따라서 선형독립. 이로써 **$\mathbb{R}[x]_{\leq n}$의 차원이 $n+1$**임이 확인된다 ($\{1, x, \ldots, x^n\}$이 생성은 정의에 의해 자명). 결과: 차수 $\leq 1000$ 다항식의 공간은 1001차원이므로, Transformer의 positional encoding을 $\sin(kx), \cos(kx)$의 유한 합으로 대체해도 표현력은 유한차원으로 제한된다.

</details>

**문제 3** (AI 연결): 신경망 파라미터 $\theta \in \mathbb{R}^N$에서 훈련 경로 $\{\theta_0, \theta_1, \ldots, \theta_T\}$의 **span**의 차원이 $\ll N$일 수 있다 (Li et al. 2018). 이 관찰이 "신경망의 intrinsic dimension이 낮다"는 주장과 어떻게 연결되는가? 또한 이를 활용해 훈련을 가속화하는 아이디어는?

<details>
<summary>힌트 및 해설</summary>

경로 span의 차원이 $k \ll N$이면, $\theta_t = \theta_0 + \sum_{i=1}^k c_{t,i} \mathbf{u}_i$ (어떤 기저 $\mathbf{u}_i$로) 로 쓸 수 있다. 즉 **원래 $N$차원 문제가 실제로는 $k$차원 affine subspace 위의 최적화**다. Li et al. (2018)은 random basis $\mathbf{u}_i$로 **intrinsic dimension**을 측정해, ImageNet 수준의 문제도 수백~수천 차원이면 충분함을 보였다. 응용: (1) 파라미터를 저차원 subspace에 제약(gradient projection), (2) Low-Rank Adaptation(LoRA, Hu et al. 2021)의 rank가 이 직관의 극단적 구현 — $\Delta W = AB$ ($A \in \mathbb{R}^{N \times r}, B \in \mathbb{R}^{r \times N}$)로 업데이트를 rank $r$로 제한하여 훈련 비용을 극적으로 줄인다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 벡터공간의 8개 공리](./01-vector-space-8-axioms.md) | [📚 README](../README.md) | [03. 선형변환과 행렬 표현 ▶](./03-linear-transformation-matrix.md) |

</div>
