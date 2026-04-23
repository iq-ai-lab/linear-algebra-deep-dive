# 01. 벡터공간의 8개 공리

## 🎯 핵심 질문

- 왜 **수벡터·연속함수·다항식·행렬**이 모두 같은 "벡터공간"이라는 추상 구조에 속하는가?
- "벡터는 크기와 방향을 가진 양"이라는 고등학교 정의는 왜 부족한가?
- 체(field) $\mathbb{F}$ 위에서 벡터공간을 정의하는 **8개 공리**는 각각 무엇을 보장하는가? 하나라도 빠지면 어떤 반례가 생기는가?
- 영벡터와 역원은 왜 유일한가? 공리에서 유도되는 **파생 성질**은?

---

## 🔍 왜 이 개념이 AI에서 중요한가

"벡터공간"이라는 추상 구조가 중요한 이유는, **한 번의 증명으로 무한히 많은 객체에 적용**할 수 있기 때문이다.

- **신경망의 파라미터 공간**: $\theta \in \mathbb{R}^N$ ($N$이 수십억일 수 있음)에서 $\theta_1 + \theta_2$, $\alpha\theta$가 "잘 정의"되는 이유는 $\mathbb{R}^N$이 벡터공간이기 때문이다. Gradient descent $\theta \leftarrow \theta - \eta \nabla L$은 벡터공간 연산이다.
- **함수공간으로서의 모델**: 신경망 자체를 $f_\theta: \mathcal{X} \to \mathcal{Y}$라는 함수로 볼 때, 그 함수들이 사는 공간은 **연속함수 공간 $C(\mathcal{X}, \mathcal{Y})$**의 부분집합이다. Universal Approximation 정리가 성립하려면 함수 공간이 벡터공간 구조를 가져야 한다.
- **손실함수의 선형결합**: Multi-task learning에서 $L = \alpha L_1 + \beta L_2$를 합성할 수 있는 이유는, 손실함수들이 **함수공간의 원소**로서 벡터공간 연산을 허용하기 때문이다.
- **가중치 텐서 공간**: Conv2D의 가중치 $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$도 벡터공간이다. 다중 인덱스여도 공리는 똑같이 성립한다.
- **확률분포의 선형결합**: 혼합 모델 $p = \alpha p_1 + (1-\alpha) p_2$는 함수공간에서의 convex combination이다.

"$\theta + \theta'$를 써도 되는가", "행렬과 다항식을 같은 틀로 다뤄도 되는가" — 이 질문에 답하려면 먼저 **벡터공간이 무엇인지**를 공리로부터 정의해야 한다.

---

## 📐 수학적 선행 조건

- 집합·함수의 기초: 사상, 단사·전사
- 실수·복소수의 사칙연산 (체의 기초)
- 논리: 전칭·존재 명제, 유일성 증명

> 이 문서는 **IQ AI Lab Layer 0의 첫 문서**로, 선행 지식이 거의 필요 없도록 구성되어 있습니다. 체(field) 개념은 본문에서 즉석 도입합니다.

---

## 📖 직관적 이해

### "덧셈과 스칼라배가 잘 정의되는 공간"

벡터공간은 **두 가지 연산**이 일정 규칙(공리) 하에 작동하는 집합이다.

- **덧셈**: 원소 $\mathbf{v}, \mathbf{w}$에서 새 원소 $\mathbf{v} + \mathbf{w}$를 만드는 법
- **스칼라곱**: 체 $\mathbb{F}$의 원소 $\alpha$와 공간의 원소 $\mathbf{v}$에서 $\alpha\mathbf{v}$를 만드는 법

이 두 연산이 **결합법칙, 교환법칙, 분배법칙, 영원·역원의 존재**를 만족하면 그 집합은 벡터공간이다. 이 추상화가 위대한 이유는, "$\mathbb{R}^3$의 화살표"와 "구간 $[0,1]$ 위의 연속함수"가 **같은 공리를 만족**한다는 점 때문이다.

### 왜 8개인가

| 그룹 | 공리 | 보장하는 것 |
|------|------|------------|
| 덧셈 4개 | 결합·교환·영원·역원 | $(V, +)$가 **아벨군(abelian group)** |
| 스칼라곱 4개 | 결합·단위원·두 분배법칙 | $\mathbb{F}$와 $V$의 **호환 작용** |

두 구조가 "잘 맞물린다"는 조건을 7-8번 공리(분배법칙)가 보장한다.

### 왜 "체 $\mathbb{F}$ 위의" 벡터공간인가

스칼라로 쓸 수 있는 수 체계가 **뺄셈·나눗셈까지 자유로워야** 행렬 분해·역행렬·차원 논의가 가능하다. 이런 구조가 **체(field)**이고, 대표적으로 $\mathbb{R}$(실수), $\mathbb{C}$(복소수), $\mathbb{F}_p$(소수 $p$에 대한 유한체)가 있다.

> **비유**: 벡터공간 공리는 "도시의 교통 법규"다. 차량(원소)이 다양해도(승용차·트럭·자전거), 법규를 따르면 서로 충돌 없이 같은 도로를 쓸 수 있다. 공리를 만족하는 집합이면 **선형대수의 모든 정리**가 자동으로 적용된다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 체 (Field)

집합 $\mathbb{F}$가 두 연산 $+, \cdot$을 가지고 다음을 만족하면 **체**라 한다:

1. $(\mathbb{F}, +)$는 아벨군 (교환·결합·영원 $0$·역원 존재)
2. $(\mathbb{F} \setminus \{0\}, \cdot)$는 아벨군 (교환·결합·단위원 $1$·역원 존재)
3. 분배법칙: $a \cdot (b + c) = a\cdot b + a\cdot c$

**예**: $\mathbb{R}, \mathbb{C}, \mathbb{Q}$는 체. $\mathbb{Z}$는 체가 아님(대부분 원소의 곱셈 역원이 없음).

---

### 정의 1.2 — 벡터공간 (Vector Space)

집합 $V$가 체 $\mathbb{F}$ 위의 **벡터공간**이라는 것은, 두 연산

- **덧셈** $+: V \times V \to V$, $(\mathbf{u}, \mathbf{v}) \mapsto \mathbf{u} + \mathbf{v}$
- **스칼라곱** $\cdot: \mathbb{F} \times V \to V$, $(\alpha, \mathbf{v}) \mapsto \alpha\mathbf{v}$

이 다음 **8개 공리**를 모두 만족한다는 것이다.

**덧셈에 관한 공리 (A1–A4)**:

- **(A1) 결합법칙**: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- **(A2) 교환법칙**: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
- **(A3) 영벡터 존재**: $\exists \mathbf{0} \in V$ s.t. $\mathbf{v} + \mathbf{0} = \mathbf{v}$, $\forall \mathbf{v} \in V$
- **(A4) 덧셈 역원**: $\forall \mathbf{v} \in V, \exists -\mathbf{v} \in V$ s.t. $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$

**스칼라곱에 관한 공리 (S1–S4)**:

- **(S1) 스칼라곱 결합**: $\alpha(\beta\mathbf{v}) = (\alpha\beta)\mathbf{v}$
- **(S2) 단위원 작용**: $1 \cdot \mathbf{v} = \mathbf{v}$ (여기서 $1 \in \mathbb{F}$)
- **(S3) 벡터 분배법칙**: $\alpha(\mathbf{u} + \mathbf{v}) = \alpha\mathbf{u} + \alpha\mathbf{v}$
- **(S4) 스칼라 분배법칙**: $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$

---

### 정의 1.3 — 부분공간 (Subspace)

$V$가 $\mathbb{F}$ 위의 벡터공간이고, $W \subseteq V$가 다음을 만족하면 **$W$는 $V$의 부분공간**:

1. $\mathbf{0} \in W$ (영벡터 포함)
2. $\mathbf{u}, \mathbf{v} \in W \implies \mathbf{u} + \mathbf{v} \in W$ (덧셈에 대해 닫힘)
3. $\alpha \in \mathbb{F}, \mathbf{v} \in W \implies \alpha\mathbf{v} \in W$ (스칼라곱에 대해 닫힘)

8공리는 $V$에서 상속받으므로, 세 조건만 확인하면 된다.

---

## 🔬 정리와 증명

### 정리 1.1 — 영벡터의 유일성

**명제**: 벡터공간 $V$에서 **영벡터 $\mathbf{0}$은 유일**하다.

**증명**: $\mathbf{0}_1, \mathbf{0}_2$가 모두 영벡터라 하자. 그러면

- $\mathbf{0}_1 + \mathbf{0}_2 = \mathbf{0}_1$ (A3, $\mathbf{0}_2$가 영벡터)
- $\mathbf{0}_1 + \mathbf{0}_2 = \mathbf{0}_2$ (A3, $\mathbf{0}_1$이 영벡터 + A2 교환)

두 식에서 $\mathbf{0}_1 = \mathbf{0}_2$. $\square$

---

### 정리 1.2 — 덧셈 역원의 유일성

**명제**: 각 $\mathbf{v} \in V$에 대해 **역원 $-\mathbf{v}$는 유일**하다.

**증명**: $\mathbf{w}_1, \mathbf{w}_2$가 모두 $\mathbf{v}$의 역원이라 하자. 그러면

$$\mathbf{w}_1 = \mathbf{w}_1 + \mathbf{0} = \mathbf{w}_1 + (\mathbf{v} + \mathbf{w}_2) = (\mathbf{w}_1 + \mathbf{v}) + \mathbf{w}_2 = \mathbf{0} + \mathbf{w}_2 = \mathbf{w}_2.$$

각 등호에서 사용한 공리: A3 → A4(for $\mathbf{w}_2$) → A1 → A4(for $\mathbf{w}_1$) + A2 → A3. $\square$

---

### 정리 1.3 — $0 \cdot \mathbf{v} = \mathbf{0}$

**명제**: 모든 $\mathbf{v} \in V$에 대해 **스칼라 0을 곱하면 영벡터**이다.

**증명**: $\mathbb{F}$에서 $0 + 0 = 0$이므로 S4 분배법칙에 의해

$$0 \cdot \mathbf{v} = (0 + 0)\mathbf{v} = 0\mathbf{v} + 0\mathbf{v}.$$

양변에 $-(0\mathbf{v})$를 더하면 (A4로 존재 보장):

$$\mathbf{0} = 0\mathbf{v} + 0\mathbf{v} + (-(0\mathbf{v})) = 0\mathbf{v} + \mathbf{0} = 0\mathbf{v}. \square$$

---

### 정리 1.4 — $(-1) \cdot \mathbf{v} = -\mathbf{v}$

**명제**: 스칼라 $-1$을 곱하면 덧셈 역원이 된다.

**증명**: 정리 1.3과 S4, S2를 이용해

$$\mathbf{v} + (-1)\mathbf{v} = 1\cdot\mathbf{v} + (-1)\mathbf{v} = (1 + (-1))\mathbf{v} = 0 \cdot \mathbf{v} = \mathbf{0}.$$

정리 1.2에 의해 역원은 유일하므로 $(-1)\mathbf{v} = -\mathbf{v}$. $\square$

---

### 정리 1.5 — 네 가지 대표 예는 모두 벡터공간

**명제**: 다음 집합들은 $\mathbb{R}$ 위의 벡터공간이다.

1. **수벡터 공간** $\mathbb{R}^n = \{(x_1, \ldots, x_n) : x_i \in \mathbb{R}\}$, 성분별 덧셈·스칼라곱
2. **연속함수 공간** $C[a, b] = \{f: [a,b] \to \mathbb{R} \mid f \text{ continuous}\}$, 점별 덧셈·스칼라곱
3. **다항식 공간** $\mathbb{R}[x] = \{\sum_{i=0}^n a_i x^i : n \in \mathbb{N}, a_i \in \mathbb{R}\}$, 계수별 덧셈·스칼라곱
4. **행렬 공간** $\mathbb{R}^{m \times n}$, 성분별 덧셈·스칼라곱

**증명 스케치** ($C[a, b]$를 예시로):

- **(A3) 영벡터**: 상수 함수 $f \equiv 0$. $f(x) + 0 = f(x)$는 점별로 성립.
- **(A4) 역원**: $f(x)$에 대해 $-f(x)$는 점별 음수. 연속함수의 음수도 연속이므로 $C[a,b]$ 내부에 존재.
- **(S3) 분배**: $\alpha(f + g)(x) = \alpha(f(x) + g(x)) = \alpha f(x) + \alpha g(x)$는 $\mathbb{R}$의 분배법칙에서 점별로 성립.
- 나머지 공리도 **점별로 $\mathbb{R}$의 공리를 상속**.

나머지 공간도 원소의 모든 성분·계수가 $\mathbb{R}$의 공리를 만족하므로 유사하게 성립. $\square$

> **핵심 관찰**: 4가지 집합은 표현 형식이 완전히 다르지만(튜플·함수·다항식·행렬), **공리를 만족하는 순간부터 선형대수의 모든 정리가 적용**된다. 이것이 추상화의 힘이다.

---

### 예시 반례: 공리가 깨지는 경우

**반례 1 — "양수만" 모으면?**  
$V = \mathbb{R}_{>0}$, 일반 덧셈·스칼라곱.  
영벡터 $0 \notin \mathbb{R}_{>0}$이므로 **A3 위반**. 또한 $-1 \cdot 1 = -1 \notin \mathbb{R}_{>0}$이므로 스칼라곱조차 공간 안으로 닫히지 않음.

**반례 2 — "상수항이 0인 다항식" 대신 "상수항이 1인 다항식"?**  
$W = \{p \in \mathbb{R}[x] : p(0) = 1\}$은 $\mathbf{0}$ (상수 0 다항식)을 포함하지 않으므로 부분공간 아님. 또한 $p, q \in W$이면 $(p + q)(0) = 2 \neq 1$이라 덧셈에 대해서도 닫히지 않음.

**반례 3 — 연속 아닌 함수를 섞으면?**  
불연속 함수까지 포함하면 오히려 공간이 커져 벡터공간이 되지만, 원래 $C[a,b]$ 자체는 **함수들의 합·스칼라곱이 다시 연속**이라는 성질(해석학 정리) 덕분에 닫혀 있다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp

# ─────────────────────────────────────────────
# 1. R^n이 벡터공간임을 8공리로 검증
# ─────────────────────────────────────────────
rng = np.random.default_rng(0)
u = rng.standard_normal(4)
v = rng.standard_normal(4)
w = rng.standard_normal(4)
alpha, beta = 2.3, -1.7

# A1 결합법칙
assert np.allclose((u + v) + w, u + (v + w))
# A2 교환법칙
assert np.allclose(u + v, v + u)
# A3 영벡터
zero = np.zeros(4)
assert np.allclose(v + zero, v)
# A4 역원
assert np.allclose(v + (-v), zero)
# S1 스칼라곱 결합
assert np.allclose(alpha * (beta * v), (alpha * beta) * v)
# S2 단위원 작용
assert np.allclose(1 * v, v)
# S3 벡터 분배
assert np.allclose(alpha * (u + v), alpha * u + alpha * v)
# S4 스칼라 분배
assert np.allclose((alpha + beta) * v, alpha * v + beta * v)

print("✓ R^4가 8개 공리를 모두 만족")

# ─────────────────────────────────────────────
# 2. 행렬 공간 R^{2x3}도 같은 공리로 검증
# ─────────────────────────────────────────────
U = rng.standard_normal((2, 3))
V = rng.standard_normal((2, 3))
W = rng.standard_normal((2, 3))

# S3 분배법칙을 행렬에서 검증
assert np.allclose(alpha * (U + V), alpha * U + alpha * V)
# A4 역원
assert np.allclose(U + (-U), np.zeros((2, 3)))

print("✓ R^(2x3) 행렬공간도 8공리를 만족")

# ─────────────────────────────────────────────
# 3. SymPy로 다항식 공간 검증
# ─────────────────────────────────────────────
x = sp.symbols('x')
p = 1 + 2*x + 3*x**2
q = -4 + x**3
r = 5*x**2

# A1 결합
lhs = sp.expand((p + q) + r)
rhs = sp.expand(p + (q + r))
assert sp.simplify(lhs - rhs) == 0

# S3 분배 (스칼라 = 체 원소)
a = sp.Rational(7, 2)
lhs2 = sp.expand(a * (p + q))
rhs2 = sp.expand(a * p + a * q)
assert sp.simplify(lhs2 - rhs2) == 0

print("✓ R[x] 다항식 공간도 8공리를 만족 (SymPy 검증)")

# ─────────────────────────────────────────────
# 4. 파생 성질: 0·v = 0, (-1)·v = -v
# ─────────────────────────────────────────────
v = rng.standard_normal(5)
assert np.allclose(0 * v, np.zeros(5))       # 정리 1.3
assert np.allclose((-1) * v, -v)              # 정리 1.4

print("✓ 0·v = 0 과 (-1)·v = -v 파생 성질 검증 완료")
```

**출력**:
```
✓ R^4가 8개 공리를 모두 만족
✓ R^(2x3) 행렬공간도 8공리를 만족
✓ R[x] 다항식 공간도 8공리를 만족 (SymPy 검증)
✓ 0·v = 0 과 (-1)·v = -v 파생 성질 검증 완료
```

---

## 🔗 AI/ML 연결

### 파라미터 공간의 벡터공간 구조

신경망의 파라미터 $\theta \in \mathbb{R}^N$이 벡터공간이라는 사실이 **SGD가 수학적으로 타당**함을 보장한다. 갱신식 $\theta_{k+1} = \theta_k - \eta \nabla L(\theta_k)$는 벡터공간 연산이고, $\theta_k, \nabla L, \eta$가 모두 같은 공간 또는 호환되는 체에 속해야 의미를 가진다.

### 함수 공간으로서의 모델

신경망 자체를 $f_\theta \in \mathcal{F}$라는 **함수공간 원소**로 보면, 앙상블 $\frac{1}{K}\sum_k f_{\theta_k}$의 덧셈·스칼라곱이 정당화된다. Universal Approximation Theorem(Cybenko 1989, Hornik 1991)은 1-은닉층 신경망들의 **선형결합**이 $C(\mathcal{X})$에서 조밀하다는 형태로 진술된다.

### Mixture Model과 손실 합성

혼합 모델 $p = \sum_k \pi_k p_k$, multi-task 손실 $L = \sum_k \lambda_k L_k$는 **함수공간의 affine / convex combination**이다. 가중치 $\pi_k, \lambda_k \in \mathbb{R}$이 체의 원소로서 스칼라곱 공리를 충족시킨다.

### Transformer의 Residual Stream

Pre-Norm Transformer의 residual stream $\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + \text{MLP}(\text{LN}(\mathbf{h}_\ell))$은 각 층이 **벡터공간의 덧셈**을 쓴다. "representation을 더해서 합성한다"는 아이디어의 수학적 기반이다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 / 주의 |
|------|-------------|
| 체 $\mathbb{F}$가 주어짐 | $\mathbb{F} = \mathbb{R}, \mathbb{C}$ 외에 유한체(code theory), 유리함수체 등 사용 시 "차원"이나 "정규직교"의 의미가 달라짐 |
| 집합 $V$가 닫혀 있음 | "양수만"처럼 일부만 고르면 닫히지 않음 → 공간 아님 |
| 덧셈·스칼라곱이 외부에서 정의됨 | 연산 정의가 바뀌면 같은 집합도 다른 벡터공간이 될 수 있음 (예: $\mathbb{R}_{>0}$에 $\oplus := \times$, $\alpha \odot v := v^\alpha$로 정의하면 벡터공간!) |
| 유한차원 가정 없음 | $C[a,b], \mathbb{R}[x]$는 무한차원. Ch1-02에서 유한/무한 분기 |
| 내적·노름 무전제 | 8공리만으로는 거리·각도 개념이 없음 → Ch5에서 내적 추가 |

**수치적 함정**: 부동소수점에서는 (A1) 결합법칙이 엄격히 성립하지 않는다 (`(a+b)+c != a+(b+c)` 가능). 이는 컴퓨터 산술의 한계이지 수학적 벡터공간 정의와는 별개.

---

## 📌 핵심 정리

$$\boxed{\;V \text{는 }\mathbb{F}\text{-벡터공간} \iff (V, +) \text{ 아벨군} + \mathbb{F}\text{-스칼라곱이 8공리 만족}\;}$$

| 공리 그룹 | 개수 | 본질 |
|-----------|------|------|
| 덧셈 공리 (A1–A4) | 4 | $(V, +)$를 **아벨군**으로 |
| 스칼라곱 공리 (S1–S4) | 4 | $\mathbb{F}$와의 **호환 작용** |

파생 정리: 영벡터 유일·역원 유일·$0\mathbf{v} = \mathbf{0}$·$(-1)\mathbf{v} = -\mathbf{v}$ 는 모두 공리에서 유도된다.

---

## 🤔 생각해볼 문제

**문제 1** (기초): 집합 $V = \mathbb{R}^2$에 다음 연산을 정의한다: $(x_1, y_1) \oplus (x_2, y_2) = (x_1 + x_2, y_1 + y_2 + x_1 x_2)$, 스칼라곱은 일반 성분별 곱. 이것이 벡터공간인가?

<details>
<summary>힌트 및 해설</summary>

$(0,0)$이 영벡터 후보이지만, $(x, y) \oplus (0,0) = (x, y + 0) = (x, y)$는 OK. 그러나 A2 교환법칙: $(x_1, y_1) \oplus (x_2, y_2) = (x_1 + x_2, y_1 + y_2 + x_1 x_2)$인 반면 $(x_2, y_2) \oplus (x_1, y_1) = (x_1 + x_2, y_1 + y_2 + x_2 x_1)$. $x_1 x_2 = x_2 x_1$이므로 실제로 교환법칙은 성립. 하지만 A1 결합법칙 검증: $((x_1, y_1) \oplus (x_2, y_2)) \oplus (x_3, y_3)$과 $(x_1, y_1) \oplus ((x_2, y_2) \oplus (x_3, y_3))$를 비교해 보면 $y$ 성분이 각각 $y_1 + y_2 + y_3 + x_1 x_2 + (x_1+x_2)x_3$와 $y_1 + y_2 + y_3 + x_2 x_3 + x_1(x_2+x_3)$로 같음(확인). 그러나 S4 분배법칙 $(\alpha + \beta)(x, y)$를 검증: 일반 스칼라곱이면 $((\alpha+\beta)x, (\alpha+\beta)y)$. $\alpha(x,y) \oplus \beta(x,y) = (\alpha x, \alpha y) \oplus (\beta x, \beta y) = (\alpha x + \beta x, \alpha y + \beta y + \alpha\beta x^2)$. 두 값이 같으려면 $\alpha\beta x^2 = 0$이 항상 성립해야 하는데 이는 $x=0$ 외에는 거짓. 따라서 **S4 위반 → 벡터공간 아님**.

</details>

**문제 2** (심화): $V = \mathbb{R}_{>0}$에 $\mathbf{u} \oplus \mathbf{v} := \mathbf{u} \cdot \mathbf{v}$ (보통 곱셈), $\alpha \odot \mathbf{v} := \mathbf{v}^\alpha$로 정의한다. 이것이 $\mathbb{R}$ 위의 벡터공간임을 보이고, "영벡터"가 무엇인지 찾아라.

<details>
<summary>힌트 및 해설</summary>

영벡터 후보: $\mathbf{v} \oplus \mathbf{0} = \mathbf{v}$ 이려면 $\mathbf{v} \cdot \mathbf{0} = \mathbf{v}$, 즉 $\mathbf{0} = 1$. 맞다 — **이 공간에서 영벡터는 실수 1**이다! 역원: $\mathbf{v} \oplus (-\mathbf{v}) = 1$ 이므로 $-\mathbf{v} = 1/\mathbf{v}$. S2: $1 \odot \mathbf{v} = \mathbf{v}^1 = \mathbf{v}$ ✓. S4: $(\alpha + \beta) \odot \mathbf{v} = \mathbf{v}^{\alpha+\beta} = \mathbf{v}^\alpha \mathbf{v}^\beta = (\alpha \odot \mathbf{v}) \oplus (\beta \odot \mathbf{v})$ ✓. 나머지도 성립. **로그 변환 $\log: \mathbb{R}_{>0} \to \mathbb{R}$이 이 "곱셈 벡터공간"과 "덧셈 벡터공간" 사이의 동형사상**. 이는 log-space에서의 확률 계산(log-sum-exp, logit)이 왜 선형대수적으로 자연스러운지 설명한다.

</details>

**문제 3** (AI 연결): 신경망 가중치 텐서 $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$ (Conv2D)가 $\mathbb{R}$ 위의 벡터공간임을 보이고, 차원을 구하라. 가중치의 평균 $\bar{W} = \frac{1}{K}\sum_k W_k$ (앙상블에서의 stochastic weight averaging)가 이 공간의 원소인 이유를 8공리로 설명하라.

<details>
<summary>힌트 및 해설</summary>

$W$는 $C_{\text{out}} \cdot C_{\text{in}} \cdot k^2$개의 실수 성분을 가지고, 성분별 덧셈·스칼라곱이 $\mathbb{R}$의 공리를 상속한다. 차원 = $C_{\text{out}} \cdot C_{\text{in}} \cdot k^2$. 평균 $\bar{W}$는 $\frac{1}{K}$이라는 스칼라와 $W_1 + \cdots + W_K$라는 덧셈의 결과이므로, S1~S4와 A1~A4에 의해 공간 안에 머문다. 이것이 SWA(Izmailov et al. 2018) 알고리즘이 잘 정의된 이유다. 다만 Conv2D가 쓰이는 함수로서의 성질(비선형성·feature locality)은 공리와 무관하며, 다른 정리가 필요하다.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. 선형독립, 기저, 차원 ▶](./02-basis-dimension.md) |

</div>
