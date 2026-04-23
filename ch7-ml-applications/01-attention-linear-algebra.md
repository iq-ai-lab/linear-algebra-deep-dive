# 7.1 Attention의 선형대수

> "Transformer는 '학습된 매트릭스 4개와 softmax 하나'로 이루어진 선형대수 기계다."

---

## 1. 학습 목표

- **Self-Attention**의 수식을 선형대수의 언어로 완전히 해부한다.
- $Q, K, V, O$ 투영 행렬의 의미, 쿼리-키 유사도, **softmax의 역할**을 엄밀히 이해한다.
- **Multi-head**가 왜 단일 head보다 풍부한지 rank와 subspace 관점에서 설명한다.
- **Attention 행렬의 저랭크 구조**(Linformer, Performer)를 관찰한다.
- **Positional encoding**, **Causal masking**의 수학적 구성을 본다.

---

## 2. Self-Attention의 정의

### 2.1 입력

토큰 임베딩 시퀀스 $X \in \mathbb{R}^{S \times d}$ ($S$: 시퀀스 길이, $d$: 임베딩 차원).

### 2.2 Q, K, V 투영

학습되는 가중치 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

$Q, K \in \mathbb{R}^{S \times d_k}$, $V \in \mathbb{R}^{S \times d_v}$.

### 2.3 Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

### 2.4 해석

- $QK^T \in \mathbb{R}^{S \times S}$: 토큰 $i$의 쿼리와 토큰 $j$의 키의 **내적**. 즉 "$i$가 $j$를 얼마나 찾고 있는가".
- $\text{softmax}$: 각 행을 확률 분포로 변환. 쿼리 $i$에 대해 모든 키를 **확률적으로 선택**.
- $\cdot V$: 선택된 가중치로 값 $V$의 **선형 결합**.

---

## 3. 왜 $\sqrt{d_k}$로 스케일하는가?

### 3.1 분산 계산

$Q, K$의 각 성분이 평균 0, 분산 $\sigma^2 = 1$인 독립 난수라 가정. 그러면 내적 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$의 분산은

$$
\text{Var}(q \cdot k) = d_k \cdot \text{Var}(q_i k_i) = d_k
$$

(독립성에서 $\text{Var}(q_i k_i) = \text{Var}(q_i)\text{Var}(k_i) = 1$).

### 3.2 Softmax의 포화

큰 값들 사이의 softmax는 **원핫(one-hot)** 에 가까워지며, 기울기가 소실된다:

$$
\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j) \approx 0 \text{ when softmax near 0/1}
$$

### 3.3 해결: $\sqrt{d_k}$ 나눗셈

스케일 후 분산은 $d_k / d_k = 1$로 유지. 기울기 흐름을 안정시킨다.

---

## 4. Softmax의 역할

### 4.1 왜 softmax인가?

선형 가중합 $AV$로도 가능해 보이지만:

1. **정규화**: 모든 행이 확률 분포 → "관심의 배분"
2. **비선형성**: 고차원 표현력 ($\to$ sigmoid/tanh의 역할)
3. **미분가능**: 역전파 가능

### 4.2 Softmax의 행별 독립성

$\text{softmax}(QK^T)$는 **각 행마다 독립적으로** 계산. 따라서 한 쿼리의 어텐션은 다른 쿼리에 영향받지 않음 (병렬화 용이).

### 4.3 다른 대안들

- **Linear Attention**: softmax 제거 (아래 8절)
- **Sparse Attention**: 일부 키만 선택 (Longformer)
- **Local Attention**: 주변 윈도우만
- **Performer**: softmax의 **랜덤 피처 근사**

---

## 5. Multi-Head Attention

### 5.1 정의

$h$개의 "헤드"를 병렬로 계산, 결과를 이어붙여 한 번 더 투영:

$$
\text{head}_i = \text{Attention}(XW^Q_i, XW^K_i, XW^V_i)
$$

$$
\text{MHA}(X) = [\text{head}_1 | \cdots | \text{head}_h] W^O
$$

$W^Q_i, W^K_i \in \mathbb{R}^{d \times d_k}$, $W^V_i \in \mathbb{R}^{d \times d_v}$, $W^O \in \mathbb{R}^{h d_v \times d}$.

### 5.2 공식 재표현

$d_k = d_v = d/h$로 선택하면 파라미터 수는 단일 head와 같다. 단:

- **한 head**: rank $\le d_k = d$ (한 번의 $QK^T$)
- **$h$개 head**: 각각 rank $\le d/h$. 이들의 합은 rank $\le d$.

차원이 같아 보이지만 **부분공간 분할**이 다르다.

### 5.3 왜 Multi-Head인가?

직관: 각 head가 **서로 다른 관계**를 학습.

- Head 1: 구문 의존 (종속절-주절)
- Head 2: 장기 의미 (주제-객체)
- Head 3: 위치 관계 (다음-이전)

### 5.4 수학적 표현력

**정리 7.1.1 (Michel-Levy-Neubig).** 학습된 Multi-head에서 상당수 head를 제거해도 성능이 크게 떨어지지 않는다. 이는 head들이 **중복**된다는 뜻이지만, 훈련 중 다양성을 확보하는 데 기여한다.

---

## 6. Attention 행렬의 저랭크 구조

### 6.1 관찰: $A = \text{softmax}(QK^T / \sqrt{d_k})$의 랭크

- 정확한 rank는 $S$일 수 있음 (full rank).
- 그러나 **유효 랭크(effective rank)** 는 훨씬 작다. 특잇값이 빠르게 감소.

### 6.2 Linformer의 아이디어

$A \in \mathbb{R}^{S \times S}$가 유효 랭크 $k \ll S$라면

$$
AV \approx E A F^T V, \quad E \in \mathbb{R}^{S \times k}, F \in \mathbb{R}^{k \times S}
$$

로 근사할 수 있다. 복잡도 $O(S^2)$가 $O(Sk)$로.

### 6.3 Nyström 근사

큰 $A$를 작은 submatrix로 근사:

$$
A \approx C W^{+} C^T
$$

$W$는 $k \times k$ 부분 행렬. Attention에서는 $k$개의 "landmark" 토큰을 선택.

---

## 7. Positional Encoding

### 7.1 문제: Attention의 순열 등분리성

Self-attention은 **입력 순서 불변**:

$$
\text{Attn}(PX) = P \text{Attn}(X) \text{ for any permutation } P
$$

따라서 위치 정보가 필요.

### 7.2 Sinusoidal Positional Encoding

$$
PE(pos, 2i) = \sin(pos / 10000^{2i/d}), \quad PE(pos, 2i+1) = \cos(pos / 10000^{2i/d})
$$

임베딩에 **덧셈**: $X' = X + PE$.

**선형성**: $PE_{pos + k}$는 $PE_{pos}$의 선형결합으로 표현 가능 (삼각함수의 덧셈 정리).

### 7.3 Learnable Position Embedding

단순히 학습 가능한 $PE \in \mathbb{R}^{S_{\max} \times d}$를 도입. BERT 등.

### 7.4 RoPE (Rotary Position Embedding)

위치 $m$의 쿼리/키를 $2$차원씩 쌍지어 회전 변환:

$$
R_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
$$

$$
q'_m = R_m q_m, \quad k'_n = R_n k_n
$$

그러면

$$
\langle q'_m, k'_n \rangle = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n
$$

**상대 위치**만 남는다! LLaMA, GPT-NeoX 등에서 사용 (자세한 내용은 7.5절).

---

## 8. 선형 Attention

### 8.1 Softmax 없애기

$$
\text{Attn} = \text{softmax}(QK^T) V \quad \to \quad \phi(Q) (\phi(K)^T V)
$$

$\phi$는 적절한 특징 함수. 이 경우

$$
\phi(K)^T V \in \mathbb{R}^{d' \times d_v}
$$

미리 계산 가능 ($O(Sd'd_v)$), 그 후 $\phi(Q)$와 곱해 $O(Sd'd_v)$. **총 $O(Sd'd_v)$**로 감소 ($O(S^2 d)$ 대비).

### 8.2 Performer (FAVOR+)

$$
\exp(q^T k) = \mathbb{E}_{\omega}[\phi_\omega(q)^T \phi_\omega(k)]
$$

으로 랜덤 피처를 사용해 softmax를 **편향되지 않게(unbiased)** 근사.

### 8.3 Kernel Attention

$\phi$를 명시적 커널의 특징 맵으로 선택 (RBF, polynomial). RKHS 해석.

---

## 9. Causal Masking

### 9.1 자기회귀 생성 (Autoregressive Generation)

GPT 같은 언어모델은 **이전 토큰만** 보고 다음 토큰을 예측. 따라서 어텐션을 하삼각으로 제한:

$$
A_{ij} = \begin{cases} \frac{\exp(s_{ij})}{\sum_{k \le i} \exp(s_{ik})} & j \le i \\ 0 & j > i \end{cases}
$$

### 9.2 구현

Mask 행렬 $M$: 상삼각 부분을 $-\infty$로:

$$
A = \text{softmax}(S + M)
$$

$M_{ij} = 0$ if $j \le i$, $M_{ij} = -\infty$ if $j > i$.

### 9.3 Attention의 "미래 보기" 방지

훈련 시 **teacher forcing**과 causal mask로 평행 훈련. 추론 시는 순차 생성 (KV cache).

---

## 10. Python 실험

### 10.1 기본 Attention

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

np.random.seed(0)
S, d = 8, 16
X = np.random.randn(S, d)

W_Q = np.random.randn(d, d) / np.sqrt(d)
W_K = np.random.randn(d, d) / np.sqrt(d)
W_V = np.random.randn(d, d) / np.sqrt(d)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = Q @ K.T / np.sqrt(d)
A = softmax(scores, axis=-1)
O = A @ V

print(f"Attention 행렬 shape: {A.shape}")
print(f"행마다 합이 1: {np.allclose(A.sum(axis=-1), 1)}")
print(f"출력 shape: {O.shape}")
```

### 10.2 스케일링 효과 확인

```python
import numpy as np

np.random.seed(0)
S = 20
for d in [4, 64, 1024]:
    Q = np.random.randn(S, d)
    K = np.random.randn(S, d)
    
    unscaled = Q @ K.T
    scaled = Q @ K.T / np.sqrt(d)
    
    print(f"d={d:4d}: unscaled std={unscaled.std():.2f}, scaled std={scaled.std():.2f}")
```

### 10.3 Attention의 특잇값 분포 (저랭크성)

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

np.random.seed(0)
S, d = 128, 64

# 랜덤 어텐션
Q = np.random.randn(S, d)
K = np.random.randn(S, d)
A = softmax(Q @ K.T / np.sqrt(d), axis=-1)

s = np.linalg.svd(A, compute_uv=False)
print(f"상위 5개 특잇값: {s[:5]}")
print(f"하위 5개 특잇값: {s[-5:]}")
print(f"유효 랭크 (90% 에너지): {np.argmax(np.cumsum(s**2) / np.sum(s**2) > 0.9) + 1}")
```

### 10.4 Causal Mask

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

np.random.seed(0)
S, d = 6, 8
Q = np.random.randn(S, d)
K = np.random.randn(S, d)
V = np.random.randn(S, d)

# Mask
mask = np.triu(np.full((S, S), -np.inf), k=1)
scores = Q @ K.T / np.sqrt(d) + mask
A = softmax(scores, axis=-1)

print("Causal mask 적용 후 Attention 행렬 (하삼각):")
print(np.round(A, 3))
```

### 10.5 Multi-head의 부분공간 다양성

```python
import numpy as np

np.random.seed(0)
d, h = 64, 8
d_h = d // h

# 각 head의 QK 투영 행렬들의 부분공간
W_Q_multi = np.random.randn(h, d, d_h)
W_K_multi = np.random.randn(h, d, d_h)

# 각 head의 QK^T 출력 부분공간의 기저
for i in range(h):
    combined = np.hstack([W_Q_multi[i], W_K_multi[i]])
    rank = np.linalg.matrix_rank(combined)
    print(f"Head {i}: combined rank = {rank}, 부분공간 차원 ≤ {2 * d_h}")

# 모든 head의 부분공간 합집합
all_Q = W_Q_multi.reshape(h, d, d_h).transpose(1, 0, 2).reshape(d, -1)
print(f"\n모든 Q 투영의 합집합 rank: {np.linalg.matrix_rank(all_Q)} (최대 {min(all_Q.shape)})")
```

### 10.6 선형 Attention

```python
import numpy as np

def linear_attention(Q, K, V):
    """softmax 없이"""
    # phi: elu + 1 (기본 선택)
    phi = lambda x: np.where(x > 0, x + 1, np.exp(x))
    Q_phi = phi(Q)
    K_phi = phi(K)
    # K^T V를 먼저 계산
    KV = K_phi.T @ V  # (d, d_v)
    numerator = Q_phi @ KV
    # 정규화
    Z = Q_phi @ K_phi.sum(axis=0)[:, None]
    return numerator / (Z + 1e-6)

def softmax_attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    scores = scores - scores.max(axis=-1, keepdims=True)
    A = np.exp(scores)
    A = A / A.sum(axis=-1, keepdims=True)
    return A @ V

np.random.seed(0)
S, d = 100, 32
Q = np.random.randn(S, d)
K = np.random.randn(S, d)
V = np.random.randn(S, d)

O_linear = linear_attention(Q, K, V)
O_soft = softmax_attention(Q, K, V)

# 복잡도 비교: S가 커질수록 linear이 유리
print(f"Softmax Attention: O(S^2 d) = O({S*S*d})")
print(f"Linear Attention:  O(S d^2) = O({S*d*d})")
print(f"출력 평균 차이: {np.linalg.norm(O_linear - O_soft) / np.linalg.norm(O_soft):.3f}")
```

---

## 11. 요약 및 다음 절 예고

### 핵심 공식

| 요소 | 공식 |
|---|---|
| Attention | $\text{softmax}(QK^T / \sqrt{d_k}) V$ |
| Multi-Head | $\text{concat}(\text{head}_1, \ldots) W^O$ |
| Causal Mask | 상삼각 = $-\infty$ |
| RoPE | $\langle R_m q, R_n k\rangle = q^T R_{n-m} k$ |
| Linear Attn | $\phi(Q)(\phi(K)^T V)$ |

### 한 줄 요약

> **Attention은 쿼리-키 내적으로 확률 분포를 만들어 값을 가중 평균하는 선형대수 절차이다.**

### 다음 절 예고

다음은 **역전파와 Vector-Jacobian Product**. Attention의 그래디언트가 어떻게 계산되는지, 그리고 왜 $O(N^2)$ 메모리가 필요한지를 선형대수적으로 본다.

---

[◀ Ch6 마무리](../ch6-tensor/05-nn-weight-tensor.md) | [📚 README](../README.md) | [02. 역전파 ▶](./02-backpropagation.md)
