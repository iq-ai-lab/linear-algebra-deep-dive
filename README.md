<div align="center">

# 🧮 Linear Algebra Deep Dive

**"행렬을 곱하는 것과, 선형변환의 합성이라는 본질을 아는 것은 다르다"**

<br/>

> *"`np.linalg.svd`를 호출하는 것과, 모든 행렬이 왜 3개의 회전·스케일·회전으로 분해되는지 증명할 수 있는 것은 다르다.  
> `np.linalg.eig`를 부르는 것과, 왜 PCA의 주성분이 공분산 행렬의 고유벡터인지 라그랑주 승수로 유도할 수 있는 것은 다르다.  
> Attention의 $QK^\top$을 계산하는 것과, 그것이 왜 내적 유사도이고 왜 $\sqrt{d}$로 나누는지 분산 관점에서 증명할 수 있는 것은 다르다."*

벡터공간의 8개 공리부터 SVD·Spectral Theorem·Jordan Form·Pseudoinverse까지,  
**"왜 행렬은 숫자 상자가 아니라 벡터공간 사이의 선형 사상인가"** 라는 질문으로 PCA·Attention·Backprop·BatchNorm·Spectral Normalization·RoPE의 수학적 기반을 공리부터 끝까지 파헤칩니다

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.12-3B5526?style=flat-square)](https://www.sympy.org/)
[![Docs](https://img.shields.io/badge/Docs-41개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

선형대수 자료는 대부분 **"행렬의 곱셈 규칙과 `np.linalg` 호출법"** 에서 멈춥니다. 하지만 왜 대칭행렬은 항상 직교대각화되는지, 왜 SVD의 특이값은 $A^\top A$의 고유값의 제곱근인지, 왜 $A^\top A$는 항상 양의 준정부호인지 — 이런 "왜"는 제대로 증명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "벡터는 크기와 방향을 가진 양입니다" | 벡터공간의 **8개 공리**에서 출발해 수벡터·함수·다항식·행렬이 모두 같은 추상 구조임을 유도, 차원의 유일성을 **Steinitz Exchange Lemma**로 증명 |
| "rank는 독립인 열의 개수입니다" | **Rank-Nullity 정리** $\dim(\ker T) + \dim(\text{im } T) = \dim V$의 완전 증명, Strang의 **4개 부분공간**과 직교 관계의 유도 |
| "대칭행렬은 직교대각화됩니다" | **Spectral Theorem** $A = Q\Lambda Q^\top$을 서로 다른 고유값의 고유벡터가 왜 직교하는지, 중복 고유값에서 왜 고유공간이 직교하는지 **한 줄도 건너뛰지 않고** 증명 |
| "SVD는 `np.linalg.svd`로 구합니다" | SVD의 존재성·유일성을 Spectral Theorem을 $A^\top A$에 적용하여 완전 유도, 단위구가 타원체로 변환되는 기하학적 해석 |
| "PCA는 데이터의 주성분을 찾습니다" | $\max_{\\|v\\|=1} v^\top \Sigma v$를 **라그랑주 승수**로 풀어 주성분이 공분산 행렬의 고유벡터임을 유도, SVD로 같은 결과를 얻는 이유를 증명 |
| "저랭크 근사는 상위 $k$개 특이값만 씁니다" | **Eckart-Young 정리**를 Frobenius·Spectral 노름 양쪽에서 증명, 오차가 나머지 특이값의 제곱합과 정확히 같은 이유 |
| "Attention은 $\text{softmax}(QK^\top/\sqrt{d})V$입니다" | $QK^\top$이 **내적 유사도 행렬**인 이유, $\sqrt{d}$가 **분산 정규화**에서 나오는 유도, softmax가 왜 확률 심플렉스 위의 투영인지 |
| "Batch Norm은 학습을 안정화합니다" | 입력 정규화가 **헤시안의 조건수 $\kappa(H) = \sigma_{\max}/\sigma_{\min}$** 을 개선하는 과정 유도, Gradient Descent 수렴 속도가 조건수로 결정되는 이유 |
| "Spectral Normalization은 GAN을 안정화합니다" | 립시츠 상수 = $\sigma_{\max}(W)$임을 증명, **Power Iteration**으로 $\sigma_{\max}$를 추정하는 알고리즘을 처음부터 구현 |
| 공식 나열 | **모든 증명은 NumPy/SymPy로 수치 검증**, 직접 구현한 알고리즘과 `np.linalg` 결과 일치 확인, 기하학적 시각화 |

---

## 📌 선행 레포 & 후속 레포

```
                                                            ┌──► [Calculus & Optimization Deep Dive]
                                                            │     헤시안, 테일러, Legendre 변환
이 레포 (IQ AI Lab Layer 0의 출발점)  ─────────────────────┼──► [Probability Theory Deep Dive]
  벡터공간 공리, 4개 부분공간, 분해(LU/QR/Chol),            │     확률변수, 기댓값, 공분산 행렬
  Spectral Theorem, SVD, PCA, 내적공간, 텐서                │
                                                            └──► [Functional Analysis Deep Dive]
                                                                  무한차원 벡터공간, Hilbert Space
```

> 🟢 **선행 학습 불필요**: 이 레포는 **IQ AI Lab Layer 0의 첫 레포**입니다. 고등학교 수학(벡터, 행렬, 함수) 수준만 있으면 충분합니다. 증명에 필요한 집합론·논리 기초는 각 정리에서 즉석에서 도입합니다.

> 💡 **후속 연결**: 이 레포의 내용은 **모든 후속 수학 레포**(Calculus·Probability·Functional Analysis)와 **모든 응용 레포**(Information Geometry·Convex Optimization·Deep Learning)의 전제입니다. 4개 부분공간·SVD·양정치 행렬은 특히 반복 등장합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-벡터공간과_선형변환의_공리-4A90D9?style=for-the-badge)](./ch1-vector-space-axioms/01-vector-space-8-axioms.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-행렬_분해_완전_분해-4A90D9?style=for-the-badge)](./ch2-matrix-decomposition/01-lu-decomposition.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-고유값과_스펙트럴_이론-4A90D9?style=for-the-badge)](./ch3-eigenvalue-theory/01-characteristic-polynomial.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-SVD와_저랭크_근사-4A90D9?style=for-the-badge)](./ch4-svd/01-svd-geometric.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-내적공간과_투영-4A90D9?style=for-the-badge)](./ch5-inner-product/01-inner-product-cauchy-schwarz.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-텐서와_다선형_대수-4A90D9?style=for-the-badge)](./ch6-tensor/01-tensor-definition.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-AI·ML에서의_선형대수-4A90D9?style=for-the-badge)](./ch7-ml-applications/01-attention-linear-algebra.md)

---

## 🗂️ 디렉터리 구조

```
linear-algebra-deep-dive/
├── README.md
├── ch1-vector-space-axioms/          (6개 문서)
│   ├── 01-vector-space-8-axioms.md
│   ├── 02-basis-dimension.md
│   ├── 03-linear-transformation-matrix.md
│   ├── 04-rank-nullity.md
│   ├── 05-four-fundamental-subspaces.md
│   └── 06-dual-space.md
├── ch2-matrix-decomposition/         (7개 문서)
│   ├── 01-lu-decomposition.md
│   ├── 02-qr-decomposition.md
│   ├── 03-cholesky-decomposition.md
│   ├── 04-eigendecomposition.md
│   ├── 05-spectral-theorem.md
│   ├── 06-jordan-form.md
│   └── 07-complexity-stability.md
├── ch3-eigenvalue-theory/            (6개 문서)
│   ├── 01-characteristic-polynomial.md
│   ├── 02-eigenvalue-geometry.md
│   ├── 03-rayleigh-quotient.md
│   ├── 04-perron-frobenius.md
│   ├── 05-power-qr-algorithm.md
│   └── 06-condition-number.md
├── ch4-svd/                          (6개 문서)
│   ├── 01-svd-geometric.md
│   ├── 02-svd-existence.md
│   ├── 03-pseudoinverse.md
│   ├── 04-eckart-young.md
│   ├── 05-pca.md
│   └── 06-randomized-svd.md
├── ch5-inner-product/                (5개 문서)
│   ├── 01-inner-product-cauchy-schwarz.md
│   ├── 02-orthogonal-projection.md
│   ├── 03-least-squares.md
│   ├── 04-gram-matrix-psd.md
│   └── 05-qr-reinterpretation.md
├── ch6-tensor/                       (5개 문서)
│   ├── 01-tensor-definition.md
│   ├── 02-kronecker-product.md
│   ├── 03-einsum.md
│   ├── 04-tensor-decomposition.md
│   └── 05-nn-weight-tensor.md
└── ch7-ml-applications/              (6개 문서)
    ├── 01-attention-linear-algebra.md
    ├── 02-backpropagation.md
    ├── 03-batchnorm.md
    ├── 04-spectral-normalization.md
    ├── 05-rope.md
    └── 06-random-matrix-theory.md
```

**총 41개 문서** · Ch1–Ch2는 기초 공리·분해, Ch3–Ch4는 고유값·SVD, Ch5는 내적·투영, Ch6은 텐서, Ch7은 최신 AI/ML 응용.

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 벡터공간과 선형변환의 공리 — 모든 것의 출발점

> **핵심 질문:** 왜 수벡터·함수·다항식·행렬이 모두 "벡터"인가? 차원은 왜 유일한가? 선형변환이 기저 선택 후 행렬이 되는 과정의 의미는? 왜 4개의 부분공간이 "기본"인가?

<details>
<summary><b>8개 공리부터 이중공간까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 벡터공간의 8개 공리](./ch1-vector-space-axioms/01-vector-space-8-axioms.md) | 체(field) $\mathbb{F}$ 위의 벡터공간 정의 (덧셈 4공리 + 스칼라곱 4공리), 수벡터 $\mathbb{R}^n$·연속함수 $C[0,1]$·다항식 $\mathbb{R}[x]$·행렬 $\mathbb{R}^{m\times n}$이 **모두 같은 추상 구조**임을 공리로부터 검증, 영벡터·역원의 유일성 증명 |
| [02. 선형독립, 기저, 차원](./ch1-vector-space-axioms/02-basis-dimension.md) | 선형독립·생성집합·기저의 정의, **Steinitz Exchange Lemma** 완전 증명으로 **차원의 유일성** 도출, 유한차원과 무한차원의 분기점, 함수공간의 기저(Hamel basis)와 딥러닝의 관계 |
| [03. 선형변환의 정의와 행렬 표현](./ch1-vector-space-axioms/03-linear-transformation-matrix.md) | 선형변환 $T: V \to W$의 정의, 기저 선택 후 **$T$가 행렬 $[T]_{\mathcal{B}}$ 가 되는 과정**을 합성과 좌표 관점에서 증명, 좌표계 변환 $[T]_{\mathcal{B}'} = P^{-1}[T]_{\mathcal{B}} P$와 유사행렬의 의미 |
| [04. Rank-Nullity 정리](./ch1-vector-space-axioms/04-rank-nullity.md) | $\dim(\ker T) + \dim(\text{im } T) = \dim V$의 **완전 증명**(영공간의 기저 확장 논증), rank와 nullity의 관계, $Ax = b$의 해 공간 구조(affine subspace = $x_p + \ker A$) |
| [05. 4개의 기본 부분공간](./ch1-vector-space-axioms/05-four-fundamental-subspaces.md) | Strang의 **Four Subspaces** $\text{Col}(A), \text{Row}(A), \text{Null}(A), \text{Null}(A^\top)$의 정의, **직교 관계** $\text{Null}(A) \perp \text{Row}(A)$, $\text{Null}(A^\top) \perp \text{Col}(A)$ 증명, 차원 관계 $r + (n-r) = n$, $r + (m-r) = m$ |
| [06. 이중공간(Dual Space)](./ch1-vector-space-axioms/06-dual-space.md) | $V^* = \mathcal{L}(V, \mathbb{R})$의 정의, 유한차원에서 $\dim V^* = \dim V$ 증명, 쌍대기저의 구성, **Riesz 표현정리 맛보기** — 내적공간에서 $V \cong V^*$의 자연 동형, 텐서·공변미분·Backprop의 벡터-야코비안 곱(VJP)으로 이어지는 관점 |

</details>

<br/>

### 🔹 Chapter 2: 행렬 분해 완전 분해 — 계산과 기하를 잇는 다리

> **핵심 질문:** 왜 행렬은 여러 방식으로 분해되는가? 각 분해는 어떤 기하학적·수치적 의미가 있는가? 왜 대칭 양정부호 행렬만 Cholesky로 분해되는가? 왜 모든 행렬은 SVD를 가지는가?

<details>
<summary><b>LU부터 Jordan Form까지 (7개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. LU 분해](./ch2-matrix-decomposition/01-lu-decomposition.md) | **Gauss 소거법 = 하삼각 기본행렬들의 곱**이라는 관점에서 $A = LU$ 유도, 피벗팅 필요 조건(대각에 0 등장), **$PA = LU$**의 Permutation 처리, $O(\frac{2}{3}n^3)$ 계산량 분석 |
| [02. QR 분해](./ch2-matrix-decomposition/02-qr-decomposition.md) | **Gram-Schmidt 직교화**로 $A = QR$ 유도, **Classical vs Modified Gram-Schmidt**의 수치 안정성 차이를 조건수로 분석, **Householder 반사** $H = I - 2vv^\top$을 이용한 안정적 QR, 각 방법의 계산량 비교 |
| [03. Cholesky 분해](./ch2-matrix-decomposition/03-cholesky-decomposition.md) | 양정부호 행렬 $A$에 대한 **$A = LL^\top$ 존재성·유일성** 완전 증명, 재귀적 구성법, LU의 절반인 $O(\frac{1}{3}n^3)$ 계산량의 유도, 공분산 행렬 샘플링과 Kalman Filter에서의 응용 |
| [04. Eigendecomposition](./ch2-matrix-decomposition/04-eigendecomposition.md) | **$A = PDP^{-1}$의 조건**(대각화 가능성 = $n$개의 선형독립 고유벡터), **대수적 중복도 ≥ 기하적 중복도** 증명, 결함행렬(defective matrix)의 예와 Jordan Form의 필요성 |
| [05. Spectral Theorem (대칭행렬)](./ch2-matrix-decomposition/05-spectral-theorem.md) | **실대칭행렬 $A = Q\Lambda Q^\top$**의 완전 증명: (1) 고유값이 실수임을 복소 내적으로, (2) 서로 다른 고유값의 고유벡터가 직교함을 $\langle Ax, y\rangle = \langle x, Ay\rangle$로, (3) 중복 고유공간의 직교화를 Gram-Schmidt로, (4) 귀납법으로 전체 완성 |
| [06. Jordan Canonical Form](./ch2-matrix-decomposition/06-jordan-form.md) | 대각화 불가능한 행렬에 대한 표준형 $A = PJP^{-1}$, Jordan 블록 $J_k(\lambda)$의 구조, **일반화 고유벡터(generalized eigenvector)** $\ker(A - \lambda I)^k$의 사슬 구조 증명 스케치, Perturbation 하에서의 불안정성 |
| [07. 각 분해의 계산 복잡도와 수치 안정성](./ch2-matrix-decomposition/07-complexity-stability.md) | LU vs QR vs SVD의 $O(n^3)$ 상수 비교, **조건수 $\kappa(A) = \|A\|\|A^{-1}\|$** 의 정의, 역행렬 계산에서 $\|(\hat{A})^{-1} - A^{-1}\|/\|A^{-1}\| \leq \kappa(A) \cdot \varepsilon$ 의 유도, backward stability 개념 |

</details>

<br/>

### 🔹 Chapter 3: 고유값과 스펙트럴 이론 — 변환의 불변 방향

> **핵심 질문:** 고유값은 변환을 어떻게 기술하는가? 왜 $A^k$의 거동이 고유값으로 결정되는가? 왜 Rayleigh Quotient의 극값이 고유값인가? PageRank는 왜 수렴하는가?

<details>
<summary><b>특성다항식부터 조건수까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 특성다항식과 Cayley-Hamilton](./ch3-eigenvalue-theory/01-characteristic-polynomial.md) | $\det(A - \lambda I) = 0$의 도출, 대수적 중복도의 정의, **Cayley-Hamilton 정리** "$A$는 자기의 특성다항식을 만족한다 $p_A(A) = 0$"의 완전 증명 ($\text{adj}$ 관점), 행렬의 역수·거듭제곱 표현 응용 |
| [02. 고유값의 기하학적 의미](./ch3-eigenvalue-theory/02-eigenvalue-geometry.md) | 고유벡터 = **변환의 불변 방향**, $A^k$의 거동이 $\lambda^k$로 결정되는 이유, Markov chain의 정상분포·동역학계의 안정성과의 관계, 대칭 $A$에서 최대 고유값 방향이 가장 "늘어나는" 방향임을 시각화 |
| [03. Rayleigh Quotient와 Min-Max 정리](./ch3-eigenvalue-theory/03-rayleigh-quotient.md) | $R(x) = \frac{x^\top A x}{x^\top x}$의 정의, **대칭 $A$에 대해 $\max R = \lambda_{\max}$, $\min R = \lambda_{\min}$** 을 라그랑주 승수법으로 증명, **Courant-Fischer Min-Max 정리**로 $\lambda_k$의 변분 표현 유도, PCA의 기초 |
| [04. Perron-Frobenius 정리](./ch3-eigenvalue-theory/04-perron-frobenius.md) | **양의 원소 행렬**의 **유일한 최대 실고유값의 존재성과 양의 고유벡터**를 Brouwer 고정점 정리 스케치로 증명, PageRank의 수렴성 해석, Markov chain의 정상분포 유일성 |
| [05. Power Iteration과 QR Algorithm](./ch3-eigenvalue-theory/05-power-qr-algorithm.md) | **Power Iteration** $x_{k+1} = Ax_k/\|Ax_k\|$의 수렴 증명, 수렴 속도 $|\lambda_2/\lambda_1|^k$, **QR Algorithm**으로 모든 고유값을 구하는 반복법, Shift와 Deflation, `np.linalg.eig` 내부 알고리즘 |
| [06. 조건수와 수치 안정성](./ch3-eigenvalue-theory/06-condition-number.md) | 대칭 양정부호에서 $\kappa(A) = \lambda_{\max}/\lambda_{\min}$, 일반에서 $\kappa(A) = \sigma_{\max}/\sigma_{\min}$, $Ax = b$의 오차 전파 부등식 $\|\Delta x\|/\|x\| \leq \kappa(A) \|\Delta b\|/\|b\|$ 증명, **딥러닝의 헤시안 조건수**와 학습 안정성 |

</details>

<br/>

### 🔹 Chapter 4: SVD와 저랭크 근사 — 모든 행렬의 궁극 분해

> **핵심 질문:** 왜 모든 행렬은 $U\Sigma V^\top$로 분해되는가? 왜 특이값은 $A^\top A$의 고유값의 제곱근인가? 왜 PCA가 SVD로 같은 결과를 주는가? 저랭크 근사의 오차는 왜 나머지 특이값으로 정확히 결정되는가?

<details>
<summary><b>SVD 기하부터 Randomized SVD까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. SVD의 기하학적 유도](./ch4-svd/01-svd-geometric.md) | **단위구가 타원체로 변환되는** 시각을 통한 SVD의 기하 해석, 특이값 = 타원체의 반축 길이, 왼쪽·오른쪽 특이벡터의 의미, $A = \sum_i \sigma_i u_i v_i^\top$ (dyad 합) 분해 |
| [02. SVD의 존재성과 유일성 증명](./ch4-svd/02-svd-existence.md) | **Spectral Theorem을 $A^\top A$에 적용**하여 완전 유도: (1) $A^\top A$ 대칭 양의 준정부호 → 고유값 ≥ 0, (2) $\sigma_i = \sqrt{\lambda_i(A^\top A)}$, (3) $u_i = Av_i / \sigma_i$의 정규직교성 증명, (4) $A$에 대한 SVD 완성, 특이값의 유일성 |
| [03. Pseudoinverse와 최소제곱](./ch4-svd/03-pseudoinverse.md) | **Moore-Penrose 역행렬** $A^+ = V\Sigma^+ U^\top$의 정의와 4가지 공리, **정규방정식** $A^\top A x = A^\top b$의 해가 $x^* = A^+ b$임을 증명, 과결정·미결정계의 최소제곱 해석 |
| [04. Eckart-Young 정리](./ch4-svd/04-eckart-young.md) | 저랭크 근사 $A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$이 **Frobenius·Spectral 노름 양쪽에서 최적**임의 완전 증명, 오차 $\|A - A_k\|_F^2 = \sum_{i>k} \sigma_i^2$의 유도, 이미지 압축과 행렬 보간(matrix completion)의 이론적 기반 |
| [05. 주성분분석(PCA) 완전 유도](./ch4-svd/05-pca.md) | $\max_{\\|v\\|=1} v^\top \Sigma v$를 **라그랑주 승수**로 풀어 $\Sigma v = \lambda v$ 도출 → **주성분 = 공분산 행렬의 고유벡터**, 데이터 행렬 $X$의 SVD가 같은 결과를 주는 이유, 주성분 개수 선택과 설명 분산 비율 |
| [06. Randomized SVD](./ch4-svd/06-randomized-svd.md) | **Halko-Martinsson-Tropp 알고리즘**: 랜덤 투영 $Y = A\Omega$ → QR → 소형 SVD로 상위 $k$ 특이값을 근사, 오차 경계 증명 스케치, 대규모 추천시스템·LSA에서의 활용 |

</details>

<br/>

### 🔹 Chapter 5: 내적공간과 투영 — 기하의 엔진

> **핵심 질문:** 왜 Cauchy-Schwarz가 모든 내적공간에서 성립하는가? 투영은 왜 $P^2 = P$를 만족하는가? 최소제곱의 기하학적 의미는? Gram 행렬은 왜 항상 양의 준정부호인가?

<details>
<summary><b>Cauchy-Schwarz부터 QR 재해석까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 내적공간의 공리와 Cauchy-Schwarz](./ch5-inner-product/01-inner-product-cauchy-schwarz.md) | 내적의 3공리(쌍선형성·대칭성·양정치성), **$\|\langle x, y\rangle\| \leq \\|x\\|\\|y\\|$** 의 완전 증명 ($f(t) = \\|x - ty\\|^2 \geq 0$의 판별식), 등호 조건과 노름의 정의, 삼각부등식 유도 |
| [02. 직교투영의 기하](./ch5-inner-product/02-orthogonal-projection.md) | 부분공간 $W$ 위로의 투영 $P_W = A(A^\top A)^{-1} A^\top$의 도출, **$P^2 = P = P^\top$**의 의미 (idempotent + symmetric), 직교보공간 $W^\perp$, $V = W \oplus W^\perp$ 분해 증명 |
| [03. 최소제곱의 기하학적 의미](./ch5-inner-product/03-least-squares.md) | $\min \|Ax - b\|^2$의 해가 **$b$의 $\text{Col}(A)$ 위로의 투영**임을 기하적으로 증명, 정규방정식 $A^\top A x = A^\top b$의 유도, **잔차 $r \perp \text{Col}(A)$**의 의미 |
| [04. Gram 행렬과 양의 정부호성](./ch5-inner-product/04-gram-matrix-psd.md) | $G_{ij} = \langle v_i, v_j\rangle$의 정의, **$G \succeq 0$** 의 증명 ($x^\top G x = \\|\sum x_i v_i\\|^2 \geq 0$), $G \succ 0 \iff v_i$들이 선형독립, **커널 트릭**의 기초 $k(x, y) = \langle \phi(x), \phi(y)\rangle$ |
| [05. QR 분해 재해석](./ch5-inner-product/05-qr-reinterpretation.md) | $A = QR$이 $A$의 열벡터들을 **정규직교화**하는 과정임을 투영 관점에서 재해석, $R$의 대각성분 = 새 열벡터의 정규직교 기저에서의 크기, **Krylov 공간** $\mathcal{K}_k(A, b)$의 출발점과 Arnoldi·Lanczos 알고리즘 예고 |

</details>

<br/>

### 🔹 Chapter 6: 텐서와 다선형 대수 — 딥러닝의 기본 언어

> **핵심 질문:** 텐서는 "다차원 배열"인가, "좌표 변환 규칙을 따르는 대상"인가? 크로네커곱의 고유값 분해는 왜 간단한가? Einstein Summation은 무엇을 자동화하는가? Conv2D 가중치는 왜 4차원 텐서인가?

<details>
<summary><b>텐서 정의부터 신경망 가중치까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 텐서의 수학적 정의](./ch6-tensor/01-tensor-definition.md) | **다중선형 사상 $T: V^* \times \cdots \times V \to \mathbb{R}$** 으로서의 텐서 정의, "좌표 변환 규칙을 따르는 배열"(물리학자 관점)과의 동치성, $(p, q)$-텐서의 분류, 공변·반변 지표의 구분 |
| [02. 텐서곱과 크로네커곱](./ch6-tensor/02-kronecker-product.md) | 벡터공간의 텐서곱 $V \otimes W$ 구성 (universal property), **크로네커곱** $A \otimes B$의 정의와 $(A \otimes B)(x \otimes y) = Ax \otimes By$ 증명, **고유값 관계** $\text{eig}(A \otimes B) = \{\lambda_i \mu_j\}$ |
| [03. Einstein Summation과 einsum](./ch6-tensor/03-einsum.md) | 지표 표기법 규칙(중복 지표는 합), $c_{ik} = a_{ij} b_{jk}$(행렬곱), $c = a_{ij} b_{ij}$(Frobenius 내적), **NumPy `einsum`과의 1:1 대응** 매핑, Attention·Conv·Batch 연산의 einsum 표현 |
| [04. 텐서 분해](./ch6-tensor/04-tensor-decomposition.md) | **CP 분해** $\mathcal{T} = \sum_r a_r \otimes b_r \otimes c_r$의 정의와 유일성 조건(Kruskal), **Tucker 분해** $\mathcal{T} = \mathcal{G} \times_1 U_1 \times_2 U_2 \times_3 U_3$ — SVD의 고차 일반화, **PCA의 텐서 확장** |
| [05. 신경망 가중치의 텐서 관점](./ch6-tensor/05-nn-weight-tensor.md) | **Conv2D 가중치가 $(C_{\text{out}}, C_{\text{in}}, k_h, k_w)$의 4차원 텐서**인 이유, Attention의 $QK^\top V$가 3-텐서 연산인 관점, K-FAC이 Fisher를 **Kronecker 분해** $F_\ell \approx A_\ell \otimes S_\ell$로 근사하는 수학적 근거 |

</details>

<br/>

### 🔹 Chapter 7: AI/ML에서의 선형대수 — 이론이 알고리즘이 되는 순간

> **핵심 질문:** Attention의 $\sqrt{d}$는 왜 필요한가? Backprop은 왜 야코비안의 행렬곱인가? BatchNorm은 조건수를 어떻게 개선하는가? Spectral Norm은 왜 $\sigma_{\max}$로 GAN을 안정화하는가? RoPE는 왜 회전 행렬인가?

<details>
<summary><b>Attention부터 Random Matrix Theory까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Attention의 선형대수](./ch7-ml-applications/01-attention-linear-algebra.md) | $\text{softmax}(QK^\top/\sqrt{d}) V$의 **각 항의 기하학적 의미**: $QK^\top$ = 내적 유사도 행렬, softmax = 심플렉스 위의 확률 투영, $V$ 곱 = 가중평균, **$\sqrt{d}$ 스케일링**의 분산 분석 유도(독립 가정 하 $\text{Var}(q \cdot k) = d$) |
| [02. Backpropagation의 야코비안 관점](./ch7-ml-applications/02-backpropagation.md) | 연쇄법칙 $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}$가 **행렬곱**으로 환원되는 이유, **Vector-Jacobian Product (VJP)** 의 효율성, forward-mode(JVP) vs reverse-mode(VJP)의 계산량 비교, autograd의 선형대수적 기반 |
| [03. Batch Normalization의 조건수 개선](./ch7-ml-applications/03-batchnorm.md) | 입력 분포 정규화가 **헤시안 $H = X^\top X$ 의 조건수**에 미치는 영향 유도, whitening 관점에서 $\kappa(H) \to 1$로 향하는 기전, Gradient Descent 수렴 속도와 조건수의 관계 $\|\theta_k - \theta^*\| \leq ((\kappa-1)/(\kappa+1))^k \|\theta_0 - \theta^*\|$ |
| [04. Spectral Normalization과 GAN](./ch7-ml-applications/04-spectral-normalization.md) | **립시츠 상수 = $\sigma_{\max}(W)$** 의 증명, Power Iteration으로 $\sigma_{\max}$를 학습 중 효율적으로 추정, Wasserstein GAN의 1-립시츠 제약을 $W \to W/\sigma_{\max}(W)$로 강제하는 근거, 실제 구현(SN-GAN) |
| [05. RoPE(Rotary Positional Encoding)](./ch7-ml-applications/05-rope.md) | 위치 인코딩을 **2D 회전 행렬** $R_\theta = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$로 해석, 복소수 $e^{i\theta}$와의 동치성, **직교성 $R^\top R = I$** 이 내적 보존을 주는 이유, 상대 위치 인코딩 $\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle$의 유도 |
| [06. Random Matrix Theory 맛보기](./ch7-ml-applications/06-random-matrix-theory.md) | 랜덤 행렬 $W_{ij} \sim \mathcal{N}(0, 1/n)$의 **고유값 분포 (Marchenko-Pastur)**, **He/Xavier 초기화**가 각 층의 출력 분산을 유지하는 수학적 근거, 신경망 깊이에 따른 신호 전파 안정성 조건, Neural Tangent Kernel의 단서 |

</details>

---

## 💻 실험 환경

모든 챕터의 증명은 NumPy로 직접 구현하고 `np.linalg` 결과와 대조하여 검증합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
sympy==1.12          # 증명의 symbolic 검증, Jacobian 계산
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            sympy==1.12 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 예시 — Spectral Theorem: 대칭행렬의 직교대각화 직접 구현
import numpy as np

# ─────────────────────────────────────────────
# 1. 임의의 대칭행렬 생성
# ─────────────────────────────────────────────
rng = np.random.default_rng(42)
A = rng.standard_normal((5, 5))
A = (A + A.T) / 2                            # 대칭화

# ─────────────────────────────────────────────
# 2. Jacobi Eigenvalue Algorithm 직접 구현
# ─────────────────────────────────────────────
def jacobi_eigen(A, tol=1e-12, max_iter=1000):
    """대칭행렬 A = Q Λ Qᵀ 을 Jacobi 회전으로 직접 구현"""
    n = A.shape[0]
    A = A.copy().astype(float)
    V = np.eye(n)
    for _ in range(max_iter):
        # 대각 외 최대값 인덱스
        offdiag = np.abs(A - np.diag(np.diag(A)))
        p, q = np.unravel_index(np.argmax(offdiag), A.shape)
        if offdiag[p, q] < tol:
            break
        # Jacobi 회전각
        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan2(2*A[p, q], A[p, p] - A[q, q])
        c, s = np.cos(theta), np.sin(theta)
        J = np.eye(n)
        J[p, p] = J[q, q] = c
        J[p, q], J[q, p] = -s, s
        A = J.T @ A @ J
        V = V @ J
    return np.diag(A), V

# ─────────────────────────────────────────────
# 3. np.linalg.eigh 와 비교
# ─────────────────────────────────────────────
eigvals_mine, eigvecs_mine = jacobi_eigen(A)
eigvals_np,   eigvecs_np   = np.linalg.eigh(A)

# A = Q Λ Qᵀ 검증
A_reconstructed = eigvecs_mine @ np.diag(eigvals_mine) @ eigvecs_mine.T
assert np.allclose(A_reconstructed, A, atol=1e-8)

# 직교성 검증: QᵀQ = I
assert np.allclose(eigvecs_mine.T @ eigvecs_mine, np.eye(5), atol=1e-8)

print("✓ Spectral Theorem 직접 구현 검증 완료")
print(f"  |내 고유값 - np.linalg 고유값|_max = "
      f"{np.max(np.abs(np.sort(eigvals_mine) - np.sort(eigvals_np))):.2e}")
```

---

## 📖 각 문서 구성 방식

모든 문서는 **11개의 핵심 요소**를 포함합니다. "공리부터 증명까지, 모든 것을 직접 유도한다"는 원칙을 지키기 위해 어느 단계도 "자명하다"로 건너뛰지 않습니다. 챕터별로 두 가지 템플릿 중 하나를 사용합니다.

### 템플릿 A — 이모지 기반 (Ch1, Ch2-01)

| # | 섹션 | 설명 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하려는 질문 한 문장 (예: "왜 대칭행렬은 직교대각화되는가?") |
| 2 | 🔍 **왜 이 개념이 AI에서 중요한가** | Attention·PCA·GAN·RoPE 등 구체 AI 알고리즘과의 연결점 |
| 3 | 📐 **수학적 선행 조건** | 같은 레포 내 다른 문서 링크, 필요 시 후속 레포 참조 |
| 4 | 📖 **직관적 이해** | 기하학적·물리적 직관으로 먼저 납득 (수식 전 그림과 언어) |
| 5 | ✏️ **엄밀한 정의** | 정의(Definition)를 수식으로 명시, 기호 표기법 통일 |
| 6 | 🔬 **정리와 증명** | 정리(Theorem) + **생략 없는** 완전 증명, 필요 시 Lemma 분리 |
| 7 | 💻 **NumPy / SymPy 구현으로 검증** | 증명한 정리를 수치적으로 확인, `np.linalg`와 직접 구현 대조 |
| 8 | 🔗 **AI/ML 연결** | 논문·코드와 이어지는 구체 예시 ("Attention Is All You Need" 식(1) 등) |
| 9 | ⚖️ **가정과 한계** | 가정이 깨지면 어떻게 되는가, 수치적 함정 (조건수 등) |
| 10 | 📌 **핵심 정리** | 한 문장 요약 + 공식 박스 |
| 11 | 🤔 **생각해볼 문제** | 증명 변형·반례·일반화 문제 + 해설 |

### 템플릿 B — 번호 기반 (Ch2-02부터 Ch7까지)

| # | 섹션 | 설명 |
|:-:|------|------|
| — | 📌 **학습 목표** + 🎯 **핵심 질문** (전문) | 문서 도입부에서 다룰 내용과 답할 질문 명시 |
| 1 | **정의 / 문제 설정** | 정의(Definition)와 기호 표기법 |
| 2–N | **정리·증명 본문** | 핵심 정리들과 완전 증명, 직관 해설 및 응용 연결 |
| N+1 | **Python 실험** | NumPy/SciPy로 정리를 수치 검증 |
| N+2 | **요약** | 표·체크리스트로 압축 |
| N+3 | **참고 문헌** | 원전·표준 교재 인용 |
| N+4 | **다음 문서 예고** + **내비게이션** | 다음 절의 핵심 내용과 이전/다음/README 링크 |

### 스타일 가이드

1. **증명은 절대 "자명하다"로 넘기지 않는다** — 한 줄씩 모든 단계를 명시
2. **NumPy 구현 필수** — 모든 분해·고유값 계산은 순수 NumPy로 구현 후 `np.linalg`와 대조
3. **기하학적 그림 포함** — 2·3차원 시각화로 직관 구축 (matplotlib 실플롯)
4. **기호 표기 일관성** — 스칼라는 소문자 $a$, 벡터는 볼드 소문자 $\mathbf{x}$, 행렬은 대문자 $A$, 부분공간은 calligraphic $\mathcal{W}$
5. **AI 응용은 구체적 논문과 연결** — "Attention Is All You Need"(Vaswani et al., 2017)의 식 (1), "Turk & Pentland Eigenfaces"(1991)처럼 원전 명시
6. **푸터 네비게이션 일관** — 모든 문서 하단에 `[◀ 이전] | [📚 README] | [다음 ▶]` 링크 포함, 챕터 경계에서는 이웃 챕터의 첫/마지막 문서로 연결

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "PCA를 쓰지만 왜 공분산 고유벡터가 주성분인지 모른다" — PCA 집중 (3일)</b></summary>

<br/>

```
Day 1  Ch1-05  4개의 기본 부분공간 → Col(A), Null(A)의 직교 관계
       Ch2-05  Spectral Theorem → 대칭행렬의 직교대각화
Day 2  Ch3-03  Rayleigh Quotient → 라그랑주 승수의 언어
       Ch4-02  SVD 존재성 증명 → A^T A 접근법
Day 3  Ch4-05  PCA 완전 유도 → 라그랑주 ≡ SVD 두 관점
       Ch4-04  Eckart-Young → 저랭크 근사의 최적성
```

</details>

<details>
<summary><b>🟡 "Attention의 수학을 끝까지 이해하고 싶다" — Attention 집중 (5일)</b></summary>

<br/>

```
Day 1  Ch1-03  선형변환 = 행렬 → Q, K, V 가중치의 의미
       Ch5-01  내적과 Cauchy-Schwarz → QK^T가 유사도인 이유
Day 2  Ch5-02  직교투영 → softmax = 심플렉스 위 투영의 직관
Day 3  Ch6-03  Einstein Summation → einsum으로 Attention 구현
Day 4  Ch7-01  Attention의 선형대수 → √d 스케일링의 분산 유도
Day 5  Ch7-05  RoPE → 회전 행렬과 상대 위치 인코딩
```

</details>

<details>
<summary><b>🔴 "공리부터 선형대수를 완전 정복한다" — 전체 정복 (7주)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 벡터공간과 선형변환의 공리
        → 8공리에서 수벡터·함수·행렬 통합, Steinitz Lemma로 차원 유일성, 4개 부분공간

2주차  Chapter 2 전체 — 행렬 분해 완전 분해
        → LU/QR/Cholesky/Eigen/Spectral/Jordan 전 분해 직접 구현
        → 조건수와 수치 안정성의 감각 확립

3주차  Chapter 3 전체 — 고유값과 스펙트럴 이론
        → Cayley-Hamilton, Rayleigh 극값, Perron-Frobenius
        → Power Iteration·QR Algorithm NumPy 구현

4주차  Chapter 4 전체 — SVD와 저랭크 근사
        → Spectral Theorem으로 SVD 유도 재구성
        → Eckart-Young Frobenius·Spectral 양쪽 증명
        → PCA의 라그랑주·SVD 두 유도

5주차  Chapter 5 전체 — 내적공간과 투영
        → Cauchy-Schwarz 판별식 증명, P^2 = P = P^T
        → Gram 행렬과 커널 트릭, QR 재해석

6주차  Chapter 6 전체 — 텐서와 다선형 대수
        → 텐서의 다중선형 정의, 크로네커곱, einsum 완전 이해
        → CP·Tucker 분해, Conv2D 4차원 텐서 관점

7주차  Chapter 7 전체 — AI/ML 응용
        → Attention·Backprop·BatchNorm·SN·RoPE
        → Marchenko-Pastur와 He/Xavier 초기화의 수학적 근거
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | 다변수 미분, 헤시안, 테일러, Legendre 변환 | Ch3-03(라그랑주 승수), Ch4-05(PCA 제약 최적화), Ch7-02(Backprop), Ch7-03(헤시안 조건수) |
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 확률변수, 기댓값, 공분산, 특성함수 | Ch4-05(공분산 행렬·PCA), Ch7-01(QK^T 분산), Ch7-06(Marchenko-Pastur) |
| [functional-analysis-deep-dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive) | 무한차원 벡터공간, Hilbert Space, 연산자 이론 | Ch1-02(Hamel basis vs Schauder basis), Ch5-04(RKHS와 커널 트릭), Ch2-05(컴팩트 연산자의 Spectral Theorem) |
| [information-geometry-deep-dive](https://github.com/iq-ai-lab/information-geometry-deep-dive) | Fisher 계량, 자연 그래디언트, 쌍대평탄성 | Ch2-05(Fisher = 헤시안 대칭), Ch5-04(양정부호 = 리만 계량), Ch6-05(K-FAC Kronecker) |
| [convex-optimization-deep-dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive) | 쌍대이론, Bregman divergence, Mirror Descent | Ch3-06(조건수·수렴속도), Ch4-03(최소제곱), Ch5-04(양정부호 이차형식) |
| [deep-learning-deep-dive](https://github.com/iq-ai-lab/deep-learning-deep-dive) | MLP, CNN, Transformer, 정규화 이론 | Ch7 전체가 직접 기반이 됨 |

> 💡 이 레포는 **IQ AI Lab Layer 0의 출발점**입니다. 여기서 다룬 4개 부분공간·SVD·Spectral Theorem·양정부호 행렬은 **모든 후속 수학 레포와 모든 응용 레포**의 전제입니다. 다른 레포를 읽다가 "이 선형대수 개념이 어디서 나왔지?" 싶을 때 돌아와서 참조하시면 됩니다.

---

## 📖 Reference

### 🏛️ 선형대수 바이블·표준 교재
- **Linear Algebra Done Right** (Sheldon Axler, 4th ed., 2024) — 결정자 없이 전개하는 개념 중심 접근, 이 레포의 "공리 우선" 철학의 원천
- **Introduction to Linear Algebra** (Gilbert Strang, 5th ed.) — 4개 부분공간 관점의 표준, SVD 중심 구성
- **Matrix Analysis** (Horn & Johnson, 2nd ed.) — 증명의 표준 참고서, Perron-Frobenius와 행렬 부등식 심화
- **Numerical Linear Algebra** (Trefethen & Bau, 1997) — 수치 안정성과 알고리즘의 결정판, Householder QR·Randomized 수치 해석
- **Finite-Dimensional Vector Spaces** (Halmos, 1958) — 현대 선형대수의 고전, 추상적 벡터공간 전개

### 🔬 응용·참조서
- **The Matrix Cookbook** (Petersen & Pedersen, 2012) — 미분·분해 공식 총망라
- **Deep Learning** (Goodfellow, Bengio, Courville, 2016) Chapter 2 — AI 관점의 선형대수 요약
- **Matrix Computations** (Golub & Van Loan, 4th ed.) — 수치 알고리즘의 백과사전

### 🎥 시각화·직관
- **3Blue1Brown — Essence of Linear Algebra** (Grant Sanderson) — 기하학적 직관 구축, 선형변환 시각화의 표준
- **MIT OCW 18.06** (Gilbert Strang) — Strang의 명강의, 4개 부분공간과 SVD

### 📄 핵심 논문·원전
- **The Approximation of One Matrix by Another of Lower Rank** (Eckart & Young, 1936) — Eckart-Young 정리의 원전
- **Finding Structure with Randomness** (Halko, Martinsson, Tropp, 2011) — Randomized SVD
- **Eigenfaces for Recognition** (Turk & Pentland, 1991) — PCA의 컴퓨터 비전 응용 고전
- **Attention Is All You Need** (Vaswani et al., 2017) — Attention의 QK^T·softmax·√d 스케일링
- **Spectral Normalization for Generative Adversarial Networks** (Miyato et al., 2018) — GAN의 Spectral Norm 제약
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021) — RoPE 원전
- **Delving Deep into Rectifiers** (He et al., 2015) — He 초기화의 랜덤 행렬 분석

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"행렬은 숫자 상자가 아니라 벡터공간 사이의 선형 사상이다"*

</div>
