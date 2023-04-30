# 제2회 ETRI 휴먼이해 인공지능 논문경진대회

## 팀명: 이모저모 😂🍉

## 연구 요약
* 대화 중 자동 감정 인식 기술은 의료계와 산업계의 다양한 분야에 적용되며 중요한 역할을 하고 있음
* 고성능의 자동화 된 **멀티모달 데이터 기반 감정 인식 기술 구현**을 위해서는 **대규모의 학습 데이터가 필요** 함
* 그러나 이러한 데이터는 **민감 정보가 담긴 경우가 많아 직접적인 공유 보다는 모델 파라미터만 공유하는 방식**이 널리 쓰임
* 모델 파라미터만을 전달 받아 새로운 태스크 데이터로 미세조정 학습(fine-tuning)을 하는 경우 **과거 태스크 데이터에 대한 예측 성능이 떨어지는 파괴적 망각(catastrophic forgetting)이 발생**
* 따라서 이 연구에서는 **태스크별 아답터를 사용하여 파괴적 망각을 방지할 수 있는 멀티모달 감정 인식 모델**을 구현하였음
## 구현 방법
* 각각의 모달리티와 태스크의 특징 벡터를 학습하는 아답터를 사용하였음
* 멀티모달 분류 모델이 아래 그림의 왼쪽 (a)와 같이 주어짐
* 텍스트 모달리티의 태스크 별 특징 학습을 위한 아답터 구조는 그림 오른쪽의 (b), 음성 모달리티의 태스크 별 특징 학습을 위한 아답터 구조는 그림 오른쪽의 (c)와 같음
* 사전학습 된 BERT와 Wav2Vec2를 이용하여 아답터 레이어들 (a)와 (b)만 태스크별 학습을 통해 파괴적 망각을 완화 함
<p align="center">
  <img src="https://raw.githubusercontent.com/WoongheeLee/continual_erc_etri/master/figures/fig1.png" height="550"/>
</p>

## 디렉토리 구조
* KEMDy20은 v1.1
```
├── README.md
├── data
│   ├── KEMDy19
│   └── KEMDy20
├── figures
├── models
├── outputs
│   ├── adapter
│   │   ├── fold_0
│   │   │   ├── KEMDy19
│   │   │   │   └── best_model.pt
│   │   │   └── KEMDy20
│   │   │       └── best_model.pt
│   │   ├── fold_1
│   │   ├── fold_2
│   │   ├── fold_3
│   │   └── fold_4
│   ├── adapter_wo_pretraining
│   ├── ewc
│   ├── finetune
│   └── task_a
└── utils
```
* 데이터 (`/data/`)
  * K. J. Noh and H. Jeong, “KEMDy19,” https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR 
  * K. J. Noh and H. Jeong, “KEMDy20,” https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR 
* 체크포인트 (`/outputs/`)
  * [링크](https://hyu-my.sharepoint.com/:f:/g/personal/onnoo_hanyang_ac_kr/EoevacD34iBOsz7w2J3bMqQBoSOIfZAN5tD6vqOTRs3NTw?e=wacat9)

## 사용 설명서
### 1) 실험 환경 :sparkling_heart:
* Python 3.8
* pytorch==1.13.1
* torchaudio==0.13.1
* install dependencies : `pip install -r requirements.txt`

### 2️) 스크립트 실행 순서 :tada:

```
bash run-init.sh
bash run-finetune.sh
bash run-ewc.sh
bash run-adapter.sh
bash run-adapter-wo-pretraining.sh
```

### 3) 성능 비교 :fire:

위 스크립트 실행 순서에 순서에 따라 학습이 완료 된 후 [result.ipynb](https://github.com/WoongheeLee/continual_erc_etri/blob/master/result.ipynb)에서 catastrophic forgetting 완화 성능을 확인 가능

|   | task_a(original) | finetune | ewc | adapter | adapter_wo_pretrain |
|---|---|---|---|---|---|
| **KEMDy19** | 0.70389 | 0.524502 | 0.599862 | 0.681204 | 0.510747 |
| **KEMDy20** | 0.87122 | 0.765711 | 0.761987 | 0.847727 | 0.833978 |

#### 파괴적 망각 방지 성능 비교
<img src="https://raw.githubusercontent.com/WoongheeLee/continual_erc_etri/master/figures/fig2.png" width="700"/>

#### 사전학습 여부에 따른 혼동 행렬
<img src="https://raw.githubusercontent.com/WoongheeLee/continual_erc_etri/master/figures/fig3.png" width="700"/>

## Reference
* Houlsby, Neil, et al. "Parameter-efficient transfer learning for NLP." International Conference on Machine Learning. *PMLR*, 2019.
* Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks." *Proceedings of the national academy of sciences* 114.13 (2017): 3521-3526.
