# 제2회 ETRI 휴먼이해 인공지능 논문경진대회

## 팀명: 이모저모

## 연구 요약
* 대화 중 자동 감정 인식 기술은 의료계와 산업계의 다양한 분야에 적용되며 중요한 역할을 하고 있음
* 고성능의 자동화 된 **멀티모달 데이터 기반 감정 인식 기술 구현**을 위해서는 **대규모의 학습 데이터가 필요** 함
* 그러나 이러한 데이터는 **민감 정보가 담긴 경우가 많아 직접적인 공유 보다는 모델 파라미터만 공유하는 방식**이 널리 쓰임
* 모델 파라미터만을 전달 받아 새로운 태스크 데이터로 미세조정 학습(fine-tuning)을 하는 경우 **과거 태스크 데이터에 대한 예측 성능이 떨어지는 파괴적 망각(catastrophic forgetting)이 발생**
* 따라서 이 연구에서는 **태스크별 아답터를 사용하여 파괴적 망각을 방지할 수 있는 멀티모달 감정 인식 모델**을 구현하였음

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
* 데이터
  * K. J. Noh and H. Jeong, “KEMDy19,” https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR 
  * K. J. Noh and H. Jeong, “KEMDy20,” https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR 
  
## 사용 설명서
### 실험 환경
* 파이썬 버전 3.8
* pytorch==1.13.1
* torchaudio==0.13.1
* install dependencies `pip install -r requirements.txt`

### 모델 학습
#### continual_learning_init.py
| argument | description |
|---|---|
| --task | 학습 task |
| --exp_name | 실험 이름 |
| --seed | seed값 |
| --num_fold | k-fold 중에서 사용할 fold 숫자 |
| --k_fold | k-fold의 fold 개수 (k=5) |
| --lr | learning rate |
| --batch_size | Batch size `32` |
| --num_epochs | epoch `40` |
| --max_text_len | 최대 text token 길이 |
| --max_seq_len | 최대 speech token 길이 |
| --cpu | cpu 동작 여부 |

```
# Continual learning의 선행 task를 학습하기 위한 학습 스크립트 실행
# 이 연구에서는 아래 두 가지 실험 세팅으로 continual learning을 수행 함
# 1) KEMDy19 -> KEMDy20 실험을 위해, KEMDy19를 학습
# 2) KEMDy20 -> KEMDy19 실험을 위해, KEMDy20을 학습
bash run-init.sh
```
#### finetune
| argument | description |
|---|---|
| --task | 학습 task |
| --exp_name | 실험 이름 |
| --seed | seed값 |
| --num_fold | k-fold 중에서 사용할 fold 숫자 |
| --k_fold | k-fold의 fold 개수 (k=5) |
| --checkpoint | 
| --lr | learning rate |
| --batch_size | Batch size `32` |
| --num_epochs | epoch `40` |
| --max_text_len | 최대 text token 길이 |
| --max_seq_len | 최대 speech token 길이 |
| --cpu | cpu 동작 여부 |
```
bash run-finetune.sh
```
#### ewc
```
bash run-ewc.sh
```
#### adapter
```
bash run-adapter.sh
```
#### without pretraining
```
bash run-adapter-wo-pretraining.sh
```

### 성능 비교

|   | task_a(original) | finetune | ewc | adapter | adapter_wo_pretrain |
|---|---|---|---|---|---|
| **KEMDy19** | 0.70389 | 0.524502 | 0.599862 | 0.681204 | 0.510747 |
| **KEMDy20** | 0.87122 | 0.765711 | 0.761987 | 0.847727 | 0.833978 |

#### 파괴적 망각 방지 성능 비교
<img src="https://raw.githubusercontent.com/WoongheeLee/continual_erc_etri/master/figures/fig2.png" width="700"/>

#### 사전학습 여부에 따른 혼동 행렬
<img src="https://raw.githubusercontent.com/WoongheeLee/continual_erc_etri/master/figures/fig3.png" width="700"/>
