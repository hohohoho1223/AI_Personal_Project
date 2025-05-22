# Using Potato Plant Diseases Data to Build CNN Modeling Comparison
[Date : 2025.05]

---

## 1. 서론

---

- 감자는 세계 주요 식량 작물 중 하나로, 특히 **역병(Late Blight, Phytophthora infestans)** 은 감자 생산량에 치명적인 영향을 미치는 대표적인 병해입니다. 특히 초기 역병과 후기 역병은 완전히 다른 질병이며 **초기 역병**은 잎의 작은 반점이나 모서리 변색 등으로 시작되어, 빠른 시간 내에 줄기 및 뿌리까지 전파된다. **후기 역병**은 이미 병이 확산된 상태에서 급격한 조직 괴사와 작물 고사 현상을 유발한다. 이 둘은 방제 시기 및 약제 선택이 다르기 때문에 **정확한 구분이 필수적이다**. 따라서 본 연구는 **감자의 초기 및 후기 역병 발생 시기의 정확한 분류 및 예측 모델**을 구축함으로써, 조기 방제 및 농가 생산성 향상에 기여하고자 한다.

## 2. 데이터셋 설명

---

- 이번 모델에서 사용한 데이터 셋은 Kaggle 사이트에 업로드 되어있는 “**Potato Plant Diseases Data”** 이다.
- 구글 드라이브에 데이터 셋을 업로드 후, 불러오는 식으로 진행하였다.    
    - 해당 데이터 셋은
        - 'Potato___Early_blight'
        - 'Potato___Late_blight'
        - 'Potato___healthy'
    
    으로 총 3개 클래스 이며, 각각 1000장 그리고 152장으로 구성 되어있다.
- 데이터양이 매우 적은 편이므로 **데이터 증강** 및 **데이터 분할**을 train : val : test = 8:1:1로 구성하여 train 학습에 좀더 집중하였다.

## 3. 모델 설명

---

- 이번 프로젝트에서 감자 식물 역병 데이터셋을 학습시키기위해 훈련한 모델은 총 5가지 모델로 ResNet50, ResNet18,  VGG16, MobileNet, GoogLeNet 등을 사용했다.
    - ResNet50
    - 조기중단(early stopping)을 적용한 모델학습 함수
  
    - 역시나 `ResNet50` 모델은 매우 높은 성능을 보여주었다.
   → ImageNet 대규모 데이터셋에서 사전학습된 가중치를 사용이 큰 이유이다.
        
    - 추가로 **경량화**를 염두하여 `ResNet18` 모델과 여러 모델들을 함께 모델학습을 진행하였다.
    
- ResNet50&18, VGG16 그리고 MobileNet 모델을 사용결과 사전학습된 모델이다 보니 성능이 매우 좋은 방면, 상대적으로 학습성능이 떨어진 **GoogLeNet 모델을 점차 강화학습을 진행**하는 방향으로 진행하였다.
- GoogLeNet 모델의 특징
    1. **`Auxiliary Classifiers (보조 분류기)`가 중간 레이어에 존재**
        - 학습 시 보조 손실로만 사용되며 **보조 분류기로 인한 기울기 소실 완화**
        - **학습 속도 및 안전성 향상 기대**
    2. **작은 모델 크기**로도 성능이 우수해서 **리소스가 제한된 환경**에 적합
    3. 학습 안정성을 위해 보조 분류기를 쓰는 점이 **소규모 데이터셋 학습에도 유리**
        
## 4. 실험 방법

---

### **GoogLeNet 모델 기반 점진적 성능 개선 전략**

- 기존 일반 GoogLeNet 모델에서의 성능이 낮은 이유는 다음과 같다.
    1. 보조 분류기(aux) 비활성화 : `aux_logit=False`
    2. `CrossEntropyLoss` 사용
- 따라서 보조분류기(aux) 활성화 및 파인튜닝을 진행하였다.
- 조기중단(early stopping)을 적용한 해당 트레이닝 함수를 설정하여 진행하였다.

---
  
### GoogLeNet 성능 개선

### 1. 1차 파인튜닝 `(use_aux=True, freeze_base=False)`
- aux1,aux2 를 출력 레이서 클래스 수 `num_classes` 에 맞춰 재정의 하였음
- **`freeze_base=False`** 덕분에 실제로는 이미 전체 파라미터가 `requires_grad=True`인 상태임   
    → 아무 것도 얼리지(freeze) 않았기 때문에, 전체 파라미터가 기본적으로 학습 대상이 된다는 뜻    
- 따라서 위 코드 함수는 유연하게 **전체 파인튜닝**도, **Gradual Unfreeze** 초기도 모두 커버할 수 있는 구조

### 2. 2차 파인튜닝 `(use_aux=True, freeze_base=True)`
- 한번에 두 모델 생성하여 두 전략을 동시에 진행하였다.
    - **Feature Extractor 방식**
        - **백본(Conv Layer)은 고정(freeze)** → `requires_grad = False`
        - **분류기(FC + aux1, aux2)만 학습**
        - 즉, **기존 pretrained weight는 그대로 유지**하면서 새로운 task에 맞는 **출력 레이어만 학습**
    - **Gradual Unfreeze**(점진적 파라미터 해제) **방식**
        - **처음에는 Feature Extractor와 동일** → `freeze_base=True`
        - **지정된 epoch 이후**, 백본도 requires_grad=True로 변경 → 전체 파인튜닝 전환
        - 학습이 **안정화된 후 백본을 조금씩 푸는 전략**
    - 각 모델 학습 후 csv파일로 저장하였다.
        
- 전체 모델 시각화
    - 각 모델을 학습 시킨 후 시각화를 진행하였음
    - 파인튜닝을 적용한 모델이 기본 모델(GoogLeNet_Pretrained) 보다 훨씬 개선된 것이 있었으며, 반대로 개선이 안되고 오히려 테스트 정확도가 떨어진 모델이 발견 되었다(**Feature Extractor).**
    - Feature Extractor 방식에서 성능이 하락한 이유
        Feature Extractor 방식은 GoogLeNet 모델의 백본(Backbone)을 고정(freeze)하고, 출력층(fc, aux1, aux2)만을 학습하는 방식이다. 이 전략은 일반적으로 데이터 양이 적고, 기존 사전학습된 특            징이 새로운 도메인과 유사할 때 유효하다. 그러나 본 실험에서는 해당 방식이 오히려 Pretrained GoogLeNet보다 낮은 정확도를 보였다(Test Acc: 79.90% vs. 82.11%).
        
        → 이러한 결과는 다음과 같은 요인으로 설명할 수 있다
        
        1. **학습 가능한 파라미터의 제한**
            
            Feature Extractor 방식은 **출력층**만 학습하기 때문에, 병해와 같은 특수한 이미지 도메인에 대해 충분한 표현 학습이 어려움 특히 패턴이 미묘하게 변화하는 감자 병해 이미지에 대해 일반적인 백본 특징만으로는 성능이 제한됨 → 과적합 발생
            
        2. **Pretrained 모델의 일반화 능력**
            
            사전학습된 GoogLeNet은 ImageNet 데이터에 기반하여 다양한 시각 특징을 학습했기 때문에, 간단한 분류 문제에서는 Feature Extractor보다 오히려 더 일반적인 표현 능력을 제공할 수 있음
            
        3. **데이터셋 도메인 간 차이 및 규모 이슈**
            
            본 프로젝트의 데이터셋은 클래스 수(3개)가 적고, 전체 이미지 수 또한 대규모가 아님 
            
            이러한 조건에서는 전체 파라미터를 조정할 수 있는 Fine-tuning 또는 Gradual Unfreeze 전략이 더 효과적으로 동작함
            
        
        따라서 Feature Extractor 전략은 학습 범위가 제한적이기 때문에, 데이터의 특성과 도메인 간 차이를 충분히 반영하기 어려웠으며, 이는 전체적인 분류 성능의 하락으로 보임
        

### 3. Loss/Scheduler 변경

- **Label Smoothing Loss 적용**
    - Loss Function은 **정답에 얼마나 가까운지 수치화 하는 것**이 핵심이다.
        - `CrossEntropyLoss` : 정답 레이블만을 기준으로 확실히 맞춰야 한다고 학습
        - `LabelSmoothingLoss` : 정답 이외 클래스도 약간의 확률을 부여해 **과적합을 줄이는 방향**으로 학습
    - **핵심 특징**
        - nn.CrossEntropyLoss(label_smoothing=0.1) 적용
        - 모델의 overconfidence 완화
        - 작은 데이터셋에 유리함
    - 코드
        - LabelSmoothingLoss 통합 학습 함수 정의(기본모델 + 1차 파인튜닝)
        - LabelSmoothingLoss통합 학습 함수 정의 (Gradual Unfreeze _ 2차 파인튜닝)
    - 원래 의도는 Label Smoothing을 도입하여 모델이 **과적합을 방지하고 일반화 성능을 높이도록** 개선하려던 것이었는데, 오히려 2차 파인튜닝(gradual unfreeze)에서는 정확도가 **낮아지는** 결과가 나왔다.
    - Gradual Unfreeze & Smoothing 조합이 안좋은 이유
        
        
        | **구분** | **Gradual Unfreeze** | **Label Smoothing** |
        | --- | --- | --- |
        | 목적 | feature 보존 + 안정적 미세조정 | 과적합 방지 + regularization |
        | 특징 | 점진적으로 학습 강도 ↑ | 학습 강도를 ↓시키는 경향 |
        | 충돌 | 학습의 **민감한 초기 단계에서**, 학습 대상이 너무 soft해져 → 유의미한 파라미터 업데이트가 어려움 |  |
        
        → Gradual Unfreeze 는 점진적인 학습이 필요한데 **Label Smoothing은 학습 자체를 더 약하게 만들기 때문에** 파라미터가 풀린 후에도 모델이 제대로 학습하지 못한것으로 확인됨
        
- **Label Smoothing + ReduceLROnPlateau 스케줄러 적용**
    - **Reduce Learning Rate On Plateau :** 학습이 평탄(plateu) 상태에 도달했을 때, 학습률(Learning Rate)를 자동으로 줄여주는 스케줄러
    - **핵심 특징**
        - 일정 epoch 동안 val_loss 개선 없을 시 learning rate 감소
        - StepLR보다 더 부드럽고 정밀한 조절
        - min_delta, factor, patience 조정 가능
    - 검증 손실이 2폭동안 개선 되지 않으면, 학습률을 현재의 50%로 감소시킨다는 뜻
    - Label Smoothing은 일반적으로 학습을 부드럽게 유도하기에 고정적인 StepLR보다 **동적으로 반응하는 ResuceLROnPlateu로 진행**
    - 코드
        - googlenet_smooth_rlrop (기존 구글넷 모델 + Smoothing + ReduceLROnPlateau)
        - GoogLeNet + Gradual Unfreeze + Label Smoothing + ReduceLROnPlateau 통합 셀
        - 테스트 정확도 평가 및 MongoDB에 저장  
        - 테스트 확인 & 시각화 진행  
    - 총 9가지 모델에 대하여 총 테스트 정확도를 시각화 하였다(아래).
        

## 5. 실험 결과

---

- 모델별 테스트 정확도 결과는 아래와 같다
    
    
    | **모델명** | **Test Accuracy (%)** |
    | --- | --- |
    | googlenet | 94.36 |
    | googlenet_finetuned | 82.60 |
    | googlenet_gradual_unfreeze | 91.42 |
    | `googlenet_smooth` | **97.06** |
    | googlenet_finetuned_smooth | 87.25 |
    | googlenet_gu_smooth | 85.78 |
    | googlenet_smooth_rlrop | 87.99 |
    | `googlenet_finetuned_smooth_rlrop` | **96.81** |
    | googlenet_gu_smooth_rlrop | 88.24 |
- 인사이트
    - **`Label Smoothing`**은 일반적인 GoogLeNet 학습보다 **분명한 성능 향상**을 보임 (Base 기준 94.36 → 97.06%)
    - **`Gradual Unfreeze`**는 정규화 계열 기법(Loss Smoothing, ReduceLROnPlateau 등)과의 조합 시 **과도한 일반화로 성능 하락** 가능성
    - **`ReduceLROnPlateau`**는 overfitting 방지, val_loss 안정화에 유리함 → 일부 모델에서 큰 폭의 정확도 상승
- 고려사항
    - `googlenet_finetuned`  모델은 오히려 정확도 저하 → **초기 학습률 / 과적합** 가능성 추정
    - Gradual Unfreeze 전략은 **label smoothing 없이 적용** 시 더 유리할 수도 있음
    - 정규화/스케줄링 기법이 항상 이점이 되지 않음을 보여주는 예시 포함

## 6. 결론

---

- 본 프로젝트는 여러 모델기법중, 낮은 정확도를 보인 GoogLeNet의 기본 구조에서 출발하여,보조 분류기 활성화, Gradual Unfreeze, Label Smoothing, 학습률 스케줄링(ReduceLROnPlateau) 기법을 점진적으로 적용하면서 모델 성능을 단계적으로 향상시키는 것을 목표로 하였음
    
    특히 **Finetuning**+**Label Smoothing + ReduceLROnPlateau** 조합은 작은 데이터셋 환경에서 효과적이었으며, GoogLeNet 1차 파인튜닝 모델에서 높은 테스트 정확도인 96.81%를 달성하였음
    
    이는 전체적인 파라미터 학습을 통해 사전학습된 백본(Backbone) + Fully Connected + 보조 분류기(aux) 가 모두 업데이트 되면서 모델의 표현력과 적응력이 최대로 발휘된 것으로 보임
    
    추가로 `ReduceLROnPlateau`가 일정 Epoch 이후 val_loss 개선이 멈추면 학습률을 절반으로 낮추어 더 정밀한 학습으로 전환된 점을 고려해 과적합 타이밍을 잡아준 것으로 보임
    
    하지만 일반 GoogLeNet모델에서는 오히려 정확도가 떨어지는 것을 확인할 수 있는데(97.06%→87.99%), 이는 데이터의 클래스 수가 3개로 적기에 오히려  ****`Label Smoothing`의 정규화 효과 과다로 보임
    
    또한 일반 GoogLeNet 모델은 보조분류기(aux)가 비활성화 된 상태인데, **Label Smoothing이 과도하게 confidence를 낮추고** ReduceLROnPlateau가 이를 바탕으로 조기에 learning rate를 떨어뜨리면서 결과적으로 **충분한 학습을 방해한 것으로 보임**
    
    따라서 최적 전략은 기본 `GoogLeNet` 모델에 **Label Smoothing 적용** (googlenet_smooth) 하거나 **1차 파인튜닝 + Label Smoothing + ReduceLROnPlateau 조합**(googlenet_finetuned_smooth_rlrop)으로 하는 것이 적절하다고 판단된다.
    

---

코드 링크:

- ResNet50 모델 학습 코드

[Google Colab](https://colab.research.google.com/drive/1KaoYjCGbDyiXGhz4ub2yvm5UlFvT-TE9?usp=sharing)

- 이외 여러 모델 학습 & 비교 코드

[Google Colab](https://colab.research.google.com/drive/1TswtCKCh0EAiYc3kv_udFya8xf6MnGlN?usp=sharing)

- 최종 모델링 & 학습 코드

[Google Colab](https://colab.research.google.com/drive/1-8qLkn-sNGnwFZhwF_BoKuInyJPNzWt8?usp=sharing)
