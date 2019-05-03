# LEARNING LATENT REPRESENTATIONS FOR STYLE CONTROL AND TRANSFER IN END-TO-END SPEECH SYNTHESIS

----

## Abstract

* introduce the Variational Autoencoder (VAE) to an end-to-end speech synthesis model, to learn the latent representation of speaking styles in an unsupervised manner
* The style representation learned through VAE
* Style transfer can be achieved in this framework by first inferring style representation through the recognition network of VAE

## Intro

The latent state plays a pretty similar role as the latent variable does in VAE

To be specific, direct manipulation can be easily imposed on the disentangled latent variable, so as to control the speaking style

On the other hand, with variational inference the latent representation of speaking style can be inferred from a reference audio, which then controls the style of synthesized speech.





Akuzawa et al. : speech synthesis + VAE (for expressive speech synthesis)

차이점  1) 전 논문은 direct sampling 을 통한 synthesize expressive speech (from prior of latent distribution at inference stage)

​			-> 본 논문은 direct manipulate latent variable or variational inference from ref. audio 를 통한 speaking style ***control***

​             2) end to end임  전 논문은 x



Sec2. VAE 모델 설명과 KL divergence collapse problem을 풀기위한 trick 소개

Sec3.  실험



## Model

#### VAE

* construct a relationship   (unobserved continuous random latent variable Z  와 observed dataset X 간의)

* true posterior density p(zjx) is intractable ====> indifferentiable marginal likelihood p(x)             (세타)

  ====> 이를 해결하기위해  recognition model q(zjx) 이 approximation to the intractable posterior 로 소개됨.   (파이)

  

* variational principle log p(x) can be rewritten

![1554787163806](C:\Users\sanghun\AppData\Roaming\Typora\typora-user-images\1554787163806.png)

​					L(; ; x) is the variational lower bound to optimize.

### Proposed Model Architecture

* propose a flexible model for style control and style transfer
* Recognition model = inference net ========>      audio -> fixed length vector (latent representation / latent variable **z**)
* 

###  Resolve KL collapse problem

* KL loss는 distinguishable representation을 배우기전에 항상 붕괴

* KL loss가  reconstruction loss의 수렴속도를 훨씬 능가 ===> 거의 0으로 빠르게 떨어지고 다시오르지도않음 ====> 즉 엔코더가 동작하지 않음
* KL annealing이 이러한 문제를 풀기위해 소개됨
*  즉 KL 텀에 가변 가중치를 추가, 웨이트는 훈련 시작시 0에 가까운수에서 점차 증가
*  또한 KL loss는  매 K 스텝마다 한번 고려됨
* 두가지 트릭을 이용하여 KL loss 는 0이아닌 채로 유지하며 붕괴가 방지됨.



## Experiment and anlaysis

Disentangled factors

* disentangled representation means that a latent variable completely controls a concept alone and is invariant to changes from other factors
* z의 차원이 피치높이, 로컬피치변화, 말하기 속도와같은 서로 다른 스타일 속성을 독립적으로 제어할수있음을 확인했다.,
* 다른 차원 냅두고 한 차원만 값을 변경시 mel 스펙에서도 다른건 고정이고 특정 한 특성만 변화가 생김



style transfer

* ref audio는 테스트셋으로부터 골라졌고 합성된 오디오는 인풋텍스트는 공유했음
* 결과보면 ref 오디오랑 피치높이 pause time 같은 패턴이 유사함을 알수있다.



### 결론

style transfer 연구는 single에서 multi speaker로 넓혀질 필요가있다.

