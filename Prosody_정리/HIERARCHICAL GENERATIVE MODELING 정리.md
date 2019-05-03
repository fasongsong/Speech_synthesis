# Hierarchical Generative Modeling for Controllable Speech Synthesis

---

#### Abstract

This paper proposes a neural sequence-to-sequence text-to-speech (TTS) model
which can control latent attributes in the generated speech that are rarely annotated
in the training data, such as speaking style, accent, background noise, and recording
conditions.



model is formulated as a conditional generative model based
on the variational autoencoder (VAE) framework, with two levels of hierarchical
latent variables

* first level : categorical variable
  * represent attribute groups , provide interpretability
* second level : multivariate Gaussian variable 
  * characterizes specific
    attribute configurations (e.g. noise level, speaking rate) and enables disentangled
    fine-grained control over these attributes

latent distribution에 대해 GMM 사용하는것과 같다





#### Intro

speech attributes

* speaker identity
* speaking sytle
* prosody ...



crowdsourced data에는 다중 latent attributes 있는게 보편적

latent representation 이 disentangled  하면 생성된 factor는 독립적으로 컨트롤 가능하다



Each latent variable is modeled in a variational auto-encoding framework using Gaussian mixture prior



two latent space

* labeled (speaker identity)
* unlabeled attributes

각 latent variable은 VAE 로 모델링된다



The resulting latent spaces

​	1) learn disentangled attribute representation, (각 차원은 different generating factor를 제어함)

​	2) discover a set of interpretable  clusters, each of which corresponds to a representative mode in the training data (하나는  clean 클러스터 다른건 노이지 스피	치 클러스터 같이) 

​	3) provide a systematic sampling mechanism from the learned prior



Experiments confirm that the proposed model is capable of controlling speaker, noise, and style independently, even when variation of all attributes is present but unannotated in the train set.



Contribution

* propose a principled probabilistic hierarchical generative mode
  * improving sampling stability , disentangled attribute control     (GST 논문보다!)
  * interpretability and quality   (VAE  논문보다)
* 모델 공식은 두 개의 혼합 분포를 사용하여 supervised speaker attributes 과 latent attributes 을 분리 된 방식으로 개별적으로 모델링하여 latent encoding을 명시 적으로 고려합니다
  * 이는 다른 reference 발화로부터 유추 된 스피커 및 latent encoding 에 대한 모델 출력을 조절하는 것을 직설적으로 만든다.
* this work is the first to train a high-quality controllable text-to-speech system on real found data containing significant variation in recording condition,  speaker identity, as well as prosody and style

#### Model

