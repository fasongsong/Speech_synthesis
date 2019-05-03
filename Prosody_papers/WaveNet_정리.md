# WaveNet

네트워크를 통해 오디오 샘플링 하는것

distribution이 아웃풋

방법 - CNN with dilation

이슈-Computational burden

---

 오디오 데이터의 분포를 학습

학습된 분포로부터 샘플을 생성하는  generate model



Audio 는 일종의 시퀀스데이터,    이전 시점의 데이터들이 갖고있는 특징과 값들이 현재와 미래에 영향을 미치는

statistical dependecy 를 모델에 반영해야한다

-> auto regressive mdoel형식사용

-> joint prob를 베이스룰을 통해 조건부확률의 곱으로 변환가능

======> wavenet은 이러한 곱을 conv layer 를 사용하는것으로 표현한다.



dilated causal conv 테크닉 사용

 causal  -> 미래데이터 사용x

dilated -> 훨씬 이전 시점의 데이터를 고려하고  receptive field를 효율적으로(적은 계산ㄴ량으로) 증가시키기 위한 기법

 exponentially increase the receptive field to model the long range temporal dependencies

 이미지 프로세싱에서 masked conv와 같은개념 (여기서는 1-D 데이터이기때문에 좀더 단순)





audio amplitude 를 regression문제로 해결하지 않고 .quantization을 통해 classification 문제로 변형

ReLu 대신 Gated Activation Unit을 사용 (PixelCNN에서 차용)







++ WaveNet 은 음성데이터의 분포를 학습하여 샘플링 하는 모델인데

여기에 text information 을 조건으로 추가시켜 조건부 데이터를 생성하는 conditional  WaveNet을  구성

이것을통해  TTS 구현하는것.

---



Raw audio 데이터는 1 time step마다 16비트 integer 값을 갖는 시퀀스

따라서 softmax layer는 65,536개의 probability 를 매 스텝마다 출력해야한다 (연산량 너무많음)

=> u-law companding transformation 방식을 사용해서 이를 256개의 값으로 줄임

간단히 얘기해서 255개의 값으로 quantize시키는 방법

-> 단순히 linear하게 quantize하는 것보다 훨씬 좋은 결과를 내주었음.





깊은 네트워크가 잘 학습되도록 (converge하도록) 큰역활을 했던 residual 방식을 사용

parameterised skip connection 방식을 사오ㅛㅇ

각 레이어에서 1x1  conv 연산을 통해 (kernel 개수를 맞추어줌) dimension 을 맞추어 준후 element wise 합을 구함

(skip connection을 사용한것은 아니고 그냥 각 각레이어에서의 결과값을(fully connected) 전부 더하는 방식을 사용)

