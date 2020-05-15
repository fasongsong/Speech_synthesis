# Investigation of Enhanced Tacotron Text-to-Speech Synthesis Systems with Self-attention for Pitch Accent language

### Abstract

JJap could be one of the most difficult languages for which to achieve end-to-end speech synthesis, largely due to its character diversity and pitch accents

최신 기법들도 다 separate text analyzer and duration model 을 필요로한다.

본 논문에선 self attention 사용 

* capture long-term dependencies related to pitch accents and compare their audio quality with classical pipeline systems 
  * under various conditions to show their pros and cons

we investigated the impacts of the presence of accentual-type labels, the use of force or predicted alignments, and acoustic features used as local condition parameters of the Wavenet vocoder.



우리의 결과는 비록 제안 된 시스템이 여전히 일본어에 대한 최고급 파이프 라인 시스템의 품질과 일치하지는 않지만 종단 간 일본어 음성 합성을 향한 중요한 발판을 보여줍니다.



#### Intro

traditional pipeline

	1. text analyzer, acoustic, duration model  =====> tacotron 은 하나의 모델 (reduce laborious feature engineering and error propagation)



tacotron2 = tacotron + wavenet

일본말 = 히라가나 카타카나 칸지

pitch accented  language and accentual-types may change the meaning of words

​	근데 일본 글자에선 그 accentual type이 명확하게 나타나지 않음

accent nucleus positions are context dependent

​	그래서 they change positions depending on adjacent words



###  Proposed architectures

기존 pipeline은 explicit duration model을 가졌는데

tacotron은 소스와 타겟 시퀀스 간의 alignment 를 implicitly learn 하는 attention mechanism 을 사용



![1556609482042](C:\Users\sanghun\AppData\Roaming\Typora\typora-user-images\1556609482042.png)



##### tacotron using phoneme and accentual-type

​	본 논문에서는 phoneme 과 accentual-type sequences을 사용 /  source로

​	그리고 mel-spectrogram을 사용 / target 으로



​	phoneme 과 accentual-type sequence는 분리된 임베딩 테이블로 다른 dimension으로 임베딩됨 => 컨캩



​	decoder에서는 encoded value가 attention based LSTM 으로 디코딩됨

​	forward attention 사용, (additive attention 대신에)

​	zoneout regularization will reduce alignment errors

	the forward attention accelerates the alignment learning speed and provides distinct and robust alignment with less training time than the original Tacotron.
``` At each timestep,
zoneout stochastically forces some hidden units to maintain their previous values.
Like dropout, zoneout uses random noise to train a pseudo-ensemble, improving
generalization. But by preserving instead of dropping hidden units, gradient
information and state information are more readily propagated through time, as
in feedforward stochastic depth networks
```

mel의  frame shift 사용  mel ---> waveform Wavenet에서



##### Extending Tacotron with self-attention

pitch accent language uses lexical pitch accents that involve F0 changes

Pitch accents have a large impact on the perceptual naturalness of speech because incorrect pitch accents may be judged as incorrect “pronunciations“

Japanese normally have mora of varying lengths

​	Since the length of an accentual phrase could be very long, we hypothesize that long-term information plays a significantly important role in TTS for pitch accent languages.



LSTM뒤에 self attention추가.  (enc, dec 둘다)              **SA-Tacotron**

self-attention relieves the high burden placed on LSTM to learn long-term dependencies to sequentially propagate information over long distances (long term 에대한 학습문제의 부담을 줄여줌)



Encoder에선

​	sequential relationships of input 을 capture.    &&&&  we do not using positional  encoding



Decoder에선

​	인코더에서나온 두개의 output을 additive / forward attention 두개의 방식으로 사용

		-----> because we want to utilize the benefits of both: forward attention accelerates alignment construction, and additive
	attention provides flexibility to select long-term information from
	any segment.
Unlike the encoder, self-attention works autoregressively at the decoder



디코딩의 각 time step에서,   self attention layer는 LSTM output 의  all past frame에 집중하고

output 은 오직 가장 최신의 프레임 (prediction output 으로서) 만 집중한다.

그리고 예측된 프레임은 다시 다음 타임스텝의 인풋으로 fed back 



##### Tacotron using vocoder paramters

MGC, F0 (vocoder parameters) 사용, 이것을 예측

5ms shift하여 MGC와 F0 추출, 

일반적으로 보코더를 기반으로 신뢰할 수있는 음성 분석을 위해서는 미세한 분석 조건이 필요합니다.

그러나이 조건은 입력 및 출력 불일치를 줄이기 위해 대개 12.5ms 프레임 shift 및 50ms 프레임 길이와 같이 거친 조건을 사용하는 Tacotron을 교육하는 데 자연스러운 선택이 아닙니다.



5 ms shift <<<<< 12.5 ms shift

* 2.5 배 긴 target vocoder parameter sequences 의 길이

  * predict target  위해 2.5배 긴  autoregressive loop iteration 필요.

    * 어려운문제다.

      

난이도를 줄이기 위해

​		we set the reduction factor to be three in order **to reduce the target length**

​		This setting results in 5/3 times longer target length compared to SA-Tacotron in the previous section



MGC 에는 L1  loss

log F0 , stop flag에는 cross entropy

weighted sum 해서 optimize





### Experiments

##### Experiment conditions

accentual label 있냐없냐 --- lexical pitch accent 학습가능한지

 mel이냐 vocoder params  냐 -----

forced alignment냐 predicted alignment냐  to understand the accuracy of duration modeling better

With forced alignment, alignments are calculated with teacher
forcing, and target acoustic parameters are predicted with the
alignments obtained with teacher forcing



We used a Japanese speech corpus from the ATR Ximera dataset
[15]. This corpus contains 28,959 utterances from a female speaker
and is around 46.9 hours in duration. The linguistic features, such as
phoneme and accentual-type label, were manually annotated, and
the phoneme label had 58 classes, including silence, pause, and
short pause [16]. To train our proposed systems, we trimmed the
beginning and ending silence from the utterances, after which the
duration of the corpus was reduced to 33.5 hours. We used 27,999
utterances for training, 480 for validation, and 142 for testing.



The JA-Tacotron and SA-Tacotron with and without
accentual-type labels were built to show whether the investigated
architectures can learn lexical pitch accents in an unsupervised
manner.



32 dimensions
for accentual-type embedding and 224 dimensions for phoneme
embedding. For the models without accentual-type embedding, 256
dimensions were allocated to phoneme embedding.



###  Objective evaluation

![1556696522565](C:\Users\sanghun\AppData\Roaming\Typora\typora-user-images\1556696522565.png)

Vertical white lines indicate accentual phrase boundaries obtained by forward attention



#### 		What does self-attention learn?

* alignment of an encoder LSTM source and mel-spectrogram target for dual source attention. (forward attention)
* alignment of an encoder self-attention source and mel-spectrogram target (additive attention)
  * It seems to be related to accentual phrase segments and phrase breaks divided by pauses.



#### 		What is the effect of accentual-type labels

mel을 예측했을때 label을 있이 학습하면 accentual phrase boundary가 잘 예측됬으나 라벨없이하면 안됬음

accentual phrase boundary 자체는 어텐션 메커니즘에의해 예측됨.



#### 		Comparison of mel-spectrogram and vocoder parameters

Non-monotonic alignment
may result in mispronunciation, some phonemes being skipped,
repetition, the same phoneme continuing, and intermediate
termination



alignment errors were found for SA-Tacotron using
vocoder parameters due to the longer length than the corresponding
mel-spectrogram. We found 19 alignment errors out of 142 test
utterances.



### Subjective evaluation

원래는 predicted 보다 forced가 도움됬지만  JA-Tacotron 에서는 pre 가 더 도움됬다.

Since Tacotron learns both spectrograms and alignments
simultaneously, it seems to produce the best spectrograms when
it infers both of them.



The best proposed
system still does not match the quality of the best pipeline system.

* One major difference of our is
  input linguistic features; our proposed systems use phoneme and
  accentual-type labels only, 

  

  but the baseline pipeline systems use
  various linguistic labels including word-level information such as
  inflected forms, conjugation types, and part-of-speech tags.

  

  In particular, an investigation on the same Japanese corpus found
  that the conjugation type of the next word is quite useful for F0
  prediction



### Conclusion

