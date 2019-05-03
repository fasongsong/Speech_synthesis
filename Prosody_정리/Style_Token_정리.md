# Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis

------

### 요약

global style tokens 제안

어떠한 explicit label 없이 embedding 이 학습됨 (넓은 범위의 acoustic expressiveness 를 배움)

Style transfer : 하나의 오디오 클립으로 그 스타일의 긴형태의 text corpus 를 묘사할수있음



### 인트로

style : intention, emotion, influence the speaker's choice of intonation and flow ,,,

GST  어떤 prosodic label 필요없이 학습 가능

internal architecture 는 스스로  soft interpretable label을 생성해냄 (이것을 통해 various style control, transfer task  가능)

+++ GST can be directly applied to noisy, unlabeled found data, highly scalable but robust speech synthesis



### Model Architecture

(grapheme, phoneme) =======> seq2seq ======> Mel spectrogram

#### training 

![1554114950735](C:\Users\sanghun\AppData\Roaming\Typora\typora-user-images\1554114950735.png)

* reference encoder : 다양한 길이의 오디오 신호의 prosody를 정해진 길이의 벡터 (reference embedding)으로 압축

* attention : 레퍼런스 임베딩된게 style token layer 통과 -> attention module의 쿼리 벡터로 사용됨

  ​                          이때 어텐션은  alignment 에 사용되지않고 레퍼런스 임베딩과 각 랜덤으로 초기화된 임베딩 (토큰)  사이의 유사도를 학습

  ​                            이 임베딩된 set을 global style tokens 라함.

* attention module은 a set of combination weight 를 출력함

  * 이때 weight 는 각  style  token이 인코딩된 레퍼런스 임베딩에 대한  contribution 을  represent 함

* seq2seq : jointly trained by the reconstruction loss from the Tacotron decoder

 

#### Inference

![1554115180281](C:\Users\sanghun\AppData\Roaming\Typora\typora-user-images\1554115180281.png)

* 1) directly condition the text encoder on certain tokens (without reference signal)     right hand side
* 2) feed a different audio signal   (style transfer)  left hand side



## Model detail

Tacotron을 조금 수정해서 사용했음 GRU -> LSTM / Griffin -> WaveNet



##### Styel token architecture

* reference encoder :  is made up of a conv stack, followed by an RNN

  * Input : Mel spectrogram

  * Ouput : reference embedding

    

* style token layer : is made up of a bank of style of token embeddings and an attention module

  *   실험은 10 개의 토큰을 사용 , 이 토큰은 훈련 데이터에서 작지만 풍부한 다양한 운율 차원을 표현하기에 충분
  * 다양한  conditioning sites  의 combination 실험 ---> replicating the style embedding and simply adding it to every text encoder state performed the best
  * multi-head attention significantly improves style transfer performance  (token의 수를 늘리는 것보다도.)



## Model Interpretation

##### End to End Clustering / Quantization

* Intuitively, the GST model can be thought of as an end-toend method for decomposing the reference embedding into a set of basis vectors or soft clusters – i.e. the style tokens. 
* As mentioned above, the contribution of each style token is represented by an attention score, but can be replaced with any desired similarity measure

#### Memory Augmented Neural Network

* GST embeddings can also be viewed as an external memory that stores style information extracted from training data.
* The reference signal guides memory writes at training time, and memory reads at inference time





 