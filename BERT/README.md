# **BERT(Bidirectional Encoder Representations from Transformers)**

![fig1](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/fig1.png?raw=true)

## **1. Introduction**
기존의 연구들은 Pre-training된 Langauge Model은 많은 NLP task에 효율적이라는 것을 보여줍니다. Pre-training을 위한 접근으로는 feature-based approch와 fine-tuning approch가 있습니다. feature-based approach는 pre-trained represetations를 추가적인 feature로 사용하는 특정 task 구조에 사용됩니다. 대표적으로 ELMo가 있는데, ELMo는 Language Model을 LTR, RTL 두 방식으로 각각 학습시켜 각 레이어에서 나온 Hidden state를 합친 모델입니다. fine-tuning approach는 특정 task 파라미터를 최소화하고 downstream task에 맞게 모든 pre-trained 파라미터를 학습시킵니다. GPT는 엄청난 양의 corpus를 이용하여 pre-training하면 적은 양의 데이터로도 충분히 좋은 성능을 달성할 수 있다는 것을 보여주었습니다.  
논문에서 Language Model의 Unidirectional한 성질이 pre-training을 위한 architecture를 제한하고, 모델의 성능을 저해하는 문제점이라고 주장합니다. GPT를 예로 들면, LTP구조는 모든 토큰이 직전의 토큰만 참조할 수 있습니다. 이러한 구조는 Sentence level에서의 성능이 저하될 수 있으며, 양방향의 문맥이 중요한 Question answering에 적절하지 않다고 말합니다. BERT는 Masked Laguage Model(MLM)을 사용하여 무작위로 입력 토큰에 마스크를 씌워 양쪽의 문맥을 파악할 수 있게하여 Deep birectional Transformer의 pre-training을 가능하게 합니다. 또한 Next Sentence Prediction(NSP)를 사용하여 문장들의 관계에 대해서 학습이 가능합니다.
![fig2](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/fig3.png?raw=true)

## **2. Related Work**  
생략

## **3. BERT**
BERT의 프레임워크는 pre-training과 fine-tuning 두 단계로 나뉩니다. pre-training에서 라벨링되지 않은 데이터로 다른 여러 pre-training task에 대해 학습하고, fine-tuning 단계에서는 pre-training단계에서 학습한 파라미터와 fine-tuning된 모든 파라미터가 라벨링된 downstream task 학습에 사용됩니다. 같은 pre-trained model이 적용되더라도 각각의 downstream task는 독립된 fine-tuning model을 가집니다.

**Model Architecture**  
BERT 모델의 구조는 Multi-layer Bidirectional Transformer Encoder를 사용하고 있습니다.
- 논문에서 Transfomer의 내용이 생략되어 간단하게 서술하자면 Transfomer의 Encoder는 문장이 입력으로 들어오면 Input Embeddings + Positional Encodings을 해서 최종 입력을 구한 뒤 같은 문장끼리 Self-Attention을 수행해서 Attention Map을 생성합니다. 이때 모델에 더 알맞는 특징을 반영하기 위해 Head를 나눠 토큰과 문장의 다양한 특성을 추출하는 것이 Multi-Head Self-Attention이라고 합니다. 이후 자연어는 입력이 매번 다르기때문에 Batch Normalization보다 Lyaer Normalization을 적용하고 Feed Foward Network를 거치면서 특징을 학습합니다. 그리고 Transformer에서는 Encoder를 6개 중첩해서 최종 출력값을 Decoder의 입력으로 보냅니다.

**BERT<sub>BASE</sub>** (L=12, H=768, A=12, Total Parameters=110M)  
BERT<sub>BASE</sub>는 OpenAI GPT와의 비교를 위해 같은 사이즈로 설계되었습니다.  
**BERT<sub>LARGE</sub>** (L=24, H=1024, A=16, Total Parameters=340M)  
L: The number of layers, H: Hidden size, A: The number of self-attention heads  

**Input/Output Representations**  
BERT가 다양한 downtream task를 다루기 위해 입력이 한 문장인지 여러 문장이 확실하게 구분할 수 있어야 합니다. BERT는 30,000개의 단어로 WordPiece embedding을 사용했습니다. 모든 문장의 첫 시작은 [CLS]라는 특별 토큰으로 시작합니다. 은닉층의 마지막과 대응되는 이 토큰은 입력 시퀀스의 모든 정보를 반영하여 분류문제를 푸는데 사용됩니다. 여러 문장은 하나의 시퀀스로 되어있습니다. 이를 구분하기 위해 두 가지 방법이 있습니다. 첫번째는 두 문장 사이에 있는 특별 토큰인 [SEP]로 구분하는 방법이고, 두번째는 모든 토큰에 A문장인지 B문장인지 가리키는 임베딩을 추가하는 방법입니다.(첫 번째 문장은 0, 두 번째 문장은 1을 입력합니다.) Input Embedding은 token, segment, position embeddings의 합으로 구해집니다.  
Figure 1.에서처럼 Input Embeddings는 $E$, [CLS] token의 final hidden vector는 $C \in R^H$, $i^th$ final hidden vector 는 $T_i \in R^H$로 표현됩니다.
 ![fig2](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/fig2.png?raw=true)  

```python
#modeling_modeling.py in transformers library
class BertEmbeddings(nn.Module):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

         embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```

### **3.1 Pre-training BERT**
BERT는 기존의 LTR, RTL 방식대신 두 비지도 학습으로  pre-trainig을 진행합니다.  

**Tsak #1: Masked LM**  
Deep bidirectional model이 LTR모델이나 LTR/RTL의 shallow concatenation 모델보다 성능이 좋은 것은 당연합니다. 그러나 불행하게도 표준 조건의 language modell은 단방향으로만 학습이 가능합니다. 때문에 모델의 타겟 단어의 예측이 정확하지 않을 수 있습니다.  
Deep bidirectional representation을 학습시키기위해 입력 토큰에 무작위로 마스크를 씌워 토큰을 예측하게 합니다. 여기서 mask token에 해당하는 final hidden vector가 출력 softmax로 들어가게 됩니다. BERT에서는 각 문장 WordPiece 토큰의 15%를 무작위로 mask합니다. denoising auto-encoder와 대조적으로 BERT는 전체 문장의 예측 대신 mask된 단어만을 예측합니다. 
 bidirectional pre-trained model을 얻었음에도 fine-tuning 단계에서는 mask token이 존재하지 않기때문에 pre-training과 fine-tuning 사이에서 mismatch가 발생할 수 있습니다. 이를 완화하기 위해 전체 mask될 15%의 토큰에서 (1) 80%는 그대로 [MASK] 토큰으로 바꾸고 (2) 10%는 무작위 토큰과 교체하고 (3) 나머지 10%는 바꾸지 않습니다. 그리고나서 final hidden vector가 cross entropy loss로 원래의 토큰을 예측하게 됩니다.  

**Tsak #2: Next Sentence Prediction(NSP)**  
많은 downstream task은 Question Answering(QA)나 Natural Language Inference(NLI)처럼 두 문장사이의 관계를 이해하는 것을 기반으로 합니다. 관계를 학습하기 위해 이진화된 next sentence prediction을 pre-training합니다. 특히 A, B 문장을 고를 때, 50%는 IsNext로 라벨링된 실제 연결된 문장을, 나머지 50%는 NotNext로 라벨링된 무작위 문장을 고릅니다. 그리고 [CLS] token의 final hidden vector가 다음 문장을 예측합니다. 이 과정은 매우 간단하면서도 QA나 NLI에 매우 효과적입니다. NSP는 Jernite et al. (2017) 과 Logeswaran and Lee (2018)이 발표한 논문과 관련이 깊습니다. 그러나 이전의 기술은 문장 임베딩만 downsteam task로 전달하지만, BERT는 모든 파라미터를 전달합니다.

**Pre-trainind data**  
pre-training 과정은 거대한 양의 말뭉치가 필요합니다. BERT모델을 pre-training하기 위해 준비한 데이터셋은 BooksCorpus(800M words), Wikipedia(2,500M words)에서 목차, 표, 헤더를 제외한 텍스트로 준비되었습니다.

### **3.2 Fine-tuning BERT**
Fine-tuning은 transformer의 self-attention 매커니즘이 많은 downstream task를 쉽게 수행할 수 있도록 합니다. 보통 텍스트 쌍을 다루기 위해 bidirectional cross attention을 적용하기 전 각각의 문장을 따로 인코딩합니다. 그러나 BERT는 self-attention을 통해 연결된 두 문장과 bidirectional cross attention을 함께 인코딩함으로써 두 단계를 한 번에 처리합니다.  
각각의 과제에서 간단하게 특정 입력과 출력을 BERT에 플러그인 하고 모든 파라미터를 end-to-end 방식으로 fine-tuning 했습니다. 입력단에서 문장 A, B는 다음과 같습니다. (1)서로 유사한 두 문장, (2) 논리적 귀결에서 가설과 전제, (3) QA에서 질문과 응답, (4) 분류나 태깅 task에서 빈 문장을 포함한 텍스트 쌍입니다. 출력단은 QA나 태깅같은 token level task에서 각 토큰이 출력 레이어에 맵핑이 되고, 논리 귀결이나 감정분석에서는 [CLS] 토큰이 classification을 위해 출력 레이어로 연결됩니다. fine-tuning은 비교적 pre-training 보다 간단합니다. 논문 내의 모든 결과들은 같은 pre-trained model을 사용했고 TPU, GPU 환경에서 몇시간 내로 모두 실행되었습니다.

## **4. Experiments**
### **4.1 GLUE(General Langauage Understading Ecaluation)**  
![table 1](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/table1.png?raw=true)
GLUE는 9가지 문장 또는 문장 쌍의 natural language understanding task를 위한 벤치마크입니다. GLUE task위해 batch size는 32, epochs는 3을 주었으며 최적의 learning rate(5e-5, 4e-5, 3e-5, 2e-5 사이에서)를 선택했습니다. BERT<sub>BASE</sub>와 BERT<sub>LARGE</sub>는 이전의 SOTA모델보다 4.5%, 7%를 웃도는 평균 정확도를 얻었습니다. BERT<sub>BASE</sub>는 attention masking을 제외하고 거의 동일한 모델인 OpenAI GPT보다도 성능이 앞서고 있습니다. 또 작은 학습데이터에서 BERT<sub>LARGE</sub>가 BERT<sub>BASE</sub>보다 모든 task에서 우수한 성능을 보여줍니다.
### **4.2 SQuAD v1.1 (Stanford Question Answering Dataset)**  
![table 2](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/table2.png?raw=true)  
SQuAD v1.1은 Wikipedia로 수집한 100k개의 질의응답 데이터셋입니다.
### **4.3 SQuAD v2.0**  
![talbe 3](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/table3.png?raw=true)  
SQuAD v2.0은 SQuAD v1.1의 한계를 해결하기위해 짧은 대답을 제거하고 문장을 현실적으로 가공하였습니다. 
### **4.4 SWAG(Situations With Adversarial Generations)**   
![talbe 4](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/table4.png?raw=true)  
SWAG는 상식적인 상황이 담긴 113k개의 문장 데이터셋에서 문장이 주어졌을 때 가장 상식적인 선택지를 고르는 task입니다.

## **5. Ablation Studies**  
이번 목차에서는 BERT의 어떤 측면이 중요하게 작용하는 지 이해하기위해 배제된 실험을 진행합니다.

### **5.1 Effect of Pre-training Tasks**
BERT의 deep bidirectionality의 확인을 위해 정확이 같은 환경에서 실험을 진행합니다.  
![table 5](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/table5.png?raw=true)  
**No NSP**  
Masked LM으로만 pre-training을 진행합니다.  
**LTR & No NSP**  
Masked LM 대신 Left-to-Right(LTR) LM으로 학습합니다. pre-train과 fine-tune 사이의 불균형을 없애기 위해 fine-tuning에서도 똑같이 적용합니다. 또 NSP task를 배제합니다.  
먼저 NSP의 영향을 살펴보면 QNLI, MNLI, SQuAD v1.1에서 성능저하가 확인되었습니다. 다음으로 "No NSP"와 "LTR & No NSP"의 비교 통해 bidirectionality의 영향을 살펴보면 모든 task에서 성능저하가 발생하였으며 특히 MRPC와 SQuAD에서 두드러지게 나타납니다. SQuAD에서 token-level의 hidden state가 오른쪽의 token에 대한 정보를 참조할 수 없기 때문에 LTR model이 token prediction에서 낮은 성능을 보이게 됩니다. 이것을 확실하게 확인하기 위해 LTR model에 무작위로 BiLSTM을 연결합니다. 그 결과 SQuAD에서 확연한 성능 향상을 보여주었지만 여전히 bidirectional model보다는 낮은 성능을 유지했습니다. 심지어 BiLSTM은 GLUE task의 성능을 저하시키는 결과를 가져왔습니다. 분리된 LTR/RTL model이나 ELMo와 같이 두 모델로 부터 합성된 token을 학습시키는 것이 가능한 것을 확인했지만, (a) 단일 bidirectional model보다 비용이 많이들고 (b) 두 문장의 관계를 이해하지 못해 QA task에서 직관적이지 않고 (3) deep bidirectional model보다 확실히 성능이 떨어집니다.  

### **5.2 Effect of Model Size**
![table 6](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/table6.png?raw=true)  
이번 목차에서는 모델의 크기가 fine-tuning의 정확도에 미치는 영향을 확인합니다. 논문에서는 layer, hidden unit, attention head의 수를 다르게하여 몇 가지 BERT모델을 학습시켰습니다.  
fine-turing을 5번 반복하여 얻은 정확도의 평균을 보면 모든 데이터셋에서 모델이 커질수록 정확도가 증가하는 것을 볼 수 있습니다. 또 기존의 발표된 충분히 큰 크기의 모델로부터 성능 향상을 달성했습니다. Vaswani et al. (2017)에서 가장 큰 모델의 사이즈는 (L=6, H=1024, A=16, Total Parameters=100M)이고, Al-Rfou et al. (2018)에서는 (L=64, H=512 A=2, Total Parameters=235M)인데 반해 BERT<sub>BASE</sub> 와 BERT<sub>LARGE</sub>의 파라미터는 110M, 340M이다.  
모델의 사이즈가 클수록 machine translation과 language modeling 같은 큰 스케일의 task에서 성능이 향상되는 것은 오래전에 알려졌으며 LM perplexity를 통해 입증되었습니다. 그러나 극단적으로 스케일링 된 모델이 매우 작은 task에서도 성능 향상된다는 것을 증명하는 하려면 모델이 확실하게 pre-trained 되어야 합니다. Peters et al. (2018b)는 downstream task에서 pre-trained model의 사이즈를 늘렸을 때의 영향에 대해 엇갈린 의견을 제시하였으며, Melamud et al. (2016)은 hidden dimention size를 200에서 600으로 늘렸을 때는 향상됐지만 1,000으로 늘렸을 때는 그렇지 않았다고 언급했습니다.



### **5.3 Feature-based Approach with BERT**
![table 7](https://github.com/chaaaaaaaaaaan/Paper/blob/main/BERT/resource/table7.png?raw=true)  
모든 BERT의 결과는 pre-trained model에 간단한 classification layer를 더한 fine-tuning approach를 사용되었다는 것을 보여줍니다. 그리고 모든 파라미터들은 downstream task에서 fine-tuning됩니다. 그러나 pre-trained model에서 feauter를 추출하는 feature-based approach는 어드밴티지가 있습니다. 첫번째로 모든 task가 transformer encoder 구조로 해결할 수 없기 때문에 특정 task model이 요구됩니다. 두번쨰로 pre-training을 통해 데이터를 한 번 학습함으로써 계산적 이익을 얻을 수 있습니다.  
이번 목차에서는 두 가지 방식으로 CoNLL-2003 Named Entity Recognition(NER) task에 BERT를 적용시킵니다. BERT를 입력으로 넣기위해 데이터에 글의 문맥을 최대한으로 포함시켜 WordPiece model을 사용했습니다. taggin task와 같이 진행되지만 CRF layter는 사용하지 않습니다. fine-tuning approach를 배제하기 위해 아나 이상의 layer에서 정보를 추출하여 feature-based approach를 적용합니다. contextual embeddings는 classification layer에 입력되기 전 BiLSTM으로 들어갑니다. 결과를 보면 BERT<sub>LARGE</sub>는 SOTA model과 근소한 차이를 보여줍니다. Feature-based approach에서는 마지막 4개의 레이어를 합쳤을 때(concatenation) fine-tuning model과 0.3차이로 가장 성능이 좋았습니다. BERT는 두 가지 방법 모두 효과적인 것을 보여줍니다.


