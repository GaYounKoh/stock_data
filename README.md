# stock_data
stock data handling and utilization <br>
- 1st: DSML test <br>
- 2nd: Capstone test <br> - pandas사용 ver ; with open 구문 ver <br>
<br><br>
정확도<br>
---- 검색어: scikit-learn 회귀모델 <br>
            'regression model 종류'는 별 도움 안됨 <br>
[다양한 회귀모델들 - 회귀 코드에 참고함](https://cleancode-ws.tistory.com/109) <br>
[다중선형회귀](https://hleecaster.com/ml-multiple-linear-regression-example/) <br>
[회귀 개념, 분류](https://bangu4.tistory.com/100) <br>
<br>
<span style = 'color: red'>/project/guri/forposter/Result1.정확도.ipynb</span> 참고 <br>
<br>
<br>
한 줄로 조건에 맞는 df 열 생성하기. <br>
** 조건에 따른 df 열 바꾸기 아님에 주의!! <br>

<예시>
``` python
df['new_col'] = np.where(df['base_col'] == 1, 'new', 'old')
```
<br>
조건을 만족하면 'new' 작성, 만족하지 않으면 'old' 작성. <br>

[np.where(condition, x, y); 조건 충족시 x, 그렇지 않으면 y 반환.](https://www.delftstack.com/ko/howto/python-pandas/how-to-create-dataframe-column-based-on-given-condition-in-pandas/) <br>

[data split - random ,,, shuffle](https://rfriend.tistory.com/519) <br>

```
비시계열 데이터는 shuffle 해도 random이랑 같음.
단, 시계열 데이터는 shuffle과 random이 다른 의미이니 주의.
(시계열 데이터는 순서가 중요하므로)
```
<br>

[회귀 cross_val_score()의 hyper param](https://nicola-ml.tistory.com/26) <br>

[cross_val_score()의 공식 doc](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) <br>

검색어 : [pipeline.fit.transform 파이썬] <br>
[transpose()가 뭔지 봐야함.1 - 읽음](https://dsbook.tistory.com/107) <br>
-- [fit & transform 과 fit_transform의 차이](https://www.inflearn.com/questions/19038) <br>
```
Q. [fit & transform 과 fit_transform의 차이?]
A. 사이킷런은 데이터를 변환하는 대부분의 로직에서 fit()과 transform()을 쌍으로 사용

 학습데이터 세트에서 변환을 위한 기반 설정
 (예를 들어 학습 데이터 세트의 최대값/최소값등)
 을 먼저 fit()을 통해서 설정한 뒤에

 이를 기반으로 학습 데이터의 transform()을 수행하되
학습 데이터에서 설정된 변환을 위한 기반 설정을 그대로 테스트 데이터에도 적용하기 위해서입니다.


즉 학습 데이터 세트로 fit() 된 Scaler를 이용하여 테스트 데이터를 변환할 경우에는
테스트 데이터에서 다시 fit()하지 않고
반드시 그대로 이 Scaler를 이용하여 transform()을 수행해야 합니다.
```
[transpose()가 뭔지 봐야함.2 - 아직 안 읽음.](https://mindsee-ai.tistory.com/61) <br>

<br>

[xticks]() <br>
[python plt arrow](https://codetorial.net/matplotlib/module_patches.html) <br>
[annotate()](https://runebook.dev/ko/docs/matplotlib/_as_gen/matplotlib.pyplot.arrow) <br>

<br>

[reg 평가지표로써의 r^2](https://velog.io/@dlskawns/Linear-Regression-%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80%EC%9D%98-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C-MAE-MSE-RMSE-R-Squared-%EC%A0%95%EB%A6%AC) <br>

<br>

# 220315
[dummy regressor 새로운 방법; 검색어 dummyregressor 사용방법](https://dlsdn73.tistory.com/820) <br>

- classification 결과 다른 이유는 y 나누는 기준값이 달랐고, 수연이 cv를 안했기 때문.
- reg dummy 결과 달랐던 이유는 scoring 방법이 달랐기 때문. + 나는 cv를 안함.

<br>

# 220316
[cross_val_score는 그냥 x,y만 나눠놓은 데이터 집어넣으면 알아서 비율 짜서 데이터 분할해서 cv 해주는가에 대하여... ==> 이미 전에도 같은 의문을 가졌었으며, 답도 적어둠. cv default == 5 (5회 cross checking)] <br>

[아 모델링이 아니라 cv면 전체 데이터를 넣어야하는거구나.](https://post.naver.com/viewer/postView.nhn?volumeNo=28007428&memberNo=18071586) <br>

cross_val_score()의 default는 r^2(결정계수) <br>
<br>

[교차 검증의 정확도를 간단하게 나타낼 때는 평균으로...](https://jhryu1208.github.io/data/2021/01/24/ML_cross_validation/) <br>

dummy에 cv가 필요한가에 대하여... <br>

<br>

* 잊지 말아햐 할 사항: 다 함수로 만들어서 사용중이기 때문에 입력이 매개변수임. <br>
매개변수가 기존에 만들어둔 변수랑 이름이 같아서 헷갈렸음.
<br>
<br>
<br>


[모델링 단계] <br>
1. data split (1차: x대 y, 2차: xy각각을 train대 test로)

2. model fitting with train data

3. predict with test data

4. scoring with test data, predict data <br>
(얼마나 맞았는지.) <br>

근데 왜 우리가 썼던 reg는 fitting을 하지 않는 가에 대하여... 내가 오로지 cv하는 것에 대한 포스팅을 찾은건가... <br>
회귀모델링 포스팅이었음.. <br>
그냥 그 분이 fitting을 빼먹으신 거인듯. <br>
[error message: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.](https://velog.io/@xdfc1745/The-least-populated-class-in-y-has-only-1-member-which-is-too-few.-The-minimum-number-of-groups-for-any-class-cannot-be-less-than-2) <br>
<br>
```
회귀는 계층적 split이 안됨. 주의!!!!!
sss (Stratified Shuffle Split) 쓰면 안되고, ss (Shuffle Split) 써야함.
```
<br>

[위 에러메세지 잘못된 해석](https://stackoverflow.com/questions/43179429/scikit-learn-error-the-least-populated-class-in-y-has-only-1-member) <br>
~~The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.~~<br>
~~==> 즉, y shape이 (1,)으로 돼있을 것이므로 reshape하란 소리임. (______,1)이 되게끔.~~ <br>
~~y.reshape(-1,1)~~
<br>

[기타 오늘 참고한 블로그들] <br>
[기타 1. cross_val_score() 와 cross_validate() - 아마 cv 할 때 fitting을 알아서 해주는건지 해서 찾아본 듯.](https://tensorflow.blog/tag/cross_val_score/) <br>
[기타 2. 문제 풀이를 통한 reg 이해 - 아마 reg는 예측할 때 fitting을 안하는건가 해서 찾아본 것일듯.](https://post.naver.com/viewer/postView.nhn?volumeNo=28007428&memberNo=18071586) <br>
[기타 3. 머신러닝 학습시 고려해야 할 것: Test data와 CV data - 아마 참고한 예시 블로그에서 cv 할 때 fitting 안한 것 때문에 cv할 때는 fitting 안하는건지, 혹은 알아서 하는건지 해서 찾아본 듯.](https://box-world.tistory.com/23) <br>
[기타 4. 선형회귀 (linear reg) - 회귀는 fitting을 안하는건지 해서 찾아봄.](https://hleecaster.com/ml-linear-regression-example/) <br>
[기타 5. 선형회귀 모델 - def의 매개변수와 함수 생성 전에 만들었던 변수가 같아서 회귀 pred는 df 전체를 사용하는건지 헷갈렸어서 찾아봄.](https://kimdingko-world.tistory.com/101) <br>
[기타 6. 회귀로 예측하기 - 회귀는 예측전에 train으로 fitting 안하나 해서 찾아봄.](https://otexts.com/fppkr/forecasting-regression.html) <br>
==> 논외로 여기 블로그 기능이 신기해서 맘에 들었음. <br>
<br>

[기타 7. 파이썬 회귀분석 기본 사용법 정리 scikit-learn, statsmodels - 회귀는 예측 전에 train으로 fitting 안하나 해서 ...](https://data-newbie.tistory.com/777) <br>
[기타 8. 파이썬으로 선형회귀 분석하기 예제 - 회귀는 예측 전에 train으로 fitting 안하나 해서 ...](https://jimmy-ai.tistory.com/33) <br>
[기타 9. 모델 평가와 성능향상 - 교차검증 - 또 cross_val_score 관련...](https://jhryu1208.github.io/data/2021/01/24/ML_cross_validation/) <br>


<br>


* *오늘의 이슈* <br>
cross_val_score <br>
회귀는 train data로 fitting을 해주지 않는가에 대하여... <br>

* *결론:* <br>
회귀도 fitting 해줘야 하는건데 참고했던 그 블로그가 fitting 과정을 안넣은것 같았음. <br>
cross_val_score는 결국 단순히 score 내는 거라서 fitting을 해주지는 않는다고 판단. <span style = 'color : red'> 아직 확실하지 않음.</span><br>


* 오늘 알게된 것: <br>
list에는 mean함수가 따로 내장돼있지 않다.<br>

* 구문 통째로 알아두면 좋을 lambda함수 사용 예시, 그리고 내장함수 sorted(): <br>
``` python
sorted(dic.items(), key = lambda t : t[1])
# dictionary의 value에 대해 작은 순으로 줄 세워짐.
```

# 220317
* issue
1. 매번 split 값이 같은지 여부 확인 후 다르다면 <br>
shuffle split을 함수 밖에서 실행하고 cv 할 것. <br>

👉 <span style = 'color: red'> 근데 random_state 지정해줘서 같았음. 따라서 새로 뭐 할 필요 없음.</span> <br>

* HW
2. scaling에 대해 공부해올 것. <br>
[교수님께서 보내주신 참고 페이지, 우리가 지금 쓰는 데이터임.](https://inhovation97.tistory.com/m/60) <br>
column 별 minmax 해줘야함. <br>

3. 위 블로그에서 말하는 3번 minmax방식을 사용하려면 <br>
현재 쓰고있는 cv 방식을 바꿀 필요가 있어보임. ~~밖에서 train test 다 나누고, idx로 해야할 것 같음.~~ (??????엥,,, x,y만 나눠야 idx로 cv 할 수 있는 것임.) <br>
==> 흠.... cv 돌리려면 for문 안에서 train test 나누는게 맞아보임. <br>

<br>

# 220320

* 오늘의 python TMI
[np.append() 사용법](https://ponyozzang.tistory.com/506) <br>
[ndarray dtype 한 번에 바꾸기; astype(np.int64)](https://rfriend.tistory.com/285) <br>
<br>

* inhovation~ blog 3단계 이해하기 <br>
[볼린저밴드와 MA20; MA20은 20일 이평선으로, 동시에 볼린저밴드의 중심선이다.](https://psystat.tistory.com/119) <br>
[볼린저밴드 수식](https://grand-unified-engine.tistory.com/21) <br>
<br>

* 한 줄 씩 csv file 생성하는 최대한 기본 함수를 이용한 코드 (결국 csv 모듈을 불러옴...): <br>

* [그냥 open ver] <br>
[현재까지 발견한 최선..? import csv 필요...](https://walknrest.tistory.com/288) <br>
[함께 보기](https://devpouch.tistory.com/55) <br>
<br>

* [with open ver] <br>
[with 문 사용하기](https://twpower.github.io/17-with-usage-in-python) <br>
<br>


``` python
import csv
# 저장할 파일명, 인코딩 타입 입력
f = open('new file.csv', 'w', encoding = 'utf-8-sig')
w = csv.writer(f)
w.writerow(list)
w.writerow(list)
w.writerow(list) # 하다보면 작성됨.

f.close()
```

<br>

* for문의 loop name을 같게해서 for loop이 계속 도는 것인가..?
<br>

# 220330
[FinanceDataReader manual](https://coding-kindergarten.tistory.com/category/%ED%8C%8C%EC%9D%B4%EC%8D%AC%20%ED%8C%A8%ED%82%A4%EC%A7%80/%EC%A3%BC%EC%8B%9D%ED%88%AC%EC%9E%90) <br>

[lambda에 if문 1, lambda는 적용시킬 함수 짜는 것,,, map 결과는 list로 볼 수 있음.](https://dojang.io/mod/page/view.php?id=2360) <br>

[lambda에 if문 2](https://wpaud16.tistory.com/55) <br>

```python
# lambda 예시
list(map(lambda x:x**2, range(5)))
```
<br>

[pandas 불러온 데이터 살펴보기](https://hogni.tistory.com/5) <br>

```python
df.head()
df.shape()
df.info()
df.describe()
df.value_counts()
df.unique()
df.nunique()
```
<br>

[(NaT) null 값 확인하기 1](https://stackoverflow.com/questions/69590754/nattype-object-has-no-attribute-isna) <br>
[(NaT) null 값 확인하기 2](https://pandas.pydata.org/pandas-docs/version/1.0.0/whatsnew/v1.0.0.html) <br>
[notnull, notempty 차이](https://055055.tistory.com/37) <br>

```python
# NaT는 어떻게 판단..?
# 정답
pd.isnull(df[col][n])
pd.notnull(df[col][n])
################################### 이 아래 코드들로는 그 열 내 요소 한 개 판단은 못 함.
pd.NA
np.where(df['col1'].isnull())
df['col1'].isna()
df['col1'].notnull()
df['col1'].notna()
```
<br>

* [파일여닫고 읽고 쓰기](https://nittaku.tistory.com/244)
open으로 파일을 열면 .close()로 닫아줘야함. <br>
with open으로 열면 안닫아줘도 됨. 알아서 닫힘. <br>

# 220407
과제: 
* 존속일 500일 이상.
* 20일 간 거래 없는 수 5일 이하,
* Trading_Value 100억 이상,
* 당일 변동폭(고가/저가) > 1.05 인 종목과 날짜 선별.


# 220409
attention lstm,,, <br>

* LSTM 층을 rnn layer라고 부르는 이유가 궁금했다. <br>
[RNN과 LSTM을 이해해보자!](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/)<br>
[TensolFlow LSTM layer 활용법](https://teddylee777.github.io/tensorflow/lstm-layer) <br>
[[머신러닝 순한맛] LSTM의 모든 것](https://box-world.tistory.com/73) <br>
<br>

[np.squeeze](https://jimmy-ai.tistory.com/101) <br>

<br>

# 220412
* numpy의 unique는 np.unique(ndarray)
<br>

# 220413
* 문자열 판단 method
> str.is~~()로 사용
[ref1_kor](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=zlatmgpdjtiq&logNo=221302490913) <br>
[ref2_eng](https://initialcommit.com/blog/python-isalpha-string-method) <br>

[isdecimal과 숫자인지 판별하는 다른 method의 차이: 지수표현을 문자로 안보고 보고...](https://it-neicebee.tistory.com/33) <br>

[그 외 다양한 str 함수](https://jhproject.tistory.com/158) <br>
<br>

```python
isalpha # 글자인지
isdigit # 숫자인지
isdecimal # 숫자인지
isnumeric # 숫자인지
isalnum # 숫자 또는 글자인지

isspace
isprintable
isidentifier
```
<br>

# 220428 ~ 220512 수업 전 (8w)
* 조건만족cd_dt (by 최종변동폭cd_dt).txt <br>
* 조건만족cd_dt (by ffin).txt <br>
<br>

# 220512 수업 중 ~ (10w)
조건df.csv <br>
<br>

# 220516
[데이터 읽고 쓰고 저장하기 .to_feather, .to_pickle, .to_csv 비교](https://data-newbie.tistory.com/359) <br>

[100GB 이하의 data에서는 partition 방식의 modin.pandas 사용하는 것이 좋음.](https://data-newbie.tistory.com/279?category=750452) <br>

<br>

``` python
## ma_df: 보조지표 추가한 df
## object보다는 category type으로 저장, 사용하는 것이 용량면에서 더 나은 선택일것.

!pip install pyarrow # feather로 저장하려면 pyarrow 설치 먼저

# feather와 pickle은 index 파라미터가 따로 없으므로 reset_index 먼저
ma_df.reset_index(inplace = True)
ma_df.drop('index', axis = 1, inplace = True)





## 용량 ftr < pkl < csv
# feather로 저장, 확장자 .ftr
ma_df.to_feather('보조지표추가_cd_nuniq=2348.ftr')

# 읽기
pd.read_feather("보조지표추가_cd_nuniq=2348.ftr", columns = None, use_threads = True)




# pickle로 저장, 확장자 .pkl
ma_df.to_pickle('보조지표추가_cd_nuniq=2348.pkl')

# 읽기
pd.read_pickle("보조지표추가_cd_nuniq=2348.pkl")

```
<br>

# 220524
[11w] 4 옆으로 10일, 시간단축.ipynb 파일로 전처리 완료, 파일명 d9d0.txt <br>
csv는 도저히 시간이 오래걸려서 포기 <br>
** [11w] 3 파일은 안봐도 됨, 옆으로 10일 하려다 시간 너무 걸려서 버림. <br>
<br>

# 220525
[11w] d9d0.txt to csv, fin_df 저장.ipynb 파일로 <br>
d9d0.csv 생성, <br>
col 구성 바꾼 fin_df.csv 생성 <br>
<br>

[머신러닝] 학습시간 단축? <br>
데이터양이 너무 많아서 시간이 너무 오래걸리는 문제... <br>
[그냥 데이터 증강이 학습속도 저하를 야기한다는 것만 나와있음.](https://www.hankyung.com/it/article/2021073013321)<br>
<span style = 'font-size : 150%'>❗❗❗❗</span>[짱 친절... 100만개 정도 데이터로 머신러닝 수행하는 경우 학습 속도 높이는 방법에 대한 질문](https://www.inflearn.com/questions/30545)<br>

```
<<<필요한 내용만 발췌>>>

따라서 100만개 정도의 데이터가 서버 메모리에 올라갈 수 있는지 부터 확인해야 합니다.
100만개 레코드이지만 Feature가 많지 않다면 충분히 8GB정도에 올라갑니다. ==> 우리 데이터의 경우 10.7GB
먼저 Pandas로 data를 로드 한 뒤에 DataFrame.memory_usage() 로 메모리 사용량을 확인해 보시면 알 수 있습니다. ==> 아직 확인 안해봄.

2. 머신러닝의 속도를 높이는 방법
    A. 속도가 빠른 알고리즘을 적용,
        Tree기반 앙상블보다는 선형 계열이 빠름.
        즉 Logistic Regression > Random Forest
        같은 Tree기반 앙상블이더라도 Random Forest > Gradient Boosting
        XGboost < LightGBM & LightGBM이 메모리도 더 적게 사용

        하지만 예측 정확도(성능)를 더 중요시 한다면 학습속도를 포기해야할 수도 있음.

    B. Multi processing 으로 알고리즘을 적용하는 것.
        서버를 여러개 Core를 가진 시스템으로 구성.
        사이킷런은 멀티 core로 병렬 처리를 지원
        n_jobs=-1을 Estimator 객체에 초기 파라미터로 설정하면 시스템이 가진 모든 CPU 코어를 병렬로 사용하여 학습하게 됨. - n_jobs = 숫자만큼 cpu 사용
        8Core CPU가 1Core CPU보다 더 빠르게 학습함.(그렇다고 8배 빠르지는 않음. 선형 성능 확장에 제약 o).


요약하자면 100만개 Record의 데이터 세트의 피처 갯수가 몇 개이든간에
메모리에만 들어온다면 1~2 시간내에 학습이 가능하며,
만약 학습 시간을 더 줄이고자 한다면 8 Core이상의 시스템에서 구동하시면 훨씬 학습 시간을 개선할 수 있을 것...
```

[딥 러닝 모델 학습을 빠르게 하기 위한 6가지 tip❗](https://info-topnews.tistory.com/7) <br>

```
<<<아직 끝까지 안읽음.>>>

1. 다른 학습률 조정 계획 사용 고려 ???
2. DataLoader 및 페이지 잠금 메모리에서 여러 보조 프로세스 사용 ???
3. 배치 크기 최대화
4. 자동 혼합 정밀 AMP 사용 ???
5. gradient 활성화 checkpoint 사용 ???
6. .tensor() 대신 .as_tensor() 사용 ???
```
<br>


* 분산학습
<br>

... 그냥 바로 lstm으로 갈까... <br>
lstm의 경우 epoch은 줄이고 batch 사이즈는 최대로 키워서 해결...? <br>
<br>

[인공지능 > 머신러닝 > 딥러닝](https://hongong.hanbit.co.kr/ai-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%B0%A8%EC%9D%B4%EC%A0%90-%EC%B4%9D%EC%A0%95%EB%A6%AC/) <br>
```
<<위 페이지에서 볼 내용만 따로 정리>>
딥러닝으로 대표되는 인공신경망은 머신러닝을 구현하는 기술의 하나로,
인간 뇌의 동작 방식에서 착안하여 개발한 학습방법...

[기존(rule-based AI)]에는 규칙을 알려줘야했음. (규칙을 프로그래밍해야 했음.)
==> [머신러닝]은 답안지를 미리 주면 알아서 규칙을 학습함. (규칙을 프로그래밍하지 않아도 됨.)
    대표 라이브러리: <사이킷런>
[인공신경망]은 기존의 머신러인 알고리즘으로 다루기 어려웠던 이미지, 음성, 텍스트 분야에서 뛰어난 성능을 발위, 종종 딥러닝이라고도 부름.
대표 라이브러리: <텐서플로>, <파이토치>

```

# 220526
[시계열 수치입력 수치예측 모델레시피](https://tykimos.github.io/2017/09/09/Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe/) <br>
[lightbgm을 이용한 회귀예측 치트코드](http://machinelearningkorea.com/2019/05/18/lightgbm%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%ED%9A%8C%EA%B7%80%EC%98%88%EC%B8%A1-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%B9%98%ED%8A%B8%EC%BD%94%EB%93%9C/) - 따라해봄.<br>
[lightgbm 공식문서 1 - 파라미터에 대한 보다 더 자세한 설명](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html) <br>
[lightgbm 공식문서 2](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html) <br>
<br>
[lightgbm은 어떻게 사용할까? - sample code](https://greatjoy.tistory.com/72) - lightgbm으로 classification (분류)하기<br>
<br>
<br>
[multi core 멀티코어 참고자료](https://machinelearningmastery.com/multi-core-machine-learning-in-python/) <br>
<br>
<br>
<br>
```python
# cpu 개수 확인 방법
import os
os.cpu_count()
```
<br>

## 220601
[무작정 따라하는 EDA](https://daje0601.tistory.com/106) <br>
[파이썬으로 주식 보조지표 구하기 TA](https://junyoru.tistory.com/136) <br>
[파이싼 주식데이터 분석, 주식 보조지표 확인하는 방법은?](https://tariat.tistory.com/955) <br>

[파이썬, 주식차트와 보조지표 그리기(plotly)](https://sjblog1.tistory.com/45) <br>
<br>

## 220602
### study multi processing
[원작자 코드 보기](https://github.com/inhovation97/Research-Stock-market-Data/blob/main/code/scaling3.py)
-> 여기에 멀티 프로세싱 관련 코드 나와있음 <br>

[검색어]: 멀티 프로세싱 예시 코드 파이썬 <br>
[판다스 멀티 프로세싱 공식 문서](https://docs.python.org/ko/3/library/multiprocessing.html) <br>
[멀티 프로세싱 구현예제 및 멀티 쓰레드와 실행시간 비교 분석](https://ddolcat.tistory.com/665) <br>
[파이썬 multi processing 사용법](https://light-tree.tistory.com/239) <br>
[[병렬 프로그래밍] 3. multi-process 사용하기 with python](https://zephyrus1111.tistory.com/113) - 줄 별로 설명, 친절<br>
4초 정도 걸리는 작업을 단축시키는 예시

[Python multiprocessing.Pool 멀티프로세싱 2](https://tempdev.tistory.com/entry/Python-multiprocessingPool-%EB%A9%80%ED%8B%B0%ED%94%84%EB%A1%9C%EC%84%B8%EC%8B%B1-2) <br>
[Python | Multiprocessing(파이썬 멀티프로세싱)](https://yeonfamily.tistory.com/5) <br>
[6주차, 병렬처리, 프로세스, 쓰레드](https://ish0301.tistory.com/60) - 코드가 예쁘게 나와있지는 않음. <br>
[wikidocs 멀티프로세싱 문서](https://wikidocs.net/85603) <br>

[파이썬 - multiprocessing 설명 및 예제](https://niceman.tistory.com/147) <br>
[multi processing python](https://velog.io/@tmvkrorl/Multi-Processing-python) <br>
[멀티 쓰레드(x) 멀티 프로세싱](https://koreapy.tistory.com/1276) <br>
❤💛💜💨 읽는 중 [Ray를 이용해 Python 병렬 처리 쉽게 하기](https://otzslayer.github.io/python/2021/10/15/multiprocesesing-using-ray.html) - 병렬처리를 하는 이유가 나와있음, 작성자가 multiprocessing 사용법이 맘에 안들었지만 그래도 써봤다고 함.<br>
<br>

멀티 프로세싱을 하는 이유: 큰 테이터에 대한 작업을 더 빠르게 하기위해.......... <br>
파이썬에서 기본으로 제공해주는 multiprocessing이라는 표준 라이브러리 <br>
<br>

multi processing의 Pool 객체 <br>
여러 입력 값에 걸쳐 함수의 실행 병렬처리 <br>
입력 데이터를 프로세스에 분산시키는 방법 제공 <br>
==> 데이터 병렬처리 <br>

```python
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
```
<br>

# 220603
[numpy 배열 저장 및 불러오기](https://gldmg.tistory.com/43) <br>
```python
import numpy as np
data = np.arange(100) # 저장하는 데이터
np.save('my_data.npy', data) # numpy.ndarray 저장. @파일명, @값
data2 = np.load('my_data.npy') # 데이터 로드. @파일명
```

💛💜💨 [사이킷런을 이용해 머신러닝 모델링 해보기](https://brunch.co.kr/@parkkyunga/66) - plot 그리는 것도 나와있음. <br>
[csv to numpy methods, methods3, 5 이용](https://linuxhint.com/python-read-csv-2d-array/) <br>
```
Method 3: Using the Pandas Dataframe
Method 5: Using Pandas Dataframe Values
```
<br>

[np.concatenate, np.stack](https://engineer-mole.tistory.com/234) <br>
```python
np.concatenate([arr1, arr2]) # 아래로 잇기 (열 수가 같아야 함.)
np.concatenate([arr1, arr2], axis = 1) # 옆으로 잇기 (행 수가 같아야 함.)
```
<br>
<br>

# 220604
머신러닝 모델링시 NaN값 있으면 안됨.(보통은 안되는데 되는 경우가 있기도 함. ex) rf...) <br>
그래서 isnull을 확인해주고, <br>
fillna를 해주는 것. <br>
<br>
[머신러닝 모델 입력에 NaN값] 이런거 검색했다가 기억 남. 검색결과 따로 보진 않음.
<br>
<br>

[머신러닝 모델 성능 평가 mse for regression, acc for classification, ...](https://nicola-ml.tistory.com/88) <br>

> RMSE / MSE / logloss # for Regression
> Accuracy / f1-score # for Classification
```
Accuracy를 평가 척도로 사용한다면 균형(Balanced) 데이터에서 사용하시길 권유
불균형 데이터 상태에서는 F1 Score를 이용
```

<br>

[머신러닝 모델 성능 평가 관련 + 예시 코드](https://bhcboy100.medium.com/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%B6%84%EB%A5%98-%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-%EC%A0%95%ED%99%95%EB%8F%84-%EC%A0%95%EB%B0%80%EB%8F%84-%EC%9E%AC%ED%98%84%EC%9C%A8-f1-%EC%8A%A4%EC%BD%94%EC%96%B4-6bf91535a01a) <br>

[회귀/ 분류시 알맞은 metric과 설명](https://mole-starseeker.tistory.com/30) - 아주 친절, 괜찮아 보임, 회귀의 경우 예시가 주식데이터<br>

## 회귀문제
---
실제 값과 모델이 예측하는 값의 차이에 기반을 둔 metric 사용. <br>

대표적으로
- RSS(단순 오차 제곱 합)
- MSE(평균 제곱 오차)
- MAE(평균 절대값 오차)
<br>

RSS : 예측값과 실제값의 오차의 제곱합 <br>
MSE : RSS를 데이터의 개수만큼 나눈 값 <br>
MAE : 예측값과 실제값의 오차의 절대값의 평균 <br>


++  RMSE와 RMAE라는 것도 있는데, <br>
각각 MSE와 MAE에 루트를 씌운 값입니다. <br>
<br>


> MSE의 경우 오차에 제곱이 되기 때문에 **이상치(outlier)를 잡아내는 데 효과적**. <br>
> 틀린 걸 더 많이 틀렸다고 알려주는 것. <br>
> MAE의 경우 **변동치가 큰 지표와 낮은 지표를 같이 예측하는 데 효과적**. <br>
> 둘 다 가장 간단한 평가 방법으로 직관적인 해석이 가능하지만, <br>
> 평균을 그대로 이용하기 때문에 데이터의 크기에 의존한다는 단점이 있음.

>> MSE는 전체 데이터의 크기에 의존하기 때문에 <br>
서로 다른 두 모델의 <u>MSE만을 비교해서</u> **어떤게 더 좋은 모델인지 판단하기 어렵다**는 단점이 있음.

- 이를 해결하기 위한 metric으로 R2 (결정계수)가 있음.

> R2는 **회귀 모델의 설명력을 표현**하는 지표 <br>
> 그 값이 <u>**1에 가까울수록 높은 성능**</u>의 모델 <br>
>> R2의 식에서 분자인 RSS의 근본은 실제값과 예측값의 차이인데, <br>
>> 그 값이 0에 가까울수록 모델이 잘 예측을 했다는 뜻이므로 <br>
>> R2값이 1에 가까워지게 됩니다. <br>

<br>
<br>

- 검색어: [linear regression 하이퍼 파라미터 튜닝] <br>
💛💜💨💫 [[ 핸즈 온 머신러닝 2판 ] pandas, sklearn을 통한 모델 학습과 튜닝은 어떻게 하는 것일까? (3)](https://box-world.tistory.com/44) - GridSearchCV, 여기도 아주 친절 <br>

이전 2개의 포스팅에 결쳐 우리는 지금까지 <u>**문제를 정의**</u>하고 <u>**데이터를 읽어들여 탐색**</u>하였습니다. <br>
그리고 데이터를 training set과 test set으로 나누고 학습을 위한 머신러닝 알고리즘에 주입할 데이터를 자동으로 <u>**전처리**</u>하고 <u>**정제**</u>하는 파이프라인까지 만들어 보았습니다. <br>
이번 포스팅에서는 머신러닝 모델을 선택하고 훈련시켜 세부적으로 튜닝하는 법까지 다뤄보겠습니다. <br>
<br>
<br>

### <현재 LinearRegressor> <br>
대부분 구역의 median house value가 
120000 ~ 265000 사이인 것을 감안하면, <br>
$68628의 오차는 그리 좋은 편은 아닌 것 같습니다. <br>
<br>
이러한 결과는 모델이 과소 적합(Underfit) 되었기 때문이며, <br>
이는 데이터가 부족하거나, 모델이 강력하지 못한 탓 <br>
<u>**우선**</u> **좀 더 복잡한 모델**을 시도해서 어떻게 되는지 확인해보겠습니다. <br>

### <이제 DecisionTreeRegressor> <br>
- 이 모델은 강력, 데이터에서 복잡한 비선형관계를 찾을 수 있음. <br>

평가시 결과가 0.0이 나옴. <br>
=> ***오차가 없다***는 뜻인데, <u>**모델이 완벽할 리는 없으므로**</u> <br>
아마 데이터가 심각하게 과대적합(Overfit) 되었을 확률이 큼. <br>
하지만 이 또한 확신할 수 없으므로 <br>
<u>**training set에서 일부**</u>를 교차검증 (cross-validation) 데이터로 **분리**시켜 <br>
<u>**모델을 평가하는데에 사용해야**</u> 함. <br>
<br>
train_test_split 함수를 사용하여 <u>**training set을**</u> 더 작은 **traing set과 cv set으로 나누고**, <br>
training set에서는 모델 훈련을, <br>
cv set에서는 모델 평가가 이루어지게 하면 됨. <br>

> 혹은 훌륭한 대안으로 sklearn의 <u>**k-fold cross-validation**</u> 기능을 사용할 수 있음. <br>
>> 이는 training set을 fold라 불리는 10개의 subset (k-fold, 작성자의 예시 코드에서 k = 10)으로 <u>**무작위 분할**</u> <br>
>> 그 후 DecisionTree 모델을 <u>**10번 훈련하고 평가**</u>하는데, <br>
>> 이때 매번 다른 하나의 fold를 사용하여 평가하고 나머지 9개는 훈련에 사용. <br>
>> 그리고 10개의 평가 점수가 담긴 배열이 결과가 됩니다. <br>

>  np.sqrt()에 -scores가 들어간 것은
> cross_val_score() 메서드의 scoring 매개변수가
>> 낮을수록 좋은 loss function이 아니라, <br>
>> 클수록 좋은 utility function을 기대하기 때문에 <br>

> 따라서 MSE의 반대 즉 **음숫값을 계산**하는 <u>**neg_mean_squared_error**</u> 함수를 사용함. <br>
그래서 제곱근 계산을 위하여 -scores로 부호를 +로 바꾼 것. <br>

<br>

[핸즈온 머신러닝(3) - 머신러닝 프로젝트 6[마무리]](https://chana.tistory.com/entry/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D3-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-6%EB%A7%88%EB%AC%B4%EB%A6%AC) <br>
[[머신러닝][교차검증, 파라미터 튜닝]](https://kpumangyou.tistory.com/80) <br>
[3.1. 선형 회귀(Linear Regression)](https://ko.d2l.ai/chapter_deep-learning-basics/linear-regression.html) <br>
[[Sklearn] 파이썬 랜덤 포레스트 모델 학습, 하이퍼파라미터 튜닝 - RandomForestClassifier](https://jimmy-ai.tistory.com/29) <br>
<br>

# 220612
[R squred](https://vitalflux.com/r-squared-explained-machine-learning/) <br>


# 220616
[분류성능평가지표 Precision, Recall, Accuracy](https://sumniya.tistory.com/26)
imbalance한 문제에서는 precision과 recall이 유용하게 사용될 수 있음. 두 지표를 동시에 잘 이용한다면 imbalance dataset이 주어진 상황에서 좀 더 좋은 모델을 선택할 수 있지 않을까
[Precision, Recall, F1 score](https://89douner.tistory.com/174)

과제의 경우 예측 모델이 오를 것이라고 예측했는데, 실제로 오르는 지를 평가해야 하므로 정밀도를 더 비중있게 살펴봐야 함.


[정밀도와 재현율 예시와, 오차행렬 안헷갈리는 방법, 분류모델 평가지표](https://jennainsight.tistory.com/entry/%EC%A0%95%EB%B0%80%EB%8F%84precision%EC%99%80-%EC%9E%AC%ED%98%84%EC%9C%A8recall%EC%9D%98-%EC%98%A4%EC%B0%A8%ED%96%89%EB%A0%AC-%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9C)

[💛F1 score가 높을 수록 정교한 모델임.](https://jennainsight.tistory.com/entry/F1-Score-Roc%EA%B3%A1%EC%84%A0-Auc-%EA%B3%84%EC%82%B0%EB%B0%A9%EB%B2%95-scikit-learn-%EC%BD%94%EB%93%9C%EB%A1%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0)

683통계적 검증 

# 220616 교수님 미팅 내용
[교차검증 및 통계적 검정 실습, 유의성 검증](https://www.youtube.com/watch?v=KBZDh6Ho8Rg&list=PLPHtWS04VkUanVXHXhvFEh0GM-dAnzlHr&index=20) <br>
[교차검증 및 통계적 검정](https://www.youtube.com/watch?v=wYW3gcSQTR4&list=PLPHtWS04VkUanVXHXhvFEh0GM-dAnzlHr&index=21)
