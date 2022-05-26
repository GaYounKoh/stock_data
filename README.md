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
