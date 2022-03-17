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

👉<span style = 'color: red'>근데 random_state 지정해줘서 같았음. 따라서 새로 뭐 할 필요 없음.</span> <br>

* HW
2. scaling에 대해 공부해올 것. <br>
[교수님께서 보내주신 참고 페이지, 우리가 지금 쓰는 데이터임.](https://inhovation97.tistory.com/m/60) <br>
column 별 minmax 해줘야함. <br>


