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
