# stock_data
stock data handling and utilization <br>
- 1st: DSML test <br>
- 2nd: Capstone test <br> - pandas사용 ver ; with open 구문 ver <br>
<br><br>
정확도<br>
---- 검색어: scikit-learn 회귀모델 <br>
            'regression model 종류'는 별 도움 안됨 <br>
[다양한 회귀모델들](https://cleancode-ws.tistory.com/109) <br>
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

[np.where(condition, x, y); 조건 충족시 x, 그렇지 않으면 y 반환.](https://www.delftstack.com/ko/howto/python-pandas/how-to-create-dataframe-column-based-on-given-condition-in-pandas/)