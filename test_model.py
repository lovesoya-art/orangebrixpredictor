import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

model = joblib.load('brix_model.joblib')
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(np.array([[0, 0, 0]]))

test = np.array([[25, 15, 12]])
test_poly = poly.transform(test)
pred = model.predict(test_poly)[0]

print('✓ 테스트 성공!')
print(f'입력: 최고기온=25°C, 최저기온=15°C, 가조시간=12시간')
print(f'예측된 Brix: {pred:.2f}')
