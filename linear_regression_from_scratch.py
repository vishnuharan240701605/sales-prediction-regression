import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("https://www.statlearning.com/s/Advertising.csv")
TV=data["TV"]
sales=data["sales"]

x=np.array(TV)
y=np.array(sales)

x_mean=np.mean(x)
y_mean=np.mean(y)

sum_xy=0
sum_x=0

for i in range(len(x)):
    sum_xy+=(x[i]-x_mean)*(y[i]-y_mean)
    sum_x+=(x[i]-x_mean)**2

m=sum_xy/sum_x
c=y_mean-(m*x_mean)

print("slope-> ",m,"\n")
print("Intercept-> ",c)

y_pred=m*x+c

def predict(x):
    return m * x + c

spend = float(input("Enter TV advertising spend (in $1000s): "))
prediction = predict(spend)

print(f"Predicted Sales: {prediction:.2f} (in $1000s)")

plt.figure(figsize=(6,6))
plt.scatter(x,y,marker="*",color="red",label=" Actual points")
plt.plot(x,y_pred,color="blue",label="Best fit line")
plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("Sales predictor")
plt.legend()
plt.show()


r2=1-sum((y_pred-y)**2)/sum((y_mean-y)**2)
print(r2)