# Collaborative Filtering with Temporal Dynamics

这篇文章的主题在于使推荐模型考虑到用户偏好以及基准偏移评分根据时间而变化的因素(所以产生的模型叫TimeSVD++模型).主要的工作就是给SVD++模型的几个参数加上了时间因素.

## 商品基准偏移方面

作者认为一个商品的基准偏移评分不会在一天之中发生过于突然的变化,因此是考虑到了一个近期的时间段的因素.改版之后的物品基准偏移评分为
<img src="http://chart.googleapis.com/chart?cht=tx&chl=b_i(t)=b_i%2Bb_{i,Bin(t)}" style="border:none;">
这里的Bin(t)表示一个近期的范围,Bin的大小可以调节,作者给的范围是在1~30之间.

## 用户基准偏移方面

对于一个用户来说,样本数量按照时间分割成bin之后可能并不均匀,因此作者是给出了一个线性的长期偏移
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\alpha_u\cdot dev_u(t)" style="border:none;">
其中alpha是可学习的系数,dev是时间偏移.  作者也尝试使用了样条曲线来做一个高次的模型,但是结果没有显著提升.

另一方面,用户的一个特点是用户的心情等方面可能会在一天内产生波动.因此作者为每个用户加入了一个"当天评分偏移"因素.最终的用户基准偏移评分模型为
<img src="http://chart.googleapis.com/chart?cht=tx&chl=b_u(t)=b_u%2B\alpha_u\cdot dev_u(t)%2Bb_{u,t}" style="border:none;">

## 用户偏好方面

和基准偏移类似,对每一个隐式偏好考虑了长期变化和当前变化,最终用户的偏好表示为
<img src="http://chart.googleapis.com/chart?cht=tx&chl=p_{uk}(t)=p_{uk}%2B\alpha_{uk}\cdot dev_u(t)%2Bp_{uk,t}\ k=1,2...,f" style="border:none;">
> 作者在这里并没有尝试将用户的偏好变化建模为高次函数,以后可以尝试一下.

## timeSVD++

最终模型的预测值为
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\hat{r}_{ui}(t)=\mu%2Bb_i(t)%2Bb_u(t)%2Bq_i^T(p_u(t)%2B|R(u)|^{-\frac{1}{2}}\sum_{j\in R(u)}y_j)" style="border:none;">
模型的形式和SVD++还是一样的,只是增加了时间参数.

## 收获
* 时间对于评分也是一个重要的因素.
* 高次模型可以不直接用多项式拟合,而是用样条曲线代替.
