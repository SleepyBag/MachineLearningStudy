# Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model
---

## 临近模型

### 传统临近模型

临近模型与SVD模型的不同之处在于:SVD直接对用户进行数值建模,而临近模型的数值模型都是针对商品的.在预测用户评分时:

* SVD模型直接计算用户与商品的相似度.
* 临近模型从用户评分过的商品中找出与被预测商品最相似的k个商品,然后以商品间的相关度为权值计算该用户对这些商品评分的加权和,作为用户对商品的预测偏好

### 本文的临近模型

与传统临近模型的区别在于:

* 先取与被预测商品相似的k个商品,然后取这些商品与用户评分过的商品的交集(这样最后纳入考虑的不一定为k个商品)
* 与受限PMF方法类似,把用户"评过"一个商品与否加入了考虑(此时不考虑评分)

## 潜在因素模型

指包括SVD模型在内的预测因素难以人工解释的模型.

### 对称SVD

本文提出的对称SVD基于SVD模型提出,与SVD模型的区别在于:

* 不对用户直接进行数值建模,而是根据用户以往的评分预测用户偏好
* 加入了"评过"因素(不考虑评分).其实与受限PMF差不多,只是把系数加了个根号,改小了一点

### 我的疑问
> 这个模型似乎和临近模型区别不大.
> 临近模型的优化目标为
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\mu%2Bb_u%2Bb_i%2B|R^k(i;u)|^{-1/2}\sum_{j\in R^k(i;u)}(r_{uj}-b_{uj})w_{ij}%2B|N^k(i;u)|^{-1/2}\sum_{j\in N^k(i;u)}c_{ij}" style="border:none;">
> 对称SVD的优化目标为
<img src="http://chart.googleapis.com/chart?cht=tx&chl=b_{ui}%2Bq_i^T(|R(u)|^{-1/2}\sum_{j\in R(u)}(r_{uj}-b_{uj})x_j%2B|N(u)|^{-1/2}\sum_{j\in N(u)}y_j)" style="border:none;">
> 可以看到当临近模型的k取无穷大时,二者的求和范围是一样的.临近模型变成
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\mu%2Bb_u%2Bb_i%2B|R(u)|^{-1/2}\sum_{j\in R(u)}(r_{uj}-b_{uj})w_{ij}%2B|N(u)|^{-1/2}\sum_{j\in N(u)}c_{ij}" style="border:none;">
> 而事实上无论对称SVD跑出什么结果,只要令临近模型中的参数
<img src="http://chart.googleapis.com/chart?cht=tx&chl=w_{ij}=q_i^Tx_j" style="border:none;">
<img src="http://chart.googleapis.com/chart?cht=tx&chl=c_{ij}=q_i^Ty_j" style="border:none;">
> 那么后面两项就完全一样了,再调整一下前面的三个参数,临近模型就能跑出和对称SVD一样的结果.
> 但是这个结论反过来是不成立的.因为对称SVD中q向量的长度,也就是潜在因素的数量,已经决定了其结果矩阵R的秩的上限.所以是不能通过赋值来使其等同于这里的临近模型的.
> 如果上面的想法是正确的,那么其实对称SVD实际上是一个不存在的模型= =只是上面这个临近模型的特例.

### SVD++

文章给出的SVD++模型进一步印证了我的疑问.
* 相比对称SVD:SVD++放弃了根据用户以往评分来对用户建模的方法,又用回了朴素SVD模型的p向量.绕了一圈又回来了...而作者给出的理由是"比对称SVD跑出的结果比较好",并没有其他解释.根据这个说法我再次认为对称SVD其实和文中增广的临近模型没有什么区别.
* 相比SVD:SVD++增加了对"评过与否"的考虑.(与受限PMF基本一致,只是系数加了一个根号来减小影响)

## 收获

* SVD++模型.
* 对临近推荐算法的了解.
