---
layout: post
title: Are you still diversifying using the Markowitz model? Welcome to the 21st century.
date: 2020-07-15
description: A better alternative to the Markowitz model is Machine Learning and Statistics.
img: markowitz.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Risk Management, Portfolio Management, Machine Learning, Python]
---

This article is Part 1 of a series of articles on Diversification.

Most retail traders aka "dumb money" don't diversify.

Contrastingly, professional portfolio managers diversify:

1. across markets;
2. across sectors;
3. across asset types;
4. across investment strategies.

Professional portfolio managers are more afraid of the occasional downswings and crashes than retails traders are. And rightly so. The best managers monitor their exposure and seek to re-balance it to according to their perceived optimum portfolio weights. They understand that diversification is key to minimizing risk.


The most popular, most used model for calculating the optimal portfolio weights is the Markowitz model and the Efficient Frontier.  
**This is now considered ancient technology.**

There are 2 main issues with the Markowitz Model:
1. **Markowitz is always overfitting the training data.**. 
    
    Markowitz's result was an optimization process for the weights that would have been profitable in the past. Markowitz model is not learning a set of rules for predicting the future. Instead, this model can be summed up as "Has this portfolio worked well in the past 10 years? Then it is sure to work well for the next 10!". You can change the 10 year for whatever timestep you like - you'll still be overfitting.
    
    
    
2. **It finds the weights that maximize Sharpe ratio. Sharpe ratio sucks.**
    
    > "What?! But everybody is using Sharpe ratio!"
    
    Using Sharpe for showcasing portfolio metrics to clients is ok; using it for daily quant work is not ok. Sharpe ratio's denominator is the standard deviation of returns, which is a good measure of variability, not a good measure of risk. Sharpe ratio penalizes large positive swings, penalizes accelerating returns, and fails to penalize deaccelerating returns. For experimental evidence, see: https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/

 

# Enter Machine Learning.

There are several ways we can frame the problem of portfolio optimization as a Machine Learning problem:

1. **Reinforcement Learning**: learning the optimal increase/decrease in portfolio weights.

2. **Supervised Learning**: learning the optimal portfolio weights for the next N days/months/years.

3. **Unsupervised Learning**: learning clusters of assets, according to their price and fundamentals similarity.

Today, I will focus on the latter problem. Learning groups of assets can identify how to diversify according to historical data.

We will be starting with the following dataset:


```bash
cd ..
```


```python
import pandas as pd

data = pd.read_csv("data/stocks.csv")
data
```




<div style="width: 100%;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>ticker</th>
      <th>returns_yoy</th>
      <th>returns_mean</th>
      <th>returns_std</th>
      <th>returns_kurt</th>
      <th>returns_skew</th>
      <th>current_assets_chg</th>
      <th>total_assets_chg</th>
      <th>current_liabilities_chg</th>
      <th>...</th>
      <th>gross_profit_chg</th>
      <th>operating_income_chg</th>
      <th>ebit_chg</th>
      <th>ebitda_chg</th>
      <th>net_income_chg</th>
      <th>cash_flow_chg</th>
      <th>gics_sector</th>
      <th>gics_industry_group</th>
      <th>gics_industry</th>
      <th>gics_sub_industry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1984</td>
      <td>ABT</td>
      <td>-0.023327</td>
      <td>-0.000081</td>
      <td>0.015751</td>
      <td>1.137280</td>
      <td>0.150682</td>
      <td>0.199012</td>
      <td>0.123607</td>
      <td>0.192155</td>
      <td>...</td>
      <td>0.091034</td>
      <td>0.119551</td>
      <td>0.119551</td>
      <td>0.142322</td>
      <td>0.158099</td>
      <td>0.185255</td>
      <td>Health Care</td>
      <td>Health Care Equipment &amp; Services</td>
      <td>Health Care Equipment &amp; Supplies</td>
      <td>Health Care Equipment</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1984</td>
      <td>ADM</td>
      <td>-0.026099</td>
      <td>-0.000104</td>
      <td>0.020297</td>
      <td>3.388723</td>
      <td>0.910988</td>
      <td>-0.010240</td>
      <td>0.016783</td>
      <td>-0.442166</td>
      <td>...</td>
      <td>0.090505</td>
      <td>0.239521</td>
      <td>0.239521</td>
      <td>0.184024</td>
      <td>0.068358</td>
      <td>0.085822</td>
      <td>Consumer Staples</td>
      <td>Food, Beverage &amp; Tobacco</td>
      <td>Food Products</td>
      <td>Agricultural Products</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1984</td>
      <td>AIR</td>
      <td>0.632523</td>
      <td>0.002427</td>
      <td>0.021781</td>
      <td>4.852271</td>
      <td>1.239041</td>
      <td>0.328328</td>
      <td>0.233089</td>
      <td>-0.053046</td>
      <td>...</td>
      <td>0.165023</td>
      <td>0.072245</td>
      <td>0.072245</td>
      <td>0.087333</td>
      <td>0.605367</td>
      <td>0.327884</td>
      <td>Industrials</td>
      <td>Capital Goods</td>
      <td>Aerospace &amp; Defense</td>
      <td>Aerospace &amp; Defense</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1984</td>
      <td>AP</td>
      <td>0.254065</td>
      <td>0.001079</td>
      <td>0.015319</td>
      <td>6.334656</td>
      <td>1.158865</td>
      <td>0.473669</td>
      <td>0.463316</td>
      <td>0.484195</td>
      <td>...</td>
      <td>0.723608</td>
      <td>-1.143189</td>
      <td>-4.665928</td>
      <td>4.393316</td>
      <td>-4.411381</td>
      <td>2.389767</td>
      <td>Materials</td>
      <td>Materials</td>
      <td>Metals &amp; Mining</td>
      <td>Steel</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1984</td>
      <td>APA</td>
      <td>0.354090</td>
      <td>0.001460</td>
      <td>0.023034</td>
      <td>1.213794</td>
      <td>0.306496</td>
      <td>-0.103468</td>
      <td>-0.083883</td>
      <td>0.176250</td>
      <td>...</td>
      <td>-0.125055</td>
      <td>-0.006917</td>
      <td>-0.058889</td>
      <td>0.037832</td>
      <td>-0.028582</td>
      <td>0.121321</td>
      <td>Energy</td>
      <td>Energy</td>
      <td>Oil, Gas &amp; Consumable Fuels</td>
      <td>Oil &amp; Gas Exploration &amp; Production</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16034</th>
      <td>2018</td>
      <td>XYL</td>
      <td>0.438349</td>
      <td>0.001772</td>
      <td>0.010812</td>
      <td>2.641754</td>
      <td>-0.159363</td>
      <td>0.011106</td>
      <td>0.052770</td>
      <td>0.262727</td>
      <td>...</td>
      <td>0.094543</td>
      <td>0.197802</td>
      <td>0.163511</td>
      <td>0.149693</td>
      <td>0.658610</td>
      <td>0.327869</td>
      <td>Industrials</td>
      <td>Capital Goods</td>
      <td>Machinery</td>
      <td>Industrial Machinery</td>
    </tr>
    <tr>
      <th>16035</th>
      <td>2018</td>
      <td>YUM</td>
      <td>0.304004</td>
      <td>0.001204</td>
      <td>0.010132</td>
      <td>8.515445</td>
      <td>0.365575</td>
      <td>-0.518548</td>
      <td>-0.222369</td>
      <td>-0.139550</td>
      <td>...</td>
      <td>0.074187</td>
      <td>-0.168417</td>
      <td>0.040901</td>
      <td>-0.024227</td>
      <td>0.150746</td>
      <td>-0.204243</td>
      <td>Consumer Discretionary</td>
      <td>Consumer Services</td>
      <td>Hotels, Restaurants &amp; Leisure</td>
      <td>Restaurants</td>
    </tr>
    <tr>
      <th>16036</th>
      <td>2018</td>
      <td>ZBH</td>
      <td>-0.105038</td>
      <td>-0.000394</td>
      <td>0.013195</td>
      <td>3.860412</td>
      <td>-0.492598</td>
      <td>-0.030100</td>
      <td>-0.072546</td>
      <td>-0.219370</td>
      <td>...</td>
      <td>-0.006197</td>
      <td>-0.958179</td>
      <td>-0.260979</td>
      <td>-0.170304</td>
      <td>-1.209064</td>
      <td>-0.605724</td>
      <td>Health Care</td>
      <td>Health Care Equipment &amp; Services</td>
      <td>Health Care Equipment &amp; Supplies</td>
      <td>Health Care Equipment</td>
    </tr>
    <tr>
      <th>16037</th>
      <td>2018</td>
      <td>ZEN</td>
      <td>0.534857</td>
      <td>0.002201</td>
      <td>0.022210</td>
      <td>3.768737</td>
      <td>0.040686</td>
      <td>0.664630</td>
      <td>1.093239</td>
      <td>0.494517</td>
      <td>...</td>
      <td>0.379028</td>
      <td>0.299418</td>
      <td>0.299418</td>
      <td>0.366408</td>
      <td>0.283363</td>
      <td>0.346874</td>
      <td>Information Technology</td>
      <td>Software &amp; Services</td>
      <td>Software</td>
      <td>Application Software</td>
    </tr>
    <tr>
      <th>16038</th>
      <td>2018</td>
      <td>ZTS</td>
      <td>0.454571</td>
      <td>0.001804</td>
      <td>0.011308</td>
      <td>5.094913</td>
      <td>0.584154</td>
      <td>0.043159</td>
      <td>0.255183</td>
      <td>0.117916</td>
      <td>...</td>
      <td>0.112302</td>
      <td>0.108197</td>
      <td>0.098801</td>
      <td>0.119920</td>
      <td>0.652778</td>
      <td>0.281915</td>
      <td>Health Care</td>
      <td>Pharmaceuticals, Biotechnology &amp; Life Sciences</td>
      <td>Pharmaceuticals</td>
      <td>Pharmaceuticals</td>
    </tr>
  </tbody>
</table>
<p>16039 rows × 23 columns</p>
</div>



Each line describes an instance. Each instance contains quantitative information about a given stock for a given year, as well as its GICS sector, industry_group, industry, and sub_industry.

We select the top50 industries with that contain the most instances:


```python
top50 = data.gics_industry.value_counts().iloc[:50]
top50
```




    Machinery                                         1379
    Oil, Gas & Consumable Fuels                       1312
    Chemicals                                          916
    Specialty Retail                                   899
    Energy Equipment & Services                        779
    Aerospace & Defense                                634
    Health Care Equipment & Supplies                   582
    Electronic Equipment, Instruments & Components     485
    Hotels, Restaurants & Leisure                      480
    Food Products                                      463
    Metals & Mining                                    452
    Commercial Services & Supplies                     424
    Health Care Providers & Services                   387
    Containers & Packaging                             362
    Textiles, Apparel & Luxury Goods                   302
    IT Services                                        301
    Construction & Engineering                         272
    Professional Services                              270
    Household Durables                                 255
    Building Products                                  250
    Auto Components                                    250
    Media                                              241
    Life Sciences Tools & Services                     226
    Multiline Retail                                   223
    Pharmaceuticals                                    221
    Household Products                                 203
    Electrical Equipment                               203
    Diversified Consumer Services                      197
    Trading Companies & Distributors                   194
    Equity Real Estate Investment Trusts (REITs)       170
    Food & Staples Retailing                           167
    Diversified Telecommunication Services             166
    Technology Hardware, Storage & Peripherals         160
    Road & Rail                                        157
    Leisure Products                                   149
    Capital Markets                                    137
    Personal Products                                  136
    Automobiles                                        134
    Paper & Forest Products                            125
    Industrial Conglomerates                           125
    Entertainment                                      118
    Software                                           115
    Marine                                             106
    Tobacco                                            103
    Beverages                                          101
    Airlines                                            90
    Construction Materials                              81
    Real Estate Management & Development                80
    Gas Utilities                                       65
    Air Freight & Logistics                             63
    Name: gics_industry, dtype: int64



And filter our data so that it only contains those GICS industries.


```python
data = data[data.gics_industry.isin(top50.index)]
```

We then standardize the numerical information:


```python
quantitative_vars = data.select_dtypes('number').columns
quantitative_data = data.loc[:, quantitative_vars]
data.loc[:, quantitative_vars] = (quantitative_data - quantitative_data.mean()) / quantitative_data.std()
```

We can now calculate and visualize the dissimilarity between industries, measured by the [Maximum Mean Discrepancy](http://jmlr.csail.mit.edu/papers/v13/gretton12a.html) between the samples of the different industries).


```python
from qclustering import dissimilarity_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

dmatrix = dissimilarity_matrix(data, 'gics_industry')

fig, ax = plt.subplots(figsize=(30,30))
cbar_kws = cbar_kws={'shrink': .8, 'label':'Distance'}
sns.heatmap(dmatrix, ax=ax, vmin=0, square=True, cmap='Blues', cbar_kws=cbar_kws)
fig.axes[-1].yaxis.label.set_size(23)
fig.axes[-1].tick_params(labelsize=18)
```


![Stock Industries Dissimilarities]({{site.baseurl}}/assets/img/industries_dissimilarities.png)


We can also frame this as a Hierarchical Clustering problem and use MMD as a linkage metric between industries and clusters of industries:


```python
from qclustering import hierarchical_clustering, plot_dendrogram
initial_clusters, linkage = hierarchical_clustering(data, 'gics_industry')
plot_dendrogram(initial_clusters, linkage, color_threshold=0.09, above_threshold_color='#CCCCCC');
```


![Stocks Clustering]({{site.baseurl}}/assets/img/stocks_clustering.png)


Based on the chart above, we can see that some industries are being clustered with industries of different sectors. Therefor, the data indicates that we're better off diversifying by industries than diversifying by sectors.
  
  
---
  
  
This concludes Part 1.

In Part 2, we will look at ML-based diversification strategies at compare their foward testing / out-of-sampling results, including some methods from the [MlFinLab python package](https://github.com/hudson-and-thames/mlfinlab).
