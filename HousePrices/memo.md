<!-- 以下のフォーマットに沿って取り組んだことと結果をまとめていこう -->

# House Prices - Advanced Regression Techniques
[Competition's URL](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

<!-- ここから -->
-------------------------------------------------------------------------------------------------
- 投稿者 : 土屋
- 日付 : 2022/01/01
- 使用データ : ver1
<br>

- **取り組み・変更点**
    - モデルの手法をXGBoostからLightGBMに変更
    - 特徴量に「XXX」、「YYY」を追加
    - ハイパーパラメータ「Z」を0.3から0.4に変更
- **結果**
    - 精度・スコア : 0.1 -> 0.3
    - Public LB : 100 -> 30
- **今後の計画**
    - Optunaを使ってハイパーパラメータ自動調整
    - LightGBMとNNのアンサンブル
- **伝達事項**
    - 特になし
- **備考**
    - 特になし
<!-- ここまで -->

-------------------------------------------------------------------------------------------------
- 投稿者 : 土屋
- 日付 : 2022/01/10
<br>

- **取り組み・変更点**
    - LGBMRegressorを使用したベースモデルの作成
- **結果**
    - Public Score : 0.16339
- **今後の計画**
    - EDAと特徴量エンジニアリング
- **伝達事項**
    - 特徴量エンジニアリングを全くしていないので、Public Scoreはまだまだ改善できる余地あり
- **備考**
    - 特になし