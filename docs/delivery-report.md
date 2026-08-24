# v1.0.0 teslim raporu

## Tamamlanan özellikler

- NumPy tabanlı `StandardScaler` ve tekrar üretilebilir `train_test_split`
- Batch gradient descent kullanan Linear Regression
- Euclidean distance ve deterministik eşit oy çözümü kullanan KNN
- Çoklu rastgele başlangıç, inertia ve boş küme yönetimi içeren K-Means
- Kovaryans matrisi ve `numpy.linalg.eigh` kullanan PCA
- MSE, R², accuracy ve confusion matrix metrikleri
- Dört algoritma için terminal özetleri ve PNG çıktıları üreten CLI
- Scikit-learn referans karşılaştırmaları ve tekrar üretilebilir deneyler

## Matematiksel kavramlar

Ortalama, varyans, standart sapma, z-skoru, vektörler, Euclidean distance,
çoğunluk oylaması, MSE, türev, gradyan, gradient descent, R², centroid,
inertia, kovaryans, özdeğer, özvektör, explained variance ve boyut azaltma.

## Test ve kalite sonucu

- Test sayısı: **47**
- Yerel sonuç: **47 passed**
- Statement coverage: **%92**
- Ruff lint: başarılı
- Ruff format kontrolü: başarılı
- Ana algoritma modüllerinde scikit-learn import edilmediğini doğrulayan test: başarılı
- Python sürümleri: 3.11, 3.12 ve 3.13 GitHub Actions matrisinde başarılı
- CI koşusu: [GitHub Actions run 32785047061](https://github.com/agirkayamemin/ml-foundations-from-scratch/actions/runs/32785047061)

## CLI örnekleri

```bash
python -m ml_foundations demo linear-regression --output-dir outputs
python -m ml_foundations demo knn --output-dir outputs
python -m ml_foundations demo kmeans --output-dir outputs
python -m ml_foundations demo pca --output-dir outputs
```

Doğrulanmış örnek sonuçlar: Linear Regression R² `0.9971`, KNN Iris accuracy
`0.9000`, sentetik K-Means ARI `1.0000`, PCA ilk iki bileşen toplam explained
variance ratio `0.9581`.

## Release

- Sürüm: `v1.0.0`
- Release: [v1.0.0](https://github.com/agirkayamemin/ml-foundations-from-scratch/releases/tag/v1.0.0)

## Bilinen sınırlamalar

- Uygulamalar büyük ölçekli performanstan çok okunabilirliğe öncelik verir.
- KNN bütün eğitim örneklerine uzaklık hesaplar.
- K-Means başlangıcı K-Means++ yerine veri içinden rastgele örnekleme kullanır.
- Linear Regression yalnızca batch gradient descent sağlar.
- PCA çok yüksek boyutlu veride doğrudan SVD kadar uygun değildir.
- Train/test ayrımı stratified değildir.

## Gelecekte yapılabilecekler

- Regularization ve mini-batch optimizasyon
- Weighted KNN ve ek uzaklık metrikleri
- K-Means++
- SVD tabanlı PCA ve whitening
- Stratified split ve cross-validation
