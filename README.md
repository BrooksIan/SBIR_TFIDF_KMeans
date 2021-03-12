## Data Science In Apache Spark
### Using TF / IDF -> K-Means Clustering & LSH (MinHash)
### SBIR

**Language**: Scala and Python

**Requirements**: 
- Spark 2.x

**Author**: Ian Brooks

**Follow** [LinkedIn - Ian Brooks PhD] (https://www.linkedin.com/in/ianrbrooksphd/)

**Source Data** [SBIR Search Site](https://www.sbir.gov/sbirsearch/award/all)

**Addtional Information**: The Small Business Innovation Research (SBIR) program is a highly competitive program that encourages domestic small businesses to engage in Federal Research/Research and Development (R/R&D) that has the potential for commercialization. Through a competitive awards-based program, SBIR enables small businesses to explore their technological potential and provides the incentive to profit from its commercialization. By including qualified small businesses in the nation's R&D arena, high-tech innovation is stimulated and the United States gains entrepreneurial spirit as it meets its specific research and development needs.

**File Upload:** Upload the source data json  files to HDFS in the /tmp directory

**Links:**
* [Term Frequency–Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* [PySpark: CountVectorizer|HashingTF](https://towardsdatascience.com/countvectorizer-hashingtf-e66f169e2d4e)
* [Spark API- Feature Extractors](https://spark.apache.org/docs/2.2.3/ml-features.html#countvectorizer)
* [Scalable Jaccard similarity using MinHash and Spark](https://towardsdatascience.com/scalable-jaccard-similarity-using-minhash-and-spark-85d00a007c5e)
* [Spark - MinHash for Jaccard Distance](https://george-jen.gitbook.io/data-science-and-apache-spark/minhash-for-jaccard-distance)
* [Detecting Abuse at Scale: Locality Sensitive Hashing at Uber Engineering](https://databricks.com/blog/2017/05/09/detecting-abuse-scale-locality-sensitive-hashing-uber-engineering.html)
* [Read multiline JSON in Apache Spark](https://stackoverflow.com/questions/38545850/read-multiline-json-in-apache-spark)
* [Effectively Pre-processing the Text Data Part 1: Text Cleaning](https://towardsdatascience.com/effectively-pre-processing-the-text-data-part-1-text-cleaning-9ecae119cb3e)


![SBIR](https://s11759.pcdn.co/wp-content/uploads/2018/04/SBIR_logo.jpg "SBIR")

**IDF / TF Overview:**
![TF](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/TFIDF.jpg "tf")

1. Import Data - Use the provided Jypter Notebook
2. Prepare Text for Models - Tokeniziation
![TF2](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/tfidf_detail.png "tf2" )
3. 

**LSH - MinHash Overview:**
![LSH1](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/MinHashBuckets.png "tf3")
![LSH2](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/benchmark-minhashlsh-algorithm-on-spark-5-638.jpg "TF4")





