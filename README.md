## Data Science In Apache Spark
## Natural Language Processing Use Case
### Using TF / IDF -> K-Means Clustering & LSH (MinHash)
### On Data From Small Business Innovation Research (SBIR)

![SBIR](https://s11759.pcdn.co/wp-content/uploads/2018/04/SBIR_logo.jpg "SBIR")

**Language**: Scala and Python

**Requirements**: 
- Spark 2.4.X (Or Greater)

**Author**: Ian Brooks

**Follow** [LinkedIn - Ian Brooks PhD] (https://www.linkedin.com/in/ianrbrooksphd/)

## Project Goals
The goal is to use Open Source tools to build a Natural Language Processing (NLP) model for Inverse Document Frequency / Term Frequency (IDF/TF) for the purpose of discovering documents that are related to each other.  In this project, the IDF/TF model will be used as to create features, from text strings in the original documents, that represent that weight of each word term in the corpus of the text documents.  

## Source of Datasets 

[SBIR Search Site](https://www.sbir.gov/sbirsearch/award/all)

The data will be sourced from SBIR's funding awards from 2019, and the hashing features will only be trained on the words in the abstract section of each of awards.

**Addtional Information**: The Small Business Innovation Research (SBIR) program is a highly competitive program that encourages domestic small businesses to engage in Federal Research/Research and Development (R/R&D) that has the potential for commercialization. 

## Background Information - Inverse Document Frequency / Term Frequency (TF-IDF)
![TF](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/TFIDF.jpg "tf")

In information retrieval, tf–idf, TFIDF, or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general. tf–idf is one of the most popular term-weighting schemes today.

![WC](word-clouds.png "WC")

## Feature Engineering - Tokenizer
The purpose of using TF/IDF for feature engineering is to help the downstream ML models understand the weight or imporpantace of a word or search term.  This will allow these models to find documents that are related to each other when they are trained on these values.  The TF/IDF model tokenizes the text terms, which looks like the following image. 
![FE2](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/tfidf_detail.png "tf2" )

Using the provided PySpark code, the results of Term Frequency hashing are displayed in the following image.  The project is using the text terms provided by the abstracts from the original documents.  The tokenized values are listed in the column called features.  

![FE1](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/featureEng.png "Fe2")

## ML Model Building

Once these features have been created, they can be used to train downstream Machine Learning models or Hashing based models that are designed to find similar documentd.  This project will be providing an example of Unsuperivsed ML K-Means Clustering and Locality-Sensitive Hashing (LSH) - MinHash.  They were selected because they are both provided in Aapche Spark ML Lib.

 ## Project Build

 ### Import Data - Use the provided Jypter Notebook or run downloadData.sh

```bash
./downloadData.sh
```

![NB](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/notebook.png)

## Option 1 - Unsupervised ML KMeans Clustering - Unsupervised ML

![Kmeans3](kmeansCLusters.jpg "kmeans3")

Document Clustering is common technque used to find similar documents based on key words or other search features.  This project will demonstrate Clustering (K-Means), and find the documents that cluster with each other.  The model is trainined on the Term Frequency Hashing values, and this unsupervised ML approach is well suited for this dataset, since there is no labeled data.  

Once the model has been trained, the results from K-Means will display the documents and their assigned cluster.  Keep in mind, these documents clustered together based on 2 factors: the words in their abstracts and the value of K used when the model was trained.  In K-Mean, K is the predetermined number of clusters that will be used as buckets.  

### KMeans - Results

In the following image, the results show the first 20 documents and their cluster assignments.  
![KMeans1](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/ClusterByDocs.png "kmeans1")

In the following image, the results displays the documents that were assigned to one particular cluster.  This example is using cluster number lucky 13.  
![KMeans](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/clusterResults.png "kmeans")

## Option 2 - Locality Sensitive Hashing - MinHash

Locality-Sensitive Hashing (LSH) is an algorithmic technique that hashes similar input items into the same "buckets" with high probability.  Since similar items end up in the same buckets, this technique can be used for data clustering and nearest neighbor search. It differs from conventional hashing techniques in that hash collisions are maximized, not minimized.

### Common LSH Applications

* Near-duplicate detection
* Audio similarity identification
* Nearest neighbor search
* Audio fingerprint
* Digital video fingerprinting

![LSH1](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/MinHashBuckets.png "lsh1")
![LSH2](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/benchmark-minhashlsh-algorithm-on-spark-5-638.jpg "lsh2")

### LSH - Set Search Terms

After the LSH model has been built, we can enter search terms and find matching values.  In this example, I have used the search terms: high, heat, and metal. 

![LSH4](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/searchTerms.png "lsh4")

### LSH - Results

In the results image, you can see the documents that are the most common based on the search terms that were entered. 

![LSH3](https://github.com/BrooksIan/SBIR_TFIDF_KMeans/blob/master/LSH_Resutls.png "lsh3")

## Additional Information Links
* [Apache Spark](https://spark.apache.org/)
* [Term Frequency–Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* [PySpark: CountVectorizer|HashingTF](https://towardsdatascience.com/countvectorizer-hashingtf-e66f169e2d4e)
* [A Friendly Introduction to Text Clustering](https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04)
* [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
* [Text clustering with K-means and tf-idf](https://medium.com/@MSalnikov/text-clustering-with-k-means-and-tf-idf-f099bcf95183)
* [Spark API- Feature Extractors](https://spark.apache.org/docs/2.2.3/ml-features.html#countvectorizer)
* [Locality-Sensitive Hashing - LSH](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
* [Scalable Jaccard similarity using MinHash and Spark](https://towardsdatascience.com/scalable-jaccard-similarity-using-minhash-and-spark-85d00a007c5e)
* [Spark - MinHash for Jaccard Distance](https://george-jen.gitbook.io/data-science-and-apache-spark/minhash-for-jaccard-distance)
* [Detecting Abuse at Scale: Locality Sensitive Hashing at Uber Engineering](https://databricks.com/blog/2017/05/09/detecting-abuse-scale-locality-sensitive-hashing-uber-engineering.html)
* [Read multiline JSON in Apache Spark](https://stackoverflow.com/questions/38545850/read-multiline-json-in-apache-spark)
* [Effectively Pre-processing the Text Data Part 1: Text Cleaning](https://towardsdatascience.com/effectively-pre-processing-the-text-data-part-1-text-cleaning-9ecae119cb3e)
