from pyspark.sql import SparkSession
from pyspark.sql import types
from pyspark.sql import DataFrame

spark  = SparkSession.builder \
    .appName("SBIR") \
    .enableHiveSupport() \
    .config("spark.hadoop.yarn.resourcemanager.principal", "ibrooks") \
    .config("spark.sql.warehouse.dir", "target/spark-warehouse") \
    .getOrCreate()
    
spark.version

df_WholeSetRaw = spark.read.option("multiline", "true").json("sbir-search-results*.json")

#df_WholeSetRaw = spark.read.option("multiLine", "true").option("mode", "PERMISSIVE").json("/home/cdsw/sbir-search-results1.json")

df_WholeSetRaw.cache()

#Create Table from DataFrame
#df_WholeSetRaw.createOrReplaceTempView("SBIR2018")

#Display resulting Infered schema 
df_WholeSetRaw.printSchema()
df_WholeSetRaw.take(1)

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, CountVectorizer, CountVectorizerModel
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

tokenizer = Tokenizer(inputCol="abstract", outputCol="words")
tokenized = tokenizer.transform(df_WholeSetRaw)

regexTokenizer = RegexTokenizer(inputCol="abstract", outputCol="words", pattern="\\\W+")
# alternatively, pattern="\\w+", gaps(False)
regexTokenized = regexTokenizer.transform(df_WholeSetRaw)

countTokens = udf(lambda words: len(words), IntegerType())

#tokenized.select("abstract", "words")\
#    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

regexTokenized.select("abstract", "words") \
    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)
  
  
#TFtokenizer = RegexTokenizer(inputCol="abstract", outputCol="words", pattern="\\W+").setGaps("false")

#TFwordsData = TFtokenizer.transform(df_WholeSetRaw)

cvModel = CountVectorizer(inputCol="words", outputCol="rawFeatures", minDF=4,  vocabSize=100000).fit(tokenized)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=400)
  
TFfeaturizedData = cvModel.transform(regexTokenized)
TFfeaturizedData = hashingTF.transform(regexTokenized)

idf = IDF(inputCol="rawFeatures", outputCol="features")

idfModel = idf.fit(TFfeaturizedData)
rescaledData = idfModel.transform(TFfeaturizedData)
rescaledData.select("abstract","features").show()

#rescaledData.select("features").createOrReplaceTempView("TFIDF")
#rescaledData.show(truncate=False)

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import lit

#from pyspark.mllib.linagl import Vectors

NumberOfClusters = 128

#Create a new feature vector to include a column for furure prediction 
#DF_TestSet = rescaledData.withColumn("prediction", lit('0').cast('int'))

#Create the K-Means model
kmeans = KMeans().setK(NumberOfClusters).setSeed(1).setFeaturesCol("features").setPredictionCol("prediction")

#Train the K-Means model with feature vector
model = kmeans.fit(rescaledData)

# Make predictions
predictions = model.transform(rescaledData)
# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
    
print("Documents by Cluster")
predictions.select("award_title","prediction").show()

predictions.filter("prediction =13").select("agency","award_title","research_keywords").show()


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinHashLSH

vocabSize = 10000

mh = MinHashLSH().setNumHashTables(vocabSize).setInputCol("rawFeatures").setOutputCol("hashValues")
MHmodel = mh.fit(rescaledData)
MHmodel.transform(rescaledData).show()

keyVal1 = cvModel.vocabulary.index("high")
keyVal2 = cvModel.vocabulary.index("heat")
keyVal3 = cvModel.vocabulary.index("metal")


One_key = Vectors.sparse(vocabSize, [200], [1.0])
Two_key = Vectors.sparse(vocabSize, [200, 398], [1.0, 1.0])
Three_key = Vectors.sparse(vocabSize, [24, 200, 398], [1.0, 1.0, 1.0])

k = 40

DF_NNMatched = MHmodel.approxNearestNeighbors(rescaledData, Three_key, k)
DF_NNMatched.cache()
DF_NNMatched.select("agency","award_title","research_keywords","distCol").show() #truncate=False
#DF_NNMatched.select("award_title","distCol").write.json("/tmp/1LSHResults.json")

