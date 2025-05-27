# Step 1: Load the dataset
getwd()  # Check current working directory
setwd("E:/UNICAS Materials/DataAnalytics/Assignment")  # Set working directory to dataset location

# Load required libraries
library(dplyr)        # Data manipulation
library(skimr)        # Enhanced summary stats
library(factoextra)   # Clustering and PCA visualization
library(ggplot2)      # Plotting

# Load the data
data <- read.table("ecoli.data", header = FALSE, sep = "", strip.white = TRUE)
View(data)

# Assign column names
colnames(data) <- c("SequenceName", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "Class")

# Remove the class label (not used for clustering)
data <- data[, -ncol(data)]

# Step 2: Provide the summary
summary(data)     # Basic summary statistics
skim(data)        # Detailed summary including missing values and distribution

# Step 3: Show the first 8 observations
head(data, n = 8)

# Step 4: Check for missing values
is.na(data)         # Check NA values across data
sum(is.na(data))    # Total count of missing values

# Step 5: Select only numerical variables
d_numeric <- data[sapply(data, is.numeric)]  # Keep only numeric columns
View(d_numeric)  # Optional: View numeric data

# Step 6: Normalize data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))  # Min-max normalization
}
data_numeric_normalized <- as.data.frame(lapply(d_numeric, normalize))
View(data_numeric_normalized)

# Step 7: Find the optimal number of clusters using the Elbow Method
data_scale <- scale(d_numeric)  # Standardize data
View(data_scale)  # Optional: View scaled data
dist_data <- dist(data_scale)  # Compute Euclidean distance matrix

fviz_nbclust(data_scale, kmeans, method = "wss") +  # Elbow method to find best k
  labs(subtitle = "ELBOW METHOD")

# Step 8: Apply k-means and Hierarchical Clustering
km.out <- kmeans(data_scale, centers = 4, nstart = 25, iter.max = 30)  # K-means clustering
km.clusters <- km.out$cluster  # Extract cluster assignments
km.out
km.clusters

fviz_cluster(list(data = data_scale, cluster = km.clusters)) +  # Visualize K-means
  labs(subtitle = "k-means Clustering")

hc.out <- hclust(dist_data, method = "complete")  # Hierarchical clustering
plot(hc.out)  # Dendrogram
rect.hclust(hc.out, k = 4, border = 2:5)  # Highlight clusters on dendrogram
hc.clusters <- cutree(hc.out, k = 4)  # Get hierarchical cluster assignments
hc.out
hc.clusters

fviz_cluster(list(data = data_scale, cluster = hc.clusters)) +  # Visualize Hierarchical
  labs(subtitle = "Hierarchical Clustering")

# Step 9: PCA and visualization
pca_result <- prcomp(data_scale, scale = FALSE)  # Run PCA
summary(pca_result)  # View PCA summary

# PCA with K-means clusters
fviz_pca_biplot(pca_result,
                geom = "point",
                habillage = km.clusters,
                addEllipses = TRUE,
                ellipse.level = 0.95,
                repel = TRUE) +
  labs(subtitle = "PCA with K-means clusters")

# PCA with Hierarchical clusters
fviz_pca_biplot(pca_result,
                geom = "point",
                habillage = hc.clusters,
                addEllipses = TRUE,
                ellipse.level = 0.95,
                repel = TRUE) +
  labs(subtitle = "PCA with Hierarchical Clustering")

# Step 10: Compare clustering results
clustering_comparison <- table(km.clusters, hc.clusters)  # Contingency table
print(clustering_comparison)  # Show how K-means and Hierarchical clusters match
