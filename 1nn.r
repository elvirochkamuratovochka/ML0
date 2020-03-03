colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col = colors[iris$Species], asp = 1, main ="1NN")

euclideanDistance <- function(u, v)
{
 sqrt(sum((u - v)^2))
}

sortObjectsByDist <- function(xl, z, metricFunction = euclideanDistance)
{
 l <- dim(xl)[1]
 n <- dim(xl)[2] - 1

 min <- c(1, metricFunction(xl[1, 1:n], z))

 for (i in 2:l)
 {
	dist <- c(i,metricFunction(xl[i, 1:n], z))
 	if (min[2]>dist[2]){
		min <- dist
		}
 }
 
 return (min);
}


z <- c(4, 2)
xl <- iris[, 3:5]
n <- dim(xl)[2] - 1
class <- xl[sortObjectsByDist(xl,z)[1], n + 1]
points(z[1], z[2], pch = 22, bg = colors[class], asp = 1)