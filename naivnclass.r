objectCounter <- 500

get_mu <- function(objects) {
  rows <- dim(objects)[1]
  cols <- dim(objects)[2]
  mu <- matrix(NA, 1, cols)
  for (col in 1:cols) {
    mu[1, col] = mean(objects[ ,col])
  }
  return(mu)
}

# Covariation matrix of normal distribution
get_matrix <- function(objects, mu) {
  rows <- dim(objects)[1]
  cols <- dim(objects)[2]
  sigma <- matrix(0, cols, cols)
  for (i in 1:rows) {
    sigma <- sigma + (t(objects[i, ] - mu) %*% (objects[i, ] - mu)) / (rows - 1)
  }
  return(sigma)
}

objects_count <- 300

# Generation of test data
n <- 300
sigma1 <- matrix(c(2, 0, 0, 2),2,2)
sigma2 <- matrix(c(1, 0, 0, 1),2,2)

mu1 <- c(0, 0)
mu2 <- c(5, 5)
xy1 <- mvrnorm(n = objects_count, mu1, sigma1)
xy2 <- mvrnorm(n = objects_count, mu2, sigma2)

xl <- rbind(cbind(xy1, 1), cbind(xy2, 2))

# Ðèñóåì îáó÷àþùóþ âûáîðêó
colors <- c("blue2", "green3")
plot(xl[ , 1], xl[ , 2], pch = 21, bg = colors[xl[ ,3]], asp = 1, xlab = "x", ylab = "y")

objects_first <- xl[xl[,3] == 1, 1:2]
objects_second <- xl[xl[,3] == 2, 1:2]
mu1 <- get_mu(objects_first)
mu2 <- get_mu(objects_second)
sigma1 <- get_matrix(objects_first, mu1)
sigma2 <- get_matrix(objects_second, mu2)

sigma <- rbind(sigma1,sigma2)

mu <- rbind(mu1,mu2)


colors <- c("green3", "blue")
plot(xl[,1],xl[,2], pch = 21,main = "Íàèâíûé áàéåñîâñêèé êëàññèôèêàòîð", col = colors[xl[,3]], asp = 1, bg=colors[xl[,3]])

naiv <- function(x,mus,sigmas,lambda,Py)
{
  n <- 2
  p <- rep(0,n)
  for(i in 1:n)
  {
    sigma <- sigmas[i]
    mu <- matrix(c(mus[i,1],mus[i,2]),1,2)
    pyj <- (1/(sqrt(2*pi*sigma^2)))*exp(-((x-mu)^2)/(2*sigma^2))
    p[i] <- log(lambda*Py)+log(pyj[1,1])+log(pyj[1,2])
    
  }
  if(p[1] > p[2])
  {
    class <- colors[1]
  }
  else
  {
    class <- colors[2]
  }
  return(class)
}

x <- -10
while(x < 40)
{
  y <- -10
  while(y < 40)
  {
    xy <- c(x,y)
    c <- naiv(xy,mu,sigma,lambda=1,Py=0.5)
    points(xy[1],xy[2], col=c)
    y <- y+0.5
  }
  x <- x+0.5
}
