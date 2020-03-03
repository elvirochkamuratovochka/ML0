 line <- function(M)
{
	determ <-det(M)
	
	a <- M[2,2]/determ
	b <- -M[2,1]/determ
	c <- -M[1,2]/determ
	d <- M[1,1]/determ
	
	m1 <- 0
	m2 <- 0
  
	x <- seq(-3, 3, 0.1)
	y <- seq(-3, 3, 0.1)
	
	A <- a
	B <- d
	C <- b+c
	D <- -2*m1*a-b*m1-c*m1
	E <- m1*b-y*m1*c-2*m2*d
	F <- a*m1^2+b*m1*m2+m1*m2*c+m2^2*d
	
	func <- function(x, y) {
    	1/(2*pi*sqrt(determ))*exp((-1/2)*(x^2*A + y^2*B + x*y*C + x*D + y*E + F))
	}
	z <- outer(x, y, func)
	contour(x, y, z)
}

line(matrix(c(1,1,0,1),2,2))
