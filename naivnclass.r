library(shiny)
library(MASS)

date <- read.csv("filename.csv")
ui <- fluidPage(
  titlePanel("Изменяемые параметры"),
  sidebarLayout(
    sidebarPanel(
      # checkboxInput("class","Отобразить классификацию", FALSE),
      # numericInput("NumberOfSamples", "Кол-во элементов:", 150,min = 1,max=500, width = '200px'),
      numericInput("mu1", "μ для первого класса", 1,min=1,max=10, width = '200px'),
      numericInput("mu2", "μ для второго класса", 4,min=1,max=10, width = '200px'),
      # checkboxInput("vib","Выборка не меняется", FALSE),
      numericInput("sigma1", "Элементы ковариационной матрицы для первого класса", 2,min=1,max=20, width = '400px'),
      numericInput("sigma2", "Элементы ковариационной матрицы для второго класса", 2,min=1,max=20, width = '400px'),
      numericInput("lmd1","Задайте степень важности для первого класса", 1,min=1,max=10, width = '400px'),
      numericInput("lmd2","Задайте степень важности для второго класса", 1,min=1,max=10, width = '400px'),
      sliderInput("p1",
                  "Задайте априорную вероятноть для первого класса",
                  min = 0,
                  max = 1,
                  value = 0.5,
                  step = 0.1         
      ),
      sliderInput("p2",
                  "Задайте априорную вероятноть для второго класса",
                  min = 0,
                  max = 1,
                  value = 0.5,
                  step = 0.1         
      )
    ),  
    # Show a plot of the generated distribution
      mainPanel(
        HTML("<center><h1><b>Наивный нормальный байесовский классификатор</b></h1>"),
        h3(textOutput("label")),
        # textOutput(outputId = "covMessage22"),
        plotOutput(outputId = "plot", height = "700px")
    )
  )
)

naivBaysDistribution <- function(x, mu, sigma, lamda, P){
  n <- 2
  res <- log(lamda*P)
  for(i in 1 : n){
    f<- (1/(sigma[i]*sqrt(2*pi))) * exp(-1 * ((x[i] - mu[i])^2)/(2*sigma[i]^2))
    res <- res + log(f)
  }
  
  return(res)
}

getMu <- function(xl){
  
  l <- dim(xl)[1] 
  return(c(sum(xl[,1])/l, sum(xl[,2])/l))
  
}

getSigma <- function(xl, mu){
  
  l <- dim(xl)[1] 
  return(c(sum((xl[,1] - mu[1])^2)/l, sum((xl[,2] - mu[2])^2)/l))
  
}
naivBaysAlgo <- function(x,lamda, mu, sigma,p){
  n <- 2
  class <- 0
  i <- 1
  if(naivBaysDistribution(x, mu[1:2], sigma[1:2], lamda[1], p[1])>naivBaysDistribution(x, mu[3:4], sigma[3:4], lamda[2], p[2] ))
   class <- 1
  else 
   class <- 2
  
  return (class)
}

server <- function(input, output) {
  
output$plot = renderPlot ({
    s1 <- input$sigma1
    s2 <- input$sigma2
  
    sigma1 <- matrix(c(s1, 0, 0, s1),2,2)
    sigma2 <- matrix(c(s2, 0, 0, s2),2,2)
  
    mu1 <- c(input$mu1,input$mu1)
    mu2 <- c(input$mu2,input$mu2)
  
    x1 <- mvrnorm(n = 150, mu1, sigma1)
    x2 <- mvrnorm(n = 150, mu2, sigma2)
    xy1 <- cbind(x1,1) 
    xy2 <- cbind(x2,2) 
    
    xl <- date # rbind(xy1,xy2)

    print(xl[])
    # write.csv(xl[], "filename.csv")
    colors <- c("#FF6660", "#333399")
    plot(xl[,1],xl[,2],xlab = "Первый признак",ylab = "Второй признак" ,pch = 20, col = colors[xl[,3]], asp = 1, bg=colors[xl[,3]])
    mu1 <- getMu(x1)
    mu2 <- getMu(x2)     
  
    sigma1 <- getSigma(x1, mu1)
    sigma2 <- getSigma(x2, mu2)
    
    print(mu1)
    print(mu2)
    output$covMessage12 = renderText({
       paste(mu1,sep=" ")
    })
   
    output$covMessage22 = renderText({
      paste(mu2,sep=" ")
    })
    
    lmd1 <- input$lmd1
    p1 <- input$p1
    lmd2 <- input$lmd2
    p2 <- input$p2
   
    s <- c(sigma1,sigma2)
    p <- c(p1,p2)
    l <- c(lmd1,lmd2)
    m <-c(mu1,mu2)
   
   # lamda <- 1 #input$lmd
   # p <-  0.5 #input$p
   
    print(p)
    x1 <- -14;
    while(x1 < 20){
      x2 <- -8;
     
      while(x2 < 13){          
        xl<-c(x1,x2)
        class <- naivBaysAlgo(xl,l, m, s,p)
       
        points(x1, x2, pch = 21, col=colors[class], asp = 1)
        x2 <- x2 + 0.2
      }
     x1 <- x1 + 0.2
   }
  
  })
}
shinyApp(ui = ui, server = server)