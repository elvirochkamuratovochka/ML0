
* [Метрические методы классификации](https://github.com/temirkayaeva/ML0#метрические-методы-классификации)
    * [Алгоритм 1NN](https://github.com/temirkayaeva/ML0#алгоритм-1nn)
    * [Алгоритм KNN](https://github.com/temirkayaeva/ML0#алгоритм-knn)
    * [Алгоритм KwNN](https://github.com/temirkayaeva/ML0#алгоритм-kwnn)
    * [Метод парзеновского окна](https://github.com/temirkayaeva/ML0/#метод-парзеновского-окна)
    * [Метод потенциальных функций](https://github.com/temirkayaeva/ML0/#метод-потенциальных-функций)
    * [Алгоритм STOLP](https://github.com/temirkayaeva/ML0/#алгоритм-stolp)
* [Байесовские алгоритмы классификации](https://github.com/temirkayaeva/ML0#байесовские-алгоритмы-классификации)
    * [Нормальный дискриминантный анализ](https://github.com/temirkayaeva/ML0#нормальный-дискриминантный-анализ)
    * [Наивный	нормальный байесовский	классификатор](https://github.com/temirkayaeva/ML0#наивныйнормальный-байесовскийклассификатор)
    * [Подстановочный алгоритм](https://github.com/temirkayaeva/ML0#подстановочный-алгоритм)
    * [Линейный дискриминант Фишера](https://github.com/temirkayaeva/ML0#линейный-дискриминант-фишера)
* [Линейные алгоритмы классификации](https://github.com/temirkayaeva/ML0#линейные-алгоритмы-классификации)
    * [Метод стохастического градианта](https://github.com/temirkayaeva/ML0#метод-стохастического-градианта)
    * [Адаптивный линейный элемент](https://github.com/temirkayaeva/ML0#адаптивный-линейный-элемент)
    * [Персептрон Розенблатта](https://github.com/temirkayaeva/ML0#персептрон-розенблатта)
    * [Логистическая регрессия](https://github.com/temirkayaeva/ML0#логистическая-регрессия)

# Метрические методы классификации

Методы обучения, основанные на анализе сходства объектов,  будем называть ***метрическими***.

*Метрический алгоритм классификации* с обучающей выборкой Xl относит объект u к тому классу y ∈ Y , для которого суммарный вес ближайших обучающих объектов Γy(u, Xl) максимален: <img src="https://github.com/temirkayaeva/ML0/raw/master/images/2.png" width="500"> где весовая функция <img src="https://github.com/temirkayaeva/ML0/raw/master/images/3.png" width="45">  оценивает степень важности *i*-го соседа для классификации объекта *u*. Обычно весовая функция <img src="https://github.com/temirkayaeva/ML0/raw/master/images/3.png" width="45">  — нертрицательная, невозврастающая по *i*, что соответствует **гипотезе компактности** (схожим объектам чаще соответствуют схожие ответы). 

 
## Алгоритм 1NN

**Алгоритм ближайшего соседа - 1NN** (nearest neighbor, NN)  является самым простым алгоритмом классификации. Он относит классифицируемый объект <img src="https://github.com/temirkayaeva/ML0/raw/master/images/4.png" width="45"> к тому классу, которому принадлежит ближайший обучающий объект: <img src="https://github.com/temirkayaeva/ML0/raw/master/images/5.png" width="100">

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/1nn1.png" width="900"> 
  
#### Достоинства метода

* Простота реализации.

#### Недостатки метода

* Неустойчивость к погрешностям (шуму, выбросам).
* Отсутствие настраиваемых параметров.
* Низкое качество классификации.
* Приходится хранить всю выборку целиком.

## Алгоритм KNN

**В алгоритме k ближайших соседей - KNN** (k nearest neighbors) объекты классифицируются  путем *голосования* по *k* ближайшим соседям. Каждый из соседей <img src="https://github.com/temirkayaeva/ML0/raw/master/images/knn1.png" width="120">  голосует за отнесение
объекта <img src="https://github.com/temirkayaeva/ML0/raw/master/images/knn2.png" width="15">  к своему классу <img src="https://github.com/temirkayaeva/ML0/raw/master/images/knn3.png" width="19">. Алгоритм относит объект  <img src="https://github.com/temirkayaeva/ML0/raw/master/images/knn2.png" width="15">  к тому классу, который
наберёт большее число голосов:
<img src="https://github.com/temirkayaeva/ML0/raw/master/images/knn4.png" width="350"> 

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/knn.png" width="900"> 

### Оптимизация числа соседей k

Оптимальное значение параметра *k* определяют по критерию скользящего контроля с *исключением объектов по одному* (leave-one-out, LOO).
 Для каждого объекта <img src="https://github.com/temirkayaeva/ML0/raw/master/images/loo1.png" width="70">  проверяется,
правильно ли он классифицируется по своим *k* ближайшим соседям.

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/loo2.png" width="400"> 

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/looo1.png" width="900">

Оптимальное *k* = 6.

#### Достоинства метода

* Простота реализации.

#### Недостатки метода

* При k = 1 неустойчивость к погрешностям. Если среди обучающих объектов *выброс* - объект, находящийся в окружении объектов чужого класса, то не только он сам будет классифицирован неверно, но и те окружающие его объекты, для которых он окажется ближайшим.

* При k = l алгоритм наоборот чрезмерно устойчив и вырождается в константу.
 
* Бедный набор параметров.

*  Максимальная сумма голосов может достигаться на нескольких классах одновременно.

## Алгоритм k взвешенных ближайших соседей

В данном алгоритме вводится строго убывающая последовательность вещественных весов <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kwnn1.png" width="19">,  задающих вклад i-го соседа в классификацию:

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/kwnn2.png" width="350"> 

#### Выбор последовательности

* <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kwnn3.png" width="90"> — линейно убывающие веса; при данном выборе последовательности неоднозначности также могут возникать (например: классов два; первый и четвёртый сосед голосуют за класс 1, второй и третий — за класс 2; суммы голосов совпадают).

* <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kwnn4.png" width="100"> —  экспоненциально убывающие веса (геометрическая прогрессия), *q* — параметр алгоритма. Его можно подбирать по критерию LOO, аналогично числу соседей k.

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/lookwnn.png" width="900"> 
Оптимальное q = 1, k = 6. 

### Сравнение методов KNN и KWNN

| Карта классификации KNN  | Карта классификации KWNN |
| ------------- | ------------- |
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/knnmap.png" width="400">  | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kwnn.png" width="400">  |

Из-за того, что алгоритм k взвешенных ближайших соседей учитывает порядок объектов при классификации, он выдаёт лучший результат, чем алгоритм k ближайших соседей (kNN). 

**Пример, показывающий	преимущество	метода kwNN над KNN:**
<img src="https://github.com/temirkayaeva/ML0/raw/master/images/6nn.png" width="420"><img src="https://github.com/temirkayaeva/ML0/raw/master/images/6wnn.png" width="420">  

## Метод парзеновского окна

Ещё один способ задать веса соседям — определить  <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kwnn1.png" width="19"> как функцию  не от ранга соседа *i*, а как функцию от расстояния <img src="https://github.com/temirkayaeva/ML0/raw/master/images/okno1.png" width="60">. Для этого вводится  функция ядра  <img src="https://github.com/temirkayaeva/ML0/raw/master/images/okno2.png" width="28"> невозрастающую на <img src="https://github.com/temirkayaeva/ML0/raw/master/images/okno3.png" width="28"> и рассматривается алгоритм: 

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/okno4.png" width="350"> 

Параметр *h* называется шириной окна и играет примерно ту же роль, что и число соседей *k*. "Окно" — это сферическая окрестность объекта *u* радиуса *h*, при попадании в которую обучающий объект <img src="https://github.com/temirkayaeva/ML0/raw/master/images/loo1.png" width="60"> "голосует" за отношение объекта *u* к классу <img src="https://github.com/temirkayaeva/ML0/raw/master/images/okno6.png" width="16">. Параметр *h* можно задавать или определять по скользящему контролю (LOO). 

Обучающие объекты  могут быть неравномерно распределены по пространству *X*. В окрестности одних объектов может оказываться очень много соседей, а в окрестности других — ни одного. В этих случаях применяется *окно переменной ширины*: 

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/okno5.png" width="350"> 

**Виды ядер** 

| Ядро <img src="https://github.com/temirkayaeva/ML0/raw/master/images/k0.gif" width="30"> | Формула |
| ------------- | ------------- |
| Епанечникова	 | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/k1.gif" width="200">  |
| Квартическое	 | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/k2.gif" width="200">  |
| Треугольное	 | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/k3.gif" width="180">  |
| Гауссовское	 | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/k4.gif" width="180">  |
| Прямоугольное	 | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/k5.gif" width="150">  |

| Графики LOO  | Карты классификаций |
| ------------- | ------------- |
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kerep.png" width="350">|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/mapkerep.png" width="540">
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kerq.png" width="350">|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/mapkerq.png" width="540">
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kerr.png" width="350">|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/mapkerr.png" width="540">
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kert.png" width="350">|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/mapkert.png" width="540">
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/kerg.png" width="350">|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/mapkerg.png" width="540">

### Сравнение методов

На примере видно, что гауссовское ядро имеет преимущество, так как классифицирует все точки. В остальных ядрах точки, не попавшие в окна, не классифицируются (на картинках они имеют серый цвет).



## Метод потенциальных функций

Ядро помещается в каждый обучающий объект <img src="https://github.com/temirkayaeva/ML0/raw/master/images/loo1.png" width="60">  и "притягивает" объект *u* к классу <img src="https://github.com/temirkayaeva/ML0/raw/master/images/okno6.png" width="16">, если он попадает в его окрестность радиуса <img src="https://github.com/temirkayaeva/ML0/raw/master/images/pfunctions1.png" width="15">:

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/pfunctions.png" width="400"> 

**Идея метода**: если обучающий объект  <img src="https://github.com/temirkayaeva/ML0/raw/master/images/pfunctions2.png" width="16"> классифицируется неверно, то потенциал класса <img src="https://github.com/temirkayaeva/ML0/raw/master/images/okno6.png" width="16"> недостаточен в точке <img src="https://github.com/temirkayaeva/ML0/raw/master/images/pfunctions2.png" width="16">, и вес <img src="https://github.com/temirkayaeva/ML0/raw/master/images/pfunctions3.png" width="16"> увеличивается на единицу. 


<img src="https://github.com/temirkayaeva/ML0/raw/master/images/pot.png" width="900">

| Ядро Епанечникова | Ядро Треугольное |
| ------------- | ------------- |
|Ядро Прямоугольное| Ядро Гауссовское|


#### Достоинства метода

* Эффективность (когда обучающие объекты поступают потоком, и хранить их в памяти нет возможности или необходимости)

#### Недостатки метода

* Очень медленно сходится

* Результат обучения зависит от порядка предъявления объектов

*  Слишком грубо (с шагом 1) настраиваются веса <img src="https://github.com/temirkayaeva/ML0/raw/master/images/pfunctions3.png" width="16"> 

* Не настраиваются параметры <img src="https://github.com/temirkayaeva/ML0/raw/master/images/pfunctions1.png" width="15">

Следовательно,  данный алгоритм не может похвастаться высоким качеством классификации.

## Алгоритм STOLP

Отступом (margin) объекта x относительно алгоритма классификации называется величина

<img src="http://1.618034.com/blog_data/math/formula.57916.png" width="400">

Отступ показывает степень типичности объекта (насколько глубоко объект погружён в свой класс). Отступ отрицателен тогда и только тогда, когда алгоритм допускает ошибку на данном объекте.

В зависимости от значений отступа обучающие объекты условно делятся на пять типов, в порядке убывания отступа: эталонные, неинформативные, пограничные, ошибочные, шумовые.

* *Эталонные* объекты имеют большой положительный отступ, плотно окружены объектами своего класса и являются наиболее типичными его представителями.
* *Неинформативные* объекты также имеют положительный отступ. Изъятие этих объектов из выборки (при условии, что эталонные объекты остаются), не влияет на качество классификации. Фактически, они не добавляют к эталонам никакой новой информации.
* *Пограничные* объекты имеют отступ, близкий к нулю. Классификация таких объектов неустойчива в том смысле, что малые изменения метрики или состава обучающей выборки могут изменять их классификацию.
* *Ошибочные* объекты имеют отрицательные отступы и классифицируются неверно.
* *Шумовые* объекты или выбросы — это небольшое число объектов с большими отрицательными отступами. Они плотно окружены объектами чужих классов и классифицируются неверно.

В алгоритме STOLP реализована идея отбора эталонных объектов.  Обозначим через <img src="http://1.618034.com/blog_data/math/formula.57917.png" width="70">  отступ объекта xi относительно алгоритма <img src="http://1.618034.com/blog_data/math/formula.57918.png" width="70">. Большой отрицательный отступ свидетельствует о том, что объект xi окружён объектами чужих классов, следовательно, является выбросом. Большой положительный отступ означает, что объект окружён объектами своего класса, то есть является либо эталонным.

**Алгоритм STOLP:**

0. Из выборки Xℓ исключить все объекты xi с отступом меньшим заданного порога δ.
0. В Ω добавить по одному наиболее типичному представителю (с наибольшим отступом) от каждого класса. 
0. Пока <img src="http://1.618034.com/blog_data/math/formula.57920.png" width="70">, выполнять: 
* выделить множество объектов, на которых алгоритм ошибается, посчитать количество элементов в нем;
* если количество ошибок меньше  заданной допустимой доли ошибок, то алгоритм останавливается;
* к Ω присоединить объект xi, имеющий минимальное значение отступа. 

В результате каждый класс будет представлен в Ω одним эталонным объектом и массой пограничных объектов, на которых отступ принимал наименьшие значения.

**Результат работы алгоритма для kNN (k=6)** 

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/stolp1.png" width="1000">
<img src="https://github.com/temirkayaeva/ML0/raw/master/images/stolp.png" width="1000">

# Байесовские алгоритмы классификации

Байесовский подход основан на теореме, утверждающей, что если плотности распределения каждого из классов известны, то искомый алгоритм можно выписать в явном аналитическом виде. На практике плотности распределения классов, как правило, не известны.
Их приходится оценивать (восстанавливать) по обучающей выборке. Чтобы классифицировать точку, для начала, нужно вычислить функции правдоподобия каждого из классов, затем вычислить апостериорные вероятности классов. Классифицируемый объект относится к тому классу, у которого апостериорная вероятность максимальна.

**Оптимальное байесовское решающее правило**: 
<img src="https://github.com/temirkayaeva/ML0/raw/master/images/baes.png" width="190">, где
* <img src="https://github.com/temirkayaeva/ML0/raw/master/images/baes1.png" width="30">  - априорные вероятности  классов (вероятности появления объектов каждого из классов)
* <img src="https://github.com/temirkayaeva/ML0/raw/master/images/baes2.png" width="35"> - функции правдоподобия классов (плотности распределения классов)
* <img src="https://github.com/temirkayaeva/ML0/raw/master/images/baes3.png" width="30"> - величина потери.

## Нормальный дискриминантный анализ

*Вероятностное распределение с плотностью* 
<img src="http://1.618034.com/blog_data/math/formula.48016.png" width="450"> 
называется n-мерным нормальным (гауссовским) распределением с вектором матожидания (центром)  <img src="http://1.618034.com/blog_data/math/formula.48017.png" width="70">  и ковариационной матрицей  <img src="http://1.618034.com/blog_data/math/formula.48019.png" width="70">. Матрица Σ симметричная, невырожденная и положительно определённая. 

**Геометрическая интерпретация нормальной плотности**. 

* Если признаки некоррелированы, <img src="http://1.618034.com/blog_data/math/formula.48021.png" width="150">  то линии уровня плотности распределения имеют форму эллипсоидов с центром <img src="http://1.618034.com/blog_data/math/formula.23640.png" width="20"> и осями, параллельными линиям координат.
* Если признаки имеют одинаковые дисперсии, <img src="http://1.618034.com/blog_data/math/formula.48022.png" width="70">, то эллипсоиды являются сферам. 
* Если признаки коррелированы, то матрица <img src="http://1.618034.com/blog_data/math/formula.29578.png" width="20">  не диагональна и линии уровня имеют форму эллипсоидов, оси которых повёрнуты относительно исходной системы координат. 

Получения коэффициентов дискриминантной функции: 

```R
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
```

|  признаки некоррелированы | признаки имеют одинаковые дисперсии | признаки коррелированы |
| ------------- | ------------- | ------------- | 
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/b2.png" width="280">|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/b1.png" width="280"> | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/b3.png" width="280">|

## Наивный	нормальный байесовский	классификатор
Допустим, что признаки <img src="http://1.618034.com/blog_data/math/formula.48025.png" width="90"> - независимые случайные величины с плотностями распределения, <img src="http://1.618034.com/blog_data/math/formula.48028.png" width="170">.

Тогда функции правдоподобия классов представимы в виде произведения одномерных плотностей по признакам: 
<img src="http://1.618034.com/blog_data/math/formula.48031.png" width="310">.

Вычислим эмпирическую плотность распределения и подставим ее в формулу оптимального байесовского классификатора (вместо истинной функции правдоподобия).  Априорную вероятность каждого из классов оценим, как долю объектов класса y ∈ Y в выборке. 
Получаем классификатор:

<img src="http://1.618034.com/blog_data/math/formula.48036.png" width="400">
<img src="http://1.618034.com/blog_data/math/formula.57969.png" width="500">

Предположение о независимости существенно упрощает задачу, так как восстановление n одномерных плотностей намного более простая задача,  чем одной n-мерной. 

```R
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
```

|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/naiv.png" width="600">|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/naive1.png" width="600">|
| ------------- | ------------- |

#### Достоинства

* Простота вычислений 
* Когда признаки действительно независимы, классификатор оптимален

#### Недостатки

* Низкое качество классификации в реальных задач

## Подстановочный алгоритм

Параметры функций правдоподобия  <img src="http://1.618034.com/blog_data/math/formula.48700.png" width="35"> и <img src="http://1.618034.com/blog_data/math/formula.48701.png" width="35"> можно оценить по частям обучающей выборки для каждого класса y отдельно.

<img src="http://1.618034.com/blog_data/math/formula.48704.png" width="400"> 

Полученные выборочные оценки непосредственно подставляются в формулу оптимального байесовского классификатора <img src="http://1.618034.com/blog_data/math/formula.48706.png" width="200">. В результате получается алгоритм классификации, который так и называется — подстановочным (plug-in).

```R
plugin <- function(x,mus,sigmas,lymda,P)
{
  n <- 2
  p <- rep(0,n)
  for(i in 1:n)
  {
    sigma <- matrix(c(sigmas[i*2-1,1],sigmas[i*2-1,2],sigmas[i*2,1],sigmas[i*2,2]),2,2)
    mu <- matrix(c(mus[i,1],mus[i,2]),1,2)
    determ <-det(sigma)
    a <- sigma[2,2]/determ
    b <- -sigma[2,1]/determ
    c <- -sigma[1,2]/determ
    d <- sigma[1,1]/determ
    
    F <-  - log(abs(det(sigma))) + mu[1]*mu[1]*a+(b+c)*mu[1]*mu[2]+d*mu[2]*mu[2]    
    A <- a
    B <- d
    C <- b+c
    D <- -2*mu[1]*a-2*mu[2]*b-mu[1]*c
    E <- -mu[1]*b-mu[1]*c-d*2*mu[2]
    
    func <- function(x, y) {
      f <- x^2*A + y^2*B + x*y*C + x*D + y*E + F
    }
    f <- func(x[1],x[2])
    p[i] <- log(lymda*P) - f
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
```
Разделяющая поверхность между двумя классами s и t задаётся следующим образом: 

<img src="http://1.618034.com/blog_data/math/formula.52395.png" width="200">

Так как <img src="http://1.618034.com/blog_data/math/formula.52396.png" width="200">, то логарифмируя плотности каждого класса, получим коэффициенты разделяющей кривой:

<img src="https://camo.githubusercontent.com/ede022e9653a71c319e7e10fa2bbe2df77a33eed/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f2535436c6e253742705f79253238782532392537442532302533442532302d253543667261632537426e253744253742322537442532302535436c6e2537423225354370692537442532302d2532302535436672616325374231253744253742322537442535436c6e2537422535436c6566742532302537432532302535435369676d615f7925323025354372696768742532302537432537442532302d253230253543667261632537423125374425374232253744253238782532302d2532302535436d755f79253239253545542532302535435369676d612535452537422d312537445f79253230253238782532302d2532302535436d755f79253239" width="500">

Получение коэффициентов подстановочного алгоритма:

```R
get_coeffs <- function(mu1, sigma1, mu2, sigma2) {
  # Line equation: a*x1^2 + b*x1*x2 + c*x2 + d*x1 + e*x2 + f = 0
  determ1 <-det(sigma1)
  determ2 <-det(sigma2)
  a <- sigma1[2,2]/determ1
  b <- -sigma1[2,1]/determ1
  c <- -sigma1[1,2]/determ1
  d <- sigma1[1,1]/determ1

  e <- sigma2[2,2]/determ2
  f <- -sigma2[2,1]/determ2
  m <- -sigma2[1,2]/determ2
  n <- sigma2[1,1]/determ2

  F <-  - log(abs(det(sigma1))) + log(abs(det(sigma2))) + mu1[1]*mu1[1]*a+(b+c)*mu1[1]*mu1[2]+d*mu1[2]*mu1[2]-mu2[1]*mu2[1]*e-(f+m)*mu2[1]*mu2[2]-mu2[2]*n
  A <- a-e
  B <- d-n
  C <- b+c+f+m
  D <- -2*mu1[1]*a-2*mu1[2]*b-mu1[1]*c+2*mu2[1]*e+f*mu2[1]+mu2[2]*m
  E <- -mu1[1]*b-mu1[1]*c-d*2*mu1[2]+f*mu2[1]+m*mu2[1]+2*mu2[2]*n
  return(c("x^2" = A, "y^2" = B, "xy" = C, "x" = D, "y" = E, "1" = F))
}
```

* Если классы равновероятны и равнозначны, ковариационные матрицы
равны, признаки некоррелированы и имеют одинаковые дисперсии, то разделяющая гиперплоскость проходит по середине между классами, ортогонально линии, соединяющей центры классов; классы имеют сферическую форму.

* Если классы равновероятны и равнозначны, ковариационные матрицы
равны, признаки некоррелированы и имеют одинаковые дисперсии, то разделяющая гиперплоскость проходит по середине между классами, ортогонально линии, соединяющей центры классов; классы имеют сферическую форму. 

* Если ковариационные матрицы не диагональны и не равны, то разделяющая поверхность становится квадратичной и «прогибается» так, чтобы менее плотный класс «охватывал» более плотный.

| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/12.png" width="600"> |  <img src="https://github.com/temirkayaeva/ML0/raw/master/images/16.png" width="600"> |
| ------------- | ------------- |
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/17.png" width="600"> | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/15.png" width="600"> |
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/photoeditorsdk-export.png" width="600"> | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/photoeditorsdk-export(1).png" width="600"> |


## Линейный дискриминант Фишера

Допустим, что ковариационные матрицы классов равными. В таком случае достаточно оценить только одну ковариационную матрицу Σ, задействовав для этого всю выборку целиком.  Оценка ковариацинной матрицы вычисляется по следующей формуле: 

<img src="http://1.618034.com/blog_data/math/formula.54530.png" width="300">

В данном случае разделяющая поверхность является линейной, а ее коэффициенты находятся из оптимального байесовского решающего правила: 

<img src="http://1.618034.com/blog_data/math/formula.54528.png" width="700">
Этот алгоритм называется линейным дискриминантом Фишера (ЛДФ). 

Разделяющая поверхность задается формулой: 
<img src="http://1.618034.com/blog_data/math/formula.54531.png" width="100">, коэффициенты находятся по следующим формулам: 

<img src="http://1.618034.com/blog_data/math/formula.54532.png" width="150">

<img src="http://1.618034.com/blog_data/math/formula.54533.png" width="170">

**Вероятность ошибки линейного дискриминанта Фишера** выражается через расстояние Махаланобиса между классами, в случае, когда классов два:

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/photoeditorsdk-export(4).png" width="500">



```R
ldf <- function(xy,m,s,lambda,Py)
{
  n <- 2
  p <- rep(0,n)
  for(i in 1:n) {
   mu <- matrix(c(m[i,1],m[i,2]),1,2)
   det <- det(sigma)
   a <- sigma[2,2]/det
   b <- -sigma[2,1]/det
   c <- -sigma[1,2]/det
   d <- sigma[1,1]/det

   A <- -2*mu[1]*a-mu[2]*b-mu[2]*c #x
   B <- -mu[1]*b-mu[2]*c-2*mu[2]*d #y
   C <- a*mu[1]^2 + mu[1]*mu[2]*b+ mu[1]*mu[2]*c+ d*mu[2]^2
    
    func <- function(x, y) {
      f<-x*A + y*B + C
    }
    f<-func(xy[1],xy[2])
    p[i] <- log(lambda*Py) - f
  }
  if(p[1] > p[2])
  {
    class<-colors[1]
  }
  else
  {
    class<-colors[2]
  }
  return(class)
}
```

|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/ldf1.png" width="600"> | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/ldf2.png" width="600"> |
| ------------- | ------------- |
|<img src="https://github.com/temirkayaeva/ML0/raw/master/images/photoeditorsdk-export(2).png" width="600"> | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/photoeditorsdk-export(3).png" width="600"> |
# Линейные методы классификации

Рассмотрим задачу классификации с двумя классами <img src="http://1.618034.com/blog_data/math/formula.56462.png" width="100">. Модель алгоритмов — параметрическое семейство отображений вида  <img src="http://1.618034.com/blog_data/math/formula.56464.png" width="200">. где w — вектор параметров. Функция f(x,w) называется дискриминантной функцией. Если f(x, w) > 0, то алгоритм a относит объект x к классу +1, иначе к классу −1. Уравнение f(x, w) = 0 задает разделяющую поверхность.

## Линейная модель классификации

Пусть <img src="http://1.618034.com/blog_data/math/formula.56466.png" width="200">. 

Если дискриминантная функция определяется как скалярное произведение вектора x и вектора пераметров w, то получается линейный
классификатор: 

 <img src="http://1.618034.com/blog_data/math/formula.56470.png" width="400">
 
 Уравнение  <img src="http://1.618034.com/blog_data/math/formula.56472.png" width="100"> задает гиперплоскость, разделяющую классы. Если вектор x находится по одну сторону гиперплоскости с ее направляющим вектором w, то объект x относится к классу +1, иначе — классу −1. 
 
 ## Метод стохастического градианта

Пусть задана обучающая выборка и требуется найти вектор параметров  w, при котором достигается минимум эмпирического риска:

<img src="https://camo.githubusercontent.com/212214890c628ad501d943664cdad24ffb1129df/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5125323877253243253230582535456c25323925323025334425323025354373756d5f25374269253230253344253230312537442535452537426c2537444c25323825334377253243253230785f69253345795f69253239" width="250">

Для этого применим **метод стохастического градиента**.В этом методе выбирается некоторое начальное приближение для w, затем запускается итерационный процесс, на каждом шаге которого вектор w изменяется в направлении наиболее быстрого убывания функционала Q. Это направление противоположно направлению вектора градиента

<img src="http://1.618034.com/blog_data/math/formula.56492.png" width="200">, <img src="http://1.618034.com/blog_data/math/formula.56493.png" width="150">, <img src="http://1.618034.com/blog_data/math/formula.56495.png" width="20"> - **темп обучения**. 

Предположим, что функция L дифференцируема, и выпишем градиент: 
<img src="http://1.618034.com/blog_data/math/formula.56499.png" width="350">.

Инициализация весов может производиться различными способами. Cтандратный способ: взять небольшие случайные значения.

<img src="http://1.618034.com/blog_data/math/formula.56501.png" width="300">.

**Алгоритм SG:**

**Вход**: <img src="http://1.618034.com/blog_data/math/formula.56503.png" width="25"> -  обуч. выборка, <img src="http://1.618034.com/blog_data/math/formula.56495.png" width="20"> - темп обучения, <img src="http://1.618034.com/blog_data/math/formula.13603.png" width="20"> - параметр сглаживания.

**Выход**: веса <img src="http://1.618034.com/blog_data/math/formula.56504.png" width="100">.

1. инициализировать веса <img src="http://1.618034.com/blog_data/math/formula.56504.png" width="100">.
2. инициализировать текущую оценку функционала: <img src="https://camo.githubusercontent.com/212214890c628ad501d943664cdad24ffb1129df/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5125323877253243253230582535456c25323925323025334425323025354373756d5f25374269253230253344253230312537442535452537426c2537444c25323825334377253243253230785f69253345795f69253239" width="250">
3. **повторять**
4. выбрать объект  <img src="http://1.618034.com/blog_data/math/formula.56505.png" width="100"> (например, случайным образом)
5. вычислить ошибку алгоритма <img src="http://1.618034.com/blog_data/math/formula.56507.png" width="150">  
6. сделать шаг градиентного спуcка <img src="http://1.618034.com/blog_data/math/formula.56499.png" width="350">
7. оценить новое значение функционала <img src="http://1.618034.com/blog_data/math/formula.56508.png" width="200">
8. **пока** значение Q не стабилизируется и/или веса w не перестанут изменятся

**В зависимости от функции потерь, которая используется в функционале эмпирического риска, будем получать различные линейные алгоритмы классификации** 


## Адаптивный линейный элемент

Возьмем квадратичную функцию потерь  <img src="http://1.618034.com/blog_data/math/formula.56509.png" width="200">, тогда 

<img src="http://1.618034.com/blog_data/math/formula.56510.png" width="400"> (производная берется по w).

 <img src="http://1.618034.com/blog_data/math/formula.56513.png" width="600">
 
 получим правило обновления весов на каждой итерации метода стохастического градиента: <img src="http://1.618034.com/blog_data/math/formula.56515.png" width="400">
 
 Это правило называется **дельта-правилом**, а сам линейный нейрон — **адаптивным линейным элементом (ADALINE)**. 
 
 Пример работы алгоритма: 
 
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/ada1.png" width="500"> | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/ada2.png" width="500"> |
| ------------- | ------------- |


## Персептрон Розенблатта

Будет считать, что признаки являются бинарными, классы могут принимать значения **+1** и **-1**. Тогда при классификации a(x) объекта x, возможны три случая:

* Если значение a(x) совпадает с y, тогда веса изменять не надо.

* Если значение а(х) = -1 и y = 1, то вектор весов увеличивается. 

* Если значение а(х) = 1 и y = -1, то вектор весов уменьшается. 

Эти три случая объединяются в **правило Хэбба**. Правило обновления весов принимает следующий вид: 

<img src="http://1.618034.com/blog_data/math/formula.56633.png" width="400"> 

Следовательно, в алгоритме стохастического градиента заменяется **шаг 6**.  В качестве функции потерь возьмем кусочно-линейную функцию.

Для данного правила доказывается теорема сходимости (Новикова), которая справедлива не только для бинарных, но и для произвольных действительных признаков


 Пример работы алгоритма: 
 
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/habbs.png" width="500"> | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/habbs1.png" width="500"> |
| ------------- | ------------- |


## Логистическая регрессия

Логистическая регрессия — линейный байесовский классификатор, использующий логарифмическую функцию потерь. Алгоритм способен помимо определения принадлежности объекта к классу определять и степень его принадлежности. 

<img src="http://1.618034.com/blog_data/math/formula.58051.png" width="400"> 

Для настройки вектора параметров **w** по выборке будем использовать принцип максимума правдоподобия:

<img src="http://1.618034.com/blog_data/math/formula.58060.png" width="500"> 

Так как <img src="http://1.618034.com/blog_data/math/formula.58051.png" width="400">, и p(x) не зависит от вектора **w**, а максимизация правдоподобия эквивалетна минимизации функционала эмпирического риска, получим следующую формулу: 

<img src="http://1.618034.com/blog_data/math/formula.58062.png" width="500"> 

Запишем градиент этого функционала, используя выражение для производной сигмоидной функции **σ'(z) = σ(z)σ(−z)** и получим новое правило обновления весов в методе SG:

<img src="http://1.618034.com/blog_data/math/formula.58064.png" width="400">

Логистическая функция потерь:

<img src="http://1.618034.com/blog_data/math/formula.58067.png" width="300">


**Достоинства:**

* Логистическая регрессия дает лучшие результаты по сравнению с линейным дискриминантом Фишера (поскольку она основана на менее жестких гипотезах), а также по сравнению с адалайном и правилом Хэбба (поскольку она использует *более правильную* функцию потерь).

* Возможность оценивать апостериорные вероятности и риски.

**Недостатки:**

* Градиентный метод обучения логистической регрессии наследует все недостатки метода стохастического градиента. Практичная реализация должна предусматривать стандартизацию данных, отсев выбросов, регуляризацию (сокращение весов), отбор признаков и т.п.. 

 Пример работы алгоритма: 
 
  
| <img src="https://github.com/temirkayaeva/ML0/raw/master/images/regress.png" width="500"> | <img src="https://github.com/temirkayaeva/ML0/raw/master/images/reg1.png" width="500"> |
| ------------- | ------------- |

### Сравнение методов

<img src="https://github.com/temirkayaeva/ML0/raw/master/images/all.png" width="600">
